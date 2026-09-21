import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_random_pivot_transfer_index(x, intervals, balanced=False):
    """Select one uniform pivot per half-open interval, independently per sample.

    Keep interval bookkeeping and position sampling on CPU to avoid a GPU
    synchronization for each pivot. Sampling follows torch.manual_seed().
    Children become eligible on the next forward pass.
    """
    transfer_index = torch.zeros_like(x, dtype=torch.bool)
    next_intervals = []
    for batch_index, sample_intervals in enumerate(intervals):
        pivots = []
        children = []
        for start, end in sample_intervals:
            trim = (end - start) // 4 if balanced else 0
            pivot = torch.randint(start + trim, end - trim, (), device='cpu').item()
            pivots.append(pivot)
            if start < pivot:
                children.append((start, pivot))
            if pivot + 1 < end:
                children.append((pivot + 1, end))
        if pivots:
            transfer_index[batch_index, pivots] = True
        next_intervals.append(children)
    return transfer_index, next_intervals


def select_confident_pivots(x, intervals, confidence, block_start, retries,
                           use_confidence=False, threshold=None, max_retries=3,
                           balanced=False):
    """Select/accept pivots, retaining a rejected interval without splitting it.

    confidence contains FP32 probabilities for predicted tokens in this block.
    Copy this small matrix once instead of synchronizing once per interval.
    Rejection counts belong to intervals; each accepted pivot's children start
    with zero rejections. A fallback selects the most confident position.
    """
    scores = confidence.detach().cpu()
    transfer_index = torch.zeros_like(x, dtype=torch.bool)
    next_intervals, next_retries = [], {}
    diagnostics = {name: [0] * x.shape[0] for name in ('deferred', 'fallback', 'forced')}
    for batch_index, sample_intervals in enumerate(intervals):
        children, pivots = [], []
        for start, end in sample_intervals:
            key = (batch_index, start, end)
            rejected = retries.get(key, 0)
            fallback = threshold is not None and max_retries is not None and rejected >= max_retries
            if use_confidence or fallback:
                local = scores[batch_index, start - block_start:end - block_start]
                pivot = start + int(local.argmax())
            else:
                trim = (end - start) // 4 if balanced else 0
                pivot = torch.randint(start + trim, end - trim, (), device='cpu').item()
            low = threshold is not None and bool(scores[batch_index, pivot - block_start] <= threshold)
            if low and not fallback:
                children.append((start, end))
                next_retries[key] = rejected + 1
                diagnostics['deferred'][batch_index] += 1
                continue
            diagnostics['fallback'][batch_index] += int(fallback)
            diagnostics['forced'][batch_index] += int(fallback and low)
            pivots.append(pivot)
            if start < pivot:
                children.append((start, pivot))
            if pivot + 1 < end:
                children.append((pivot + 1, end))
        if pivots:
            transfer_index[batch_index, pivots] = True
        next_intervals.append(children)
    return transfer_index, next_intervals, next_retries, diagnostics


def contains_token_sequence(tokens, sequence):
    if sequence.numel() == 0 or tokens.numel() < sequence.numel():
        return False
    for start in range(tokens.numel() - sequence.numel() + 1):
        if torch.equal(tokens[start:start + sequence.numel()], sequence):
            return True
    return False


def get_next_sequence_token_id(tokens, position, sequence, context_start):
    max_prefix_length = min(sequence.numel() - 1, position - context_start)
    for prefix_length in range(max_prefix_length, 0, -1):
        if torch.equal(tokens[position - prefix_length:position], sequence[:prefix_length]):
            return sequence[prefix_length].item()
    return sequence[0].item()


def apply_end_think_logit_boost(logits, tokens, candidate_mask_index, context_start,
                                total_gen_length, end_think_token_ids=None,
                                end_think_logit_boost=0., end_think_boost_power=2.):
    """Gradually encourage an end-of-thinking token sequence during generation."""
    if end_think_logit_boost <= 0 or not end_think_token_ids or total_gen_length <= 0:
        return logits
    if any(token_id < 0 or token_id >= logits.shape[-1] for token_id in end_think_token_ids):
        return logits

    sequence = torch.tensor(end_think_token_ids, dtype=torch.long, device=logits.device)
    logits = logits.clone()

    for batch_index in range(tokens.shape[0]):
        if contains_token_sequence(tokens[batch_index, context_start:], sequence):
            continue

        candidate_positions = torch.nonzero(
            candidate_mask_index[batch_index], as_tuple=False
        ).flatten()
        if candidate_positions.numel() == 0:
            continue

        position = candidate_positions[0].item()
        generated_length = position - context_start + 1
        progress = min(max(generated_length / float(total_gen_length), 0.), 1.)
        boost = end_think_logit_boost * (progress ** end_think_boost_power)
        token_id = get_next_sequence_token_id(
            tokens[batch_index], position, sequence, context_start
        )
        logits[batch_index, position, token_id] += boost

    return logits


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False,
             confidence_eos_eot_inf=False, end_think_token_ids=None, end_think_logit_boost=0.,
             end_think_boost_power=2., end_think_context_start=None,
             end_think_total_gen_length=None, return_stats=False,
             pivot_balanced_steps=0, pivot_confidence_steps=0,
             pivot_confidence_threshold=None, pivot_max_retries=3,
             pivot_max_block_steps=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length. Ignored for random_pivot.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: 'low_confidence', 'random', or 'random_pivot'. random_pivot
            simultaneously unmasks one random position per non-empty interval
            and recursively splits each interval until the block is complete.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        return_stats: Return (tokens, stats) with actual unmask counts per forward.
        pivot_balanced_steps: First K steps of EACH block sample from its
            intervals' central half (trim floor(length/4) positions at each end).
        pivot_confidence_steps: First K steps of EACH block select the highest
            predicted-token probability position in each interval.
        pivot_confidence_threshold: Defer pivots with probability <= threshold.
        pivot_max_retries: After this many interval rejections, select and commit
            its most confident token. None disables forced acceptance.
        pivot_max_block_steps: Optional hard cap. A capped incomplete block ends
            generation immediately with residual MASKs, reported in stats.
    '''
    if pivot_balanced_steps < 0 or pivot_confidence_steps < 0:
        raise ValueError('Initial pivot step counts must be non-negative.')
    if pivot_confidence_threshold is not None and not 0 <= pivot_confidence_threshold <= 1:
        raise ValueError('Pivot confidence threshold must be between 0 and 1.')
    if pivot_max_retries is not None and pivot_max_retries < 0:
        raise ValueError('pivot_max_retries must be non-negative or None.')
    if pivot_max_block_steps is not None and pivot_max_block_steps <= 0:
        raise ValueError('pivot_max_block_steps must be positive.')
    if pivot_confidence_threshold is not None and pivot_max_retries is None and pivot_max_block_steps is None:
        raise ValueError('Unbounded deferral requires a finite pivot_max_block_steps.')
    base_model = getattr(model, 'module', model)
    model_name = getattr(getattr(base_model, 'config', None), '_name_or_path', '')
    if 'illada' in model_name.lower():
        assert prompt.shape[0] == 1, 'iLLaDA currently does not support padded batch generation.'

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    unmasked_per_step = []
    pivot_diagnostics = {name: [] for name in ('deferred', 'fallback', 'forced')}
    capped_incomplete = False

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if remasking != 'random_pivot':
        assert steps % num_blocks == 0
        steps = steps // num_blocks

    if end_think_context_start is None:
        end_think_context_start = prompt.shape[1]
    if end_think_total_gen_length is None:
        end_think_total_gen_length = gen_length

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        if remasking == 'random_pivot':
            intervals = [[(block_start, block_end)] for _ in range(x.shape[0])]
            retries = {}
            # Even a maximally unbalanced split finishes within block_length rounds.
            block_steps = block_length
            if pivot_confidence_threshold is not None and pivot_max_retries is not None:
                block_steps *= pivot_max_retries + 1
            if pivot_max_block_steps is not None:
                block_steps = pivot_max_block_steps
        else:
            block_mask_index = (x[:, block_start:block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            block_steps = steps
        for i in range(block_steps):
            if remasking == 'random_pivot':
                if not any(intervals):
                    break
                needs_pivot_confidence = i < pivot_confidence_steps or pivot_confidence_threshold is not None
                if not needs_pivot_confidence:
                    transfer_index, intervals = get_random_pivot_transfer_index(
                        x, intervals, balanced=i < pivot_balanced_steps)
            mask_index = (x == mask_id)
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length
            candidate_mask_index = mask_index.clone()
            candidate_mask_index[:, :block_start] = False
            candidate_mask_index[:, block_end:] = False
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits = apply_end_think_logit_boost(
                logits,
                x,
                candidate_mask_index,
                end_think_context_start,
                end_think_total_gen_length,
                end_think_token_ids=end_think_token_ids,
                end_think_logit_boost=end_think_logit_boost,
                end_think_boost_power=end_think_boost_power,
            )

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            if remasking == 'random_pivot':
                # A selected pivot must become an actual token, never another MASK.
                logits[:, :, mask_id] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'random_pivot':
                if needs_pivot_confidence:
                    # Normalize over vocabulary, not over positions. Only the
                    # current block needs probabilities; use FP32 for thresholds.
                    block_logits = logits[:, block_start:block_end].float()
                    chosen = x0[:, block_start:block_end].unsqueeze(-1)
                    pivot_confidence = torch.exp(
                        block_logits.gather(-1, chosen).squeeze(-1)
                        - torch.logsumexp(block_logits, dim=-1))
                    transfer_index, intervals, retries, diagnostics = select_confident_pivots(
                        x, intervals, pivot_confidence, block_start, retries,
                        use_confidence=i < pivot_confidence_steps,
                        threshold=pivot_confidence_threshold, max_retries=pivot_max_retries,
                        balanced=i < pivot_balanced_steps)
                    if return_stats:
                        for name, values in diagnostics.items():
                            pivot_diagnostics[name].append(values)
                    del block_logits, pivot_confidence
                elif return_stats:
                    for values in pivot_diagnostics.values():
                        values.append([0] * x.shape[0])
                if return_stats:
                    unmasked_per_step.append((transfer_index & mask_index & (x0 != mask_id)).sum(dim=1))
                x[transfer_index] = x0[transfer_index]
                continue
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            if return_stats:
                unmasked_per_step.append((transfer_index & mask_index & (x0 != mask_id)).sum(dim=1))
            x[transfer_index] = x0[transfer_index]

        if remasking == 'random_pivot' and any(intervals):
            capped_incomplete = True
            break

    if return_stats:
        counts = torch.stack(unmasked_per_step).cpu().tolist() if unmasked_per_step else []
        stats = {'forward_steps': len(counts), 'unmasked_per_step': counts}
        if remasking == 'random_pivot':
            stats.update({f'{name}_per_step': values for name, values in pivot_diagnostics.items()})
            stats['capped_incomplete'] = capped_incomplete
        return x, stats
    return x


@torch.no_grad()
def var_generate(model, tokenizer, prompt, steps, gen_length, block_length,
                 temperature=0., cfg_scale=0., remasking='low_confidence',
                 mask_id=5, stop_tokens=None, end_think_token_ids=None,
                 end_think_logit_boost=0., end_think_boost_power=2.):
    """Generate one block at a time without adding future mask blocks."""
    assert gen_length % block_length == 0

    x = prompt.clone()
    stop_tokens = stop_tokens or []
    for _ in range(gen_length // block_length):
        x = generate(
            model, x, steps=steps, gen_length=block_length,
            block_length=block_length, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id,
            end_think_token_ids=end_think_token_ids,
            end_think_logit_boost=end_think_logit_boost,
            end_think_boost_power=end_think_boost_power,
            end_think_context_start=prompt.shape[1],
            end_think_total_gen_length=gen_length,
        )
        text = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)
        if any(stop_token in text for stop_token in stop_tokens):
            break
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='random_pivot')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
