"""Random pivot decoding with confidence-filtered candidates.

For each non-empty masked interval, first keep positions whose predicted-token
probability is >= confidence_threshold, then choose uniformly among them.
An interval with no candidate waits without splitting. If an entire sample has
no eligible position in the current block, commit its most confident masked
position to guarantee progress. All selected pivots use one shared forward.

Example (model and prompt already loaded):
    from decode_confidence import generate
    output, stats = generate(
        model, prompt, gen_length=512, block_length=32,
        confidence_threshold=0.8, return_stats=True,
    )

This module does not load models or start experiments when imported or executed.
"""

import math

import torch

from decode import add_gumbel_noise


def select_pivots(confidence, intervals, confidence_threshold):
    """Select pivots from eligible positions, then split only accepted intervals.

    confidence: (batch, block_length) predicted-token probabilities.
    intervals: per-sample lists of half-open, block-relative masked intervals.
    CPU position sampling follows torch.manual_seed(). Copy the small confidence
    matrix once per step; avoid a separate GPU synchronization for each interval.
    """
    scores = confidence.detach().cpu()
    selected = torch.zeros(scores.shape, dtype=torch.bool)
    next_intervals = []
    diagnostics = {key: [] for key in ('candidates', 'deferred', 'fallback')}

    for batch_index, sample_intervals in enumerate(intervals):
        pivots = {}
        candidate_count = 0
        for interval_index, (start, end) in enumerate(sample_intervals):
            eligible = torch.nonzero(
                scores[batch_index, start:end] >= confidence_threshold,
                as_tuple=True,
            )[0]
            candidate_count += eligible.numel()
            if eligible.numel():
                choice = torch.randint(eligible.numel(), (), device='cpu').item()
                pivots[interval_index] = start + eligible[choice].item()

        fallback = bool(sample_intervals) and not pivots
        if fallback:
            # Only unresolved intervals participate; never select a prompt,
            # previously committed token, or a future block. Ties go leftmost.
            best_score = float('-inf')
            for interval_index, (start, end) in enumerate(sample_intervals):
                position = start + scores[batch_index, start:end].argmax().item()
                value = scores[batch_index, position].item()
                if value > best_score:
                    best_score = value
                    best_interval, best_position = interval_index, position
            if not math.isfinite(best_score):
                raise ValueError('No finite predicted-token confidence in an active sample.')
            pivots[best_interval] = best_position

        children = []
        for interval_index, (start, end) in enumerate(sample_intervals):
            if interval_index not in pivots:
                children.append((start, end))
                continue
            pivot = pivots[interval_index]
            selected[batch_index, pivot] = True
            if start < pivot:
                children.append((start, pivot))
            if pivot + 1 < end:
                children.append((pivot + 1, end))
        next_intervals.append(children)
        diagnostics['candidates'].append(candidate_count)
        diagnostics['deferred'].append(len(sample_intervals) - len(pivots))
        diagnostics['fallback'].append(int(fallback))

    return selected.to(confidence.device), next_intervals, diagnostics


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=None, gen_length=128,
             block_length=32, temperature=0., cfg_scale=0., mask_id=126336,
             confidence_threshold=0.8, logits_eos_inf=False, eos_token_id=126081,
             return_stats=False):
    """Generate using one confidence-filtered random pivot per eligible interval.

    Blocks are processed sequentially while the model attends to the full
    sequence. Within a step, every eligible interval commits exactly one pivot;
    intervals without candidates wait. The global-best fallback is per sample
    and occurs only when none of its intervals has candidates. It may commit a
    token below the threshold and is counted in fallback_per_step.

    confidence_threshold is an absolute probability cutoff in [0, 1], applied
    at every step. Confidence is the FP32 vocabulary softmax probability of the
    proposed token, after CFG and MASK/optional EOS suppression, before sampling
    noise. With temperature > 0, score the sampled token, not necessarily argmax.

    steps is accepted for compatibility with callers of decode.generate but is
    ignored: generation finishes when no masked intervals remain. Each active
    sample commits at least one token per step, so a block needs at most
    block_length forward calls. Previously committed tokens are not remasked.

    return_stats returns (output, stats). Per-step counts have shape
    (forward_steps, batch_size), including zero counts for finished samples.
    The default threshold 0.8 is configurable and has not been tuned here.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('confidence_threshold must be between 0 and 1.')
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError('Positive gen_length must be divisible by positive block_length.')
    if prompt.ndim != 2 or prompt.shape[0] == 0:
        raise ValueError('prompt must have shape (non-empty batch, prompt_length).')
    if attention_mask is not None and attention_mask.shape != prompt.shape:
        raise ValueError('attention_mask must have the same shape as prompt.')
    if temperature < 0 or cfg_scale < 0:
        raise ValueError('temperature and cfg_scale must be non-negative.')

    base_model = getattr(model, 'module', model)
    model_name = getattr(getattr(base_model, 'config', None), '_name_or_path', '')
    if 'illada' in model_name.lower() and prompt.shape[0] != 1:
        raise ValueError('iLLaDA currently does not support padded batch generation.')

    device = model.device
    batch_size, prompt_length = prompt.shape
    x = torch.full((batch_size, prompt_length + gen_length), mask_id,
                   dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt.to(device)
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask.to(device),
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device),
        ], dim=-1)
    prompt_index = x != mask_id
    unmasked_per_step = []
    diagnostics_per_step = {key: [] for key in ('candidates', 'deferred', 'fallback')}
    steps_per_block = []
    forward_steps = 0

    for block_start in range(prompt_length, prompt_length + gen_length, block_length):
        block_end = block_start + block_length
        intervals = [[(0, block_length)] for _ in range(batch_size)]
        block_steps = 0
        while any(intervals):
            if cfg_scale > 0:
                unconditioned = x.clone()
                unconditioned[prompt_index] = mask_id
                inputs = torch.cat([x, unconditioned], dim=0)
                masks = (torch.cat([attention_mask, attention_mask], dim=0)
                         if attention_mask is not None else None)
                logits = model(inputs, attention_mask=masks).logits[:, block_start:block_end]
                conditioned, unconditioned_logits = logits.chunk(2, dim=0)
                logits = unconditioned_logits + (cfg_scale + 1) * (conditioned - unconditioned_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits[:, block_start:block_end]

            logits[..., mask_id] = -torch.inf
            if logits_eos_inf:
                logits[..., eos_token_id] = -torch.inf
            predictions = add_gumbel_noise(logits, temperature).argmax(dim=-1)
            float_logits = logits.float()
            confidence = torch.exp(
                float_logits.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
                - torch.logsumexp(float_logits, dim=-1)
            )
            selected, intervals, diagnostics = select_pivots(
                confidence, intervals, confidence_threshold)
            block = x[:, block_start:block_end]
            block[selected] = predictions[selected]
            forward_steps += 1
            block_steps += 1
            if return_stats:
                unmasked_per_step.append(selected.sum(dim=1))
                for key, values in diagnostics.items():
                    diagnostics_per_step[key].append(values)
            del logits, float_logits, confidence, predictions
        steps_per_block.append(block_steps)

    if not return_stats:
        return x
    stats = {
        'forward_steps': forward_steps,
        'steps_per_block': steps_per_block,
        'unmasked_per_step': torch.stack(unmasked_per_step).cpu().tolist(),
        'confidence_threshold': confidence_threshold,
        'capped_incomplete': False,
    }
    stats.update({f'{key}_per_step': values for key, values in diagnostics_per_step.items()})
    return x, stats
