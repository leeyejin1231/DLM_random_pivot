"""Confidence-filtered random pivots with bulk completion of small intervals.

Intervals of length <= small_interval_size at the start of a step commit all
their tokens, regardless of confidence. Larger intervals use the original
confidence-filtered random pivot rule. Children created by a pivot are processed
only on the next forward, so their predictions can use the committed pivot.

Example (model and prompt already loaded):
    from decode_confidence_v2 import generate
    output, stats = generate(
        model, prompt, gen_length=256, block_length=32,
        confidence_threshold=0.8, small_interval_size=2, return_stats=True,
    )

This module does not load models or start experiments when imported or executed.
"""

import math

import torch

from decode import add_gumbel_noise


_DIAGNOSTICS = ('candidates', 'deferred', 'fallback', 'bulk_intervals', 'bulk_tokens')


def select_pivots(confidence, intervals, confidence_threshold, small_interval_size=2,
                  pivots_per_interval=1, nonadjacent_pivots=False):
    """Complete existing small intervals and select pivots per eligible large one.

    Each large interval commits up to pivots_per_interval distinct positions,
    sampled uniformly without replacement from its above-threshold candidates.
    With nonadjacent_pivots, candidates are visited in that random order and a
    candidate is skipped when it neighbours an already chosen pivot, so no two
    pivots of one interval are committed side by side from the same forward.

    confidence: (batch, block_length) predicted-token probabilities.
    intervals: per-sample lists of half-open, block-relative masked intervals.
    CPU position sampling follows torch.manual_seed(). Copy the small confidence
    matrix once per step; avoid a separate GPU synchronization for each interval.

    The per-sample global-best fallback runs only when neither bulk completion
    nor an eligible pivot makes progress. candidates counts above-threshold
    positions in all input intervals, including those completed in bulk.
    deferred counts only intervals carried forward without any token committed.
    """
    scores = confidence.detach().cpu()
    selected = torch.zeros(scores.shape, dtype=torch.bool)
    next_intervals = []
    diagnostics = {key: [] for key in _DIAGNOSTICS}

    for batch_index, sample_intervals in enumerate(intervals):
        pivots = {}
        completed = set()
        candidate_count = 0
        bulk_tokens = 0
        for interval_index, (start, end) in enumerate(sample_intervals):
            eligible = torch.nonzero(
                scores[batch_index, start:end] >= confidence_threshold,
                as_tuple=True,
            )[0]
            candidate_count += eligible.numel()
            if end - start <= small_interval_size:
                selected[batch_index, start:end] = True
                completed.add(interval_index)
                bulk_tokens += end - start
            elif eligible.numel():
                count = min(pivots_per_interval, eligible.numel())
                order = torch.randperm(eligible.numel(), device='cpu')
                if nonadjacent_pivots:
                    chosen = []
                    for candidate in eligible[order].tolist():
                        if all(abs(candidate - other) > 1 for other in chosen):
                            chosen.append(candidate)
                            if len(chosen) == count:
                                break
                else:
                    chosen = eligible[order[:count]].tolist()
                pivots[interval_index] = sorted(start + offset for offset in chosen)

        fallback = bool(sample_intervals) and not completed and not pivots
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
            pivots[best_interval] = [best_position]

        children = []
        for interval_index, (start, end) in enumerate(sample_intervals):
            if interval_index in completed:
                continue
            if interval_index not in pivots:
                children.append((start, end))
                continue
            # Even small children wait for a new forward conditioned on these pivots.
            child_start = start
            for pivot in pivots[interval_index]:
                selected[batch_index, pivot] = True
                if child_start < pivot:
                    children.append((child_start, pivot))
                child_start = pivot + 1
            if child_start < end:
                children.append((child_start, end))
        next_intervals.append(children)
        diagnostics['candidates'].append(candidate_count)
        diagnostics['deferred'].append(len(sample_intervals) - len(pivots) - len(completed))
        diagnostics['fallback'].append(int(fallback))
        diagnostics['bulk_intervals'].append(len(completed))
        diagnostics['bulk_tokens'].append(bulk_tokens)

    return selected.to(confidence.device), next_intervals, diagnostics


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=None, gen_length=128,
             block_length=32, temperature=0., cfg_scale=0., mask_id=126336,
             confidence_threshold=0.8, logits_eos_inf=False, eos_token_id=126081,
             return_stats=False, small_interval_size=2, pivots_per_interval=1,
             nonadjacent_pivots=False):
    """Generate with confidence-filtered random pivots and small-interval completion.

    Blocks are processed sequentially while the model attends to the full
    sequence. Each step uses one shared forward for all active intervals:
      * Existing intervals of length <= small_interval_size commit every token,
        even below confidence_threshold.
      * Larger intervals commit up to pivots_per_interval uniformly sampled
        above-threshold pivots (without replacement), or wait when there is
        no candidate. nonadjacent_pivots forbids two pivots of the same
        interval at neighbouring positions; bulk completion is unaffected.
      * Newly created children become eligible only on the next forward.
    When a sample would otherwise commit nothing, its most confident masked
    token is committed as a fallback, even below the confidence threshold.

    small_interval_size is a non-negative integer. The default is 2; 1 bypasses
    confidence-based waiting for singleton intervals, and 0 disables bulk
    completion to recover the original confidence-pivot rule.

    confidence_threshold is an absolute probability cutoff in [0, 1]. Confidence
    is the FP32 vocabulary softmax probability of the proposed token, after CFG
    and MASK/optional EOS suppression, before sampling noise. With temperature
    > 0, score the sampled token, not necessarily argmax.

    steps is accepted for compatibility but ignored: generation finishes when
    no masked intervals remain. Each active sample commits at least one token
    per step, so a block needs at most block_length forward calls. Previously
    committed tokens are not remasked.

    return_stats returns (output, stats). Per-step counts have shape
    (forward_steps, batch_size), including zero counts for finished samples.
    bulk_intervals_per_step and bulk_tokens_per_step count intervals and tokens
    completed by the small-interval rule, regardless of their confidence.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('confidence_threshold must be between 0 and 1.')
    if (isinstance(small_interval_size, bool)
            or not isinstance(small_interval_size, int) or small_interval_size < 0):
        raise ValueError('small_interval_size must be a non-negative integer.')
    if (isinstance(pivots_per_interval, bool)
            or not isinstance(pivots_per_interval, int) or pivots_per_interval < 1):
        raise ValueError('pivots_per_interval must be a positive integer.')
    if not isinstance(nonadjacent_pivots, bool):
        raise ValueError('nonadjacent_pivots must be a bool.')
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
    diagnostics_per_step = {key: [] for key in _DIAGNOSTICS}
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
                confidence, intervals, confidence_threshold, small_interval_size,
                pivots_per_interval, nonadjacent_pivots)
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
        'small_interval_size': small_interval_size,
        'pivots_per_interval': pivots_per_interval,
        'nonadjacent_pivots': nonadjacent_pivots,
        'capped_incomplete': False,
    }
    stats.update({f'{key}_per_step': values for key, values in diagnostics_per_step.items()})
    return x, stats
