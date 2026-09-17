"""Hierarchy-dLLM baseline (Qi et al., ICLR 2026), paper algorithm only.

Ported from dInfer's non-cached reference path
(python/dinfer/decoding/generate_hierarchy.py, decoding='hierarchy_remasking',
python/dinfer/decoding/parallel_strategy.py::get_transfer_index_hierarchy_remask).
Excluded on purpose: KV/dual cache, vicinity/look-ahead, credit fusion,
torch.compile, iteration smoothing, block-diffusion serving. Every step is a full
forward over the whole sequence, exactly like the vanilla and random-pivot
decoders in decode.py, so timing is comparable.

Per step inside the current block (Eq. 9-12 / Algorithm 1 of the paper):
  1. x0 = argmax logits, c = p(x0) (confidence, Eq. 8) for every block position.
  2. Remask candidates: already-decoded positions with c < tau_remask (Eq. 12).
  3. Sub-decoding areas = maximal contiguous runs of (masked | remask-candidate)
     positions. In each area the most confident position is selected (Eq. 10),
     kept only if c > tau_low; every position with c > tau_high is selected (Eq. 9).
  4. Progress guarantee (Eq. 11): if fewer than (#remask + 1) positions were
     selected, add the most confident remaining positions to make up the gap.
  5. Remask candidates that were not selected become [MASK]; selected positions
     take x0. Repeat until the block contains no [MASK].
"""
import numpy as np
import torch
import torch.nn.functional as F

from decode import add_gumbel_noise


def hierarchy_transfer_index(logits, temperature, mask_index, x, mask_id,
                             threshold, low_threshold, remask_threshold):
    """Verbatim port of dInfer get_transfer_index_hierarchy_remask (threshold mode).

    logits/mask_index/x cover the current block only, shape (B, block_length[, V]).
    Returns x0 (tokens to write, [MASK] for remasked positions) and transfer_index.
    """
    if temperature != 0:
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l

    lower_index = x0_p < remask_threshold
    remask_index = torch.logical_and(lower_index, torch.logical_not(mask_index))
    mask_new = torch.logical_or(lower_index, mask_index)

    confidence = torch.where(mask_new, x0_p, float('-inf'))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    remask_cnt = remask_index.sum(dim=1)

    for i in range(mask_new.shape[0]):
        mask_i = mask_new[i].int()
        conf_i = confidence[i]

        diff = torch.diff(torch.cat([mask_i[:1] * 0, mask_i, mask_i[-1:] * 0]))
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]

        if len(starts) > 0:
            max_indices = [s + torch.argmax(conf_i[s:e])
                           for s, e in zip(starts.tolist(), ends.tolist())]
            transfer_index[i, max_indices] = True

        if low_threshold is not None:
            transfer_index[i] = torch.logical_and(transfer_index[i], conf_i > low_threshold)

        if threshold is not None:
            transfer_index[i] = torch.logical_or(transfer_index[i], conf_i > threshold)

        gap = int((remask_cnt[i] + 1 - transfer_index[i].sum()).item())
        if gap > 0:
            conf_i[transfer_index[i]] = float('-inf')
            values, indices = torch.topk(conf_i, gap, largest=True, sorted=False)
            transfer_index[i][indices] = True

    remask_index = torch.logical_and(remask_index, torch.logical_not(transfer_index))
    x0[remask_index] = mask_id
    transfer_index[remask_index] = True

    return x0, transfer_index


@torch.no_grad()
def generate_hierarchy(model, prompt, attention_mask=None, gen_length=128, block_length=32,
                       temperature=0., mask_id=126336, threshold=0.82, low_threshold=0.3,
                       remask_threshold=0.3, return_stats=False, max_block_steps=None,
                       **unused):
    """Semi-autoregressive block decoding with Hierarchy-dLLM token selection.

    unused absorbs shared evaluator kwargs (steps, cfg_scale) that this method ignores.
    max_block_steps is a safety cap only (the released code has none); a capped
    block ends generation with residual masks, reported in stats.
    """
    assert gen_length % block_length == 0
    if unused.get('cfg_scale', 0):
        raise NotImplementedError('CFG is not part of the Hierarchy-dLLM baseline.')
    if max_block_steps is None:
        max_block_steps = 8 * block_length
    num_blocks = gen_length // block_length

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    unmasked_per_step, remasked_per_step, edited_per_step = [], [], []
    capped_incomplete = False

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        for _ in range(max_block_steps):
            x_block = x[:, block_start:block_end]
            mask_index = (x_block == mask_id)
            if not mask_index.any():
                break
            logits = model(x, attention_mask=attention_mask).logits
            x0, transfer_index = hierarchy_transfer_index(
                logits[:, block_start:block_end], temperature, mask_index, x_block, mask_id,
                threshold, low_threshold, remask_threshold)
            if return_stats:
                became_token = transfer_index & (x0 != mask_id)
                unmasked_per_step.append((became_token & mask_index).sum(dim=1))
                remasked_per_step.append((transfer_index & (x0 == mask_id) & ~mask_index).sum(dim=1))
                edited_per_step.append((became_token & ~mask_index & (x0 != x_block)).sum(dim=1))
            x_block[transfer_index] = x0[transfer_index]
        if (x[:, block_start:block_end] == mask_id).any():
            capped_incomplete = True
            break

    if return_stats:
        stats = {
            'forward_steps': len(unmasked_per_step),
            'unmasked_per_step': torch.stack(unmasked_per_step).cpu().tolist() if unmasked_per_step else [],
            'remasked_per_step': torch.stack(remasked_per_step).cpu().tolist() if remasked_per_step else [],
            'edited_per_step': torch.stack(edited_per_step).cpu().tolist() if edited_per_step else [],
            'capped_incomplete': capped_incomplete,
        }
        return x, stats
    return x
