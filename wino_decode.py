"""WINO baseline (Hong et al., ICLR 2026), ported from WINO-DLLM/LLaDA/decoding.py::decoding_wino.

WINO appends a block_length "shadow" copy of the current block to the sequence. Shadow
positions share the block's position ids and attend to everything except their own
counterpart, so their predictions score already-committed tokens (back-confidence).
Each step accepts masked tokens with confidence > threshold (capped at
clamp(0.7 * #masked, 5, 20); the argmax token if none qualify) and, when more than one
token was accepted, revokes committed tokens whose back-confidence < threshold_back
(capped at last_accept - 1 lowest). The block ends when it contains no [MASK].

The shadow block needs a 4-D boolean attention mask and explicit position ids, which the
Hugging Face LLaDA modeling does not accept, so load_model() builds the model with WINO's
own modeling_llada.py (same weights, no KV cache). Batch size 1 only, as in the original.

dream=True adapts the trick to Dream, which predicts position i from the hidden state at
i-1 (its sampler shifts logits right by one). Shadow slot k therefore takes position id
b0+k-1 and is forbidden to attend to block column b0+k, so its *unshifted* logits are the
leave-one-out prediction of block token b0+k; the main sequence uses shifted logits as
usual. Pass the raw Dream model (not DreamForwardAdapter), mask_id=151666.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from decode import add_gumbel_noise

WINO_DIR = Path(__file__).resolve().parent / 'WINO-DLLM' / 'LLaDA'


def load_model(path, device='cuda'):
    if str(WINO_DIR) not in sys.path:
        sys.path.insert(0, str(WINO_DIR))
    from modeling_llada import LLaDAModelLM
    return LLaDAModelLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                        local_files_only=True).to(device).eval()


@torch.no_grad()
def generate(model, prompt, attention_mask=None, gen_length=128, block_length=32, temperature=0.,
             mask_id=126336, threshold=0.6, threshold_back=0.9, return_stats=False, dream=False,
             **unused):
    """Verbatim WINO decoding with per-step accounting; unused absorbs steps/cfg_scale."""
    if unused.get('cfg_scale', 0):
        raise NotImplementedError('CFG is not part of the WINO baseline.')
    if prompt.shape[0] != 1:
        raise ValueError('WINO decoding supports batch size 1 only.')
    assert gen_length % block_length == 0
    device = model.device
    P = prompt.shape[1]
    x_block = torch.full((1, P + gen_length + block_length), mask_id, dtype=torch.long, device=device)
    x_block[:, :P] = prompt.clone()
    num_blocks = gen_length // block_length
    unmasked_per_step, remasked_per_step = [], []

    for num_block in range(num_blocks):
        b0, b1 = P + num_block * block_length, P + (num_block + 1) * block_length
        mask_index_block = (x_block == mask_id)
        mask_index_block[:, b1:] = False
        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:, -block_length:] = ~mask_index_block[:, b0:b1]
        shadow_offset = -1 if dream else 0
        position_ids = torch.cat([torch.arange(P + gen_length, device=device),
                                  torch.arange(b0 + shadow_offset, b1 + shadow_offset, device=device)])
        if dream:
            position_ids = position_ids.unsqueeze(0)
        attn = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool, device=device)
        attn[:, :, :, -block_length:] = False
        attn[:, :, -block_length:, -block_length:] = True
        attn[:, :, -block_length:, b0:b1] = ~torch.eye(block_length, dtype=torch.bool, device=device)
        last_accept = 30
        while mask_index_block.any():
            max_accept = min(max(int(mask_index_block.sum()) * 7 // 10, 5), 20)
            logits = model(x_block, attention_mask=attn, position_ids=position_ids).logits
            if dream:
                # Main sequence: Dream's right shift. Shadow slots already predict b0+k directly.
                main = P + gen_length
                logits = torch.cat([logits[:, :1], logits[:, :main - 1], logits[:, main:]], dim=1)
            x0 = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)
            shift_left = torch.zeros_like(unmask_index_block)
            shift_left[:, b0:b1] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[shift_left]

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(mask_index_block, x0, x_block)
            confidence = torch.where(mask_index_block, x0_p, -np.inf)
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)

            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            elif not transfer_index.any():
                transfer_index.view(-1)[torch.argmax(confidence)] = True
            x_block[transfer_index] = x0[transfer_index]
            num_accept = int(transfer_index.sum())

            if num_accept > 1:
                remask_index = confidence_back < threshold_back
                if remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    flat = confidence_back.view(-1)
                    temp = torch.zeros_like(flat, dtype=torch.bool)
                    _, indices = torch.topk(flat, k=num_remask, largest=False)
                    temp[indices] = True
                    remask_index = temp.view(confidence_back.shape)
            else:
                remask_index = torch.zeros_like(transfer_index)
            remask_shift = torch.zeros_like(remask_index)
            remask_shift[:, b0:b1] = remask_index[:, -block_length:]
            x_block[remask_shift] = mask_id
            mask_index_block[transfer_index] = False
            mask_index_block[remask_shift] = True
            transfer_shift = torch.zeros_like(transfer_index)
            transfer_shift[:, -block_length:] = transfer_index[:, b0:b1]
            unmask_index_block[transfer_shift] = True
            unmask_index_block[remask_index] = False
            last_accept = num_accept
            if return_stats:
                unmasked_per_step.append([num_accept])
                remasked_per_step.append([int(remask_shift.sum())])

    x = x_block[:, :P + gen_length]
    if not return_stats:
        return x
    return x, {
        'forward_steps': len(unmasked_per_step),
        'unmasked_per_step': unmasked_per_step,
        'remasked_per_step': remasked_per_step,
        'threshold': threshold, 'threshold_back': threshold_back,
        'capped_incomplete': False,
    }
