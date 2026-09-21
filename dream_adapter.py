"""Run the LLaDA-style decoders on Dream-v0-Instruct-7B without changing them.

Dream differs from LLaDA in two ways that matter to the decoders in this repo:
  * its forward wants attention_mask="full" (or a 4-D boolean mask) rather than a 2-D
    padding mask, and
  * it predicts the token at position i from the logits at position i-1, so its own
    sampler shifts logits right by one (generation_utils.DreamGenerationMixin._sample).
DreamForwardAdapter wraps the model so that `adapter(x, attention_mask=None).logits`
returns already-shifted logits over a fully bidirectional forward. Batch size 1 only
(no padding), which is what every evaluator here uses. Token ids: mask 151666,
end-of-turn 151645 (<|im_end|>), eos/pad 151643 (<|endoftext|>).
"""
from types import SimpleNamespace

import torch
from transformers import AutoModel

MASK_ID = 151666
END_IDS = (151643, 151645)


class DreamForwardAdapter:
    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, input_ids, attention_mask=None, **unused):
        if attention_mask is not None:
            raise ValueError('DreamForwardAdapter expects attention_mask=None (unpadded batch of 1).')
        logits = self.model(input_ids, attention_mask='full', position_ids=None).logits
        return SimpleNamespace(logits=torch.cat([logits[:, :1], logits[:, :-1]], dim=1))


def load_model(path, device='cuda'):
    model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    return DreamForwardAdapter(model)
