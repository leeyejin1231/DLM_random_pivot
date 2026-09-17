"""CPU-only behavioral checks; no real model loading or benchmark execution."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from decode import generate, get_random_pivot_transfer_index, select_confident_pivots


class MockModel:
    device = torch.device('cpu')
    config = SimpleNamespace(_name_or_path='mock')

    def __init__(self, uncertain=False):
        self.inputs = []
        self.uncertain = uncertain

    def __call__(self, x, attention_mask=None):
        self.inputs.append(x.clone())
        logits = torch.zeros(*x.shape, 8)
        if not self.uncertain:
            logits[..., 3] = torch.arange(x.shape[1], dtype=torch.float32) * .1 + 1
        return SimpleNamespace(logits=logits)


class PivotTests(unittest.TestCase):
    def test_baseline_regression(self):
        old_path = Path(__file__).parent / 'results/gsm8k_100_g256_b32_s256_seed42/source/decode.py'
        spec = importlib.util.spec_from_file_location('old_decode', old_path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for mode in ('low_confidence', 'random', 'random_pivot'):
            runs = []
            for fn in (old.generate, generate):
                model = MockModel()
                torch.manual_seed(71)
                result, stats = fn(model, torch.tensor([[1, 2], [1, 2]]), steps=24,
                                   gen_length=24, block_length=8, mask_id=5,
                                   remasking=mode, return_stats=True)
                runs.append((result, stats, model.inputs))
            self.assertTrue(torch.equal(runs[0][0], runs[1][0]))
            self.assertEqual(runs[0][1]['unmasked_per_step'], runs[1][1]['unmasked_per_step'])
            self.assertEqual(len(runs[0][2]), len(runs[1][2]))
            for before, after in zip(runs[0][2], runs[1][2]):
                self.assertTrue(torch.equal(before, after))

    def test_balanced_bounds(self):
        x = torch.full((2, 40), 5)
        for seed in range(20):
            torch.manual_seed(seed)
            selected, _ = get_random_pivot_transfer_index(x, [[(2, 34)], [(2, 34)]], balanced=True)
            for b in range(2):
                positions = selected[b].nonzero().flatten().tolist()
                self.assertEqual(len(positions), 1)
                self.assertTrue(10 <= positions[0] < 26)

    def test_threshold_boundary_and_fallback(self):
        x = torch.full((2, 4), 5)
        intervals = [[(1, 2)], [(2, 3)]]
        confidence = torch.tensor([[0., .3, 0., 0.], [0., 0., .31, 0.]])
        selected, intervals, retries, diagnostics = select_confident_pivots(
            x, intervals, confidence, 0, {}, threshold=.3, max_retries=3)
        self.assertFalse(selected[0].any())
        self.assertTrue(selected[1, 2])
        self.assertEqual(intervals, [[(1, 2)], []])
        self.assertEqual(diagnostics['deferred'], [1, 0])
        for _ in range(2):
            selected, intervals, retries, _ = select_confident_pivots(
                x, intervals, confidence, 0, retries, threshold=.3, max_retries=3)
            self.assertFalse(selected.any())
        selected, intervals, retries, diagnostics = select_confident_pivots(
            x, intervals, confidence, 0, retries, threshold=.3, max_retries=3)
        self.assertTrue(selected[0, 1])
        self.assertEqual(intervals, [[], []])
        self.assertEqual(diagnostics['forced'], [1, 0])

    def test_fallback_chooses_best_and_resets_children(self):
        x = torch.full((1, 6), 5)
        selected, intervals, retries, stats = select_confident_pivots(
            x, [[(1, 5)]], torch.tensor([[0., .1, .2, .25, .1, 0.]]), 0,
            {(0, 1, 5): 3}, threshold=.3, max_retries=3)
        self.assertEqual(selected.nonzero().tolist(), [[0, 3]])
        self.assertEqual(intervals, [[(1, 3), (4, 5)]])
        self.assertEqual(retries, {})
        self.assertEqual(stats['forced'], [1])

    def test_warm_start_resets_per_block_and_keeps_context(self):
        for kwargs in ({'pivot_balanced_steps': 3}, {'pivot_confidence_steps': 3}):
            model = MockModel()
            torch.manual_seed(42)
            out, stats = generate(model, torch.tensor([[1, 2], [1, 2]]), steps=16,
                                  gen_length=16, block_length=8, mask_id=5,
                                  remasking='random_pivot', return_stats=True, **kwargs)
            self.assertTrue((out[:, 2:] == 3).all())
            self.assertEqual(torch.tensor(stats['unmasked_per_step']).sum(0).tolist(), [16, 16])
            states = model.inputs + [out]
            rounds = {0: 0, 1: 0}
            for before, after in zip(states, states[1:]):
                block = 0 if (before[:, 2:10] == 5).any() else 1
                left, right = 2 + block * 8, 10 + block * 8
                changed = before != after
                self.assertFalse(changed[:, :left].any())
                self.assertFalse(changed[:, right:].any())
                for batch in range(2):
                    position = left
                    while position < right:
                        if before[batch, position] != 5:
                            position += 1
                            continue
                        start = position
                        while position < right and before[batch, position] == 5:
                            position += 1
                        pivots = changed[batch, start:position].nonzero().flatten()
                        self.assertEqual(pivots.numel(), 1)
                        pivot = start + int(pivots[0])
                        if rounds[block] < 3:
                            if 'pivot_balanced_steps' in kwargs:
                                trim = (position - start) // 4
                                self.assertTrue(start + trim <= pivot < position - trim)
                            else:
                                self.assertEqual(pivot, position - 1)
                rounds[block] += 1

    def test_all_low_confidence_finishes_with_bounded_fallback(self):
        model = MockModel(uncertain=True)
        out, stats = generate(model, torch.tensor([[1, 2]]), gen_length=8,
                              block_length=4, mask_id=5, remasking='random_pivot',
                              pivot_confidence_threshold=.3, pivot_max_retries=3,
                              return_stats=True)
        self.assertEqual(stats['forward_steps'], 32)
        self.assertFalse((out[:, 2:] == 5).any())
        self.assertEqual(sum(sum(x) for x in stats['deferred_per_step']), 24)
        self.assertEqual(sum(sum(x) for x in stats['forced_per_step']), 8)
        self.assertEqual(sum(sum(x) for x in stats['unmasked_per_step']), 8)

    def test_optional_cap_reports_incomplete(self):
        out, stats = generate(MockModel(uncertain=True), torch.tensor([[1, 2]]),
                              gen_length=8, block_length=4, mask_id=5, remasking='random_pivot',
                              pivot_confidence_threshold=.3, pivot_max_retries=None,
                              pivot_max_block_steps=2, return_stats=True)
        self.assertEqual(stats['forward_steps'], 2)
        self.assertTrue(stats['capped_incomplete'])
        self.assertTrue((out[:, 2:] == 5).all())


if __name__ == '__main__':
    unittest.main()
