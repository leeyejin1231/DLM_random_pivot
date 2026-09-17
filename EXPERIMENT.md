# GSM8K 100-question pilot

Compare `low_confidence` and `random_pivot` using the same cached
LLaDA-8B-Instruct checkpoint and the same 100 GSM8K test examples.

- Select 100 examples without replacement with Python `random.Random(42)`;
  save test indices, questions, and references in `samples.json`.
- Zero-shot chat prompt: request step-by-step reasoning ending in `#### <answer>`.
  This is a pilot protocol, not a reproduction of the published few-shot score.
- Generate 256 positions with sequential blocks of 32. Vanilla uses 256
  forward calls; pivot stops when all intervals have been consumed.
- BF16, batch size 1, temperature 0, CFG 0, no KV cache, no EOS early stop,
  no EOS suppression or end-of-thinking boost.
- Use one GPU for both methods, alternating method order for each question.
  Run full-size warmup once per method, excluded from reported time.
- Seed each question independently (`42 + test_index`) so a resumed run keeps
  the same pivot choices. Save code hashes and library versions in `config.json`.

## Metrics

`accuracy_percent`: numeric comparison using the final `####` numeric answer,
falling back to the last number if the marker is missing. Also report strict
marker accuracy and missing-marker counts. Preserve full outputs for inspection.

`unmasked_tokens_per_second`: actual mask-to-token transitions, including
special tokens and positions after EOS, divided by CUDA-synchronized decoding
wall time. Loading, warmup, tokenization, parsing, and writing results are excluded.
The optional decoding counters are included in timing for both methods.

`answer_tokens_per_second`: non-special tokens before the first EOS/EOT divided
by the same decoding time. This avoids treating post-EOS filler as answer text.

`unmasked_tokens_per_step`: total actual mask-to-token transitions divided by
total forward calls, with batch size 1. Per-forward counts are also saved.

`possible_truncation_count`: outputs lacking both an EOS/EOT and a final answer
marker. This is a heuristic, not proof of truncation; inspect saved outputs.
`no_end_token_count` is saved separately. Residual masks are counted explicitly.

## Outputs and resume

Results are appended after every answer in `results.jsonl`; `summary.json`
is refreshed after each answer and may contain partial results during a run.
`paired.n` counts questions completed by both methods. A complete run has
100 records per method, 200 records total, and `paired.n = 100`.

Run the following from this directory (use a GPU available on the host):

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
HF_MODULES_CACHE=/home/yejin/.cache/huggingface/modules \
python -u eval_gsm8k.py \
  --model /mnt/shared/huggingface-cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07 \
  --dataset /home/yejin/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow \
  --output /home/yejin/contents/DLM_EF_random_pivot/results/gsm8k_100_g256_b32_s256_seed42
```

Rerunning with the same code and configuration resumes completed records;
do not start a second process against an output directory with an active run.

The implementation suppresses MASK predictions for random pivots so intervals
always finish. Vanilla retains its original token selection logic; check the
reported residual mask count when interpreting results.

## Random pivot variants

`pivot_variants.json` defines three independent conditions on the same 100
questions. Use the original completed random-pivot run as the comparison;
do not rerun that control.

| Label | Rule |
|---|---|
| `pivot_balanced_first3` | First 3 forward steps of each block: trim floor(interval length / 4) positions from each end and choose uniformly from the remainder. Subsequent steps: original random pivot. |
| `pivot_defer_p03_retry3` | Random pivot with predicted-token probability <= 0.3 is left masked, and its interval is retained without splitting. Next step draws a fresh random pivot. After 3 rejections of an interval, its next attempt commits its most confident token even if its probability is low. Children start with zero rejections. |
| `pivot_confidence_first3` | First 3 forward steps of each block: select the highest-confidence predicted token in each non-empty interval. Subsequent steps: original random pivot. |

These are separate ablations, not cumulative combinations. Confidence is the
predicted token's vocabulary probability, evaluated in FP32 after suppressing
MASK. All pivots in a step use the same forward output. Deferrals never expose
their proposed token values to the next model call.

The first-3-step schedule restarts in each of the eight sequential 32-token
blocks. Ties in confidence select the leftmost position. Original random pivot
keeps its previous RNG sequence and decoding outputs with a fixed seed.

The deferral run records `deferred_pivots`, `fallback_pivots`, and `forced_pivots`:
respectively rejected attempts, best-confidence selections after 3 rejections,
and those fallback selections that are still <= 0.3. Deferrals count as forward
steps but contribute zero unmasked tokens. These runs do not remask already
committed tokens. `final_tokens_per_second` and `final_tokens_per_step` also
count distinct final non-MASK positions, for an explicit useful-work metric.

Use the command above with:

```text
--variant-config /home/yejin/contents/DLM_EF_random_pivot/pivot_variants.json
--output /home/yejin/contents/DLM_EF_random_pivot/results/gsm8k_100_pivot_variants_g256_b32_seed42
```

A complete variants run has 300 records (100 per condition).
The previous 82% vanilla and 77% random-pivot results remain available in the
original run directory. The initial extra-control run was interrupted on user
request; its files are preserved in `interrupted_with_control`, and its seven
completed variant records are retained in the resumed run. Control records
are excluded from the resumed results. Timing comparisons use separate runs.
Every new run saves its decoder and evaluator source files alongside code hashes.
