"""Aggregate HumanEval runs (vanilla / random pivot / hierarchy) into one comparison table.

Usage: python report_humaneval.py RESULTS_DIR [RESULTS_DIR ...]
All runs must cover the same task set; paired statistics use test_index.
"""
import json
import random
import sys
from pathlib import Path


def load(dirs):
    rows = []
    for d in dirs:
        for line in (Path(d) / 'results.jsonl').read_text().splitlines():
            r = json.loads(line)
            r['run'] = Path(d).name
            rows.append(r)
    return rows


def mode_summary(subset):
    n = len(subset)
    sec = sum(r['decode_seconds'] for r in subset)
    steps = sum(r['forward_steps'] for r in subset)
    unmasked = sum(r['unmasked_tokens'] for r in subset)
    answer = sum(r['answer_tokens'] for r in subset)
    final = sum(r['final_unmasked_tokens'] for r in subset)
    stats = [r['decoder_stats'] for r in subset]
    out = {
        'n': n,
        'pass@1 (%)': 100 * sum(r['passed'] for r in subset) / n,
        'passed': sum(r['passed'] for r in subset),
        'forward/problem': steps / n,
        'sec/problem': sec / n,
        'unmasked tok/step': unmasked / steps,
        'answer tok/step': answer / steps,
        'unmasked tok/s': unmasked / sec,
        'answer tok/s': answer / sec,
        'final tok/step': final / steps,
        'answer tokens/problem': answer / n,
        'no_end_token': sum(not r['has_end_token'] for r in subset),
        'residual_masks': sum(r['residual_mask_tokens'] for r in subset),
        'capped': sum(s.get('capped_incomplete', False) for s in stats),
        'status': {},
    }
    for r in subset:
        out['status'][r['status']] = out['status'].get(r['status'], 0) + 1
    if all('remasked_per_step' in s for s in stats):
        out['remasked tokens/problem'] = sum(sum(c[0] for c in s['remasked_per_step']) for s in stats) / n
        out['edited tokens/problem'] = sum(sum(c[0] for c in s['edited_per_step']) for s in stats) / n
    return out


def paired(a, b, seed=0, boots=10000):
    """a, b: dict test_index -> passed. Returns difference b-a in points with bootstrap CI and discordant counts."""
    keys = sorted(set(a) & set(b))
    diffs = [int(b[k]) - int(a[k]) for k in keys]
    rng = random.Random(seed)
    n = len(keys)
    samples = []
    for _ in range(boots):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100)
    samples.sort()
    return {
        'n': n,
        'diff_pp': 100 * sum(diffs) / n,
        'ci95': (samples[int(0.025 * boots)], samples[int(0.975 * boots) - 1]),
        'only_a': sum(d == -1 for d in diffs),
        'only_b': sum(d == 1 for d in diffs),
        'both': sum(a[k] and b[k] for k in keys),
    }


def main():
    rows = load(sys.argv[1:])
    modes = list(dict.fromkeys(r['mode'] for r in rows))
    summaries = {m: mode_summary([r for r in rows if r['mode'] == m]) for m in modes}
    metrics = ['n', 'pass@1 (%)', 'passed', 'forward/problem', 'sec/problem', 'unmasked tok/step',
               'answer tok/step', 'final tok/step', 'unmasked tok/s', 'answer tok/s',
               'answer tokens/problem', 'no_end_token', 'residual_masks', 'capped',
               'remasked tokens/problem', 'edited tokens/problem']
    print('| metric | ' + ' | '.join(modes) + ' |')
    print('|---|' + '---|' * len(modes))
    for k in metrics:
        vals = []
        for m in modes:
            v = summaries[m].get(k, '')
            vals.append(f'{v:.2f}' if isinstance(v, float) else str(v))
        print(f'| {k} | ' + ' | '.join(vals) + ' |')
    print('\nstatus counts:')
    for m in modes:
        print(f'  {m}: {summaries[m]["status"]}')
    if 'low_confidence' in modes:
        base = {r['test_index']: r['passed'] for r in rows if r['mode'] == 'low_confidence'}
        print('\npaired vs low_confidence (vanilla):')
        for m in modes:
            if m == 'low_confidence':
                continue
            other = {r['test_index']: r['passed'] for r in rows if r['mode'] == m}
            p = paired(base, other)
            print(f"  {m}: n={p['n']} diff={p['diff_pp']:+.2f}pp CI95=[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] "
                  f"vanilla_only={p['only_a']} {m}_only={p['only_b']} both={p['both']}")
    if 'random_pivot' in modes:
        base = {r['test_index']: r['passed'] for r in rows if r['mode'] == 'random_pivot'}
        print('\npaired vs random_pivot:')
        for m in modes:
            if m in ('random_pivot', 'low_confidence'):
                continue
            other = {r['test_index']: r['passed'] for r in rows if r['mode'] == m}
            p = paired(base, other)
            print(f"  {m}: n={p['n']} diff={p['diff_pp']:+.2f}pp CI95=[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] "
                  f"pivot_only={p['only_a']} {m}_only={p['only_b']} both={p['both']}")
    json.dump(summaries, open(Path(sys.argv[1]).parent / 'humaneval_comparison.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
