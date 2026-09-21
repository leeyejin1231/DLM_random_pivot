"""Final per-benchmark table: vanilla / Hierarchy-dLLM / WINO / ours, with McNemar vs ours.

Usage: python final_table.py [--family llada|dream] [--markdown]
Metrics: performance (accuracy, pass@1, or IFEval prompt-level strict accuracy),
tokens/step (unmasked positions per forward), throughput (unmasked tokens per second).
Shards are merged; partial runs are flagged.
"""
import argparse
import glob
import json
from math import comb
from pathlib import Path

PERF = {'gsm8k': 'acc', 'humaneval': 'pass@1', 'math500': 'acc', 'mbpp': 'pass@1', 'ifeval': 'prompt-strict'}
TOTAL = {'gsm8k': 1319, 'humaneval': 164, 'math500': 500, 'mbpp': 500, 'ifeval': 541}
SUCCESS = {'gsm8k': lambda r: r['correct'], 'humaneval': lambda r: r['passed'],
           'math500': lambda r: r['correct'], 'mbpp': lambda r: r['passed'],
           'ifeval': lambda r: r['ifeval']['prompt_strict']}
NAMES = {'vanilla': 'Vanilla', 'hier': 'Hierarchy-dLLM', 'wino': 'WINO', 'ours': 'Ours'}

# family -> task -> (directory pattern with {cfg}, {method: cfg})
FAMILY = {
    'llada': {
        'gsm8k': ('results/gsm8k_full_g128_b32_{cfg}',
                  {'vanilla': 'vanilla_s128', 'hier': 'hier_r03', 'wino': 'wino', 'ours': 't075_p32_s2'}),
        'humaneval': ('results/humaneval_g256_b32_{cfg}',
                      {'vanilla': 'vanilla_s256', 'hier': 'hier_r03', 'wino': 'wino', 'ours': 't075_p32_s2'}),
        'math500': ('results/math500_g256_b32_{cfg}',
                    {'vanilla': 'vanilla', 'hier': 'hier', 'wino': 'wino', 'ours': 'ours'}),
        'mbpp': ('results/mbpp_g256_b32_{cfg}',
                 {'vanilla': 'vanilla', 'hier': 'hier', 'wino': 'wino', 'ours': 'ours'}),
        'ifeval': ('results/ifeval_g512_b32_{cfg}',
                   {'vanilla': 'vanilla', 'hier': 'hier', 'wino': 'wino', 'ours': 'ours'}),
    },
    'dream': {task: (f'results/dream_{task}_g{gen}_b32_{{cfg}}',
                     {m: m for m in ('vanilla', 'hier', 'wino', 'ours')})
              for task, gen in (('gsm8k', 128), ('humaneval', 256), ('math500', 256), ('mbpp', 256), ('ifeval', 512))},
}


def load(pattern, cfg):
    base = pattern.format(cfg=cfg)
    rows = {}
    for d in sorted(glob.glob(base) + glob.glob(base + '_shard*')):
        path = Path(d) / 'results.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                rows[r['test_index']] = r
    return rows


def metrics(task, rows):
    R = list(rows.values())
    ok = SUCCESS[task]
    return {'n': len(R), 'perf': 100 * sum(ok(r) for r in R) / len(R),
            'tps': sum(r['unmasked_tokens'] for r in R) / sum(r['forward_steps'] for r in R),
            'tok_s': sum(r['unmasked_tokens'] for r in R) / sum(r['decode_seconds'] for r in R)}


def mcnemar(task, a, b):
    ok = SUCCESS[task]
    common = set(a) & set(b)
    x = sum(ok(a[i]) and not ok(b[i]) for i in common)
    y = sum(ok(b[i]) and not ok(a[i]) for i in common)
    n = x + y
    return x, y, (min(1.0, sum(comb(n, k) for k in range(min(x, y) + 1)) * 2 / 2 ** n) if n else 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', default='llada', choices=sorted(FAMILY))
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()
    spec = FAMILY[args.family]
    for task, (pattern, cfgs) in spec.items():
        rows = {m: load(pattern, cfg) for m, cfg in cfgs.items()}
        print(f'\n### {task} ({PERF[task]}, n={TOTAL[task]})')
        if args.markdown:
            print('| method | perf | tok/step | tok/s | n | p vs ours (other:ours) |')
            print('|---|---|---|---|---|---|')
        for key, r in rows.items():
            if not r:
                print(f'| {NAMES[key]} | (no results) | | | | |' if args.markdown else f'{NAMES[key]:15s} (no results)')
                continue
            m = metrics(task, r)
            partial = '' if m['n'] >= TOTAL[task] else ' (partial)'
            if key != 'ours' and rows['ours']:
                x, y, p = mcnemar(task, r, rows['ours'])
                pv = f'{p:.3f} ({x}:{y})'
            else:
                pv = '—'
            if args.markdown:
                print(f"| {NAMES[key]} | {m['perf']:.2f} | {m['tps']:.2f} | {m['tok_s']:.1f} | {m['n']}{partial} | {pv} |")
            else:
                print(f"{NAMES[key]:15s} perf={m['perf']:6.2f} tok/step={m['tps']:5.2f} tok/s={m['tok_s']:6.1f} "
                      f"n={m['n']:4d}{partial}  p_vs_ours={pv}")


if __name__ == '__main__':
    main()
