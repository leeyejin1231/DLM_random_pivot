"""Final per-benchmark table: vanilla / Hierarchy-dLLM / WINO / ours, with McNemar vs ours.

Usage: python final_table.py [--ours t075_p32_s2] [--markdown]
Metrics: performance (accuracy, pass@1, or IFEval prompt-level strict accuracy),
tokens/step (unmasked positions per forward), throughput (unmasked tokens per second).
"""
import argparse

from select_config import BENCH, BASELINES, load, metrics, mcnemar

OURS_DIRS = {'t075_p32_s2': {'math500': 'ours', 'mbpp': 'ours', 'ifeval': 'ours'}}
NAMES = {'vanilla': 'Vanilla', 'hier': 'Hierarchy-dLLM', 'wino': 'WINO', 'ours': 'Ours'}
PERF = {'gsm8k': 'acc', 'humaneval': 'pass@1', 'math500': 'acc', 'mbpp': 'pass@1', 'ifeval': 'prompt-strict'}
TOTAL = {t: v[2] for t, v in BENCH.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours', default='t075_p32_s2')
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()
    for task in BENCH:
        rows = {}
        for key in ('vanilla', 'hier'):
            rows[key] = load(task, BASELINES[task][key])
        rows['wino'] = load(task, 'wino')
        rows['ours'] = load(task, OURS_DIRS.get(args.ours, {}).get(task, args.ours))
        print(f'\n### {task} ({PERF[task]}, n={TOTAL[task]})')
        if args.markdown:
            print('| method | perf | tok/step | tok/s | n | p vs ours (a:b) |')
            print('|---|---|---|---|---|---|')
        for key, r in rows.items():
            if not r:
                print(f'| {NAMES[key]} | (no results) |' if args.markdown else f'{NAMES[key]:15s} (no results)')
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
                print(f"{NAMES[key]:15s} perf={m['perf']:6.2f} tok/step={m['tps']:5.2f} tok/s={m['tok_s']:6.1f} n={m['n']:4d}{partial}  p_vs_ours={pv}")


if __name__ == '__main__':
    main()
