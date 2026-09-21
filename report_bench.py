"""Merge eval_bench.py outputs (including shards) and print a per-task comparison table.

Usage: python report_bench.py [--tasks math500 mbpp ifeval] [--methods vanilla hier ours]
Reads results/<task>_g*_b32_<method>[_shard*]/results.jsonl. Performance is accuracy
(math500), pass@1 (mbpp) or prompt-level strict accuracy (ifeval). Paired McNemar exact
tests are computed on the test indices shared by both methods.
"""
import argparse
import glob
import json
from math import comb
from pathlib import Path


def load(task, method):
    rows = {}
    dirs = sorted(glob.glob(f'results/{task}_g*_b32_{method}') + glob.glob(f'results/{task}_g*_b32_{method}_shard*'))
    for d in dirs:
        path = Path(d) / 'results.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows[r['test_index']] = r
    return rows, dirs


def success(task, r):
    return {'math500': lambda: r['correct'], 'mbpp': lambda: r['passed'],
            'ifeval': lambda: r['ifeval']['prompt_strict']}[task]()


def mcnemar(a, b, task):
    common = set(a) & set(b)
    x = sum(success(task, a[i]) and not success(task, b[i]) for i in common)
    y = sum(success(task, b[i]) and not success(task, a[i]) for i in common)
    n = x + y
    p = min(1.0, sum(comb(n, k) for k in range(min(x, y) + 1)) * 2 / 2 ** n) if n else 1.0
    return len(common), x, y, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['math500', 'mbpp', 'ifeval'])
    parser.add_argument('--methods', nargs='+', default=['vanilla', 'hier', 'ours'])
    args = parser.parse_args()
    total = {'math500': 500, 'mbpp': 500, 'ifeval': 541}
    for task in args.tasks:
        print(f'\n=== {task} ===')
        data = {}
        for method in args.methods:
            rows, dirs = load(task, method)
            if not rows:
                print(f'{method:8s} (no results yet)')
                continue
            data[method] = rows
            R = list(rows.values())
            steps = sum(r['forward_steps'] for r in R)
            secs = sum(r['decode_seconds'] for r in R)
            unm = sum(r['unmasked_tokens'] for r in R)
            perf = 100 * sum(success(task, r) for r in R) / len(R)
            extra = ''
            if task == 'ifeval':
                n_inst = sum(len(r['ifeval']['strict_follow']) for r in R)
                extra = (f" inst_strict={100*sum(sum(r['ifeval']['strict_follow']) for r in R)/n_inst:.2f}"
                         f" prompt_loose={100*sum(r['ifeval']['prompt_loose'] for r in R)/len(R):.2f}")
            elif task == 'mbpp':
                sc = {}
                for r in R:
                    sc[r['status']] = sc.get(r['status'], 0) + 1
                extra = f' status={sc}'
            elif task == 'math500':
                extra = f" missing_boxed={sum(r['prediction'] is None for r in R)}"
            print(f"{method:8s} n={len(R):3d}/{total[task]} perf={perf:6.2f} tok/step={unm/steps:5.2f} "
                  f"tok/s={unm/secs:6.1f} steps={steps/len(R):6.1f} no_end={sum(not r['has_end_token'] for r in R)}{extra}")
        if 'ours' in data:
            for ref in ('vanilla', 'hier'):
                if ref in data:
                    n, x, y, p = mcnemar(data[ref], data['ours'], task)
                    print(f'  McNemar ours vs {ref:8s}: n={n} {ref}_only={x} ours_only={y} p={p:.4f}')


if __name__ == '__main__':
    main()
