"""Cross-benchmark table for the p32/s2 tau sweep and a Pareto-style selection.

For each config and benchmark: performance, tokens/step, tokens/s, and McNemar exact
p-value vs vanilla (and vs Hierarchy-dLLM). Then lists configs with no significant loss
against vanilla on any completed benchmark, ranked by mean tokens/s relative to hier.
"""
import glob
import json
from math import comb
from pathlib import Path

BENCH = {  # task: (dir pattern with {cfg}, success key, n)
    'gsm8k': ('results/gsm8k_full_g128_b32_{cfg}', lambda r: r['correct'], 1319),
    'humaneval': ('results/humaneval_g256_b32_{cfg}', lambda r: r['passed'], 164),
    'math500': ('results/math500_g256_b32_{cfg}', lambda r: r['correct'], 500),
    'mbpp': ('results/mbpp_g256_b32_{cfg}', lambda r: r['passed'], 500),
    'ifeval': ('results/ifeval_g512_b32_{cfg}', lambda r: r['ifeval']['prompt_strict'], 541),
}
BASELINES = {'gsm8k': {'vanilla': 'vanilla_s128', 'hier': 'hier_r03'},
             'humaneval': {'vanilla': 'vanilla_s256', 'hier': 'hier_r03'},
             'math500': {'vanilla': 'vanilla', 'hier': 'hier'},
             'mbpp': {'vanilla': 'vanilla', 'hier': 'hier'},
             'ifeval': {'vanilla': 'vanilla', 'hier': 'hier'}}
CONFIGS = {'t07_p32_s2': {'humaneval': 't07_p32_s2', 'gsm8k': 't07_p32_s2'},
           't075_p32_s2': {'gsm8k': 't075_p32_s2', 'humaneval': 't075_p32_s2',
                           'math500': 'ours', 'mbpp': 'ours', 'ifeval': 'ours'},
           't08_p32_s2': {}, 't085_p32_s2': {}, 't07_p32_s0': {}, 't075_p32_s0': {}, 't08_p32_s0': {}, 't075_p32_s1': {}}


def load(task, cfg):
    pattern = BENCH[task][0].format(cfg=cfg)
    rows = {}
    for d in sorted(glob.glob(pattern) + glob.glob(pattern + '_shard*')):
        path = Path(d) / 'results.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                rows[r['test_index']] = r
    return rows


def metrics(task, rows):
    R = list(rows.values())
    ok = BENCH[task][1]
    return {'n': len(R), 'perf': 100 * sum(ok(r) for r in R) / len(R),
            'tps': sum(r['unmasked_tokens'] for r in R) / sum(r['forward_steps'] for r in R),
            'tok_s': sum(r['unmasked_tokens'] for r in R) / sum(r['decode_seconds'] for r in R)}


def mcnemar(task, a, b):
    ok = BENCH[task][1]
    common = set(a) & set(b)
    x = sum(ok(a[i]) and not ok(b[i]) for i in common)
    y = sum(ok(b[i]) and not ok(a[i]) for i in common)
    n = x + y
    return x, y, (min(1.0, sum(comb(n, k) for k in range(min(x, y) + 1)) * 2 / 2 ** n) if n else 1.0)


def main():
    base = {t: {k: load(t, v) for k, v in BASELINES[t].items()} for t in BENCH}
    print(f"{'config':12s} {'bench':9s} {'n':>4s} {'perf':>6s} {'tok/step':>8s} {'tok/s':>6s} | vanilla p | hier p | hier perf/tok/s")
    table = {}
    for cfg, overrides in CONFIGS.items():
        for task in BENCH:
            rows = load(task, overrides.get(task, cfg))
            if not rows:
                continue
            m = metrics(task, rows)
            xv, yv, pv = mcnemar(task, base[task]['vanilla'], rows)
            xh, yh, ph = mcnemar(task, base[task]['hier'], rows)
            hm = metrics(task, base[task]['hier'])
            complete = m['n'] >= BENCH[task][2]
            table.setdefault(cfg, {})[task] = {**m, 'p_vanilla': pv, 'p_hier': ph, 'complete': complete,
                                              'hier_tok_s': hm['tok_s'], 'hier_perf': hm['perf']}
            flag = '' if complete else ' (partial)'
            print(f"{cfg:12s} {task:9s} {m['n']:4d} {m['perf']:6.2f} {m['tps']:8.2f} {m['tok_s']:6.1f} | "
                  f"{pv:.3f} ({xv}:{yv}) | {ph:.3f} ({xh}:{yh}) | {hm['perf']:.2f} / {hm['tok_s']:.1f}{flag}")
    print('\n=== selection: no significant loss vs vanilla (p >= 0.05) on every completed benchmark ===')
    for cfg, per in table.items():
        losses = [t for t, v in per.items() if v['complete'] and v['p_vanilla'] < 0.05 and v['perf'] < metrics(t, base[t]['vanilla'])['perf']]
        speed = sum(v['tok_s'] / v['hier_tok_s'] for v in per.values()) / len(per)
        print(f"{cfg:12s} benchmarks={len(per)} sig_losses={losses or 'none'} mean(tok/s ÷ hier)={speed:.3f}")


if __name__ == '__main__':
    main()
