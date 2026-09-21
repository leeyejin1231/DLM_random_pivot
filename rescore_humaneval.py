"""Re-score existing HumanEval results.jsonl files without regenerating.

The original runs stored correct programs but executed them with a doubled
relative path (run_tests received a relative workdir while using cwd=tmp), so
every row was marked failed. This re-runs the tests from the saved programs
and rewrites passed/status/error_tail, results.jsonl, and summary.json.

Usage: python rescore_humaneval.py <result_dir> [<result_dir> ...]
"""
import json
import sys
from pathlib import Path

from datasets import Dataset

from eval_humaneval import run_tests, summarize, write_json

DATASET = ('/scratch/ssgyejin/datasets/openai_humaneval/openai_humaneval/0.0.0/'
           '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow')


def main():
    dataset = Dataset.from_file(DATASET)
    by_index = {i: dataset[i] for i in range(len(dataset))}
    for arg in sys.argv[1:]:
        out_dir = Path(arg).resolve()
        workdir = out_dir / 'exec_tmp'
        workdir.mkdir(exist_ok=True)
        records_path = out_dir / 'results.jsonl'
        rows = [json.loads(line) for line in records_path.read_text().splitlines()]
        changed = 0
        for row in rows:
            sample = by_index[row['test_index']]
            assert sample['task_id'] == row['task_id']
            passed, status, error_tail = run_tests(
                row['program'], sample['test'], sample['entry_point'], 10.0, workdir)
            if (passed, status) != (row['passed'], row['status']):
                changed += 1
            row.update({'passed': passed, 'status': status, 'error_tail': error_tail})
        backup = records_path.with_suffix('.jsonl.pre_rescore')
        if not backup.exists():
            backup.write_text(records_path.read_text())
        records_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        write_json(out_dir / 'summary.json', summarize(rows))
        passed_count = sum(r['passed'] for r in rows)
        print(f'{out_dir.name}: {len(rows)} rows, {changed} changed, '
              f'pass@1={100 * passed_count / len(rows):.2f}%', flush=True)


if __name__ == '__main__':
    main()
