"""Paired, resumable HumanEval (164 tasks) benchmark for the local LLaDA decoders (one GPU).

Protocol mirrors eval_gsm8k.py: same checkpoint, gen 256 / block 32 / steps 256,
temperature 0, BF16, batch 1, per-question alternating method order, per-question
seed 42 + index. pass@1 uses one greedy sample per task and the official
HumanEval test harness logic (prompt + completion + test + check(entry_point))
executed in a subprocess with a timeout.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

from decode import generate
from hierarchy_decode import generate_hierarchy
from decode_confidence_v2 import generate as generate_confidence_pivot_v2
from wino_decode import generate as generate_wino, load_model as load_wino_model
from dream_adapter import load_model as load_dream_model, MASK_ID as DREAM_MASK_ID, END_IDS as DREAM_END_IDS


MODES = ('low_confidence', 'random_pivot')
STOP_SEQUENCES = ('\nif __name__', '\nprint(', '\nassert ')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def extract_code(text):
    """Return (code, source). Prefer the first fenced python block, then any fence, else raw text."""
    fenced = re.findall(r'```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)', text, flags=re.S)
    if fenced:
        return fenced[0], 'fence'
    return text, 'raw'


def build_program(prompt, code, entry_point):
    """Combine prompt and model code the way lm-eval/Fast-dLLM instruct HumanEval evals do."""
    code = code.rstrip()
    defines = re.search(rf'^\s*def\s+{re.escape(entry_point)}\s*\(', code, flags=re.M) is not None
    if defines:
        # The prompt's function body is docstring-only, so re-defining it below is valid.
        return prompt.rstrip() + '\n\n\n' + code + '\n', 'full_function'
    # Continuation: cut at typical trailing test/print blocks and indent if needed.
    for stop in STOP_SEQUENCES:
        cut = code.find(stop)
        if cut != -1:
            code = code[:cut]
    lines = code.splitlines()
    first = next((line for line in lines if line.strip()), '')
    if first and not first.startswith((' ', '\t')):
        lines = ['    ' + line if line.strip() else line for line in lines]
    return prompt + '\n'.join(lines) + '\n', 'continuation'


def run_tests(program, test, entry_point, timeout, workdir):
    """Execute program + test + check(entry_point) in a fresh interpreter. Returns (passed, status, stderr_tail)."""
    source = program + '\n\n' + test + '\n\n' + f'check({entry_point})\n'
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        path = Path(tmp) / 'candidate.py'
        path.write_text(source)
        try:
            proc = subprocess.run(
                [sys.executable, '-I', str(path)], cwd=tmp, capture_output=True, text=True,
                timeout=timeout, env={'PATH': os.environ.get('PATH', ''), 'PYTHONHASHSEED': '0'})
        except subprocess.TimeoutExpired:
            return False, 'timeout', ''
    if proc.returncode == 0:
        return True, 'passed', ''
    tail = proc.stderr.strip().splitlines()
    status = 'failed'
    if tail and 'SyntaxError' in tail[-1] or any('SyntaxError' in line for line in tail[-3:]):
        status = 'syntax_error'
    elif tail and 'AssertionError' in tail[-1]:
        status = 'assertion_failed'
    return False, status, '\n'.join(tail[-5:])


def summarize(rows):
    summary = {}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        subset = [r for r in rows if r['mode'] == mode]
        seconds = sum(r['decode_seconds'] for r in subset)
        steps = sum(r['forward_steps'] for r in subset)
        unmasked = sum(r['unmasked_tokens'] for r in subset)
        statuses = {}
        for r in subset:
            statuses[r['status']] = statuses.get(r['status'], 0) + 1
        summary[mode] = {
            'n': len(subset),
            'pass_at_1_percent': 100 * sum(r['passed'] for r in subset) / len(subset),
            'passed': sum(r['passed'] for r in subset),
            'status_counts': statuses,
            'decode_seconds': seconds,
            'mean_seconds_per_problem': seconds / len(subset),
            'unmasked_tokens_per_second': unmasked / seconds,
            'answer_tokens_per_second': sum(r['answer_tokens'] for r in subset) / seconds,
            'unmasked_tokens_per_step': unmasked / steps,
            'mean_forward_steps': steps / len(subset),
            'mean_answer_tokens': sum(r['answer_tokens'] for r in subset) / len(subset),
            'code_source_counts': {k: sum(r['code_source'] == k for r in subset) for k in ('fence', 'raw')},
            'program_kind_counts': {k: sum(r['program_kind'] == k for r in subset) for k in ('full_function', 'continuation')},
            'no_end_token_count': sum(not r['has_end_token'] for r in subset),
            'residual_mask_tokens': sum(r['residual_mask_tokens'] for r in subset),
            'peak_allocated_gib': max(r['peak_allocated_gib'] for r in subset),
        }
        if all('decoder_stats' in r for r in subset):
            summary[mode]['capped_incomplete_count'] = sum(
                r['decoder_stats'].get('capped_incomplete', False) for r in subset)
            for name in ('remasked', 'edited'):
                if all(f'{name}_per_step' in r['decoder_stats'] for r in subset):
                    summary[mode][f'{name}_tokens_total'] = sum(
                        sum(c[0] for c in r['decoder_stats'][f'{name}_per_step']) for r in subset)
        net = sum(r['final_unmasked_tokens'] for r in subset)
        summary[mode].update({
            'answer_tokens_per_step': sum(r['answer_tokens'] for r in subset) / steps,
            'final_tokens_per_second': net / seconds,
            'final_tokens_per_step': net / steps,
        })
    if all(mode in summary for mode in MODES):
        paired = {}
        for row in rows:
            paired.setdefault(row['test_index'], {})[row['mode']] = row
        pairs = [p for p in paired.values() if all(m in p for m in MODES)]
        if pairs:
            summary['paired'] = {
                'n': len(pairs),
                'decode_speedup': sum(p['low_confidence']['decode_seconds'] for p in pairs) / sum(p['random_pivot']['decode_seconds'] for p in pairs),
                'vanilla_only_passed': sum(p['low_confidence']['passed'] and not p['random_pivot']['passed'] for p in pairs),
                'pivot_only_passed': sum(p['random_pivot']['passed'] and not p['low_confidence']['passed'] for p in pairs),
                'both_passed': sum(p['random_pivot']['passed'] and p['low_confidence']['passed'] for p in pairs),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True, help='Cached HumanEval test Arrow file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen-length', type=int, default=256)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--exec-timeout', type=float, default=10.0)
    parser.add_argument('--limit', type=int, default=None, help='Evaluate only the first N tasks (debug)')
    parser.add_argument('--model-family', choices=('llada', 'dream'), default='llada',
                        help='dream: Dream-v0-Instruct-7B via dream_adapter (shifted logits, full attention)')
    parser.add_argument('--variant-config', help='JSON mapping experiment labels to generate keyword arguments')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this benchmark requires a GPU.')
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_file(args.dataset)
    indices = list(range(len(dataset)))[:args.limit]
    variants = (json.loads(Path(args.variant_config).read_text()) if args.variant_config
                else {mode: {'remasking': mode} for mode in MODES})
    for label, kwargs in variants.items():
        if not isinstance(kwargs, dict) or 'remasking' not in kwargs:
            raise ValueError(f'{label}: specify generate keyword arguments including remasking.')
    modes_to_run = tuple(variants)
    prompt_template = ('Complete the following Python function. Reply with the complete '
                       'function definition in a ```python code block.\n\n```python\n{prompt}```')
    config = vars(args).copy()
    config.update({
        'model_revision': Path(args.model).name,
        'num_tasks': len(indices), 'batch_size': 1, 'num_fewshot': 0,
        'variants': variants,
        **({'model_family': args.model_family} if args.model_family != 'llada' else {}),
        'temperature': 0, 'cfg_scale': 0, 'dtype': 'bfloat16',
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'prompt_template': prompt_template,
        'decode_sha256': hashlib.sha256(Path(__file__).with_name('decode.py').read_bytes()).hexdigest(),
        'hierarchy_decode_sha256': hashlib.sha256(Path(__file__).with_name('hierarchy_decode.py').read_bytes()).hexdigest(),
        'decode_confidence_v2_sha256': hashlib.sha256(Path(__file__).with_name('decode_confidence_v2.py').read_bytes()).hexdigest(),
        'wino_decode_sha256': hashlib.sha256(Path(__file__).with_name('wino_decode.py').read_bytes()).hexdigest(),
        'eval_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metrics': {
            'pass_at_1': 'One greedy sample per task; prompt + extracted code + test + check(entry_point) executed in a subprocess with timeout.',
            'timing': 'CUDA-synchronized wall time around generate; excludes tokenization, loading, warmup, code extraction, and test execution.',
            'tokens_per_second': 'Actual unmasked positions including special tokens and post-EOS positions / decode seconds.',
            'answer_tokens_per_second': 'Non-special tokens before first EOS/EOT / decode seconds.',
            'tokens_per_step': 'Actual unmasked positions / forward calls, batch size 1.',
        },
    })
    config_path = out_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing output has a different configuration; use a new output directory.')
    write_json(config_path, config)
    source_dir = out_dir / 'source'
    source_dir.mkdir(exist_ok=True)
    for filename in ('decode.py', 'hierarchy_decode.py', 'decode_confidence_v2.py', 'wino_decode.py', Path(__file__).name):
        (source_dir / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    workdir = out_dir / 'exec_tmp'
    workdir.mkdir(exist_ok=True)
    records_path = out_dir / 'results.jsonl'
    rows = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    done = {(r['test_index'], r['mode']) for r in rows}
    if len(done) == len(indices) * len(modes_to_run):
        print(json.dumps(summarize(rows), indent=2), flush=True)
        return
    print(f'Loading {args.model} on {config["gpu"]}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    uses_wino = any(v['remasking'] == 'wino' for v in variants.values())
    if args.model_family == 'dream':
        if uses_wino:
            raise ValueError('WINO needs a custom attention mask and is not ported to Dream.')
        model = load_dream_model(args.model)
        mask_id, end_ids = DREAM_MASK_ID, set(DREAM_END_IDS)
    else:
        if uses_wino:
            model = load_wino_model(args.model)
        else:
            model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                            local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
        mask_id, end_ids = 126336, {126081, 126348}
    eos = tokenizer.eos_token_id
    if eos is not None:
        end_ids.add(eos)
    special_ids = set(tokenizer.all_special_ids) | end_ids | {mask_id}

    def encode(prompt):
        messages = [{'role': 'user', 'content': prompt_template.format(prompt=prompt)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')

    def run(encoded, mode):
        kwargs = dict(variants[mode])
        remasking = kwargs.pop('remasking')
        decoder = {'hierarchy': generate_hierarchy,
                   'confidence_pivot_v2': generate_confidence_pivot_v2,
                   'wino': generate_wino}.get(remasking, generate)
        if decoder is generate:
            kwargs['remasking'] = remasking
        return decoder(model, encoded['input_ids'],
                       attention_mask=encoded['attention_mask'] if args.model_family == 'llada' else None,
                       mask_id=mask_id,
                       gen_length=args.gen_length, block_length=args.block_length, steps=args.steps,
                       temperature=0, cfg_scale=0, return_stats=True, **kwargs)

    warmup = encode('def add(a: int, b: int) -> int:\n    """Return the sum of a and b.\n    >>> add(1, 2)\n    3\n    """\n')
    for mode in modes_to_run:
        torch.manual_seed(args.seed)
        run(warmup, mode)
        torch.cuda.synchronize()
        print(f'Warmup complete: {mode}', flush=True)
    del warmup

    with records_path.open('a', buffering=1) as output:
        for ordinal, index in enumerate(indices):
            sample = dataset[index]
            encoded = encode(sample['prompt'])
            offset = ordinal % len(modes_to_run)
            modes = modes_to_run[offset:] + modes_to_run[:offset]
            for mode in modes:
                if (index, mode) in done:
                    continue
                sample_seed = args.seed + index
                torch.manual_seed(sample_seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                tokens, stats = run(encoded, mode)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                generated = tokens[0, encoded['input_ids'].shape[1]:].tolist()
                end = next((i for i, token in enumerate(generated) if token in end_ids), len(generated))
                text = tokenizer.decode(generated[:end], skip_special_tokens=True)
                code, code_source = extract_code(text)
                program, program_kind = build_program(sample['prompt'], code, sample['entry_point'])
                passed, status, error_tail = run_tests(program, sample['test'], sample['entry_point'],
                                                       args.exec_timeout, workdir)
                counts = [count[0] for count in stats['unmasked_per_step']]
                row = {
                    'test_index': index, 'task_id': sample['task_id'], 'mode': mode, 'seed': sample_seed,
                    'entry_point': sample['entry_point'],
                    'passed': passed, 'status': status, 'error_tail': error_tail,
                    'code_source': code_source, 'program_kind': program_kind,
                    'has_end_token': end < len(generated),
                    'text': text, 'code': code, 'program': program,
                    'raw_text': tokenizer.decode(generated, skip_special_tokens=False),
                    'generated_ids': generated, 'prompt_tokens': encoded['input_ids'].shape[1],
                    'answer_tokens': sum(token not in special_ids for token in generated[:end]),
                    'unmasked_tokens': sum(counts), 'unmasked_per_step': counts,
                    'final_unmasked_tokens': sum(token != mask_id for token in generated),
                    'decoder_stats': stats,
                    'forward_steps': stats['forward_steps'], 'decode_seconds': elapsed,
                    'residual_mask_tokens': generated.count(mask_id),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                }
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
                done.add((index, mode))
                write_json(out_dir / 'summary.json', summarize(rows))
                print(f'[{len(done)}/{len(indices) * len(modes_to_run)}] {sample["task_id"]} {mode}: '
                      f'passed={passed} ({status}) steps={row["forward_steps"]} '
                      f'time={elapsed:.2f}s tokens/s={row["unmasked_tokens"]/elapsed:.2f} '
                      f'tokens/step={row["unmasked_tokens"]/row["forward_steps"]:.2f}', flush=True)
                del tokens
    print('COMPLETE\n' + json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == '__main__':
    main()
