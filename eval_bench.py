"""Resumable MATH-500 / MBPP / IFEval benchmark for the local LLaDA decoders (one GPU).

Protocol mirrors eval_gsm8k.py: same checkpoint, block 32, temperature 0, BF16,
batch 1, per-question rotating method order, per-question seed 42 + index.

Tasks
  math500  HuggingFaceH4/MATH-500 test (500). Accuracy of the last \\boxed{} answer
           under Hendrycks-style string normalization (plus numeric equality).
  mbpp     google-research-datasets/mbpp full test (500). pass@1: extracted code +
           test_setup_code + the three test_list asserts, run in a subprocess.
  ifeval   google/IFEval (541). Prompt-level strict accuracy (primary), plus
           instruction-level strict and both loose variants, using the checkers
           vendored from lm-evaluation-harness in ifeval_lib/.

--shard k/n evaluates only indices with index % n == k into its own output
directory; report_bench.py merges shards.
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


# ----------------------------------------------------------------------------- math500

def last_boxed(text):
    """Return the content of the last \\boxed{...} / \\fbox{...} with balanced braces, else None."""
    start = max(text.rfind('\\boxed'), text.rfind('\\fbox'))
    if start < 0:
        return None
    i = text.find('{', start)
    if i < 0:
        tail = text[start:].split('$')[0]
        tail = tail.replace('\\boxed', '').replace('\\fbox', '').strip()
        return tail or None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += '\\frac'
            if substr and substr[0] == '{':
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != '{':
                    new_str += '{' + a + '}{' + b + '}' + substr[2:]
                else:
                    new_str += '{' + a + '}' + b + substr[2:]
    return new_str


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a, b = string.split('/')
    try:
        a, b = int(a), int(b)
        assert string == f'{a}/{b}'
        return '\\frac{' + str(a) + '}{' + str(b) + '}'
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string):
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != '{':
            new_string += '\\sqrt{' + split[0] + '}' + split[1:]
        else:
            new_string += '\\sqrt' + split
    return new_string


def strip_string(string):
    """Hendrycks MATH normalization (math_equivalence.strip_string) with a few extras."""
    string = str(string).replace('\n', '').replace('\\!', '').replace('\\\\', '\\')
    string = string.replace('tfrac', 'frac').replace('dfrac', 'frac')
    string = string.replace('\\left', '').replace('\\right', '')
    string = string.replace('^{\\circ}', '').replace('^\\circ', '')
    string = string.replace('\\$', '').replace('$', '')
    string = re.sub(r'\\text\{\s*([^}]*)\}', r'\1', string) if '\\text{ ' not in string else _remove_right_units(string)
    string = string.replace('\\%', '').replace('%', '')
    string = string.replace(' .', ' 0.').replace('{.', '{0.')
    if string and string[0] == '.':
        string = '0' + string
    if len(string.split('=')) == 2 and len(string.split('=')[0]) <= 2:
        string = string.split('=')[1]
    string = _fix_sqrt(string)
    string = string.replace(' ', '')
    string = _fix_fracs(string)
    if string == '0.5':
        string = '\\frac{1}{2}'
    string = _fix_a_slash_b(string)
    string = string.replace(',\\!', '').rstrip('.')
    return string


def math_equal(prediction, gold):
    if prediction is None:
        return False
    a, b = strip_string(prediction), strip_string(gold)
    if a == b:
        return True
    try:
        return abs(float(a.replace(',', '')) - float(b.replace(',', ''))) < 1e-6
    except ValueError:
        return False


# ----------------------------------------------------------------------------- code

def extract_code(text):
    fenced = re.findall(r'```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)', text, flags=re.S)
    return (fenced[0], 'fence') if fenced else (text, 'raw')


def run_program(source, timeout, workdir):
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
    if any('SyntaxError' in line for line in tail[-3:]):
        status = 'syntax_error'
    elif tail and 'AssertionError' in tail[-1]:
        status = 'assertion_failed'
    return False, status, '\n'.join(tail[-5:])


# ----------------------------------------------------------------------------- ifeval

def ifeval_score(sample, response):
    from ifeval_lib import instructions_registry
    ids = sample['instruction_id_list']
    strict, loose = [], []
    variants = [response]
    stripped = response.replace('*', '')
    lines = response.split('\n')
    no_first = '\n'.join(lines[1:]).strip()
    no_last = '\n'.join(lines[:-1]).strip()
    no_both = '\n'.join(lines[1:-1]).strip()
    for v in (no_first, no_last, no_both, stripped, no_first.replace('*', ''),
              no_last.replace('*', ''), no_both.replace('*', '')):
        variants.append(v)
    for index, instruction_id in enumerate(ids):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        kwargs = {k: v for k, v in sample['kwargs'][index].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=sample['prompt'])
        strict.append(bool(response.strip()) and instruction.check_following(response))
        loose.append(any(v.strip() and instruction.check_following(v) for v in variants))
    return {'strict_follow': strict, 'loose_follow': loose,
            'prompt_strict': all(strict), 'prompt_loose': all(loose)}


# ----------------------------------------------------------------------------- shared

TASKS = {
    'math500': {
        'question': lambda s: s['problem'],
        'suffix': '\n\nSolve this problem step by step. Put your final answer within \\boxed{}.',
        'warmup': 'What is $2 + 3$?',
    },
    'mbpp': {
        'question': lambda s: ('You are an expert Python programmer, and here is your task: '
                               + s['text'].strip() + ' Your code should pass these tests:\n\n'
                               + '\n'.join(s['test_list'])),
        'suffix': '\n\nReply with the complete solution in a ```python code block.',
        'warmup': ('You are an expert Python programmer, and here is your task: Write a function '
                   'to add two numbers. Your code should pass these tests:\n\nassert add(1, 2) == 3'),
    },
    'ifeval': {
        'question': lambda s: s['prompt'],
        'suffix': '',
        'warmup': 'Write a short poem about the sea. Do not use any commas.',
    },
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def summarize(rows, task):
    summary = {}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        subset = [r for r in rows if r['mode'] == mode]
        seconds = sum(r['decode_seconds'] for r in subset)
        steps = sum(r['forward_steps'] for r in subset)
        unmasked = sum(r['unmasked_tokens'] for r in subset)
        entry = {
            'n': len(subset),
            'decode_seconds': seconds,
            'mean_seconds_per_problem': seconds / len(subset),
            'unmasked_tokens_per_second': unmasked / seconds,
            'answer_tokens_per_second': sum(r['answer_tokens'] for r in subset) / seconds,
            'unmasked_tokens_per_step': unmasked / steps,
            'mean_forward_steps': steps / len(subset),
            'mean_answer_tokens': sum(r['answer_tokens'] for r in subset) / len(subset),
            'no_end_token_count': sum(not r['has_end_token'] for r in subset),
            'residual_mask_tokens': sum(r['residual_mask_tokens'] for r in subset),
            'peak_allocated_gib': max(r['peak_allocated_gib'] for r in subset),
        }
        if task == 'math500':
            entry['accuracy_percent'] = 100 * sum(r['correct'] for r in subset) / len(subset)
            entry['missing_boxed_count'] = sum(r['prediction'] is None for r in subset)
        elif task == 'mbpp':
            entry['pass_at_1_percent'] = 100 * sum(r['passed'] for r in subset) / len(subset)
            statuses = {}
            for r in subset:
                statuses[r['status']] = statuses.get(r['status'], 0) + 1
            entry['status_counts'] = statuses
        else:
            n_inst = sum(len(r['ifeval']['strict_follow']) for r in subset)
            entry.update({
                'prompt_strict_accuracy_percent': 100 * sum(r['ifeval']['prompt_strict'] for r in subset) / len(subset),
                'prompt_loose_accuracy_percent': 100 * sum(r['ifeval']['prompt_loose'] for r in subset) / len(subset),
                'instruction_strict_accuracy_percent': 100 * sum(sum(r['ifeval']['strict_follow']) for r in subset) / n_inst,
                'instruction_loose_accuracy_percent': 100 * sum(sum(r['ifeval']['loose_follow']) for r in subset) / n_inst,
            })
        if all('decoder_stats' in r for r in subset):
            entry['capped_incomplete_count'] = sum(r['decoder_stats'].get('capped_incomplete', False) for r in subset)
            for diagnostic in ('deferred', 'fallback', 'bulk_tokens'):
                entry[f'{diagnostic}_total'] = sum(
                    sum(sum(c) for c in r['decoder_stats'].get(f'{diagnostic}_per_step', [])) for r in subset)
        summary[mode] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, choices=sorted(TASKS))
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True, help='Cached Arrow file of the task split')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen-length', type=int, default=256)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--exec-timeout', type=float, default=10.0)
    parser.add_argument('--limit', type=int, default=None, help='Evaluate only the first N tasks (debug)')
    parser.add_argument('--shard', default=None, help='k/n: evaluate indices with index %% n == k')
    parser.add_argument('--variant-config', required=True,
                        help='JSON mapping experiment labels to generate keyword arguments')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this benchmark requires a GPU.')
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    task = TASKS[args.task]
    dataset = Dataset.from_file(args.dataset)
    indices = list(range(len(dataset)))
    if args.shard:
        k, n = (int(v) for v in args.shard.split('/'))
        indices = [i for i in indices if i % n == k]
    indices = indices[:args.limit]
    variants = json.loads(Path(args.variant_config).read_text())
    for label, kwargs in variants.items():
        if not isinstance(kwargs, dict) or 'remasking' not in kwargs:
            raise ValueError(f'{label}: specify generate keyword arguments including remasking.')
    modes_to_run = tuple(variants)
    config = vars(args).copy()
    config.update({
        'model_revision': Path(args.model).name,
        'num_tasks': len(indices), 'batch_size': 1, 'num_fewshot': 0,
        'variants': variants,
        'temperature': 0, 'cfg_scale': 0, 'dtype': 'bfloat16',
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'prompt_suffix': task['suffix'],
        'decode_sha256': hashlib.sha256(Path(__file__).with_name('decode.py').read_bytes()).hexdigest(),
        'hierarchy_decode_sha256': hashlib.sha256(Path(__file__).with_name('hierarchy_decode.py').read_bytes()).hexdigest(),
        'decode_confidence_v2_sha256': hashlib.sha256(Path(__file__).with_name('decode_confidence_v2.py').read_bytes()).hexdigest(),
        'wino_decode_sha256': hashlib.sha256(Path(__file__).with_name('wino_decode.py').read_bytes()).hexdigest(),
        'eval_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metrics': {
            'timing': 'CUDA-synchronized wall time around generate; excludes tokenization, loading, warmup, and scoring.',
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
        print(json.dumps(summarize(rows, args.task), indent=2), flush=True)
        return
    print(f'Loading {args.model} on {config["gpu"]}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if any(v['remasking'] == 'wino' for v in variants.values()):
        model = load_wino_model(args.model)
    else:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                        local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
    eos = tokenizer.eos_token_id
    end_ids = {126081, 126348}
    if eos is not None:
        end_ids.add(eos)
    special_ids = set(tokenizer.all_special_ids) | end_ids | {126336}

    def encode(question):
        messages = [{'role': 'user', 'content': question + task['suffix']}]
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
        return decoder(model, encoded['input_ids'], attention_mask=encoded['attention_mask'],
                       gen_length=args.gen_length, block_length=args.block_length, steps=args.steps,
                       temperature=0, cfg_scale=0, return_stats=True, **kwargs)

    warmup = encode(task['warmup'])
    for mode in modes_to_run:
        torch.manual_seed(args.seed)
        run(warmup, mode)
        torch.cuda.synchronize()
        print(f'Warmup complete: {mode}', flush=True)
    del warmup

    with records_path.open('a', buffering=1) as output:
        for ordinal, index in enumerate(indices):
            sample = dataset[index]
            encoded = encode(task['question'](sample))
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
                counts = [count[0] for count in stats['unmasked_per_step']]
                row = {
                    'test_index': index, 'mode': mode, 'seed': sample_seed,
                    'text': text, 'raw_text': tokenizer.decode(generated, skip_special_tokens=False),
                    'has_end_token': end < len(generated),
                    'generated_ids': generated, 'prompt_tokens': encoded['input_ids'].shape[1],
                    'answer_tokens': sum(token not in special_ids for token in generated[:end]),
                    'unmasked_tokens': sum(counts), 'unmasked_per_step': counts,
                    'final_unmasked_tokens': sum(token != 126336 for token in generated),
                    'decoder_stats': stats,
                    'forward_steps': stats['forward_steps'], 'decode_seconds': elapsed,
                    'residual_mask_tokens': generated.count(126336),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                }
                if args.task == 'math500':
                    prediction = last_boxed(text)
                    row.update({'unique_id': sample['unique_id'], 'gold': sample['answer'],
                                'prediction': prediction, 'correct': math_equal(prediction, sample['answer'])})
                    score = f'correct={row["correct"]}'
                elif args.task == 'mbpp':
                    code, code_source = extract_code(text)
                    for stop in ('\nif __name__', '\nprint(', '\nassert '):
                        cut = code.find(stop)
                        if cut != -1:
                            code = code[:cut]
                    program = (sample['test_setup_code'] or '') + '\n' + code.rstrip() + '\n\n' + '\n'.join(sample['test_list']) + '\n'
                    passed, status, error_tail = run_program(program, args.exec_timeout, workdir)
                    row.update({'task_id': sample['task_id'], 'code': code, 'code_source': code_source,
                                'program': program, 'passed': passed, 'status': status, 'error_tail': error_tail})
                    score = f'passed={passed} ({status})'
                else:
                    result = ifeval_score(sample, text)
                    row.update({'key': sample['key'], 'ifeval': result})
                    score = f'strict={result["prompt_strict"]} loose={result["prompt_loose"]}'
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
                done.add((index, mode))
                write_json(out_dir / 'summary.json', summarize(rows, args.task))
                print(f'[{len(done)}/{len(indices) * len(modes_to_run)}] index={index} {mode}: {score} '
                      f'steps={row["forward_steps"]} time={elapsed:.2f}s '
                      f'tokens/s={row["unmasked_tokens"]/elapsed:.2f} '
                      f'tokens/step={row["unmasked_tokens"]/row["forward_steps"]:.2f}', flush=True)
                del tokens
    print('COMPLETE\n' + json.dumps(summarize(rows, args.task), indent=2), flush=True)


if __name__ == '__main__':
    main()
