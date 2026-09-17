"""Paired, resumable GSM8K pilot for the local LLaDA decoders (one GPU)."""
import argparse
import hashlib
import json
import random
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

from decode import generate
from hierarchy_decode import generate_hierarchy
from decode_confidence import generate as generate_confidence_pivot


NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
MODES = ('low_confidence', 'random_pivot')


def normalize_number(value):
    try:
        return str(Decimal(value.replace(',', '').strip()).normalize())
    except InvalidOperation:
        return None


def extract_answer(text):
    marked = re.findall(r'####\s*\$?\s*(' + NUMBER + r')', text)
    if marked:
        return normalize_number(marked[-1]), True
    numbers = re.findall(NUMBER, text)
    return (normalize_number(numbers[-1]) if numbers else None), False


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def summarize(rows):
    summary = {}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        subset = [r for r in rows if r['mode'] == mode]
        if not subset:
            continue
        seconds = sum(r['decode_seconds'] for r in subset)
        steps = sum(r['forward_steps'] for r in subset)
        unmasked = sum(r['unmasked_tokens'] for r in subset)
        summary[mode] = {
            'n': len(subset),
            'accuracy_percent': 100 * sum(r['correct'] for r in subset) / len(subset),
            'strict_marker_accuracy_percent': 100 * sum(r['correct'] and r['has_answer_marker'] for r in subset) / len(subset),
            'decode_seconds': seconds,
            'mean_seconds_per_problem': seconds / len(subset),
            'unmasked_tokens_per_second': unmasked / seconds,
            'answer_tokens_per_second': sum(r['answer_tokens'] for r in subset) / seconds,
            'unmasked_tokens_per_step': unmasked / steps,
            'mean_forward_steps': steps / len(subset),
            'missing_answer_marker_count': sum(not r['has_answer_marker'] for r in subset),
            'no_end_token_count': sum(not r['has_end_token'] for r in subset),
            'possible_truncation_count': sum(r['possible_truncation'] for r in subset),
            'residual_mask_tokens': sum(r['residual_mask_tokens'] for r in subset),
            'peak_allocated_gib': max(r['peak_allocated_gib'] for r in subset),
        }
        if all('final_unmasked_tokens' in r for r in subset):
            net = sum(r['final_unmasked_tokens'] for r in subset)
            summary[mode].update({
                'final_tokens_per_second': net / seconds,
                'final_tokens_per_step': net / steps,
            })
        if all('decoder_stats' in r for r in subset):
            for diagnostic in ('deferred', 'fallback', 'forced'):
                summary[mode][f'{diagnostic}_pivots'] = sum(
                    sum(sum(counts) for counts in r['decoder_stats'].get(f'{diagnostic}_per_step', []))
                    for r in subset)
            summary[mode]['capped_incomplete_count'] = sum(
                r['decoder_stats'].get('capped_incomplete', False) for r in subset)
    if all(mode in summary for mode in MODES):
        paired = {}
        for row in rows:
            paired.setdefault(row['test_index'], {})[row['mode']] = row
        pairs = [p for p in paired.values() if all(m in p for m in MODES)]
        if pairs:
            summary['paired'] = {
                'n': len(pairs),
                'decode_speedup': sum(p['low_confidence']['decode_seconds'] for p in pairs) / sum(p['random_pivot']['decode_seconds'] for p in pairs),
                'vanilla_only_correct': sum(p['low_confidence']['correct'] and not p['random_pivot']['correct'] for p in pairs),
                'pivot_only_correct': sum(p['random_pivot']['correct'] and not p['low_confidence']['correct'] for p in pairs),
            }
    if 'random_pivot' in summary:
        control = {r['test_index']: r for r in rows if r['mode'] == 'random_pivot'}
        comparisons = {}
        for mode in dict.fromkeys(r['mode'] for r in rows):
            if mode in MODES:
                continue
            paired_rows = [(control[r['test_index']], r) for r in rows
                           if r['mode'] == mode and r['test_index'] in control]
            if paired_rows:
                comparisons[mode] = {
                    'n': len(paired_rows),
                    'decode_speedup_vs_random_pivot': sum(c['decode_seconds'] for c, r in paired_rows) / sum(r['decode_seconds'] for c, r in paired_rows),
                    'accuracy_difference_percentage_points': 100 * sum(int(r['correct']) - int(c['correct']) for c, r in paired_rows) / len(paired_rows),
                    'control_only_correct': sum(c['correct'] and not r['correct'] for c, r in paired_rows),
                    'variant_only_correct': sum(r['correct'] and not c['correct'] for c, r in paired_rows),
                }
        if comparisons:
            summary['comparisons_vs_random_pivot'] = comparisons
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True, help='Cached GSM8K test Arrow file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen-length', type=int, default=256)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--variant-config', help='JSON mapping experiment labels to generate keyword arguments')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this benchmark requires a GPU.')
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_file(args.dataset)
    indices = sorted(random.Random(args.seed).sample(range(len(dataset)), args.num_samples))
    variants = (json.loads(Path(args.variant_config).read_text()) if args.variant_config
                else {mode: {'remasking': mode} for mode in MODES})
    if not isinstance(variants, dict) or not variants:
        raise ValueError('Variant configuration must be a non-empty object.')
    for label, kwargs in variants.items():
        if not isinstance(kwargs, dict) or 'remasking' not in kwargs:
            raise ValueError(f'{label}: specify generate keyword arguments including remasking.')
        if set(kwargs) & {'gen_length', 'block_length', 'steps', 'temperature', 'cfg_scale', 'return_stats', 'model', 'prompt', 'attention_mask'}:
            raise ValueError(f'{label}: shared evaluation settings cannot be overridden.')
    modes_to_run = tuple(variants)
    config = vars(args).copy()
    config.update({
        'model_revision': Path(args.model).name,
        'test_indices': indices, 'batch_size': 1, 'num_fewshot': 0,
        'variants': variants,
        'temperature': 0, 'cfg_scale': 0, 'dtype': 'bfloat16',
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'prompt_suffix': '\n\nSolve this problem step by step. End your response with "#### <answer>", where <answer> is the final numerical answer.',
        'decode_sha256': hashlib.sha256(Path(__file__).with_name('decode.py').read_bytes()).hexdigest(),
        'hierarchy_decode_sha256': hashlib.sha256(Path(__file__).with_name('hierarchy_decode.py').read_bytes()).hexdigest(),
        'decode_confidence_sha256': hashlib.sha256(Path(__file__).with_name('decode_confidence.py').read_bytes()).hexdigest(),
        'eval_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metrics': {
            'accuracy': 'Last #### numeric answer, falling back to last number; numeric equality.',
            'strict_accuracy': 'Requires a #### numeric answer and numeric equality.',
            'timing': 'CUDA-synchronized wall time around generate; excludes tokenization, loading, warmup, and answer parsing.',
            'tokens_per_second': 'Actual unmasked positions including special tokens and post-EOS positions / decode seconds.',
            'answer_tokens_per_second': 'Non-special tokens before first EOS/EOT / decode seconds.',
            'tokens_per_step': 'Actual unmasked positions / forward calls, batch size 1.',
            'possible_truncation': 'Neither EOS/EOT nor #### answer marker; heuristic, not confirmed truncation.',
        },
    })
    config_path = out_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing output has a different configuration; use a new output directory.')
    write_json(config_path, config)
    source_dir = out_dir / 'source'
    source_dir.mkdir(exist_ok=True)
    for filename in ('decode.py', 'hierarchy_decode.py', 'decode_confidence.py', 'eval_gsm8k.py'):
        (source_dir / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    write_json(out_dir / 'samples.json', [{'test_index': i, **dataset[i]} for i in indices])
    records_path = out_dir / 'results.jsonl'
    rows = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    done = {(r['test_index'], r['mode']) for r in rows}
    if len(done) == args.num_samples * len(modes_to_run):
        print(json.dumps(summarize(rows), indent=2), flush=True)
        return
    print(f'Loading {args.model} on {config["gpu"]}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                    local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
    eos = tokenizer.eos_token_id
    end_ids = {126081, 126348}
    if eos is not None:
        end_ids.add(eos)
    special_ids = set(tokenizer.all_special_ids) | end_ids | {126336}

    def encode(question):
        messages = [{'role': 'user', 'content': question + config['prompt_suffix']}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')

    def run(encoded, mode):
        kwargs = dict(variants[mode])
        remasking = kwargs.pop('remasking')
        decoder = {'hierarchy': generate_hierarchy,
                   'confidence_pivot': generate_confidence_pivot}.get(remasking, generate)
        if decoder is generate:
            kwargs['remasking'] = remasking
        return decoder(model, encoded['input_ids'], attention_mask=encoded['attention_mask'],
                       gen_length=args.gen_length, block_length=args.block_length, steps=args.steps,
                       temperature=0, cfg_scale=0, return_stats=True, **kwargs)

    # Full-shape warmup on an unscored question, for both decoders.
    warmup = encode('A box contains 3 red balls and 4 blue balls. How many balls are in the box?')
    for mode in modes_to_run:
        torch.manual_seed(args.seed)
        run(warmup, mode)
        torch.cuda.synchronize()
        print(f'Warmup complete: {mode}', flush=True)
    del warmup

    with records_path.open('a', buffering=1) as output:
        for ordinal, index in enumerate(indices):
            sample = dataset[index]
            gold, _ = extract_answer(sample['answer'])
            encoded = encode(sample['question'])
            # Rotate order to reduce a systematic warmup/thermal ordering effect.
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
                predicted, marked = extract_answer(text)
                counts = [count[0] for count in stats['unmasked_per_step']]
                row = {
                    'test_index': index, 'mode': mode, 'seed': sample_seed,
                    'question': sample['question'], 'reference': sample['answer'],
                    'gold': gold, 'prediction': predicted,
                    'correct': predicted is not None and predicted == gold,
                    'has_answer_marker': marked, 'has_end_token': end < len(generated),
                    'possible_truncation': end == len(generated) and not marked,
                    'text': text, 'raw_text': tokenizer.decode(generated, skip_special_tokens=False),
                    'generated_ids': generated, 'prompt_tokens': encoded['input_ids'].shape[1],
                    'answer_tokens': sum(token not in special_ids for token in generated[:end]),
                    'unmasked_tokens': sum(counts), 'unmasked_per_step': counts,
                    'final_unmasked_tokens': sum(token != 126336 for token in generated),
                    'decoder_stats': stats,
                    'forward_steps': stats['forward_steps'], 'decode_seconds': elapsed,
                    'residual_mask_tokens': generated.count(126336),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                }
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
                done.add((index, mode))
                summary = summarize(rows)
                write_json(out_dir / 'summary.json', summary)
                print(f'[{len(done)}/{args.num_samples * len(modes_to_run)}] index={index} {mode}: '
                      f'correct={row["correct"]} steps={row["forward_steps"]} '
                      f'time={elapsed:.2f}s tokens/s={row["unmasked_tokens"]/elapsed:.2f} '
                      f'tokens/step={row["unmasked_tokens"]/row["forward_steps"]:.2f}', flush=True)
                del tokens
    print('COMPLETE\n' + json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == '__main__':
    main()
