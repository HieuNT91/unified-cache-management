#!/usr/bin/env python3
"""Prepare pre-generated RULER rows (sparse_attention layout) for `run.sh sweep`.

Each row's `input` is already chat-formatted and ends with RULER's answer
prefix, so it is tokenized verbatim (no chat template is applied twice).
The final question starts at the last '\\nWhat ' (NIAH) or '\\nQuestion:'
(CWE) and ends at the user turn's <|im_end|>. Everything from the question to
the end of the prompt forms the fresh suffix (>=256 tokens, block rounded);
the body before it is cut into 4096-token CacheBlend chunks by
runner.cache.cacheblend_prompt. Nothing is truncated.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.cache import aligned_prompt, cacheblend_prompt
from runner.config import validate_sample
from runner.profiles import PROFILES
from runner.setups import atomic_json, file_hash, fingerprint

# RULER generation lengths used by sparse_attention (ShadowKV/data/dataset.py).
GEN_LEN = {'niah': 128, 'cwe': 120}


def query_span(content, task):
    marker = '\nWhat ' if task.startswith('niah_') else '\nQuestion:'
    begin = content.rfind(marker)
    end = content.find('<|im_end|>', begin)
    if begin < 0 or end < 0:
        raise ValueError(f'Cannot locate complete final question for {task}')
    return begin + 1, end


def format_sample(tokenizer, row, task, window, output, layout='padded'):
    content = row['input']
    begin, end = query_span(content, task)
    encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    positions = [i for i, (a, b) in enumerate(offsets) if b > begin and a < end]
    if not positions:
        raise ValueError('Question token span is empty')
    suffix = ((max(256, len(ids) - positions[0]) + 63) // 64) * 64
    marker = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    if marker is None or marker in ids:
        raise ValueError('Expected Qwen3 <|endoftext|> delimiter absent from the source prompt')
    if layout == 'aligned':
        chunks, tokens, suffix = aligned_prompt(ids, tokenizer, 4096, suffix)
    else:
        chunks, tokens = cacheblend_prompt(ids, tokenizer, marker, 4095, suffix)
    bounds = [0]
    for chunk in chunks:
        bounds.append(bounds[-1] + len(chunk))
    bounds.append(len(tokens))
    shift = len(tokens) - len(ids)
    query = [p + shift for p in positions]
    if tokens[-suffix:] != ids[-suffix:] or [tokens[p] for p in query] != [ids[p] for p in positions]:
        raise RuntimeError('Question or fresh suffix changed')
    sample = dict(token_ids=tokens, boundaries=bounds, question_positions=query, thinking=False,
                  max_output_tokens=output, task=task, question=content[begin:end],
                  source_tokens=len(ids), source_index=row['index'], source_row_sha256=fingerprint(row),
                  truncated=False, answer_prefix_policy='RULER answer prefix kept verbatim from source row',
                  layout=layout)
    if layout == 'aligned':
        sample.update(chunk_end_token_id=-1, forbidden_token_id=marker)
    validate_sample(sample, 4096, window, window)
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='.../ruler/data/qwen/<length>')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layout', choices=('padded', 'aligned'), default='padded',
                        help='padded: <|endoftext|> pads each chunk (repo default); '
                             'aligned: chunks cut to whole blocks, no delimiter or padding')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU preparation requires CUDA_VISIBLE_DEVICES=''")
    from transformers import AutoTokenizer
    from run import check_model
    args.model, args.output = args.model.resolve(), args.output.resolve()
    window = PROFILES[check_model(args.model)]['window']
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config_hash = file_hash(args.model / 'config.json')
    manifest, lengths, files = [], [], {}
    for task in args.tasks:
        source = args.data / task / 'validation.jsonl'
        files[str(source)] = file_hash(source)
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()][:args.samples]
        if len(rows) != args.samples:
            raise ValueError(f'Expected {args.samples} rows in {source}')
        output = GEN_LEN['niah' if task.startswith('niah_') else task]
        for ordinal, row in enumerate(rows):
            sample = format_sample(tokenizer, row, task, window, output, args.layout)
            sample['source_row'] = ordinal
            sample['model_config_sha256'] = config_hash
            relative = f'samples/{task}-{ordinal:03d}.json'
            atomic_json(args.output / relative, sample)
            refs = row['outputs'] if isinstance(row['outputs'], list) else [row['outputs']]
            manifest.append(dict(id=f'{task}-{ordinal:03d}', prepared=relative, subtask=task,
                                 references=refs, scoring='ruler_all'))
            lengths.append(len(sample['token_ids']))
    # Task-interleaved order so every index % shards slice holds every task.
    order = sorted(range(len(manifest)), key=lambda i: (i % args.samples, i // args.samples))
    path = args.output / 'manifest.jsonl'
    path.write_text(''.join(json.dumps(manifest[i]) + '\n' for i in order))
    summary = dict(prompts=len(manifest), subtasks=dict(Counter(r['subtask'] for r in manifest)),
                   min_tokens=min(lengths), max_tokens=max(lengths), window=window, layout=args.layout,
                   sources=files, model=str(args.model), tokenizer_config=config_hash,
                   adapter=file_hash(Path(__file__)), chunker=file_hash(ROOT / 'runner/cache.py'))
    atomic_json(args.output / 'preparation.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
