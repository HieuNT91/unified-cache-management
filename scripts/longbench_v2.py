#!/usr/bin/env python3
"""CPU-only LongBench v2 adapter; preserves the repository's middle truncation.

Formatting/truncation ported from benchmarks/longbenchv2_qwen3_32b_yarn4/prepare.py.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.config import validate_sample
from runner.setups import atomic_json, bytes_per_shard, file_hash, fingerprint, setup_lock

INPUT = 114688

def render(row, template):
    before, separator, after = template.partition('$DOC$')
    if not separator:
        raise ValueError('Missing document placeholder')
    fields = {'$Q$': 'question', **{f'$C_{c}$': f'choice_{c}' for c in 'ABCD'}}
    tail = re.sub(r'\$(?:Q|C_[ABCD])\$', lambda m: row[fields[m[0]]].strip(), after)
    context = row['context'].strip()
    text = before + context + tail
    query_begin = len(before) + len(context) + tail.index('What is the correct answer to this question:')
    query_end = len(before) + len(context) + tail.rindex('\n\nFormat your response')
    return text, text[query_begin:query_end], text[query_begin:]


def layout(tokenizer, content, query, protected):
    # Locate the structurally rendered tail, avoiding a duplicate question in the document.
    if not content.endswith(protected):
        raise ValueError('Middle truncation removed protected question/instruction text')
    begin = len(content) - len(protected)
    text = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
    offset = text.index(content)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
                 if b > offset + begin and a < offset + begin + len(query)]
    if not positions:
        raise ValueError('Empty question')
    from runner.layout import boundaries_for, stamp_sample
    bounds = boundaries_for(len(ids), positions[0])
    return stamp_sample(dict(token_ids=ids, tokens=len(ids), boundaries=bounds,
                fresh_suffix_tokens=len(ids)-bounds[-2], chat_tokens=len(ids),
                formatted_text=text, rendered_user_prompt=content,
                query=dict(text=query, positions=positions),
                thinking_enabled=True, prompt_sha256=fingerprint(ids)), tokenizer.chat_template)


def prepare_prompt(tokenizer, row, template, limit=INPUT):
    content, query, protected = render(row, template)
    original = tokenizer.encode(content, add_special_tokens=False)
    encoded = layout(tokenizer, content, query, protected)
    original_formatted = encoded['tokens']
    half = None
    iterations = 0
    while encoded['tokens'] > limit:
        # Equal beginning/end slices of the rendered user prompt, as in official pred.py.
        # Leave fitting inputs intact. Re-decode, reformat and recount on every reduction.
        excess = encoded['tokens'] - limit
        half = min(len(original) // 2 - 1, limit // 2) if half is None else half - max(1, (excess + 1) // 2)
        if half <= 0:
            raise ValueError('Protected tail cannot fit the input budget')
        kept = original[:half] + original[-half:]
        text = tokenizer.decode(kept, skip_special_tokens=True)
        encoded = layout(tokenizer, text, query, protected)
        iterations += 1
    spans = [[0, len(original)]] if half is None else [[0, half], [len(original)-half, len(original)]]
    retained = sum(b-a for a,b in spans)
    encoded['truncation'] = dict(truncated=half is not None, original_user_tokens=len(original),
        original_formatted_tokens=original_formatted, final_user_tokens=len(tokenizer.encode(encoded['rendered_user_prompt'], add_special_tokens=False)),
        final_formatted_tokens=encoded['tokens'], retained_source_token_spans=spans,
        removed_source_tokens=len(original)-retained, iterations=iterations,
        source_user_sha256=fingerprint(original), protected_tail=protected)
    return encoded


def prepare(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU preparation requires CUDA_VISIBLE_DEVICES=''")
    from runner.preparation import check_model
    from transformers import AutoTokenizer
    check_model(args.model)
    rows = json.loads(args.data.read_text())
    if len(rows) != 503 or len({row['_id'] for row in rows}) != 503:
        raise ValueError('Expected all 503 distinct LongBench v2 source rows')
    template_path = ROOT / 'scripts/longbench_0shot.txt'
    spec = dict(data_sha256=file_hash(args.data), model_config_sha256=file_hash(args.model/'config.json'),
                tokenizer_files={p.name:file_hash(p) for p in args.model.iterdir() if p.is_file() and
                                 p.suffix in ('.json','.txt','.jinja','.model','.tiktoken')},
                template_sha256=file_hash(template_path), adapter_sha256=file_hash(Path(__file__)),
                chunker_sha256=file_hash(ROOT/'runner/cache.py'), input_limit=INPUT, output_limit=16384)
    args.output = args.output.resolve()
    with setup_lock(args.output, build=True):
        receipt_path = args.output/'preparation.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['spec'] != spec:
                raise ValueError('Preparation settings changed; use another output directory')
            for name, digest in receipt['files'].items():
                if file_hash(args.output/name) != digest:
                    raise ValueError(f'Prepared file changed: {name}')
            print(json.dumps(receipt['summary'],indent=2))
            return
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = template_path.read_text()
        manifest, files, lengths, context_lengths, truncated = [], {}, [], [], 0
        for index, row in enumerate(rows):
            sample = prepare_prompt(tokenizer, row, template)
            sample.update(question_positions=sample['query']['positions'], thinking=True, max_output_tokens=16384,
                          model_config_sha256=spec['model_config_sha256'], source_row=index,
                          source_id=row['_id'], source_row_sha256=fingerprint(row),
                          source_metadata={k:row[k] for k in ('domain','sub_domain','difficulty','length')})
            validate_sample(sample)
            path = args.output/'samples'/f'{index:03d}.json'
            if path.exists() and json.loads(path.read_text()) != sample:
                raise ValueError('Interrupted preparation differs; use another directory')
            atomic_json(path, sample)
            files[str(path.relative_to(args.output))] = file_hash(path)
            manifest.append(dict(id=f'longbench-v2-{index:03d}', prepared=str(path.relative_to(args.output)),
                subtask=row['domain']+' / '+row['sub_domain'], references=[row['answer']], scoring='longbench_v2'))
            lengths.append(len(sample['token_ids']))
            context_lengths.append(sample['boundaries'][-2])
            truncated += sample['truncation']['truncated']
            if (index+1)%20 == 0:
                print(f'Prepared {index+1}/503',flush=True)
        manifest_path=args.output/'manifest.jsonl'
        temporary=manifest_path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row)+'\n' for row in manifest))
        temporary.replace(manifest_path)
        files['manifest.jsonl']=file_hash(manifest_path)
        # Only one prompt per group is retained by the remote sweep.
        kv_bytes=max(context_lengths)//64*bytes_per_shard(args.model,4)*4
        summary=dict(samples=503, measurements=503*13,truncated=truncated,
                     min_input_tokens=min(lengths),max_input_tokens=max(lengths),
                     mean_input_tokens=sum(lengths)/503,context_tokens=sum(context_lengths),
                     max_temporary_kv_bytes_per_group=kv_bytes,
                     max_temporary_kv_GiB_two_groups=2*kv_bytes/2**30,
                     subtasks=dict(Counter(row['subtask'] for row in manifest)))
        atomic_json(receipt_path,dict(spec=spec,files=files,summary=summary))
        print(json.dumps(summary,indent=2))


def disk(args):
    # One prompt per group, not a retained cache for all 503 inputs. Account for
    # temporary writes by reserving a second full payload, plus 16 GiB scratch.
    from runner.setups import manifest_entries
    rows = manifest_entries(args.prepared/'manifest.jsonl')
    maximum = max(json.loads(row['prepared'].read_text())['boundaries'][-2] for row in rows)
    payload = maximum * 262144 * args.groups  # Qwen3-32B BF16, across all TP ranks
    required = 2*payload + 16*2**30
    args.cache_root.mkdir(parents=True,exist_ok=True)
    free = shutil.disk_usage(args.cache_root).free
    print(f'Temporary KV payload ceiling: {payload/2**30:.2f} GiB across {args.groups} groups; '
          f'free: {free/2**30:.2f} GiB; required with write/scratch reserve: {required/2**30:.2f} GiB',flush=True)
    print('Results/diagnostics need additional space on EXPERIMENT_DIR; KV is deleted after each prompt.',flush=True)
    if free < required:
        raise RuntimeError('Insufficient temporary KV space; put CACHE_ROOT on a larger filesystem')


def status(args):
    completed=0
    for directory in sorted(args.output.glob('*')):
        path=directory/'live_aggregation.json'
        if not path.exists():
            continue
        report=json.loads(path.read_text()); m=report['overall']; completed+=m['completed']
        print(f"{directory.name}: {report['status']} {m['completed']}/{m['expected']} "
              f"TTFT={m['mean_ttft_seconds']}s accuracy={m['accuracy_percent']}% "
              f"thinking={m['mean_thinking_tokens']} answer={m['mean_answer_tokens']}")
    print(f'Total measured prompts: {completed}/6539')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--model',type=Path,required=True)
    prep.add_argument('--data',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    prep.set_defaults(func=prepare)
    check=sub.add_parser('disk')
    check.add_argument('--prepared',type=Path,required=True)
    check.add_argument('--cache-root',type=Path,required=True)
    check.add_argument('--groups',type=int,choices=(1,2),default=2)
    check.set_defaults(func=disk)
    progress=sub.add_parser('status')
    progress.add_argument('--output',type=Path,required=True)
    progress.set_defaults(func=status)
    args=parser.parse_args()
    args.func(args)

if __name__=='__main__':
    main()
