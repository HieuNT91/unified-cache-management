"""Validated per-job and combined reporting; no inference or GPU imports."""
import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import statistics

from common import CASES, RULER_TASKS, dump, load


def summaries(rows):
    labels = sorted({r['label'] for r in rows})
    groups = [(label, [r for r in rows if r['label']==label]) for label in labels]
    groups.append(('ruler_all', [r for r in rows if r['dataset']=='ruler']))
    result = []
    for label, members in groups:
        baseline = {r['sample_id']:r for r in members if r['case']=='baseline'}
        for case in CASES:
            subset = [r for r in members if r['case']==case]
            paired = [r for r in subset if r['sample_id'] in baseline]
            mean = lambda key: statistics.mean(r[key] for r in subset) if subset else None
            ratio = (statistics.mean(baseline[r['sample_id']]['ttft_seconds'] for r in paired) /
                     statistics.mean(r['ttft_seconds'] for r in paired)) if paired else None
            result.append(dict(task=label, method=case, samples=len(subset),
                accuracy_percent=100*mean('score') if subset else None,
                mean_ttft_seconds=mean('ttft_seconds'),
                median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in subset) if subset else None,
                paired_samples=len(paired), paired_mean_ttft_speedup=ratio,
                mean_generation_seconds=mean('generation_seconds'), mean_output_tokens=mean('output_tokens'),
                mean_cache_build_seconds=mean('cache_build_seconds'), mean_prime_seconds=mean('prime_seconds'),
                mean_prompt_tokens=mean('prompt_tokens'),
                length_limited=sum(r['finish_reason']=='length' for r in subset)))
    return result


def write_report(directory, rows, target, complete, protocols):
    directory.mkdir(parents=True, exist_ok=True)
    table = summaries(rows)
    with (directory/'summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (directory/'raw_records.jsonl').open('w') as stream:
        for record in rows:
            stream.write(json.dumps(record)+'\n')
    sessions = {r['session_path'] for r in rows}
    dump(directory/'engine_sessions.json', [load(path) for path in sorted(sessions)])
    lines = [f'# Qwen3-32B / YaRN2 / ProphetKV with expansion', '',
        f'Validated {len(rows)}/{target} measurements. Complete: {complete}.', '',
        'BF16 TP2; two GPU pairs 4–5 and 6–7; max input 65,536; engine allocation 65,792.',
        'The position table extends 256 positions past nominal YaRN2 using unchanged frequencies and magnitude. '
        'Original 65,536 entries are verified bitwise equal. This is extrapolation beyond nominal 2×.', '',
        'RULER: ten tasks, 100 unique prompts/task, non-thinking, greedy, 256-token cap. '
        'The generator reserves formatting overhead; actual token counts are retained. No inference input is truncated.',
        'LongBench v2: every eligible fully formatted input, non-thinking only, greedy, 256-token cap.', '',
        'Six methods share frozen prompts and GPU assignment. Expansion anchor budgets are 75% of total budgets. '
        'Rates apply after the exact first cached chunk and before the fresh question suffix (at least 256 tokens).',
        'TTFT is submission to first token, including online probing/selection. Total generation latency is reported separately. '
        'Model loading, warmup, offline cache construction, readiness, priming, export and retirement are outside TTFT.',
        'Storage is buffered local warm cache. Results are the local UCM port with modified chunk-formatted benchmark inputs. '
        'No separate qualification or smoke run is included.', '',
        '| Task | Method | N | Accuracy % | Mean TTFT s | Median TTFT s | Speedup | Length-limited |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        if not row['samples']:
            continue
        fmt = lambda key: f'{row[key]:.3f}' if row[key] is not None else '—'
        lines.append(f"| {row['task']} | {row['method']} | {row['samples']} | {fmt('accuracy_percent')} | "
                     f"{fmt('mean_ttft_seconds')} | {fmt('median_ttft_seconds')} | {fmt('paired_mean_ttft_speedup')} | {row['length_limited']} |")
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(directory/'validation.json',dict(complete=complete,validated=len(rows),target=target,
        owned_engine_groups_exited=True, protocols=protocols))


def collect(root, protocol):
    from suite import check_orphan, valid
    from prepare import verify_frozen
    check_orphan(root)
    verify_frozen(root, protocol)
    rows = []
    for meta in protocol['samples']:
        for case in CASES:
            path = root/'records'/meta['id']/(case+'.json')
            if valid(root,protocol,meta,case,path):
                rows.append(load(path))
    return rows


def report_job(root, protocol):
    from suite import progress
    rows = collect(root, protocol)
    complete = len(rows)==protocol['measured_requests']
    directory = root/('final' if complete else 'partial')
    write_report(directory,rows,protocol['measured_requests'],complete,[str(root/'protocol.json')])
    progress(root,protocol,'complete' if complete else 'partial')
    print(f'REPORT {directory}: {len(rows)}/{protocol["measured_requests"]}',flush=True)


def check_matrix(protocols):
    if len(protocols)!=2 or {p['settings']['job_id'] for p in protocols}!={'0','1'}:
        raise ValueError('Expected both independent job protocols')
    a,b=protocols
    for key in ('sources_sha256','runtime_versions','tokenizer_sha256','case_parameters','cases',
                'decoding','longbench_counts','ruler_generation','max_model_len'):
        if a[key]!=b[key]:
            raise ValueError('Jobs used different '+key)
    samples=[m for p in protocols for m in p['samples']]
    if any(m.get('thinking_enabled') is not False or m.get('max_output_tokens') != 256 for m in samples):
        raise ValueError('Unexpected thinking mode or output budget in matrix')
    ids=[m['id'] for m in samples]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate samples across jobs')
    for task in RULER_TASKS:
        if sorted(m['source_row'] for m in samples if m['label']==task)!=list(range(100)):
            raise ValueError('Incomplete RULER input coverage: '+task)
    for mode in ('non_thinking',):
        if sum(m['label']=='longbench_v2_'+mode for m in samples)!=a['longbench_counts'][mode]:
            raise ValueError('Incomplete LongBench mode coverage')
    return len(samples)*len(CASES)


def aggregate(parent, allow_partial=False):
    from suite import locked, live_supervisor
    roots=[parent/f'job-{i}' for i in range(2)]
    with ExitStack() as stack:
        for root in roots:
            stack.enter_context(locked(root))
            if live_supervisor(root):
                raise RuntimeError('Job still running: '+str(root))
        protocols=[load(root/'protocol.json') for root in roots]
        target=check_matrix(protocols)
        censuses=[load(root/'longbench-census.json') for root in roots]
        if censuses[0]!=censuses[1]:
            raise ValueError('LongBench source/eligibility differs between jobs')
        for thinking in (False,):
            expected=sorted(c['source_row'] for c in censuses[0]['census']
                            if c['eligible'] and c['thinking_enabled']==thinking)
            actual=sorted(m['source_row'] for p in protocols for m in p['samples']
                          if m['dataset']=='longbench_v2' and m['thinking_enabled']==thinking)
            if actual!=expected:
                raise ValueError('LongBench samples differ from the full eligibility census')
        rows=[row for root,p in zip(roots,protocols) for row in collect(root,p)]
        complete=len(rows)==target
        if not complete and not allow_partial:
            raise ValueError(f'Only {len(rows)}/{target} validated; resume missing work or explicitly use --partial')
        directory=parent/('combined' if complete else 'combined-partial')
        write_report(directory,rows,target,complete,[str(root/'protocol.json') for root in roots])
        print(f'COMBINED {directory}: {len(rows)}/{target}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('RESULT_ROOT','.')))
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    aggregate(args.root.resolve(),args.partial)
