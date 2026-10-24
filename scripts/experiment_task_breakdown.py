#!/usr/bin/env python3
"""Print committed results by task: python SCRIPT EXPERIMENT_DIR.

Supports RULER and LongBench v2, including copied baseline-source/ results.
With a copied baseline, metrics use exact shared input/seed pairs. Read-only; no env, model,
prepared-input access, report writes or GPU execution. Uses repository readers.
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner.reporting import aggregate
from runner.setups import file_hash, fingerprint
from scripts.collection_v3 import answer_metrics, seeds


def read(path):
    return json.loads(path.read_text())


def metadata(base, pins):
    first = read(base/'settings.json')
    role = ('tree' if first.get('schema') == 'tree-prophetkv-thinking-collection-v1'
            else 'ruler' if first.get('dataset') == 'ruler' else 'primary')
    paths = [base/name for name in ('settings.json', 'plan.json', 'rows.json', f'{role}/protocol.json')]
    pins.update({path: file_hash(path) for path in paths})
    settings, plan, rows, protocol = map(read, paths)
    if settings != first:
        raise ValueError('Settings changed during read')
    if settings.get('dataset') not in ('ruler', 'longbench-v2') or protocol.get('dataset') != settings['dataset']:
        raise ValueError('Expected matching RULER or LongBench v2 metadata')
    for name in ('settings', 'rows'):
        if pins[base/f'{name}.json'] != plan[f'{name}_sha256']:
            raise ValueError(f'Changed {name}.json')
    if protocol.get('plan_sha256') not in (fingerprint(plan), pins[base/'plan.json']):
        raise ValueError('Protocol/plan identity mismatch')
    cases = [case for case in protocol['scheduled_actions'] if case != 'probe']
    if not cases or len(cases) != len(set(cases)):
        raise ValueError('Missing or duplicate scheduled answer methods')
    row_map = {r['id']: r for r in rows}
    if len(row_map) != len(rows):
        raise ValueError('Duplicate prompt IDs')
    return role, settings, rows, protocol, cases


def records(base, role, rows, protocol, cases, pins, selected_seeds=None, selected_ids=None):
    from runner.corpus_records import accepted
    row_map = {r['id']: r for r in rows}
    scheduled_seeds = seeds(protocol)
    # Freeze commit boundaries before reading; in-flight results are excluded.
    receipts = {case: sorted((base/role/'records'/case).glob(
        '*/validated.json' if scheduled_seeds == [None] else '*/seed-*/validated.json'))
        for case in cases}
    data = {case: [] for case in cases}
    for case, paths in receipts.items():
        for path in paths:
            pins[path] = file_hash(path)
            if scheduled_seeds == [None]:
                pid, seed = path.parent.name, None
            else:
                pid, seed = path.parent.parent.name, int(path.parent.name.removeprefix('seed-'))
            if pid not in row_map or seed not in scheduled_seeds:
                raise ValueError(f'Unexpected answer identity: {path}')
            if ((selected_seeds is not None and seed not in selected_seeds)
                    or (selected_ids is not None and pid not in selected_ids)):
                continue
            record = accepted(base/role, case, row_map[pid], protocol, seed=seed)
            if record is None:
                raise ValueError(f'Committed answer disappeared: {path}')
            value = answer_metrics(record)
            value['prompt_id'] = json.dumps([pid, seed])
            data[case].append(value)
    return data


def copied_baseline(base, settings, rows, protocol, data, pins):
    from runner.thinking_budget import KEY
    source = base/'baseline-source'
    role, original, original_rows, original_protocol, cases = metadata(source, pins)
    if 'nocache' not in cases:
        raise ValueError('baseline-source does not declare nocache')
    if original['dataset'] != settings['dataset']:
        raise ValueError('Baseline dataset differs')
    if (not protocol.get('generation_protocol')
            or original_protocol.get('generation_protocol') != protocol['generation_protocol']):
        raise ValueError('Baseline generation protocol differs')
    wanted_seeds = seeds(protocol)
    if not set(wanted_seeds) <= set(seeds(original_protocol)):
        raise ValueError('Baseline does not contain the requested seed schedule')
    by_id = {r['id']: r for r in original_rows}
    for row in rows:
        other = by_id.get(row['id'])
        if other is None:
            raise ValueError('Baseline is missing input: '+row['id'])
        if not re.fullmatch('[a-f0-9]{64}', row.get('sha256', '')) or not row.get(KEY):
            raise ValueError('Missing input hash or thinking policy: '+row['id'])
        for key in ('sha256', 'subtask', 'ordinal', 'evaluation_protocol', KEY, 'references', 'scoring'):
            if other.get(key) != row.get(key):
                raise ValueError(f'Baseline input/policy differs: {row["id"]} ({key})')
    baseline = records(source, role, original_rows, original_protocol, ['nocache'], pins,
                       selected_seeds=wanted_seeds, selected_ids={r['id'] for r in rows})
    # Equal counts or IDs alone are insufficient: also match the submitted tokens.
    token_hashes = {}
    for values in [baseline['nocache'], *data.values()]:
        for value in values:
            digest = value.get('token_sha256', '')
            if not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Missing committed token identity: '+value['prompt_id'])
            previous = token_hashes.setdefault(value['prompt_id'], digest)
            if previous != digest:
                raise ValueError('Baseline/method token identity differs: '+value['prompt_id'])
    return dict(nocache=baseline['nocache'], **data)


def snapshot(base):
    base = Path(base).resolve()
    pins = {}
    role, settings, rows, protocol, cases = metadata(base, pins)
    expected = [dict(prompt_id=json.dumps([r['id'], seed]), subtask=r['subtask'])
                for r in rows for seed in seeds(protocol)]
    data = records(base, role, rows, protocol, cases, pins)
    if role == 'tree' and (base/'baseline-source').exists():
        data = copied_baseline(base, settings, rows, protocol, data, pins)
    if any(file_hash(path) != digest for path, digest in pins.items()):
        raise ValueError('Experiment metadata or committed receipts changed during read')
    return expected, data


def report(base, same_count=False):
    from runner.matched_status import match_records
    expected, data = snapshot(base)
    imported = (Path(base)/'baseline-source').exists() and 'nocache' in data and any(
        case.startswith('tree-prophetkv-') for case in data)
    same_count = same_count or imported
    # Validate all committed metrics, including records outside the intersection.
    summaries = {case: aggregate(records, expected, {}, 'running') for case, records in data.items()}
    matched, matching = match_records(data, expected)
    if same_count:
        summaries = {case: aggregate(records, expected, {}, 'running') for case, records in matched.items()}
    tables = {'overall': {case: summary['overall'] for case, summary in summaries.items()}}
    tasks = sorted({r['subtask'] for r in expected})
    tables.update({'task/'+task: {case: summary['subtasks'][task] for case, summary in summaries.items()}
                   for task in tasks})
    lines = [
        f'Experiment: {Path(base).resolve()}',
        f'Committed answers: {sum(map(len, data.values()))}/{len(expected)*len(data)}',
        f'Shared prompt/seed pairs: {matching["matched_samples"]}/{len(expected)} across {len(data)} methods.',
        'Metrics: '+('exact shared prompt/seed pairs only.' if same_count else 'all available answers per method.'),
        'Done/total uses the full planned cohort. Accuracy is percent; TTFT is seconds.',
    ]
    if imported:
        lines += ['Baseline: '+str(Path(base).resolve()/'baseline-source'),
                  'Baseline keeps its original hardware provenance; TTFT may come from a different server.']
    elif any(case.startswith('tree-prophetkv-') for case in data):
        lines.append('No baseline-source/ found: reporting tree methods only.')
    fmt = lambda value: 'N/A' if value is None else f'{value:.3f}'
    for scope, table in tables.items():
        lines += ['', scope, '',
                  '| Method | Done/total | Scored | Accuracy % | TTFT s | Thinking tokens | Answer tokens |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for case, value in table.items():
            cells = [case, f'{value["completed"]}/{value["expected"]}', str(value['accuracy_scored'])]
            cells += [fmt(value[key]) for key in (
                'accuracy_percent', 'mean_ttft_seconds', 'mean_thinking_tokens', 'mean_answer_tokens')]
            lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)+'\n', tables, matching


def main(same_count=False):
    parser = argparse.ArgumentParser(description=(
        'Print task breakdown on exact shared prompts across all scheduled methods.'
        if same_count else __doc__))
    parser.add_argument('experiment_dir', type=Path)
    args = parser.parse_args()
    try:
        text, _, _ = report(args.experiment_dir, same_count=same_count)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Cannot report {args.experiment_dir}: {error}\n')
    print(text, end='')


if __name__ == '__main__':
    main()
