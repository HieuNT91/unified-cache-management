"""Read-only AdaTree reports from committed answer records."""
import json
from pathlib import Path
import re

from runner.cache.setups import file_hash, fingerprint
from runner.corpus.dataset import relative
from runner.reporting.aggregate import aggregate
from runner.reporting.status import match_records
from scripts.collection.collection_v3 import answer_metrics, seeds


def read(path):
    return json.loads(Path(path).read_text())


def snapshot(base):
    """Freeze commit boundaries without loading inputs, models or diagnostics."""
    from runner.corpus.records import accepted

    base = Path(base).resolve()
    paths = [base / name for name in ('settings.json', 'plan.json', 'rows.json', 'tree/protocol.json')]
    pins = {path: file_hash(path) for path in paths}
    settings, plan, rows, protocol = map(read, paths)
    if (settings.get('schema') != 'tree-prophetkv-thinking-collection-v1'
            or protocol.get('schema') != settings['schema']
            or settings.get('dataset') not in ('ruler', 'longbench-v2')
            or protocol.get('dataset') != settings['dataset']):
        raise ValueError('Expected a RULER or LongBench v2 AdaTree run')
    for name in ('settings', 'rows'):
        if pins[base / (name + '.json')] != plan.get(name + '_sha256'):
            raise ValueError('Changed ' + name + '.json')
    if protocol.get('plan_sha256') not in (fingerprint(plan), pins[base / 'plan.json']):
        raise ValueError('Protocol/plan identity mismatch')
    cases = protocol.get('scheduled_actions', [])
    if (not cases or len(cases) != len(set(cases))
            or not set(cases) <= {'adatree-v1', 'nocache'}):
        raise ValueError('Expected scheduled AdaTree/no-cache methods')
    row_map = {row['id']: row for row in rows}
    if len(row_map) != len(rows):
        raise ValueError('Duplicate prepared prompt IDs')
    scheduled_seeds = seeds(protocol)
    expected = [dict(prompt_id=json.dumps([row['id'], seed]), subtask=row['subtask'])
                for row in rows for seed in scheduled_seeds]
    # List all receipts first. Answers committed afterward appear on the next call.
    pattern = '*/validated.json' if scheduled_seeds == [None] else '*/seed-*/validated.json'
    receipts = {case: sorted((base / 'tree/records' / case).glob(pattern)) for case in cases}
    data = {case: [] for case in cases}
    identities = {}
    for case, paths in receipts.items():
        for path in paths:
            pins[path] = file_hash(path)
            if scheduled_seeds == [None]:
                pid, seed = path.parent.name, None
            else:
                pid = path.parent.parent.name
                seed = int(path.parent.name.removeprefix('seed-'))
            if pid not in row_map or seed not in scheduled_seeds:
                raise ValueError('Unexpected committed prompt/seed: ' + str(path))
            record = accepted(base / 'tree', case, row_map[pid], protocol, seed=seed)
            if record is None:
                raise ValueError('Committed answer disappeared: ' + str(path))
            token_hash = record.get('token_sha256', '')
            if not re.fullmatch('[a-f0-9]{64}', token_hash):
                raise ValueError('Missing committed token identity: ' + pid)
            if row_map[pid].get('token_sha256', token_hash) != token_hash:
                raise ValueError('Result token identity differs from prepared row: ' + pid)
            key = json.dumps([pid, seed])
            if identities.setdefault(pid, token_hash) != token_hash:
                raise ValueError('Methods used different input tokens: ' + pid)
            receipt = read(path)
            pins[path.parent / 'result.json'] = receipt['files']['result.json']
            pins[relative(base / 'tree', record['initialization'])] = record['initialization_sha256']
            value = answer_metrics(record)
            value['prompt_id'] = key
            data[case].append(value)
    if any(file_hash(path) != digest for path, digest in pins.items()):
        raise ValueError('Experiment metadata or committed results changed during reporting')
    return settings, expected, data


def report(base, mode='full'):
    """Return text, tables and matching metadata; full never limits to shared rows."""
    if mode not in ('full', 'same-input'):
        raise ValueError('Report mode must be full or same-input')
    settings, expected, data = snapshot(base)
    # Validate every available result, including results outside the intersection.
    summaries = {case: aggregate(values, expected, {}, 'running') for case, values in data.items()}
    matched, matching = match_records(data, expected)
    matching.update(mode=mode, matching='exact shared prompt, seed and token identity')
    if mode == 'same-input':
        summaries = {case: aggregate(values, expected, {}, 'running') for case, values in matched.items()}
    tables = {'overall': {case: value['overall'] for case, value in summaries.items()}}
    for task in sorted({row['subtask'] for row in expected}):
        tables['task/' + task] = {case: value['subtasks'][task] for case, value in summaries.items()}
    chunk = settings.get('kv_chunk_size', 4096)
    lines = [f'Run: {Path(base).resolve()}',
             f'Dataset: {settings["dataset"]}; chunk size: {chunk}; mode: {mode}.',
             f'Committed answers: {sum(map(len, data.values()))}/{len(expected) * len(data)}.',
             f'Shared prompt/seed pairs: {matching["matched_samples"]}/{len(expected)} across {len(data)} methods.',
             ('Metrics use all available answers per method.' if mode == 'full' else
              'Metrics use only inputs and seeds completed by every scheduled method.'),
             'Included/plan counts answer requests; accuracy uses scored answers; token columns are content-token means.',
             'TTFT is request submission to first generated token (seconds), including online routing; separate KV preparation is excluded.']
    if 'nocache' not in data:
        lines.append('No-cache is absent from this run\'s schedule; no baseline comparison is available.')
    for scope, table in tables.items():
        lines += ['', scope.replace('\n', ' '), '',
                  '| Method | Available | Included/plan | Scored | Accuracy % | TTFT s | Thinking tokens | Answer tokens |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for case, value in table.items():
            available = (len(data[case]) if scope == 'overall' else
                         sum(row['subtask'] == scope.removeprefix('task/') for row in data[case]))
            cells = [case, str(available), f'{value["completed"]}/{value["expected"]}', str(value['accuracy_scored'])]
            cells += ['N/A' if value[name] is None else f'{value[name]:.3f}' for name in
                      ('accuracy_percent', 'mean_ttft_seconds', 'mean_thinking_tokens', 'mean_answer_tokens')]
            lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n', tables, matching
