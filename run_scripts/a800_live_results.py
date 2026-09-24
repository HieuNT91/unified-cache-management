#!/usr/bin/env python3
"""Read provisional A800 benchmark results without locking or changing a run.

Uses only the Python standard library. Reads accepted record JSON and verifies
its receipt hash; full diagnostic/log validation remains the final reporter's job.
This file deliberately lives outside the hash-pinned a800_expansion directory.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys


FIELDS = ('task', 'method', 'samples', 'accuracy_percent', 'mean_ttft_seconds',
          'median_ttft_seconds', 'paired_samples', 'paired_mean_ttft_speedup',
          'length_limited')


def load(path):
    return json.loads(path.read_bytes())


def read_job(directory):
    protocol_bytes = (directory / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    samples = {sample['id']: sample for sample in protocol['samples']}
    cases = protocol['cases']
    rows = []
    # Snapshot receipt paths once. A just-completed request can appear next time.
    markers = sorted((directory / 'records').glob('*/*.validated.json'))
    for marker in markers:
        receipt = load(marker)
        if receipt.get('complete') is not True:
            continue
        sid = marker.parent.name
        case = marker.name.removesuffix('.validated.json')
        if sid not in samples or case not in cases:
            raise ValueError(f'Accepted record outside prepared matrix: {marker}')
        if receipt.get('sample_id') != sid or receipt.get('case') != case:
            raise ValueError(f'Receipt identity mismatch: {marker}')
        path = marker.with_name(case + '.json')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt.get('hashes', {}).get('.json'):
            raise ValueError(f'Accepted record hash mismatch: {path}')
        record = json.loads(raw)
        expected = dict(sample_id=sid, case=case, label=samples[sid]['label'],
                        dataset=samples[sid]['dataset'], protocol_sha256=protocol_hash,
                        input_sha256=samples[sid]['input_sha256'])
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Record identity/provenance mismatch: {path}')
        if (not math.isfinite(record['score']) or not 0 <= record['score'] <= 1
                or not math.isfinite(record['ttft_seconds']) or record['ttft_seconds'] <= 0
                or record['finish_reason'] not in ('stop', 'length')):
            raise ValueError(f'Invalid metrics: {path}')
        rows.append(record)
    progress_path = directory / 'progress.json'
    progress = load(progress_path) if progress_path.exists() else {}
    info = dict(job=directory.name, accepted=len(rows), target=protocol['measured_requests'],
                recorded_state=progress.get('state', 'unknown'))
    return rows, cases, info


def summarize(rows, cases):
    seen = set()
    for row in rows:
        key = row['sample_id'], row['case']
        if key in seen:
            raise ValueError(f'Duplicate sample/method across jobs: {key}')
        seen.add(key)
    groups = [(task, [r for r in rows if r['label'] == task])
              for task in sorted({r['label'] for r in rows})]
    ruler = [r for r in rows if r['dataset'] == 'ruler']
    if ruler:
        groups.append(('ruler_all', ruler))
    table = []
    for task, members in groups:
        baseline = {r['sample_id']: r for r in members if r['case'] == 'baseline'}
        for case in cases:
            subset = [r for r in members if r['case'] == case]
            paired = [r for r in subset if r['sample_id'] in baseline]
            speedup = (statistics.mean(baseline[r['sample_id']]['ttft_seconds'] for r in paired)
                       / statistics.mean(r['ttft_seconds'] for r in paired)) if paired else None
            table.append(dict(task=task, method=case, samples=len(subset),
                accuracy_percent=100 * statistics.mean(r['score'] for r in subset) if subset else None,
                mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in subset) if subset else None,
                median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in subset) if subset else None,
                paired_samples=len(paired), paired_mean_ttft_speedup=speedup,
                length_limited=sum(r['finish_reason'] == 'length' for r in subset)))
    return table


def print_table(table):
    def display(value):
        if value is None:
            return '—'
        return f'{value:.3f}' if isinstance(value, float) else str(value)
    headers = ('Task', 'Method', 'N', 'Accuracy %', 'Mean TTFT s', 'Median TTFT s',
               'Paired N', 'Speedup', 'Length-limited')
    cells = [headers, *[tuple(display(row[key]) for key in FIELDS) for row in table]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    for row in cells:
        print('  '.join(value.ljust(width) for value, width in zip(row, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=os.environ.get('RESULT_ROOT'),
                        help='Parent containing job-0 and job-1; defaults to RESULT_ROOT')
    parser.add_argument('--job', choices=('0', '1'), help='Read one job instead of both')
    parser.add_argument('--format', choices=('table', 'csv', 'json'), default='table')
    args = parser.parse_args(argv)
    if args.root is None:
        parser.error('Set RESULT_ROOT or pass --root /path/to/results')
    try:
        rows, cases, progress = [], [], []
        for job in ([args.job] if args.job is not None else ['0', '1']):
            records, methods, info = read_job(args.root / ('job-' + job))
            if cases and methods != cases:
                raise ValueError('Jobs declare different method lists')
            rows.extend(records)
            cases = methods
            progress.append(info)
        table = summarize(rows, cases)
        for info in progress:
            print(f"{info['job']}: {info['accepted']}/{info['target']} accepted; "
                  f"recorded state={info['recorded_state']}", file=sys.stderr)
        print('PROVISIONAL snapshot: sample counts can differ by method. '
              'Record hashes checked; diagnostics/logs are not revalidated. '
              'Recorded state does not prove process liveness.', file=sys.stderr)
        if args.format == 'csv':
            writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(table)
        elif args.format == 'json':
            print(json.dumps(dict(provisional=True, jobs=progress, summary=table), indent=2))
        else:
            print_table(table)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Cannot read live results: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
