#!/usr/bin/env python3
"""CPU-only live reports; usable from a separate checkout against a running run."""
import argparse
import csv
import fcntl
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.corpus_records import protocol_identity
from runner.matched_status import committed_records, match_records
from runner.reporting import COLUMNS, aggregate
from runner.router_process import alive
from runner.setups import atomic_json
from runner.tree_policy import actions


def collect(root, prepared=None, same_count=False):
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text())
    prepared = Path(prepared or protocol['prepared'])
    rows = [json.loads(line) for line in (prepared/'manifest.jsonl').read_text().splitlines() if line.strip()]
    if protocol['dataset'] == 'ruler':
        if protocol['kind'] == 'inference':
            wanted = set(protocol['selected_ids'])
            rows = [r for r in rows if r['id'] in wanted]
            if {r['id'] for r in rows} != wanted:
                raise ValueError('Missing selected manifest rows')
        else:
            limit = json.loads((root/'tranche.json').read_text())['limit_per_task']
            rows = [r for r in rows if r['ordinal'] < limit]
    expected = [dict(prompt_id=r['id'], subtask=r['subtask']) for r in rows]
    cases = ['nocache', 'router'] if protocol['kind'] == 'inference' else list(actions(protocol['actions']))
    records = committed_records(root, cases, protocol_identity(protocol))
    # Validate all records even when a matched report excludes some of them.
    _, matching = match_records(records, expected)
    available = {case: len(rs) for case, rs in records.items()}
    if protocol['kind'] == 'collection':
        probes = committed_records(root, ['probe'], protocol_identity(protocol))['probe']
        allowed = {r['prompt_id'] for r in expected}
        if any(r['prompt_id'] not in allowed for r in probes):
            raise ValueError('Unexpected probe identity')
        probe_count = len(probes)
    else:
        probe_count = len(records['router'])
    reports = {}
    for case in cases:
        # Fixed actions report answer-engine TTFT; router reports the measured
        # complete routing interval, including its independent probe.
        rs = []
        for record in records[case]:
            timing = dict(record['timings'])
            if case != 'router':
                timing['ttft_seconds'] = timing['answer_engine_ttft_seconds']
            rs.append(dict(record, timings=timing))
        report = aggregate(rs, expected, {}, 'live')
        if same_count:
            ids = set(matching['prompt_ids'])
            report = aggregate([r for r in rs if r['prompt_id'] in ids],
                               [r for r in expected if r['prompt_id'] in ids], {}, 'live')
        reports[case] = report
    state = {}
    for name in ('supervisor', 'tranche', 'progress-group0', 'progress-group1'):
        path = root/(name+'.json')
        if path.exists():
            state[name] = json.loads(path.read_text())
    if 'supervisor' in state:
        state['supervisor']['alive'] = alive(state['supervisor'])
    result = dict(schema_version=1, status='live', dataset=protocol['dataset'],
                  scheduled_prompts=len(expected), expected_answers=len(expected)*len(cases),
                  accepted_answers=sum(available.values()), accepted_probes=probe_count,
                  available_counts=available, reports=reports, state=state,
                  validation='Accepted result hashes and protocol checked; attention/diagnostics not replayed.',
                  timing='Fixed actions: answer-engine TTFT. Router: measured live total TTFT including probe. '
                         'Construction, readiness and priming excluded.',
                  comparison='Ordinary live rows can cover different prompts; use status_same_count for paired coverage.')
    if same_count:
        result['matching'] = matching
    return result


def render(report):
    lines = ['# RULER corpus / tree inference live summary', '',
             f"Accepted answers: {report['accepted_answers']}/{report['expected_answers']}; "
             f"accepted probes: {report['accepted_probes']}/{report['scheduled_prompts']}.", '',
             report['timing'], '', report['validation'], '', report['comparison'], '']
    if 'matching' in report:
        lines += [f"Matched prompt IDs: {report['matching']['matched_samples']}; all methods use the same prompts.", '']
    table = []
    for case, value in report['reports'].items():
        table.append(dict(method=case, subtask='overall', **value['overall']))
        table.extend(dict(method=case, subtask=task, **metrics) for task, metrics in value['subtasks'].items())
    lines += ['| Method | Task | Done/total | Accuracy (%) | Mean TTFT (s) | Mean thinking tokens | Mean answer tokens |',
              '|---|---|---:|---:|---:|---:|---:|']
    def number(value):
        return 'N/A' if value is None else f'{value:.3f}'
    for row in table:
        cells = [row['method'], row['subtask'], f"{row['completed']}/{row['expected']}"]
        cells += [number(row[k]) for k in ('accuracy_percent', 'mean_ttft_seconds', 'mean_thinking_tokens', 'mean_answer_tokens')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'Overall means are prompt-weighted. CSV/JSON include scored/unscored counts, control tokens and output caps.',
              'Refreshed on status invocation. This live snapshot does not certify final completion.', '']
    return '\n'.join(lines), table


def write_summary(root, prepared=None, same_count=False):
    root = Path(root)
    # Serialize report writers, without locking collection or changing receipts.
    with (root/'corpus-status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = collect(root, prepared, same_count)
        markdown, table = render(report)
        stem = 'same_count_summary' if same_count else 'live_summary'
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=('method', 'subtask', *COLUMNS))
        writer.writeheader()
        writer.writerows(table)
        for suffix, content in (('md', markdown), ('csv', stream.getvalue())):
            target = root/(stem+'.'+suffix)
            temp = target.with_suffix(target.suffix+'.tmp')
            temp.write_text(content)
            temp.replace(target)
        atomic_json(root/(stem+'.json'), report)
    print(f"live: {report['accepted_answers']}/{report['expected_answers']} answers; "
          f"{report['accepted_probes']}/{report['scheduled_prompts']} probes; {root/(stem+'.md')}")
    state = report['state'].get('supervisor')
    if state:
        print(f"Supervisor: pid={state.get('pid')} alive={state['alive']} state={state.get('state', 'unknown')}")
    for case, value in report['reports'].items():
        metrics = value['overall']
        print(f"{case}: {metrics['completed']}/{metrics['expected']}; "
              f"accuracy (%)={metrics['accuracy_percent']}; mean TTFT (s)={metrics['mean_ttft_seconds']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--prepared', type=Path, help='Override relocated manifest directory')
    parser.add_argument('--same-count', action='store_true')
    args = parser.parse_args()
    write_summary(args.root, args.prepared, args.same_count)


if __name__ == '__main__':
    main()
