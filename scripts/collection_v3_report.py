"""Answer-weighted, sequence/seed matched reports for collection v3."""
import csv
import fcntl
import io
import statistics
from pathlib import Path
from runner.reporting import metrics
from runner.setups import atomic_json
from scripts.collection_v3 import ACTIONS, Snapshot, seeds, answer_metrics


def key(record):
    return record['sequence_id'], record['seed']


def summarize(data, rows, settings, same_count=False):
    planned = {(r['id'], seed) for r in rows for seed in seeds(settings)}
    for case, values in data.items():
        keys = [key(r) for r in values]
        if len(set(keys)) != len(keys) or not set(keys) <= planned:
            raise ValueError('Duplicate/unexpected answer identity')
    common = set.intersection(*(set(map(key, values)) for values in data.values()))
    selected = {case: [r for r in values if not same_count or key(r) in common] for case, values in data.items()}
    scopes = [('overall', rows)]
    if settings['dataset'] == 'ruler':
        from scripts.ruler import TASKS
        scopes += [('task/'+task, [r for r in rows if r['subtask'] == task]) for task in TASKS]
    else:
        scopes += [('length/'+label, [r for r in rows if r['length'] == label]) for label in ('short', 'medium', 'long')]
        scopes += [('domain/'+domain, [r for r in rows if r['domain'] == domain]) for domain in sorted({r['domain'] for r in rows})]
        scopes += [('category/'+domain+' / '+sub, [r for r in rows if (r['domain'], r['sub_domain']) == (domain, sub)])
                   for domain, sub in sorted({(r['domain'], r['sub_domain']) for r in rows})]
    tables = {}
    for seed in [None] + (seeds(settings) if settings['dataset'] == 'longbench-v2' else []):
        for scope, members in scopes:
            ids = {r['id'] for r in members}
            base = {key(r): r for r in selected['nocache'] if r['sequence_id'] in ids and (seed is None or r['seed'] == seed)}
            table = {}
            for case, records in selected.items():
                values = [r for r in records if r['sequence_id'] in ids and (seed is None or r['seed'] == seed)]
                value = metrics(values, len(members)*(len(seeds(settings)) if seed is None else 1))
                paired = [r for r in values if key(r) in base]
                denom = statistics.mean(r['timings']['ttft_seconds'] for r in paired) if paired else 0
                value['ttft_speedup'] = statistics.mean(base[key(r)]['timings']['ttft_seconds'] for r in paired)/denom if denom else None
                value['paired_with_baseline'] = len(paired)
                value['thinking_cap_reached'] = sum(r.get('thinking_closure') == 'forced' for r in values)
                value['answer_cap_reached'] = sum(r['output_cap_reached'] for r in values)
                table[case] = value
            tables[scope if seed is None else f'seed-{seed}/'+scope] = table
    return dict(tables=tables, same_count=same_count, matched_answers=len(common),
                matched_pairs=[list(k) for k in sorted(common)])


def markdown(table):
    lines = ['| method | done/total | accuracy | ttft | speedup | thinking tokens | answers tokens | capped thinking | capped answers |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'N/A' if v is None else f'{v:.3f}'
    for case, v in table.items():
        name = 'no cache' if case == 'nocache' else case.removeprefix('prophetkv-')
        cells = [name, f"{v['completed']}/{v['expected']}"] + [fmt(v[k]) for k in
            ('accuracy_percent', 'mean_ttft_seconds', 'ttft_speedup', 'mean_thinking_tokens', 'mean_answer_tokens')]
        cells += ([str(v['thinking_cap_reached']), str(v['answer_cap_reached'])] if v['completed'] else ['N/A', 'N/A'])
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)+'\n'


def report(base, same_count=False, final=False, emit=True):
    snapshot = Snapshot(base, include_probes=False)
    role = 'ruler' if snapshot.settings['dataset'] == 'ruler' else 'primary'
    data = {a['id']: [] for a in ACTIONS}
    for row in snapshot.rows:
        for case in data:
            for seed in seeds(snapshot.settings):
                record = snapshot.record(role, case, row, seed)
                if record is not None:
                    data[case].append(answer_metrics(record))
    if final and any(len(v) != len(snapshot.rows)*len(seeds(snapshot.settings)) for v in data.values()):
        raise ValueError('Cannot finalize incomplete controls')
    result = summarize(data, snapshot.rows, snapshot.settings, same_count)
    result.update(expected_answers=snapshot.plan['answers'], available_answers=sum(map(len, data.values())),
                  final=final, report_validation='committed-results-only-v1')
    snapshot.unchanged()
    text = '\n'.join(scope+'\n\n'+markdown(table) for scope, table in result['tables'].items())
    csvrows = [dict(scope=scope, method=case, **v) for scope, table in result['tables'].items() for case, v in table.items()]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(csvrows[0])); writer.writeheader(); writer.writerows(csvrows)
    base = Path(base)
    with (base/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stem = 'status_same_count' if same_count else 'final' if final else 'status'
        for suffix, content in (('md', text), ('csv', stream.getvalue())):
            target = base/f'{stem}.{suffix}'; temporary = target.with_suffix(target.suffix+'.tmp')
            temporary.write_text(content); temporary.replace(target)
        atomic_json(base/f'{stem}.json', result)
    if emit:
        print(markdown(result['tables']['overall']), end='', flush=True)
    return result
