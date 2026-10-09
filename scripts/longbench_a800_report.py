"""Committed-result reports across both A800 groups, including official lengths."""
import csv
import io
from pathlib import Path
import statistics
from runner.setups import atomic_json
from runner.corpus_records import protocol_identity
from runner.matched_status import committed_records, match_records
from runner.reporting import aggregate, metrics
from runner.router_process import alive
from scripts.longbench_a800_data import ACTIONS, SCHEDULE, LENGTHS, load, stage, read, roles, scheduled, actions_for


def records(base):
    settings, _, rows = load(base)
    result = {a['id']: [] for a in actions_for(settings)}
    for role in roles(settings):
        if role == 'features' or not (Path(base)/role/'protocol.json').exists():
            continue
        protocol, _ = stage(base, role)
        current = committed_records(Path(base)/role, scheduled(settings, role), protocol_identity(protocol))
        by_id = {r['id']: r for r in rows}
        for case, values in current.items():
            for record in values:
                row = by_id.get(record['prompt_id'])
                group = row['ordinal'] % len(protocol['groups']) if row else None
                if (row is None or record['input_sha256'] != row['sha256'] or record['group'] != group
                        or record['gpu_uuids'] != protocol['groups'][group] or record.get('executed_action') != case):
                    raise ValueError('Committed answer belongs to another prompt/group/action')
        result.update(current)
    return result


def summarize(by_action, rows, same_count=False):
    expected = [dict(prompt_id=r['id'], subtask=r['subtask']) for r in rows]
    for values in by_action.values():
        aggregate(values, expected, {}, 'running')
    matching = None
    if same_count:
        by_action, matching = match_records(by_action, expected)
    categories = {r['id']: r.get('length') for r in rows}
    reports = {}
    for case, values in by_action.items():
        report = aggregate(values, expected, {}, 'running')
        report['lengths'] = {label: metrics([r for r in values if categories[r['prompt_id']] == label],
                                          sum(r.get('length') == label for r in rows)) for label in LENGTHS}
        baseline = {r['prompt_id']: r for r in by_action.get('nocache', [])}
        for label, value in [('overall', report['overall']), *report['lengths'].items(), *report['subtasks'].items()]:
            paired = [r for r in values if r['prompt_id'] in baseline
                      and (label == 'overall' or categories[r['prompt_id']] == label or r['subtask'] == label)]
            denominator = statistics.mean(r['timings']['ttft_seconds'] for r in paired) if paired else 0
            value['paired_with_baseline'] = len(paired)
            value['ttft_speedup'] = (statistics.mean(baseline[r['prompt_id']]['timings']['ttft_seconds'] for r in paired)
                                     /denominator) if denominator > 0 else None
            value['paired_accuracy_delta_pp'] = (100*statistics.mean(
                r['accuracy']-baseline[r['prompt_id']]['accuracy'] for r in paired)) if paired else None
        reports[case] = report
    return dict(methods=reports, matching=matching)


def report(base, same_count=False, final=False, emit=True):
    base = Path(base); settings, plan, rows = load(base)
    data = records(base)
    if final and any(len(values) != len(rows) for values in data.values()):
        raise ValueError('Cannot finalize incomplete controls')
    result = summarize(data, rows, same_count)
    result['all_methods_matched'] = summarize(data, rows, True)
    if settings['dataset'] == 'longbench-v2' and not settings.get('naive_reuse'):
        result['primary_matched'] = summarize({c: data[c] for c in SCHEDULE['primary']}, rows, True)
    result['waiting'] = {role: dict(configured=(base/role/'devices.json').exists(),
        missing_answers=len(rows)*len(scheduled(settings, role))-sum(len(data[c]) for c in scheduled(settings, role)))
        for role in roles(settings) if role != 'features'}
    result.update(expected_answers=plan['answers'], available_answers=sum(map(len, data.values())),
                  same_count=same_count, final=final, report_validation='committed-results-only-v1',
                  timing=('Native answer TTFT excludes construction/readiness/priming and later feature capture; '
                          + ('TP4 groups' if settings['tp'] == 4 else 'TP2 pairs') + ' are separate timing cohorts.'),
                  length_definition=('Original LongBench v2 source length labels, before any official middle truncation'
                      if settings['dataset'] == 'longbench-v2' else 'RULER: overall and all 13 upstream tasks'))
    result['processes'] = {}
    for role in roles(settings):
        path = base/role/'supervisor.json'
        if path.exists():
            state = read(path); result['processes'][role] = dict(state, alive=alive(state))
    result['feature_probes'] = len(list((base/'features/records/probe').glob('*/validated.json')))
    training = base/'training/summary.json'
    if training.exists():
        result['training'] = read(training)
    thinking = settings.get('execution_profile') == 'ruler-thinking'
    lines = [f"{settings['dataset']}: {result['available_answers']}/{plan['answers']} accepted answers; feature probes {result['feature_probes']}/{plan['probes']}",
             result['timing'], result['length_definition']]
    if result['matching']:
        lines.append(f"Same-count: {result['matching']['matched_samples']} exact shared prompt IDs across {len(data)} methods.")
    lines += [f'{role}: configured={item["configured"]}, waiting for {item["missing_answers"]} answers'
              for role, item in result['waiting'].items()]
    if thinking:
        lines.append('Thinking RULER: TTFT retains first-output-token semantics; answer-content latency excludes forced prefix. Separate content caps and closure counts follow.')
    csvrows = []
    for label in ('overall', *(LENGTHS if settings['dataset'] == 'longbench-v2' else sorted({r['subtask'] for r in rows}))):
        lines += ['', label.upper(), '| Method | Done/total | Accuracy % | TTFT s | Speedup | Thinking tokens | Answer tokens | Capped |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for case, item in result['methods'].items():
            value = item['overall'] if label == 'overall' else (item['lengths'][label] if settings['dataset'] == 'longbench-v2' else item['subtasks'][label])
            value = dict(value)
            value.setdefault('ttft_speedup', None)
            csvrows.append(dict(scope=label, method=case, **value))
            fmt = lambda v: 'N/A' if v is None else f'{v:.3f}'
            cells = [case, f"{value['completed']}/{value['expected']}"] + [fmt(value[k]) for k in
                ('accuracy_percent', 'mean_ttft_seconds', 'ttft_speedup', 'mean_thinking_tokens', 'mean_answer_tokens')]
            lines.append('| '+' | '.join(cells+[str(value['output_cap_reached'])])+' |')
    if thinking:
        lines += ['', '| Method | Generated answer tokens | Forced tokens | First answer content s | Thinking capped | Answer capped | Natural / forced closure |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for case,item in result['methods'].items():
            v=item['overall']
            lines.append('| '+' | '.join([case,fmt(v['mean_generated_answer_tokens']),fmt(v['mean_forced_tokens']),
                fmt(v['mean_first_answer_content_seconds']),str(v['thinking_cap_reached']),str(v['answer_cap_reached']),
                f"{v['natural_thinking_closures']} / {v['forced_thinking_closures']}"] )+' |')
    comparisons = [(f'ALL {len(data)} METHODS MATCHED', result['all_methods_matched'])]
    if 'primary_matched' in result:
        comparisons.insert(0, ('PRIMARY MATCHED (seven methods)', result['primary_matched']))
    for title, matched in comparisons:
        lines += ['', title+f"; {matched['matching']['matched_samples']} exact shared prompt IDs"]
        labels = ('overall', *LENGTHS) if settings['dataset'] == 'longbench-v2' else ('overall', *sorted({r['subtask'] for r in rows}))
        for label in labels:
            lines += [label.upper(), '| Method | Matched | Accuracy % | TTFT s |', '|---|---:|---:|---:|']
            for case, item in matched['methods'].items():
                v = item['overall'] if label == 'overall' else item['lengths' if settings['dataset'] == 'longbench-v2' else 'subtasks'][label]
                lines.append(f"| {case} | {v['completed']} | {fmt(v['accuracy_percent'])} | {fmt(v['mean_ttft_seconds'])} |")
    text = '\n'.join(lines)+'\n'
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(csvrows[0]))
    writer.writeheader(); writer.writerows(csvrows)
    # Each launcher owns its own report lock; shared report publication is serialized.
    import fcntl
    with (base/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stem = 'status_same_count' if same_count else 'final' if final else 'status'
        for suffix, content in (('md', text), ('csv', stream.getvalue())):
            target = base/f'{stem}.{suffix}'; temporary = target.with_suffix(target.suffix+'.tmp')
            temporary.write_text(content); temporary.replace(target)
        atomic_json(base/f'{stem}.json', result)
    if emit:
        print(text, flush=True)
    return result
