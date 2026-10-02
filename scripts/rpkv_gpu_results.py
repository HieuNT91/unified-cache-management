#!/usr/bin/env python3
"""Summarize committed RULER/LongBench results without replaying inference evidence."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runner.setups import atomic_json,file_hash
from runner.reporting import metrics
from runner.corpus_records import accepted,answer_validation,NATIVE_ANSWER_VALIDATION,PROBE_ANSWER_VALIDATION
from scripts.ruler import TASKS
from runner.result_comparison import summarize as comparison_summary


def expected_scope(base):
    path=base/'experiment.json'
    scope=json.loads(path.read_text()) if path.exists() else dict(ruler_samples_per_task=5,longbench_samples=5,seed=42)
    for name in ('ruler_samples_per_task','longbench_samples'):
        if type(scope.get(name)) is not int or scope[name]<1:raise ValueError('Invalid experiment sample count: '+name)
    return scope


def validate_rows(dataset,rows,scope):
    count=len(TASKS)*scope['ruler_samples_per_task'] if dataset=='ruler' else scope['longbench_samples']
    if len(rows)!=count or len({r['id'] for r in rows})!=count:raise ValueError('Incomplete or duplicate cohort: '+dataset)
    if dataset=='ruler' and {t:sum(r['subtask']==t for r in rows) for t in TASKS}!=dict.fromkeys(TASKS,scope['ruler_samples_per_task']):
        raise ValueError('Incorrect RULER task counts')
    return count


def report(base):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Reporting is CPU-only')
    summary={};details=[];pins={};audits={};scope=expected_scope(base)
    for dataset in ('ruler','longbench-v2'):
        root=base/'results'/dataset;protocol=json.loads((root/'protocol.json').read_text());rows=json.loads((root/'rows.json').read_text())
        count=validate_rows(dataset,rows,scope)
        replay=answer_validation(protocol)==PROBE_ANSWER_VALIDATION
        if not replay and any((root/'records/probe').glob('*/validated.json')):
            raise ValueError('Unexpected independent probe in native answer controls')
        assert (root/'complete.json').exists()
        if 'experiment_sha256' in protocol:
            assert protocol['experiment_sha256']==file_hash(base/'experiment.json')
            assert protocol['rows_sha256']==file_hash(root/'rows.json')
        assert protocol['actions']==[dict(id='nocache',method='baseline',ratio=None),dict(id='prophetkv-1',method='prophetkv',ratio=.01),dict(id='prophetkv-5',method='prophetkv',ratio=.05)]
        for path in (root/'processes').glob('*-exit.json'):assert json.loads(path.read_text())['owned_engines_exited']
        bymethod={a['id']:[] for a in protocol['actions']}
        for row in rows:
            for action in protocol['actions']:
                result=accepted(root,action['id'],row,protocol);assert result is not None
                bymethod[action['id']].append(result)
                details.append(dict(dataset=dataset,prompt_id=row['id'],task=row['subtask'],method=action['id'],input_tokens=result['prompt_tokens'],output_cap=result['max_output_tokens'],accuracy=result['accuracy'],ttft_seconds=result['timings']['ttft_seconds'],engine_ttft_seconds=result['timings']['answer_engine_ttft_seconds'],thinking_tokens=result['thinking_tokens'],answer_tokens=result['answer_tokens'],control_tokens=result['control_tokens'],output_cap_reached=result['output_cap_reached'],token_sha256=result['token_sha256'],prediction=result['prediction']))
        summary[dataset]={}
        for action,records in bymethod.items():
            summary[dataset][action]=dict(overall=metrics(records,count),comparison=comparison_summary(records),tasks={t:metrics([r for r in records if r['subtask']==t],sum(r['subtask']==t for r in rows)) for t in sorted({r['subtask'] for r in rows})})
        for path in root.glob('records/*/*/validated.json'):pins[str(path.relative_to(base))]=file_hash(path)
        audits[dataset]=dict(answers=count*3,independent_probes=count if replay else 0,answer_validation=answer_validation(protocol),
            source='committed-runtime-results',diagnostic_replay=False,independent_scoring_replay=False)
    output=base/'final';output.mkdir(exist_ok=False)
    total_probes=sum(a['independent_probes'] for a in audits.values());total_answers=sum(a['answers'] for a in audits.values())
    atomic_json(output/'report.json',summary);atomic_json(output/'completion.json',dict(complete=True,report_validation='committed-results-only-v1',answers=total_answers,probes=total_probes,audits=audits,accepted_receipts=pins))
    with (output/'measurements.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    longbench_description=scope.get('longbench_description',f"{scope['longbench_samples']} seed{scope['seed']} random inputs from the explicitly approved full-text fitting pool, native thinking and16384 output cap; no truncation. See prepared/longbench-v2-fitting/selection.json for the full eligibility inventory.")
    lines=['# rpkv GPU controls','', 'Qwen3-32B BF16, TP4; native YaRN4. Hardware is recorded in each dataset protocol. All methods use identical prepared tokens and metadata-only chunk boundaries.','',
        f"RULER: 13 tasks ×{scope['ruler_samples_per_task']}, seed{scope['seed']} task batches, original 65536 total window and task caps; non-thinking. LongBench v2: {longbench_description}",'',
        'TTFT is measured submission-to-first-token plus TP action synchronization and includes native ProphetKV token selection. Offline cache construction, readiness and ordinary priming are separate. Thinking/answer lengths count newly generated content tokens; special/control tokens are separate. Sparse controls ran before connector-free baselines.',
        f'Legacy independent diagnostic probes: {total_probes}, excluded from answer TTFT.' if total_probes else 'No independent diagnostic probes were run. Answers retain native diagnostics, cache and retirement validation.','',
        '| Dataset / task | Method | N | Accuracy % | Mean TTFT s | Mean thinking tokens | Mean answer tokens | Capped |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset,methods in summary.items():
        for task in ['overall']+sorted(next(iter(methods.values()))['tasks']):
            for action,values in methods.items():
                m=values['overall'] if task=='overall' else values['tasks'][task]
                lines.append(f"| {dataset} / {task} | {action} | {m['completed']} | {m['accuracy_percent']:.2f} | {m['mean_ttft_seconds']:.4f} | {m['mean_thinking_tokens']:.2f} | {m['mean_answer_tokens']:.2f} | {m['output_cap_reached']} |")
    lines+=['','Historical comparison on identical input tokens (timing excluded):','',
        '| Dataset | Method | Compared | Exact matches | Differences | Incompatible | New inputs |','|---|---|---:|---:|---:|---:|---:|']
    for dataset,methods in summary.items():
        for action,values in methods.items():
            c=values['comparison'];lines.append(f"| {dataset} | {action} | {c['compared']} | {c['matched']} | {c['different']} | {c['incompatible']} | {c['new_inputs']} |")
    lines+=['',f'All {total_answers} committed answers are included. This report uses the recorded scores and token counts; it performs no diagnostic or independent scoring replay. Results describe this frozen cohort.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();report(args.root.resolve())
if __name__=='__main__':main()
