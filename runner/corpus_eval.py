"""Offline measured-outcome joins; never infer unmeasured action results."""
from collections import Counter
import csv
import html
from pathlib import Path
from runner.setups import atomic_json,file_hash,fingerprint
from runner.tree_policy import validate,decide
from runner.corpus import relative
from runner.corpus_records import protocol_identity


def summarize(rows):
    if not rows:raise ValueError('Empty evaluation')
    def metric(group):
        n=len(group);avg=lambda key:sum(r[key] for r in group)/n
        return dict(samples=n,accuracy=avg('accuracy'),baseline_accuracy=avg('baseline_accuracy'),
            paired_accuracy_difference=avg('accuracy')-avg('baseline_accuracy'),
            answer_ttft_seconds=avg('answer_ttft_seconds'),estimated_router_ttft_seconds=avg('estimated_router_ttft_seconds'),
            baseline_ttft_seconds=avg('baseline_ttft_seconds'),
            estimated_speedup=avg('baseline_ttft_seconds')/avg('estimated_router_ttft_seconds'),
            actions=dict(Counter(r['action'] for r in group)))
    tasks={t:metric([r for r in rows if r['task']==t]) for t in sorted({r['task'] for r in rows})}
    return dict(overall=metric(rows),tasks=tasks,task_coverage=len(tasks),
        macro_accuracy=sum(r['accuracy'] for r in tasks.values())/len(tasks),
        macro_baseline_accuracy=sum(r['baseline_accuracy'] for r in tasks.values())/len(tasks),
        timing='estimated router TTFT: measured fixed-action answer plus independent probe/export/features/retirement; not live router latency')


def report(rows,output,metadata=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(summarize(rows),metadata=metadata or {})
    atomic_json(output/'report.json',result);atomic_json(output/'decisions.json',rows)
    with (output/'decisions.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import json
    text=json.dumps(result,indent=2)
    (output/'report.txt').write_text(text+'\n');(output/'index.html').write_text('<meta charset="utf-8"><pre>'+html.escape(text)+'</pre>')
    return result


def replay(corpus,decisions,output,metadata=None):
    rows=[];seen=set()
    for d in decisions:
        pid=d['sample_id'];action=d['action']
        if pid in seen:raise ValueError('Duplicate sample decision')
        seen.add(pid)
        if pid not in corpus.rows:raise ValueError('Unknown sample')
        outcome=corpus.outcome(pid,action);baseline=corpus.outcome(pid,'nocache');probe=corpus.probe(pid)
        if outcome is None or baseline is None or probe is None:raise ValueError('Decision has no accepted measured outcome/probe')
        answer=outcome['timings']['answer_engine_ttft_seconds']
        rows.append(dict(sample_id=pid,task=corpus.rows[pid]['subtask'],action=action,accuracy=outcome['accuracy'],
            baseline_accuracy=baseline['accuracy'],answer_ttft_seconds=answer,
            estimated_router_ttft_seconds=answer+probe['timings']['routing_overhead_seconds'],
            baseline_ttft_seconds=baseline['timings']['answer_engine_ttft_seconds']))
    return report(rows,output,metadata)


def evaluation_ids(corpus,tree,snapshot=None):
    validate(tree)
    if tree['actions']!=corpus.protocol['actions']:raise ValueError('Tree inventory differs from measured corpus')
    if tree['provenance']['corpus_protocol_sha256']!=protocol_identity(corpus.protocol):raise ValueError('Tree belongs to a different corpus')
    if snapshot is None:
        ids=tree['heldout_ids'];pins=tree.get('heldout_hashes',{})
        if not pins:raise ValueError('Missing frozen held-out artifact hashes')
    else:
        if snapshot['protocol_sha256']!=protocol_identity(corpus.protocol):raise ValueError('Evaluation snapshot protocol mismatch')
        ids=[r['id'] for r in snapshot['samples']];pins=snapshot['acceptance_hashes']
    if not ids or len(set(ids))!=len(ids) or set(ids)&set(tree['training_ids']):raise ValueError('Empty, duplicate, or training-overlapping evaluation')
    for name,expected in pins.items():
        if file_hash(relative(corpus.root,name))!=expected:raise ValueError('Frozen evaluation evidence changed')
    for pid in ids:
        for case in ['probe',*corpus.actions]:
            if f'records/{case}/{pid}/validated.json' not in pins:raise ValueError('Incomplete evaluation snapshot')
    return ids


def test_tree(corpus,tree,output,snapshot=None):
    ids=evaluation_ids(corpus,tree,snapshot)
    decisions=[dict(sample_id=pid,action=decide(tree,corpus.features(pid))['action']) for pid in ids]
    return replay(corpus,decisions,output,dict(tree_sha256=tree['payload_sha256'],frozen_heldout=snapshot is None,
        training_overlap=False,inference='offline',refit=False))
