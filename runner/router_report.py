"""Paired cohort reporting; router1 stays primary regardless of test results."""
import csv
import html
import json
from pathlib import Path
import statistics
from runner.router_policy import load_policy, rules
from runner.router_sweep import CASES, accepted
from runner.setups import atomic_json, file_hash


def summarize(records, expected_rows):
    import numpy as np
    expected={r['id']:r for r in expected_rows}
    tasks=sorted({r['subtask'] for r in expected_rows})
    keyed={}
    for record in records:
        key=(record['method'],record['prompt_id'])
        if key in keyed or record['method'] not in CASES or record['prompt_id'] not in expected:
            raise ValueError('Duplicate or unexpected answer')
        if record['subtask']!=expected[record['prompt_id']]['subtask']:
            raise ValueError('Task identity mismatch')
        if not 0<=record['accuracy']<=1 or any(not np.isfinite(record['timings'][k]) or record['timings'][k]<0 for k in ('ttft_seconds','answer_engine_ttft_seconds','routing_overhead_seconds')):
            raise ValueError('Invalid metric')
        keyed[key]=record
    if len(keyed)!=len(expected)*7:
        raise ValueError('Incomplete paired cohort')
    rng=np.random.default_rng(20260928)
    summaries=[];policies={};paired=[]
    task_ids={t:sorted(k for k,v in expected.items() if v['subtask']==t) for t in tasks}
    for method in CASES:
        macro_boot=[];macro_dense_time=[];macro_method_time=[]
        for task in tasks:
            ids=task_ids[task];dense=[keyed['baseline',k] for k in ids];rows=[keyed[method,k] for k in ids]
            delta=np.array([r['accuracy']-b['accuracy'] for r,b in zip(rows,dense)])
            draws=rng.integers(0,len(ids),size=(2000,len(ids)))
            boot=delta[draws].mean(1)
            dense_time=np.array([r['timings']['ttft_seconds'] for r in dense])[draws].mean(1)
            method_time=np.array([r['timings']['ttft_seconds'] for r in rows])[draws].mean(1)
            macro_dense_time.append(dense_time);macro_method_time.append(method_time)
            macro_boot.append(boot)
            summaries.append(metric_row(task,method,rows,dense,np.quantile(boot,[.025,.975]).tolist(),np.quantile(dense_time/method_time,[.025,.975]).tolist()))
            paired.extend(dict(prompt_id=r['prompt_id'],subtask=task,method=method,score_delta=float(d),
                total_ttft=r['timings']['ttft_seconds'],dense_ttft=b['timings']['ttft_seconds']) for r,b,d in zip(rows,dense,delta))
        dense=[keyed['baseline',k] for k in sorted(expected)];rows=[keyed[method,k] for k in sorted(expected)]
        result=metric_row('macro',method,rows,dense,np.quantile(np.mean(macro_boot,axis=0),[.025,.975]).tolist(),
            np.quantile(np.mean(macro_dense_time,axis=0)/np.mean(macro_method_time,axis=0),[.025,.975]).tolist())
        # Explicit task-macro scores (TTFT remains the mean over paired prompts).
        per_task=[r for r in summaries if r['method']==method]
        result['score']=statistics.mean(r['score'] for r in per_task)
        result['score_delta']=statistics.mean(r['score_delta'] for r in per_task)
        summaries.append(result)
        if method.startswith('router'):
            policies[method]=dict(primary=method=='router1',macro_loss=-result['score_delta'],speedup=result['speedup'],
                loss_target_met=-result['score_delta']<=.02+1e-12,ttft_target_met=result['speedup']>=4-1e-12,
                both_targets_met=-result['score_delta']<=.02+1e-12 and result['speedup']>=4-1e-12,
                fallback_count=sum(r['routing']['decision']['action']=='baseline' for r in rows),
                actions={a:sum(r['routing']['decision']['action']==a for r in rows) for a in CASES[:4]})
    probes=sum(r.get('routing') is not None for r in records)
    if probes!=len(expected)*3 or any(r['routing']['internal_tokens']!=1 for r in records if r.get('routing')):
        raise ValueError('Independent router probe count mismatch')
    return dict(primary='router1',answers=len(records),router_probes=probes,summaries=summaries,paired=paired,
        policies=policies,caveat='Frozen 1300-prompt cohort only. Descriptive paired 95% bootstrap intervals. Training and clean A800 inference use different runtimes. No test-based refit, recalibration, or policy selection.')


def metric_row(task,method,rows,dense,ci,speed_ci):
    import numpy as np
    times=[r['timings']['ttft_seconds'] for r in rows]
    base=statistics.mean(r['timings']['ttft_seconds'] for r in dense)
    return dict(task=task,method=method,prompts=len(rows),score=statistics.mean(r['accuracy'] for r in rows),
        score_delta=statistics.mean(r['accuracy']-b['accuracy'] for r,b in zip(rows,dense)),paired_delta_ci95=ci,speedup_ci95=speed_ci,
        mean_total_ttft=statistics.mean(times),median_total_ttft=statistics.median(times),p95_total_ttft=float(np.percentile(times,95)),
        mean_answer_engine_ttft=statistics.mean(r['timings']['answer_engine_ttft_seconds'] for r in rows),
        mean_routing_overhead=statistics.mean(r['timings']['routing_overhead_seconds'] for r in rows),
        speedup=base/statistics.mean(times),output_caps=sum(r['output_cap_reached'] for r in rows),
        mean_construction_seconds=statistics.mean(r['timings'].get('construction_seconds',0) for r in rows),
        mean_readiness_seconds=statistics.mean(r['timings'].get('readiness_seconds',0) for r in rows),
        mean_priming_seconds=statistics.mean(r['timings'].get('priming_seconds',0) for r in rows))


def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def plots(folder,report,policy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
    macro=[r for r in report['summaries'] if r['task']=='macro']
    fig,ax=plt.subplots(figsize=(10,6))
    for row in macro:
        ax.scatter(row['mean_total_ttft'],row['score']*100)
        ax.annotate(row['method'],(row['mean_total_ttft'],row['score']*100),fontsize=8)
    baseline=macro[0]
    ax.axvline(baseline['mean_total_ttft']/4,ls='--',color='red')
    ax.axhline(baseline['score']*100-2,ls='--',color='red')
    ax.set(xlabel='Mean total TTFT (seconds)',ylabel='Macro score (%)')
    save(fig,'accuracy-ttft')
    fig,ax=plt.subplots(figsize=(11,5))
    labels=[r['method'] for r in macro];engine=[r['mean_answer_engine_ttft'] for r in macro]
    ax.bar(labels,engine,label='Answer engine');ax.bar(labels,[r['mean_routing_overhead'] for r in macro],bottom=engine,label='Routing')
    ax.tick_params(axis='x',labelrotation=25);ax.set_ylabel('Mean total TTFT (seconds)');ax.legend()
    save(fig,'timing')
    for row in policy['policies']:
        fig,ax=plt.subplots(figsize=(15,7));ax.axis('off');ax.set_title(row['id']+(' (primary)' if row['id']=='router1' else ''))
        def draw(tree,x,y,width):
            leaf='action' in tree
            label=tree['action'] if leaf else f"{tree['feature']}\n<= {tree['threshold']!r}"
            ax.text(x,y,label,ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round',facecolor='#e5f4e5' if leaf else '#e5eefc'))
            if not leaf:
                for key,shift in [('le',-width/2),('gt',width/2)]:
                    ax.plot([x,x+shift],[y-.025,y-.2],color='#888',lw=1)
                    ax.text(x+shift/2,y-.09,'yes' if key=='le' else 'no',fontsize=7)
                    draw(tree[key],x+shift,y-.23,width/2)
        draw(row['tree'],.5,.9,.48);ax.set(xlim=(-.05,1.05),ylim=(0,1))
        save(fig,'tree-'+row['id'])


def publish(folder, report, records, policy):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'summary.json',report);atomic_json(folder/'raw-records.json',records)
    write_csv(folder/'summary.csv',report['summaries']);write_csv(folder/'paired.csv',report['paired'])
    write_csv(folder/'raw-records.csv',records)
    lines=['RULER13 / native clean router / router1 primary',report['caveat'],'']
    for row in report['summaries']:
        lines.append(f"{row['task']} {row['method']}: score {row['score']*100:.2f}%; delta {row['score_delta']*100:+.2f}pp; TTFT {row['mean_total_ttft']:.3f}s; speedup {row['speedup']:.3f}x; paired 95% CI {row['paired_delta_ci95']}")
    lines.extend(f'{key}: {value}' for key,value in report['policies'].items())
    (folder/'results.txt').write_text('\n'.join(lines)+'\n');(folder/'trees.txt').write_text(rules(policy))
    plots(folder,report,policy)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RULER13 native router</title><pre>'+html.escape('\n'.join(lines))+'</pre>'+''.join(
        f'<p><a href="{name}.pdf"><img style="max-width:100%" src="{name}.png"></a></p>' for name in ['accuracy-ttft','timing',*('tree-'+p['id'] for p in policy['policies'])]))


def finalize(root):
    from scripts.router_inputs import verify_prepared
    from runner.router_process import alive,group_alive
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    cleanup=json.loads((root/'cleanup.json').read_text())
    if not cleanup['complete'] or not cleanup['owned_engines_exited']:
        raise ValueError('Owned engines must exit before final reporting')
    for path in (root/'processes').glob('*.json'):
        state=json.loads(path.read_text())
        if alive(state) or group_alive(state['pid']):raise ValueError('Owned engine process is alive')
    rows=verify_prepared(protocol['prepared'])
    records=[accepted(root,case,row,protocol) for case in CASES for row in rows]
    if any(r is None for r in records):raise ValueError('Incomplete accepted results')
    report=summarize(records,rows)
    if report['answers']!=9100 or report['router_probes']!=3900:raise ValueError('Wrong final scope')
    folder=root/'final';publish(folder,report,records,load_policy(root/'trees.json'))
    atomic_json(root/'final-validation.json',dict(complete=True,answers=9100,router_probes=3900,
        owned_engines_exited=True,sha256={str(p.relative_to(root)):file_hash(p) for p in folder.iterdir() if p.is_file()}))
    return report
