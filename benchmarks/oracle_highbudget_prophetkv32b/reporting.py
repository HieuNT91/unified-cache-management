"""Independent phase reports published only after all phase records and engine exit."""
import csv,html,json,statistics,time,shutil
from pathlib import Path
import numpy as np
from prepare import D,sha
from common import load,dump,group_alive
from query_config import configuration,TASKS,TARGETS,ORACLE_CASES,HIGH_CASES
from validate import validate

def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def check_complete(records,p):
    expected={(m['id'],c) for m in p['samples'] for c in p['cases']}
    assert len(records)==TARGETS[p['stage']] and {(r['sample_id'],r['case']) for r in records}==expected
    for r in records:
        cfg=configuration(r['case'])
        assert r['selection_mode']==cfg['selection_mode'] and r['layer_scope']==cfg['layer_scope'] and r['budget']==cfg['budget']
        if p['stage']=='highbudget':assert not r['oracle_eligible_tokens'] and not r['oracle_exact_prefix_tokens'] and not r['oracle_added_tokens']

def summarize(records,p):
    check_complete(records,p);rows=[]
    for case in p['cases']:
        cfg=configuration(case)
        for task in (*TASKS[p['stage']],'ALL'):
            rs=[r for r in records if r['case']==case and (task=='ALL' or r['label']==task)]
            mean=lambda k:statistics.mean(r[k] for r in rs)
            rows.append(dict(case=case,task=task,samples=len(rs),selection_mode=cfg['selection_mode'],layer_scope=cfg['layer_scope'],budget=cfg['budget'],
                accuracy_percent=100*mean('score'),mean_ttft_seconds=mean('ttft_seconds'),median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in rs),
                mean_generation_seconds=mean('generation_seconds'),output_cap_count=sum(r['output_tokens']==256 for r in rs),length_finish_count=sum(r['finish_reason']=='length' for r in rs),
                min_prompt_tokens=min(r['prompt_tokens'] for r in rs),max_prompt_tokens=max(r['prompt_tokens'] for r in rs),
                mean_selected_context_tokens=mean('selected_context_tokens'),mean_effective_recompute_percent=100*mean('effective_recompute_ratio'),
                min_effective_recompute_percent=100*min(r['effective_recompute_ratio'] for r in rs),max_effective_recompute_percent=100*max(r['effective_recompute_ratio'] for r in rs),
                mean_oracle_added_tokens=mean('oracle_added_tokens'),total_already_exact_prefix_target_tokens=sum(r['oracle_exact_prefix_tokens'] for r in rs)))
    return rows

def pairs(records,p):
    index={(r['sample_id'],r['case']):r for r in records};rows=[]
    for r in records:
        cfg=configuration(r['case']);comparisons=[]
        if p['stage']=='oracle' and cfg['selection_mode']=='target_union':comparisons.append(('union_vs_target_only','target_only-all64-0'))
        if cfg['layer_scope']=='selected5':comparisons.append(('selected5_vs_all64',r['case'].replace('-selected5-','-all64-')))
        for name,other in comparisons:
            base=index[r['sample_id'],other]
            rows.append(dict(comparison=name,sample_id=r['sample_id'],task=r['label'],case=r['case'],reference_case=other,budget=cfg['budget'],
                accuracy_difference_pp=100*(r['score']-base['score']),method_accuracy=r['score'],reference_accuracy=base['score'],
                reference_to_method_ttft_speedup=base['ttft_seconds']/r['ttft_seconds'],reference_ttft_seconds=base['ttft_seconds'],method_ttft_seconds=r['ttft_seconds'],
                method_prediction=r['prediction'],reference_prediction=base['prediction'],references=r['references'],
                method_selected_tokens=r['selected_context_tokens'],reference_selected_tokens=base['selected_context_tokens']))
    return rows

def pair_summary(rows):
    result=[]
    groups=sorted({(r['comparison'],r['case']) for r in rows})
    for comp,case in groups:
        for task in (*dict.fromkeys(r['task'] for r in rows),'ALL'):
            rs=[r for r in rows if r['comparison']==comp and r['case']==case and (task=='ALL' or r['task']==task)]
            if not rs:continue
            result.append(dict(comparison=comp,case=case,task=task,n=len(rs),accuracy_difference_pp=statistics.mean(r['accuracy_difference_pp'] for r in rs),
                improved=sum(r['accuracy_difference_pp']>0 for r in rs),regressed=sum(r['accuracy_difference_pp']<0 for r in rs),
                mean_paired_ttft_speedup=statistics.mean(r['reference_to_method_ttft_speedup'] for r in rs),
                ratio_of_mean_ttft=statistics.mean(r['reference_ttft_seconds'] for r in rs)/statistics.mean(r['method_ttft_seconds'] for r in rs)))
    return result

def plots(out,summary,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for column,task in enumerate(TASKS[p['stage']]):
        for scope in ['all64','selected5']:
            rs=sorted([r for r in summary if r['task']==task and r['layer_scope']==scope and r['selection_mode']!='target_only'],key=lambda r:r['budget'])
            axes[0,column].plot([r['budget'] for r in rs],[r['accuracy_percent'] for r in rs],marker='o',label=scope)
            axes[1,column].plot([r['budget'] for r in rs],[r['mean_ttft_seconds'] for r in rs],marker='o',label=scope)
        if p['stage']=='oracle':
            base=next(r for r in summary if r['task']==task and r['selection_mode']=='target_only')
            axes[0,column].axhline(base['accuracy_percent'],ls='--',label='target only',color='gray')
            axes[1,column].axhline(base['mean_ttft_seconds'],ls='--',label='target only',color='gray')
        axes[0,column].set(title=task,xlabel='Native ProphetKV budget (%)',ylabel='Accuracy (%)')
        axes[1,column].set(xlabel='Native ProphetKV budget (%)',ylabel='Mean TTFT (seconds)')
        for ax in axes[:,column]:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Oracle diagnostic: target union; actual budgets may exceed nominal' if p['stage']=='oracle' else 'Full-question ProphetKV: all64 versus selected5')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('comparison.'+ext),dpi=150)
    plt.close(fig)

def publish(out,records,p):
    summary=summarize(records,p);paired=pairs(records,p);ps=pair_summary(paired);out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('records',records),('summary',summary),('paired',paired),('paired_summary',ps)]:
        dump(out/(name+'.json'),rows);csv_write(out/(name+'.csv'),rows)
    oracle=p['stage']=='oracle'
    intro=('Oracle diagnostic on10 exact prompts each of MK1/MK2/MK3. Target-only once per prompt, then union of all eligible target key/value tokens with native ProphetKV top10/30/50/70/90%, using both all64 and selected5 scoring. '
           'This experiment deliberately uses ground-truth answer information to form an oracle mask; it is not deployable retrieval accuracy. Union can exceed the nominal budget; actual counts/fractions are reported. '
           'MK2 source row8 has10 target tokens in the exact first chunk. Per user instruction these remain reused and are recorded as already exact, not recomputed. Target-only does no attention scoring; it still aligns caches and recomputes the fresh suffix. '
           if oracle else
           'Full-question ProphetKV on50 exact prompts each of MK2/MK3/CWE. Fresh all64 and selected5 measurements in90%,80%,95% order, all64 then selected5 at each budget. No oracle annotations enter inference selection. ')
    intro+=('Fixed selected layers [45,48,50,56,58]. Qwen3-32B BF16 TP4 eager, non-thinking greedy256, YaRN2 unchanged original65536 RoPE entries, allocation65920,1031 KV blocks, chunk/memory tile4096 and fresh256. '
            'All64 uses ascending native FP32 sum/64; selected5 uses native FP32 sum; unchanged TP sum/4 and all-context-key softmax. Exact floor budgets and ascending ties. '
            'No new baseline inference or separate qualification. Every reported measurement is fresh. Existing prompts are copied byte-for-byte with no truncation. '
            'TTFT excludes construction, readiness, priming, exports and retirement; those durations plus generation time, raw predictions and token IDs appear in records. Timed inference has no capture observer. '
            'Cache masks denote recomputation: unselected context K/V remains available. CWE accuracy uses the preserved reference-substring metric. '
            'Sequential configurations have different timing cohorts. First five rows/task overlap the layer-selection cohort; results are descriptive, not a new held-out layer-selection test.\n')
    columns=list(summary[0]);text=intro+'\n'+'\t'.join(columns)+'\n'+'\n'.join('\t'.join(str(r[k]) for k in columns) for r in summary)+'\n'
    (out/'results.txt').write_text(text);plots(out,summary,p)
    page='<meta charset="utf-8"><title>'+html.escape(p['study'])+'</title><h1>'+html.escape(p['study'])+'</h1><p>'+html.escape(intro)+'</p>'
    page+=' '.join(f'<a href="{n}">{n}</a>' for n in ['results.txt','summary.csv','paired.csv','paired_summary.csv','records.csv','records.json'])
    page+='<table border="1"><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr>'
    page+=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in columns)+'</tr>' for r in summary)+'</table><img width="100%" src="comparison.png">'
    page+='<h2>Raw predictions</h2>'+''.join('<details><summary>'+html.escape(r['sample_id']+' / '+r['case'])+'</summary><pre>'+html.escape(json.dumps({k:r[k] for k in ['score','prediction','references','output_token_ids']},indent=2))+'</pre></details>' for r in records)
    (out/'index.html').write_text(page)
    return summary,paired

def finalize(root):
    root=Path(root);state=load(root/'stage.json');assert state['state']=='complete'
    assert load(root/'cleanup.json')['complete'] and not group_alive(state['last_engine_pid'])
    p=load(root/'protocol.json');entries=load(root/'schedule.json');records=[]
    assert len(entries)==TARGETS[p['stage']]
    for m in entries:
        path=Path(m['output']);cert=load(path.with_suffix('.validated.json'));assert cert['complete']
        for name,h in cert['sha256'].items():assert sha(name)==h,name
        records.append(validate(root,m,p,path)|dict(measurement_origin='fresh',timing_cohort=str(root),source_record=str(path)))
    check_complete(records,p)
    for name,h in load(D/'preserved-prior.json').items():assert sha(name)==h,name
    stage=root/'report-staging';assert not (root/'final').exists()
    if stage.exists():shutil.rmtree(stage)
    publish(stage,records,p);stage.rename(root/'final')
    dump(root/'final-validation.json',dict(complete=True,stage=p['stage'],validated=len(records),engine_exited=True,prior_hashes_unchanged=True,
        artifact_sha256={str(f):sha(f) for f in (root/'final').rglob('*') if f.is_file()},at=time.time()))
