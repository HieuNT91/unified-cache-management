"""Publish only after 435 combined validations, required fresh work and owned engine exit."""
import csv,html,json,statistics,time,shutil
from pathlib import Path
import numpy as np
from prepare import H,D,sha
from common import load,dump,identity,group_alive
from query_config import CASES,TASKS,BUDGETS,configuration
from validate import validate
from diagnosis import evidence_metrics,overlap

def csv_write(path,rows):
    assert rows
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def check_completeness(records,p):
    expected={(m['id'],c) for m in p['samples'] for c in ('baseline',*CASES)}
    assert len(records)==435 and {(r['sample_id'],r['case']) for r in records}==expected

def summarize(records,p):
    check_completeness(records,p)
    baseline={r['sample_id']:r for r in records if r['case']=='baseline'}
    rows=[]
    for case in ('baseline',*CASES):
        cfg=dict(query_scope='none',layer_scope='none',budget=0) if case=='baseline' else configuration(case)
        for task in (*TASKS,'ALL'):
            rs=[r for r in records if r['case']==case and (task=='ALL' or r['label']==task)]
            mean=statistics.mean(r['ttft_seconds'] for r in rs);bs=[baseline[r['sample_id']] for r in rs]
            rows.append(dict(case=case,task=task,query_scope=cfg['query_scope'],layer_scope=cfg['layer_scope'],budget=cfg['budget'],samples=len(rs),
                accuracy_percent=100*statistics.mean(r['score'] for r in rs),mean_ttft_seconds=mean,
                median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in rs),
                baseline_to_method_mean_ttft_speedup=statistics.mean(b['ttft_seconds'] for b in bs)/mean,
                mean_paired_baseline_speedup=statistics.mean(b['ttft_seconds']/r['ttft_seconds'] for b,r in zip(bs,rs)),
                baseline_accuracy_difference_pp=100*statistics.mean(r['score']-b['score'] for b,r in zip(bs,rs)),
                output_cap_count=sum(r['output_tokens']==256 for r in rs),length_finish_count=sum(r['finish_reason']=='length' for r in rs),
                min_prompt_tokens=min(r['prompt_tokens'] for r in rs),max_prompt_tokens=max(r['prompt_tokens'] for r in rs),
                retained=sum(r['measurement_origin']=='retained' for r in rs),fresh=sum(r['measurement_origin']=='fresh' for r in rs),
                timing_cohorts=sorted({r['timing_cohort'] for r in rs})))
    return rows

def pair_records(records,diagnostics):
    index={(r['sample_id'],r['case']):r for r in records};di={(r['sample_id'],r['case']):r for r in diagnostics};pairs=[]
    for r in records:
        if not r['case'].startswith('focus-'):continue
        cfg=configuration(r['case']);other='full_question-'+cfg['layer_scope']+'-'+str(cfg['budget']);f=index[r['sample_id'],other]
        d=di[r['sample_id'],r['case']];q=di[r['sample_id'],other]
        delta=r['score']-f['score']
        row=dict(sample_id=r['sample_id'],task=r['label'],budget=cfg['budget'],layer_scope=cfg['layer_scope'],
            focused_accuracy=r['score'],full_question_accuracy=f['score'],accuracy_difference_pp=100*delta,
            answer_change='improvement' if delta>0 else 'regression' if delta<0 else 'unchanged',
            focused_ttft_seconds=r['ttft_seconds'],full_question_ttft_seconds=f['ttft_seconds'],
            full_to_focus_ttft_speedup=f['ttft_seconds']/r['ttft_seconds'],focused_origin=r['measurement_origin'],full_question_origin=f['measurement_origin'],
            focused_timing_cohort=r['timing_cohort'],full_question_timing_cohort=f['timing_cohort'],
            focused_prediction=r['prediction'],full_question_prediction=f['prediction'],
            focused_output_token_ids=r['output_token_ids'],full_question_output_token_ids=f['output_token_ids'],
            references=r['references'],prompt_tokens=r['prompt_tokens'])
        for k in ['key_recall','answer_recall','harmonic_recall','complete_statement_retention','evidence_recall',
                  'key_attention_mass','answer_attention_mass','evidence_attention_mass']:
            row['focused_'+k]=d[k];row['full_question_'+k]=q[k]
            row[k+'_difference']=None if d[k] is None or q[k] is None else d[k]-q[k]
        row.update(overlap(d['selected_positions'],q['selected_positions']));pairs.append(row)
    assert len(pairs)==210
    return pairs

def paired_summary(pairs):
    result=[]
    for layer in ('all64','selected5'):
        for budget in BUDGETS:
            for task in (*TASKS,'ALL'):
                rs=[r for r in pairs if r['layer_scope']==layer and r['budget']==budget and (task=='ALL' or r['task']==task)]
                row=dict(task=task,layer_scope=layer,budget=budget,samples=len(rs),
                    accuracy_difference_pp=statistics.mean(r['accuracy_difference_pp'] for r in rs),
                    mean_full_to_focus_ttft_speedup=statistics.mean(r['full_question_ttft_seconds'] for r in rs)/statistics.mean(r['focused_ttft_seconds'] for r in rs),
                    mean_paired_full_to_focus_speedup=statistics.mean(r['full_to_focus_ttft_speedup'] for r in rs),
                    improved=sum(r['answer_change']=='improvement' for r in rs),regressed=sum(r['answer_change']=='regression' for r in rs),
                    unchanged=sum(r['answer_change']=='unchanged' for r in rs),mean_mask_jaccard=statistics.mean(r['mask_jaccard'] for r in rs))
                for k in ['key_recall','answer_recall','harmonic_recall','complete_statement_retention','evidence_recall','key_attention_mass','answer_attention_mass','evidence_attention_mass']:
                    vals=[r[k+'_difference'] for r in rs if r[k+'_difference'] is not None]
                    row[k+'_difference']=statistics.mean(vals) if vals else None
                result.append(row)
    return result

def findings(pairs):
    lines=[]
    for task in TASKS:
        rs=[r for r in pairs if r['task']==task];primary='evidence_recall' if task=='cwe' else 'harmonic_recall'
        acc=statistics.mean(r['accuracy_difference_pp'] for r in rs);coverage=statistics.mean(r[primary+'_difference'] for r in rs)
        improved=sum(r['accuracy_difference_pp']>0 for r in rs);regressed=sum(r['accuracy_difference_pp']<0 for r in rs)
        joint=sum(r['accuracy_difference_pp']>0 and r[primary+'_difference']>0 for r in rs)
        if coverage>0 and acc>0:verdict='Directionally consistent with query-token dilution on this cohort.'
        elif coverage>0:verdict='Coverage increases without an average answer gain; this does not establish dilution as the answer-error explanation.'
        else:verdict='The aggregate coverage result does not support a benefit from narrowing query rows.'
        lines.append(f'{task}: mean paired accuracy difference {acc:+.3f} percentage points; {primary} difference {coverage:+.6f}; {improved} improved, {regressed} regressed, {joint} improvements accompanied by better coverage. {verdict}')
    lines.append('These averages repeat five prompts across budgets and layer sets; they are descriptive, not independent-sample significance tests or causal proof. Inspect each prompt/budget in paired.csv. CWE focuses a task phrase, not an entity key; top-word annotations omit distractor counts needed to identify the most common words.')
    return '\n'.join(lines)

def plots(out,summary,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in TASKS:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for layer in ('all64','selected5'):
            for query in ('focus','full_question'):
                rows=sorted([r for r in summary if r['task']==task and r['layer_scope']==layer and r['query_scope']==query],key=lambda r:r['budget'])
                axes[0].plot([r['budget'] for r in rows],[r['accuracy_percent'] for r in rows],marker='o',label=query+'/'+layer)
                axes[1].plot([r['budget'] for r in rows],[r['mean_ttft_seconds'] for r in rows],marker='o',label=query+'/'+layer)
            rs=[r for r in pairs if r['task']==task and r['layer_scope']==layer];k='evidence_recall_difference' if task=='cwe' else 'harmonic_recall_difference'
            axes[2].scatter([r[k] for r in rs],[r['accuracy_difference_pp'] for r in rs],label=layer,alpha=.6)
        axes[0].set(xlabel='Budget (%)',ylabel='Answer accuracy (%)',title=task+' — five prompts')
        axes[1].set(xlabel='Budget (%)',ylabel='Mean TTFT (seconds)',title='Timing cohorts differ')
        axes[2].set(xlabel='Focused − full evidence coverage',ylabel='Answer change (percentage points)',title='Each prompt × budget')
        axes[2].axhline(0,color='gray',lw=.6);axes[2].axvline(0,color='gray',lw=.6)
        for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout()
        for extension in ['png','pdf']:fig.savefig(out/(task+'.'+extension),dpi=150)
        plt.close(fig)
    # Separate mass/coverage plots retain the distinction between attention and selection.
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for column,task in enumerate(TASKS):
        for layer in ('all64','selected5'):
            for suffix,style in [('key','-'),('answer','--')] if task!='cwe' else [('evidence','-')]:
                for row,metric in enumerate([suffix+'_attention_mass_difference',suffix+'_recall_difference']):
                    values=[statistics.mean(r[metric] for r in pairs if r['task']==task and r['layer_scope']==layer and r['budget']==b) for b in BUDGETS]
                    axes[row,column].plot(BUDGETS,values,style,marker='o',label=layer+'/'+suffix)
                    axes[row,column].set(xlabel='Budget (%)',ylabel='Focused − full-question',title=task+(' attention mass' if row==0 else ' token recall'))
        for row in range(2):axes[row,column].axhline(0,color='gray',lw=.6);axes[row,column].legend(fontsize=7);axes[row,column].grid(alpha=.2)
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(out/('evidence_changes.'+extension),dpi=150)
    plt.close(fig)

def publish(out,records,p,diagnostics):
    check_completeness(records,p);assert len(diagnostics)==420
    summary=summarize(records,p);pairs=pair_records(records,diagnostics);ps=paired_summary(pairs)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('records',records),('summary',summary),('paired',pairs),('paired_summary',ps)]:
        dump(out/(name+'.json'),rows);csv_write(out/(name+'.csv'),rows)
    # Exact masks kept separately from compact tabular diagnostics.
    for d in diagnostics:
        directory=out/'masks'/d['sample_id'];directory.mkdir(parents=True,exist_ok=True)
        np.save(directory/(d['case']+'.npy'),d['selected_positions'])
    compact=[{k:v for k,v in d.items() if k!='selected_positions'} for d in diagnostics]
    dump(out/'evidence.json',compact);csv_write(out/'evidence.csv',compact)
    intro=('Query-key ablation on five exact prompts each for MK2, MK3 and CWE, from the cohort already used to select layers [45,48,50,56,58]. '
        '420 method measurements plus 15 retained latest vanilla baselines; retained/fresh provenance and timing cohorts are explicit in every record. '
        'Budgets ascend 5/10/20/30/40/50/60%; focused all64 then focused selected5 then missing full-question controls. '
        'Qwen3-32B BF16 TP4 eager, YaRN2 with original 65536 entries unchanged and table extended to 65920, 1031 KV blocks, chunk/memory tile4096, fresh256, greedy non-thinking cap256; no truncation. '
        'Only query rows change: MK2 complete key phrase, MK3 complete UUID including hyphens, CWE the task phrase “10 most common words”. '
        'Softmax includes all cached context keys. Layer aggregation retains native FP32 arithmetic (all64 sum/64, selected5 sum, TP sum/4). '
        'TTFT includes inference probing but excludes construction, readiness, ordinary priming, export and retirement; all separate timings are in records. Timed inference has no capture observer. '
        'Speedups use matched prompts, with both ratios of means and mean paired ratios. Cross-cohort timing and configuration order limit causal interpretation. '
        'Evidence metrics exclude cached prefix and fresh suffix; selected5 attention mass is divided by five only for offline comparison. This is a small in-cohort ablation, not held-out generalization.\n')
    conclusion=findings(pairs);columns=list(summary[0]);text=intro+'\n'+conclusion+'\n\n'+'\t'.join(columns)+'\n'
    text+='\n'.join('\t'.join(str(r[k]) for k in columns) for r in summary)+'\n'
    (out/'results.txt').write_text(text);(out/'findings.txt').write_text(conclusion+'\n');plots(out,summary,pairs)
    page='<meta charset="utf-8"><title>Query-key attention ablation</title><h1>Query-key attention ablation</h1><p>'+html.escape(intro)+'</p><pre>'+html.escape(conclusion)+'</pre>'
    page+=' '.join(f'<a href="{n}">{n}</a>' for n in ['results.txt','records.json','records.csv','summary.csv','paired.csv','paired_summary.csv','evidence.json','evidence.csv'])
    page+='<table border="1"><tr>'+''.join('<th>'+k+'</th>' for k in columns)+'</tr>'
    page+=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in columns)+'</tr>' for r in summary)+'</table>'
    page+=''.join(f'<p><img width="100%" src="{task}.png"></p>' for task in [*TASKS,'evidence_changes'])
    page+='<h2>Every paired answer</h2>'+''.join('<details><summary>'+html.escape(f"{r['sample_id']} / {r['layer_scope']} / {r['budget']}%: {r['answer_change']}")+'</summary><pre>'+html.escape(json.dumps(r,indent=2))+'</pre></details>' for r in pairs)
    (out/'index.html').write_text(page)
    return summary,pairs

def finalize():
    s=load(D/'supervisor.json');assert s['state']=='complete' and identity(s['pid'])!=s['identity']
    assert load(D/'cleanup.json')['complete'] and not group_alive(load(D/'active.json')['pid'])
    p=load(D/'protocol.json');inventory=load(D/'control-inventory.json');records=[];sources={}
    for item in inventory['retained']:
        assert sha(item['receipt'])==item['receipt_sha256']
        for name,h in item['sha256'].items():assert sha(name)==h,name
        r=load(item['path']);sources[r['sample_id'],item['case']]=Path(item['path'])
        records.append(r|dict(case=item['case'],source_case=item['source_case'],source_record=item['path'],
            measurement_origin='retained',timing_cohort=item['source_root']))
    for m in inventory['fresh']:
        path=D/'records'/m['id']/(m['case']+'.json');receipt=load(path.with_suffix('.validated.json'));assert receipt['complete']
        for name,h in receipt['sha256'].items():assert sha(name)==h,name
        r=validate(m,p,path);sources[m['id'],m['case']]=path
        records.append(r|dict(source_case=m['case'],source_record=str(path),measurement_origin='fresh',timing_cohort=str(D)))
    assert len(inventory['fresh'])==inventory['target_new'];check_completeness(records,p)
    for name,h in load(D/'preserved-prior.json').items():assert sha(name)==h,name
    diagnostics=[];samples={m['id']:m for m in p['samples']}
    for r in records:
        if r['case']=='baseline':
            r.update(query_scope='none',layer_scope='none',budget=0);continue
        cfg=configuration(r['case']);r.update({k:cfg[k] for k in ['query_scope','layer_scope','budget']})
        path=sources[r['sample_id'],r['case']];workers=load(path.with_suffix('.diagnostics.json'))
        d=next(e for e in workers[0]['diagnostics'] if e['kind']=='prophetkv_selection')
        metrics=evidence_metrics(samples[r['sample_id']],d['selected_positions'],d['scores'],cfg['layer_scope'])
        diagnostics.append(dict(sample_id=r['sample_id'],case=r['case'],task=r['label'],query_scope=cfg['query_scope'],layer_scope=cfg['layer_scope'],
            budget=cfg['budget'],selected_positions=d['selected_positions'],**metrics))
    stage=D/'report-staging';assert not (D/'final').exists()
    if stage.exists():shutil.rmtree(stage)
    publish(stage,records,p,diagnostics);stage.rename(D/'final')
    shutil.copy2(D/'final/results.txt',D/'querykey_prophetkv32b_results.txt')
    dump(D/'final-validation.json',dict(complete=True,validated_new=inventory['target_new'],combined=435,methods=420,baseline=15,
        engine_exited=True,prior_hashes_unchanged=True,same_selection_cohort=True,
        artifact_sha256={str(f):sha(f) for f in (D/'final').rglob('*') if f.is_file()},at=time.time()))
