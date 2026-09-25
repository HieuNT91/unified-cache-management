"""Final reports are published only after every record validates and engines exit."""
import csv,html,json,statistics,time
from pathlib import Path
from prepare import H,D,sha
from common import load,dump,identity,group_alive
from validate import validate

def summarize(records,cases,tasks):
    rows=[];paired=[]
    baseline={r['sample_id']:r for r in records if r['case']=='baseline'}
    for case in cases:
        for task in [*tasks,'ALL']:
            rs=[r for r in records if r['case']==case and (task=='ALL' or r['label']==task)]
            bs=[baseline[r['sample_id']] for r in rs]
            mean=statistics.mean(r['ttft_seconds'] for r in rs)
            row=dict(task=task,case=case,samples=len(rs),accuracy_percent=100*statistics.mean(r['score'] for r in rs),
                mean_ttft_seconds=mean,median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in rs),
                baseline_to_method_mean_ttft_speedup=statistics.mean(b['ttft_seconds'] for b in bs)/mean,
                mean_paired_speedup=statistics.mean(b['ttft_seconds']/r['ttft_seconds'] for b,r in zip(bs,rs)),
                accuracy_difference_percentage_points=100*statistics.mean(r['score']-b['score'] for b,r in zip(bs,rs)),
                output_cap_count=sum(r['output_tokens']==256 for r in rs),
                length_finish_count=sum(r['finish_reason']=='length' for r in rs))
            rows.append(row)
        for r in (r for r in records if r['case']==case):
            b=baseline[r['sample_id']]
            paired.append(dict(sample_id=r['sample_id'],task=r['label'],case=case,
                ttft_speedup=b['ttft_seconds']/r['ttft_seconds'],accuracy_difference=r['score']-b['score'],
                baseline_ttft_seconds=b['ttft_seconds'],method_ttft_seconds=r['ttft_seconds']))
    return rows,paired

def csv_write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def compare_selective(records,prior_records,tasks):
    prior={(r['case'],r['sample_id']):r for r in prior_records}
    paired=[]
    for r in records:
        case=r['case'];other='baseline' if case=='baseline' else case.replace('prophetkv-','selective_prophetkv-')
        old=prior[(other,r['sample_id'])]
        assert r['prompt_tokens']==old['prompt_tokens'] and r['references']==old['references'] and r['max_output_tokens']==old['max_output_tokens']==256
        paired.append(dict(case=case,sample_id=r['sample_id'],task=r['label'],
            vanilla_accuracy=r['score'],selective_accuracy=old['score'],
            accuracy_difference_percentage_points=100*(r['score']-old['score']),
            vanilla_ttft_seconds=r['ttft_seconds'],selective_ttft_seconds=old['ttft_seconds'],
            vanilla_to_selective_ttft_ratio=r['ttft_seconds']/old['ttft_seconds']))
    summary=[]
    for case in dict.fromkeys(r['case'] for r in records):
        for task in [*tasks,'ALL']:
            rows=[r for r in paired if r['case']==case and (task=='ALL' or r['task']==task)]
            avg=lambda k:statistics.mean(r[k] for r in rows)
            summary.append(dict(case=case,task=task,samples=len(rows),
                vanilla_accuracy_percent=100*avg('vanilla_accuracy'),selective_accuracy_percent=100*avg('selective_accuracy'),
                accuracy_difference_percentage_points=avg('accuracy_difference_percentage_points'),
                vanilla_mean_ttft_seconds=avg('vanilla_ttft_seconds'),selective_mean_ttft_seconds=avg('selective_ttft_seconds'),
                vanilla_to_selective_mean_ttft_ratio=avg('vanilla_ttft_seconds')/avg('selective_ttft_seconds')))
    return summary,paired

def finalize():
    s=load(D/'supervisor.json')
    assert s['state']=='complete' and identity(s['pid'])!=s['identity']
    assert load(D/'cleanup.json')['engine_exited'] and not group_alive(load(D/'active.json')['pid'])
    p=load(D/'protocol.json');records=[];receipts=[]
    for case in p['cases']:
        for m in p['samples']:
            path=D/'records'/m['id']/(case+'.json');receipt=load(path.with_suffix('.validated.json'))
            assert receipt['complete']
            for name,h in receipt['sha256'].items():assert sha(name)==h,name
            records.append(validate(m|dict(case=case),p,path));receipts.append(load(path.with_suffix('.validated.json')))
    assert len(records)==360 and len({(r['case'],r['sample_id']) for r in records})==360
    for name,h in load(D/'preserved-prior.json').items():assert sha(name)==h,name
    tasks=list(dict.fromkeys(m['label'] for m in p['samples']));rows,paired=summarize(records,p['cases'],tasks)
    out=D/'final';out.mkdir(exist_ok=True)
    dump(out/'records.json',records);dump(out/'summary.json',rows);dump(out/'paired.json',paired)
    csv_write(out/'summary.csv',rows);csv_write(out/'paired.csv',paired)
    csv_write(out/'records.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records])
    arithmetic=[dict(sample=r['sample'],case=r['case'],**r['arithmetic']) for r in receipts if r['arithmetic']]
    csv_write(out/'arithmetic.csv',arithmetic);dump(out/'arithmetic.json',arithmetic)
    comparison,comparison_pairs=compare_selective(records,load(Path(p['comparison_root'])/'final/records.json'),tasks)
    csv_write(out/'versus_selective.csv',comparison);dump(out/'versus_selective.json',comparison)
    csv_write(out/'versus_selective_paired.csv',comparison_pairs)
    intro=('Vanilla ProphetKV: 360 fresh measurements on the same 40-prompt cohort used to select layers [45,48,50,56,58]. '
           'Direct comparisons with the preserved selective experiment are in versus_selective.csv and versus_selective_paired.csv. Timing cohorts differ; a fresh baseline measures current conditions. This is an in-cohort evaluation, not held-out evidence. Native FP32 mean over all 64 layers and TP averaging; all 64 layers used for inference. '
           'Qwen3-32B BF16 TP4, YaRN2 unchanged-frequency table extended to 65920 positions (384 beyond nominal), 1031 KV blocks, '
           'memory tiling 4096, greedy non-thinking output cap 256. Prompt lengths 61376–65664; no truncation. '
           'Sequential persistent engines in baseline/5/10/20/30/40/50/60/1% order introduce a timing-order limitation. '
           'TTFT is submission to first token, excluding load, cache construction/readiness, priming, artifact export and retirement; '
           'generation and all those per-request durations are provided in records. Timed inference has no attention-capture observer. '
           'Speedup is baseline mean TTFT divided by method mean TTFT; paired speedups are also reported. '
           'Output-cap counts and actual prompt lengths are retained. Prior 128-token answers are not an equivalence requirement.\n')
    columns=list(rows[0]);lines=[intro,'\t'.join(columns)]
    lines += ['\t'.join(f'{r[k]:.6f}' if isinstance(r[k],float) else str(r[k]) for k in columns) for r in rows]
    text='\n'.join(lines)+'\n';(out/'results.txt').write_text(text)
    links=['summary.csv','paired.csv','records.csv','records.json','arithmetic.csv','versus_selective.csv','versus_selective_paired.csv']
    page='<meta charset="utf-8"><title>Vanilla ProphetKV results</title><h1>Vanilla ProphetKV</h1><p>'+html.escape(intro)+'</p>'
    page+=' '.join(f'<a href="{f}">{f}</a>' for f in links)+'<table border="1"><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in columns)+'</tr>'
    page+=''.join('<tr>'+''.join(f'<td>{r[k]:.6f}</td>' if isinstance(r[k],float) else f'<td>{html.escape(str(r[k]))}</td>' for k in columns)+'</tr>' for r in rows)+'</table>'
    (out/'index.html').write_text(page)
    (D/'vanilla_prophetkv32b_results.txt').write_text(text)
    dump(D/'final-validation.json',dict(complete=True,validated=360,engine_exited=True,prior_hashes_unchanged=True,
        same_selection_cohort=True,artifact_sha256={str(f):sha(f) for f in out.iterdir() if f.is_file()},at=time.time()))
