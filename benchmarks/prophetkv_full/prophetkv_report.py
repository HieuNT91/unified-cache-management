"""Dataset/task breakdowns with paired uncertainty and full-prefill TTFT speedup."""
import csv,json,math,random,statistics
from prophetkv_common import dump
from prophetkv_gpu import ROOT
from cacheblend_prophetkv import CASES

def stats(rows):
    output=[];base={r['sample_id']:r for r in rows if r['case']=='baseline'}
    for dataset in ('ruler','longbench_v2'):
        tasks=sorted({r['label'] for r in rows if r['dataset']==dataset})
        for task in ('overall',*tasks):
            for case in CASES:
                group=[r for r in rows if r['dataset']==dataset and r['case']==case and (task=='overall' or r['label']==task)]
                if not group:continue
                paired=[r for r in group if r['sample_id'] in base];rng=random.Random(42)
                speed=statistics.mean(base[r['sample_id']]['ttft_seconds'] for r in paired)/statistics.mean(r['ttft_seconds'] for r in paired) if paired else None
                delta=statistics.mean(r['score']-base[r['sample_id']]['score'] for r in paired) if paired else None
                boot_t=[];boot_a=[]
                strata=[[r for r in paired if r['label']==label] for label in tasks]
                for _ in range(1000 if paired else 0):
                    draw=[r for g in strata if g for r in rng.choices(g,k=len(g))]
                    boot_t.append(statistics.mean(base[r['sample_id']]['ttft_seconds'] for r in draw)/statistics.mean(r['ttft_seconds'] for r in draw))
                    boot_a.append(statistics.mean(r['score']-base[r['sample_id']]['score'] for r in draw))
                ci=lambda a:[sorted(a)[int(.025*len(a))],sorted(a)[min(len(a)-1,int(.975*len(a)))]] if a else [None,None]
                baseline_accuracy=statistics.mean(base[r['sample_id']]['score'] for r in paired) if paired else 0
                output.append(dict(dataset=dataset,task=task,case=case,n=len(group),paired_n=len(paired),chunk_size=4096,
                    accuracy=statistics.mean(r['score'] for r in group),mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in group),
                    ttft_speedup=speed,paired_geomean_speedup=math.exp(statistics.mean(math.log(base[r['sample_id']]['ttft_seconds']/r['ttft_seconds']) for r in paired)) if paired else None,
                    paired_accuracy_delta_pp=100*delta if delta is not None else None,
                    relative_accuracy_improvement_percent=100*delta/baseline_accuracy if baseline_accuracy else None,
                    ttft_speedup_ci95=ci(boot_t),accuracy_delta_ci95=[100*x if x is not None else None for x in ci(boot_a)],
                    length_limited=sum(r['finish_reason']=='length' for r in group),
                    unparseable=sum(r.get('extracted_answer') is None for r in group) if dataset=='longbench_v2' else None,
                    actual_tokens_min=min(r['prompt_tokens'] for r in group),actual_tokens_max=max(r['prompt_tokens'] for r in group)))
    return output

def report(rows,state,final=False):
    directory=ROOT/('final' if final else 'diagnostic');directory.mkdir(exist_ok=True)
    p=json.loads((ROOT/'protocol.json').read_text());table=stats(rows)
    dump(directory/'raw-records.json',rows);dump(directory/'tables.json',table)
    costs=[]
    for path in sorted((ROOT/'cache-builds').rglob('*.json')):
        if 'attempts' in path.parts:continue
        r=json.loads(path.read_text())
        if 'cache_build_seconds' in r:costs.append(dict(path=str(path),**r))
    dump(directory/'offline-costs.json',costs)
    if table:
        with (directory/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    text=f"# ProphetKV UCM/vLLM port\n\nState: {state}; {len(rows)}/{p['measured_requests']} validated measurements.\n\n"
    text+='RULER: 100 samples per task, seven tasks. LongBench v2: every document with fewer than 65,536 context tokens under the pinned tokenizer; no truncation. Official zero-shot MCQ prompting uses native non-thinking chat.\n\n'
    text+='All methods share frozen numbered, block-padded 4096-token chunk layouts. ProphetKV probes the complete fresh query tail (at least 256 tokens), scores only question tokens (including MCQ choices), and applies 10%/20% budgets to eligible cached tokens excluding the exact first chunk and fresh suffix. These are prompt/budget adaptations of the UCM/vLLM port.\n\n'
    text+='TTFT speedup is paired mean full-prefill TTFT / mean method TTFT; the separate geometric metric is labeled explicitly. TTFT includes online probing, transfers, selection and recomputation. Model load, cache build and warmup are excluded and reported in raw records/offline costs. Storage is buffered local warm cache.\n\n'
    text+='Accuracy uses official RULER substring/cleanup scoring and official LongBench v2 answer extraction. Overall accuracy is sample-weighted; RULER has equal task sizes when complete. Confidence intervals use 1,000 paired task-stratified bootstrap draws, seed 42. Length-limited and unparseable outputs are counted.\n\n'
    text+='| Dataset | Task | Method | n | Accuracy | TTFT (s) | Speedup | Capped |\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for r in table:text+=f"| {r['dataset']} | {r['task']} | {r['case']} | {r['n']} | {100*r['accuracy']:.2f}% | {r['mean_ttft_seconds']:.3f} | {r['ttft_speedup'] or 0:.3f}x | {r['length_limited']} |\n"
    (directory/'REPORT.md').write_text(text);(ROOT/'REPORT.md').write_text(text)
    overall=[r for r in table if r['task']=='overall']
    if overall:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for metric in ('accuracy','mean_ttft_seconds','ttft_speedup','paired_accuracy_delta_pp','length_limited'):
            fig,ax=plt.subplots(figsize=(10,5));ax.bar(range(len(overall)),[r[metric] or 0 for r in overall])
            ax.set_xticks(range(len(overall)),[r['dataset']+' '+r['case'] for r in overall],rotation=45,ha='right');ax.set_title(metric)
            fig.tight_layout();fig.savefig(directory/f'{metric}.png',dpi=150);plt.close(fig)
    return directory
