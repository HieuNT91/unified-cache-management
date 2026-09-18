"""Paired task-stratified reports; CPU-only."""
import csv
import json
import math
from pathlib import Path
import random
import statistics
from prophetkv_common import dump
from prophetkv_gpu import ROOT
from prophetkv_data import TASKS,LENGTHS,CHUNKS
from cacheblend_prophetkv import CASES

def stats(rows):
    base={r['sample_id']:r for r in rows if r['case']=='baseline'}
    output=[]
    for length in LENGTHS:
        for chunk in CHUNKS:
            for task in ('macro',*TASKS):
                for case in CASES:
                    group=[r for r in rows if r['context_target']==length and r['chunk_size']==chunk and r['case']==case and (task=='macro' or r['label']==task)]
                    paired=[r for r in group if r['sample_id'] in base]
                    if not group:continue
                    by_task={t:[r for r in group if r['label']==t] for t in TASKS}
                    accuracy=statistics.mean(statistics.mean(r['score'] for r in g) for g in by_task.values() if g)
                    logs=[math.log(base[r['sample_id']]['ttft_seconds']/r['ttft_seconds']) for r in paired]
                    differences=[r['score']-base[r['sample_id']]['score'] for r in paired]
                    rng=random.Random(42);boot_a=[];boot_t=[]
                    if paired:
                        strata=[[r for r in paired if r['label']==t] for t in TASKS]
                        for _ in range(1000):
                            draw=[rng.choices(g,k=len(g)) for g in strata if g]
                            boot_a.append(statistics.mean(statistics.mean(r['score']-base[r['sample_id']]['score'] for r in g) for g in draw))
                            boot_t.append(math.exp(statistics.mean(math.log(base[r['sample_id']]['ttft_seconds']/r['ttft_seconds']) for g in draw for r in g)))
                    ci=lambda x: [sorted(x)[int(.025*len(x))],sorted(x)[min(len(x)-1,int(.975*len(x)))]] if x else [None,None]
                    output.append(dict(context_target=length,chunk_size=chunk,task=task,case=case,n=len(group),paired_n=len(paired),
                        accuracy=accuracy,mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in group),
                        paired_speedup=math.exp(statistics.mean(logs)) if logs else None,
                        paired_accuracy_delta=statistics.mean(differences) if differences else None,
                        accuracy_delta_ci95=ci(boot_a),speedup_ci95=ci(boot_t),
                        length_limited=sum(r['finish_reason']=='length' for r in group),
                        actual_tokens_min=min(r['prompt_tokens'] for r in group),actual_tokens_max=max(r['prompt_tokens'] for r in group)))
    return output

def report(rows,state,final=False):
    directory=ROOT/('final' if final else 'diagnostic');directory.mkdir(exist_ok=True)
    table=stats(rows)
    dump(directory/'raw-records.json',rows);dump(directory/'tables.json',table)
    costs=[]
    for path in sorted((ROOT/'cache-builds').rglob('*.json')):
        if 'attempts' in path.parts:continue
        item=json.loads(path.read_text())
        if 'cache_build_seconds' in item:costs.append(dict(path=str(path),**item))
    dump(directory/'offline-costs.json',costs)
    if table:
        with (directory/'comparison.csv').open('w') as out:
            w=csv.DictWriter(out,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    text=f'# ProphetKV UCM/vLLM port\n\nState: **{state}**. {len(rows)}/200 validated measurements.\n\n'
    text+='Question-only scores use context-key softmax and sum across 36 layers. Both cached methods use original-position causal attention. Native chat/chunk markers and eligible-region budgets are port adaptations.\n\n'
    text+='TTFT includes the online probe, cache transfer/alignment and selection. Storage uses buffered local warm caches. Loading, cache construction and warmup are excluded and retained separately in raw records.\n\n'
    text+='Confidence intervals use 1,000 paired, task-stratified bootstrap draws with seed 42. One measurement per prompt/method; substring scores do not prove coherent completion.\n\n'
    macro=[r for r in table if r['task']=='macro']
    text+='| Target | Chunk | Method | n | Accuracy | TTFT (s) | Paired speedup | Length-limited |\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for r in macro:text+=f"| {r['context_target']} | {r['chunk_size']} | {r['case']} | {r['n']} | {r['accuracy']:.3f} | {r['mean_ttft_seconds']:.3f} | {r['paired_speedup'] or 0:.3f} | {r['length_limited']} |\n"
    for p in sorted((ROOT/'gates').glob('pilot-*.json')):
        g=json.loads(p.read_text());text+=f"\nPilot {g['context_target']}: passed={g['passed']}; passing methods={g['passing_rates']}.\n"
    (directory/'REPORT.md').write_text(text);(ROOT/'REPORT.md').write_text(text)
    if macro:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for metric,title in [('accuracy','Macro accuracy'),('mean_ttft_seconds','Mean TTFT (seconds)'),('paired_speedup','Paired geometric TTFT speedup'),('length_limited','Length-limited outputs'),('paired_accuracy_delta','Paired accuracy difference')]:
            fig,ax=plt.subplots(figsize=(12,5))
            labels=[f"{r['context_target']//1024}K/{r['chunk_size']} {r['case']}" for r in macro]
            ax.bar(range(len(macro)),[r[metric] or 0 for r in macro]);ax.set_xticks(range(len(macro)),labels,rotation=70,ha='right')
            ax.set_title(title);fig.tight_layout();fig.savefig(directory/f'{metric}.png',dpi=150);plt.close(fig)
    return directory
