#!/usr/bin/env python3
"""Maintain the user-edited kv_repair.csv with per-section chunk baselines."""
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

import fill_kv_repair_report as base
ROOT=base.ROOT/'chunks'
CSV=base.REPO/'kv_repair.csv'


def refresh():
    rows=list(csv.reader((ROOT/'original-template.csv').open(newline='')))
    rows[1][0]='GPU: NVIDIA RTX 4500 Ada; chunk sizes specified per section; n=10/task/method'
    current=None;chunk=512;groups=[];pending=[];filled=0;sections=[]
    for row in rows:
        row.extend(['']*(17-len(row)))
        if row[0].startswith('Context Length:'):
            current=int(re.search(r'(\d+)k',row[0],re.I)[1])*1024
            match=re.search(r'chunk size\s+(\d+)',row[0],re.I)
            chunk=int(match[1]) if match else 512
            row[0]=f'Context Length: {current//1024}k - chunk size {chunk}'
            sections.append(dict(context_target=current,chunk_size=chunk))
        elif row[1]=='NIAH-Single-1':row[13]='CWE'
        elif row[1]=='TTFT (s)':
            for col in (2,6,10,14):row[col]='TTFT Speedup (x)'
        elif row[0].startswith(('Full Prefill','ProphetKV ')):
            case='baseline' if row[0].startswith('Full Prefill') else 'prophetkv-'+re.search(r'(\d+)%',row[0])[1]
            for i,task in enumerate(base.TASKS):
                col=1+4*i;row[col:col+4]=['']*4
                g=base.group(current,chunk,task,case);b=base.group(current,chunk,task,'baseline')
                if not g:
                    pending.append(dict(context_target=current,chunk_size=chunk,task=task,case=case,expected_samples=10));continue
                ttft=statistics.mean(x['record']['ttft_seconds'] for x in g)
                acc=statistics.mean(x['record']['score'] for x in g)
                row[col]=f'{ttft:.3f}';row[col+2]=f'{100*acc:.2f}';filled+=2
                item=dict(context_target=current,chunk_size=chunk,task=task,case=case,n=10,mean_ttft_seconds=ttft,accuracy=acc,
                    length_limited=sum(x['record']['finish_reason']=='length' for x in g),records=g)
                if b:
                    assert all(x['record']['prompt_sha256']==y['record']['prompt_sha256'] for x,y in zip(g,b))
                    bt=statistics.mean(x['record']['ttft_seconds'] for x in b);ba=statistics.mean(x['record']['score'] for x in b)
                    speedup=bt/ttft
                    row[col+1]=f'{speedup:.2f}';filled+=1
                    item.update(ttft_speedup=speedup,baseline_records=[x['path'] for x in b],
                        paired_ttft_speedup=math.exp(statistics.mean(math.log(y['record']['ttft_seconds']/x['record']['ttft_seconds']) for x,y in zip(g,b))),
                        accuracy_difference_percentage_points=100*(acc-ba),same_physical_gpu_pairs=sum(x['gpu']==y['gpu'] for x,y in zip(g,b)))
                    if ba>0:
                        improvement=100*(acc-ba)/ba;row[col+3]=f'{improvement:.2f}';filled+=1;item['accuracy_improvement_percent']=improvement
                    else:item['accuracy_improvement_undefined']='Baseline accuracy is zero.'
                groups.append(item)
    text=io.StringIO(newline='');csv.writer(text,lineterminator='\r\n').writerows(rows);base.atomic(CSV,text.getvalue())
    complete=not pending;total=len(sections)*len(base.TASKS)*len(base.CASES)*4
    base.dump(ROOT/'report-metadata.json',dict(updated_at=time.time(),csv=str(CSV),csv_sha256=base.sha(CSV),complete=complete,
        filled_numeric_cells=filled,total_metric_cells=total,sections=sections,groups=groups,pending_groups=pending,
        ttft_formula='mean baseline TTFT/mean method TTFT',
        accuracy_formula='100*(mean method accuracy-mean baseline accuracy)/mean baseline accuracy',
        baseline_matching='Same task, source rows0–9, context target, chunk size and frozen prompt hashes. No cross-chunk baseline substitution.'))
    jobs=[]
    for p in pending:
        owner=base.ACTIVE if p['context_target']==65536 else base.BACKFILL
        assert (p['context_target']==65536 and p['chunk_size'] in (512,4096)) or p['chunk_size']==512
        jobs.append({**p,'result_directory':str(owner),'scheduling':'already scheduled','source_rows':list(range(10))})
    base.dump(ROOT/'jobs.json',dict(additional_jobs_required=0,groups=jobs))
    base.atomic(ROOT/'README.md',f'''# Chunk-specific KV repair report

Output: kv_repair.csv. Filled {filled}/{total} numeric cells; {len(pending)} groups pending.

Sections preserve the user template:8K/512,32K/512,64K/512,64K/4096.
Every populated group requires ten accepted source rows0–9. TTFT is mean seconds;
accuracy is percent. TTFT speedup is mean full-prefill TTFT/mean method TTFT
(baseline1.00x; larger is faster). Accuracy improvement remains a signed relative
percentage against full prefill with the SAME chunk layout and identical prompt hashes.
Undefined zero-baseline accuracy improvement remains blank. No estimated results.

The512-token source results and supplemental queue retain their prior provenance.
The64K/4096 results come from the active1200-request GPU3/4 sweep. No additional
GPU jobs are required: all pending cells have existing jobs (see jobs.json).
The CPU-only updater refreshes every30seconds. The separately maintained
kv_repair_result.csv remains the original512-token report.

Report metadata includes raw record paths/hashes, paired speedups, sample counts
and length-limited outputs. Substring scores do not certify completed responses.
Changing chunk size also changes prompt formatting and exact-prefix reuse.
''')
    print(json.dumps(dict(filled_numeric_cells=filled,total_metric_cells=total,pending_groups=len(pending),complete=complete)),flush=True)
    return complete


def watch():
    with (ROOT/'report.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.dump(ROOT/'supervisor.json',dict(pid=os.getpid(),started_at=time.time(),state='running',cpu_only=True))
        while True:
            if refresh():
                base.dump(ROOT/'supervisor.json',dict(pid=os.getpid(),ended_at=time.time(),state='complete',cpu_only=True));return
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('refresh','watch'),default='refresh',nargs='?');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.command=='watch':watch()
    else:refresh()
