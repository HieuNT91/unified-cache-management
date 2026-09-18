#!/usr/bin/env python3
"""Refresh the user CSV from complete, receipt-validated ten-sample groups."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'.results/kv-repair-report-20260918'
CSV=REPO/'kv_repair_result.csv'
OLD=REPO/'.results/prophetkv-qwen3-ruler30-20260918'
ACTIVE=REPO/'.results/prophetkv-qwen3-ruler10-64k-chunks-gpu34-20260918'
BACKFILL=REPO/'.results/prophetkv-csv-backfill-20260918'
TASKS=('niah_single_1','niah_multivalue','vt','cwe')
CASES=('baseline','prophetkv-20','prophetkv-30','prophetkv-40')
LENGTHS=(8192,32768,65536)
FATAL=re.compile(r'Traceback|\[UC\]\[E\]|load kv cache failed|dump kv cache failed|CUDA out of memory|EngineDeadError|ERROR\s|\b(?:RuntimeError|AssertionError|ValueError):')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def atomic(p,s):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp');temp.write_text(s);temp.replace(p)

def dump(p,obj):atomic(p,json.dumps(obj,indent=2)+'\n')

def accepted(root,length,chunk,task,case,row):
    sid=f'{task}-{length}-{row:03d}-c{chunk}';path=root/'records'/sid/f'{case}.json'
    try:
        receipt=json.loads(path.with_suffix('.validated.json').read_text())
        for suffix,key in [('.json','record_sha256'),('.log','log_sha256'),('.diagnostics.json','diagnostics_sha256')]:
            if sha(path.with_suffix(suffix))!=receipt[key]:return None
        r=json.loads(path.read_text());sample_path=root/'samples'/sid/'input.json';sample=json.loads(sample_path.read_text())
        if r['protocol_sha256']!=sha(root/'protocol.json') or r['input_sha256']!=sha(sample_path):return None
        if any(r.get(k)!=v for k,v in dict(sample_id=sid,case=case,context_target=length,chunk_size=chunk,label=task,source_row=row,smoke=False,max_output_tokens=128,timing_source='engine_step_first_token_monotonic').items()):return None
        if r['prompt_sha256']!=sample['prompt_sha256'] or r['references']!=sample['source_metadata']['references']:return None
        if not math.isfinite(r['ttft_seconds']) or r['ttft_seconds']<=0:return None
        log=path.with_suffix('.log').read_text(errors='replace')
        if FATAL.search(log) or 'WORKER_COMPLETE' not in log:return None
        if case!='baseline' and not r['cache_verification']['complete']:return None
        return dict(record=r,path=str(path),sha256=sha(path),gpu=r['physical_gpu'])
    except (OSError,KeyError,ValueError,TypeError):return None


def group(length,chunk,task,case):
    # Older pilots supply the baseline/P20 groups; new runs fill only missing cells.
    roots=(ACTIVE,) if length==65536 else (OLD,BACKFILL)
    for root in roots:
        records=[accepted(root,length,chunk,task,case,row) for row in range(10)]
        if all(records):return records
    return None


def refresh():
    ROOT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.reader((ROOT/'original-template.csv').open(newline='')))
    rows[1][0]='GPU: NVIDIA RTX 4500 Ada; chunk size: 512 tokens; n=10/task/method'
    current=None;metadata=[];pending=[];filled=0
    for row in rows:
        row.extend(['']*(17-len(row)))
        if row[0].startswith('Context Length:'):
            current=int(re.search(r'(\d+)k',row[0],re.I)[1])*1024
        elif row[1]=='NIAH-Single-1':row[13]='CWE'
        elif row[1]=='TTFT':
            for j in (1,5,9,13):
                row[j:j+4]=['TTFT (s)','TTFT Speedup (x)','Accuracy (%)','Accuracy improvement (%)']
        elif row[0].startswith('Full Prefill') or row[0].startswith('ProphetKV '):
            case='baseline' if row[0].startswith('Full Prefill') else 'prophetkv-'+re.search(r'(\d+)%',row[0])[1]
            for i,task in enumerate(TASKS):
                j=1+4*i;row[j:j+4]=['']*4
                g=group(current,512,task,case);b=group(current,512,task,'baseline')
                if not g:
                    pending.append(dict(context_target=current,chunk_size=512,task=task,case=case,expected_samples=10));continue
                mean_t=statistics.mean(x['record']['ttft_seconds'] for x in g)
                acc=statistics.mean(x['record']['score'] for x in g)
                row[j]=f'{mean_t:.3f}';row[j+2]=f'{100*acc:.2f}';filled+=2
                entry=dict(context_target=current,chunk_size=512,task=task,case=case,n=10,mean_ttft_seconds=mean_t,accuracy=acc,
                    length_limited=sum(x['record']['finish_reason']=='length' for x in g),records=g)
                if b:
                    assert all(x['record']['prompt_sha256']==y['record']['prompt_sha256'] for x,y in zip(g,b)), 'Mismatched baseline prompts'
                    bt=statistics.mean(x['record']['ttft_seconds'] for x in b);ba=statistics.mean(x['record']['score'] for x in b)
                    speedup=bt/mean_t
                    row[j+1]=f'{speedup:.2f}';filled+=1
                    entry.update(baseline_records=[x['path'] for x in b],ttft_speedup=speedup,
                        accuracy_difference_percentage_points=100*(acc-ba),same_physical_gpu_pairs=sum(x['gpu']==y['gpu'] for x,y in zip(g,b)))
                    if ba>0:
                        improvement=100*(acc-ba)/ba;row[j+3]=f'{improvement:.2f}';filled+=1;entry['accuracy_improvement_percent']=improvement
                    else:entry['accuracy_improvement_undefined']='Baseline accuracy is zero; relative percentage change is undefined.'
                metadata.append(entry)
    out=io.StringIO(newline='');csv.writer(out,lineterminator='\r\n').writerows(rows);atomic(CSV,out.getvalue())
    complete=not pending
    dump(ROOT/'report-metadata.json',dict(updated_at=time.time(),csv_sha256=sha(CSV),complete=complete,filled_numeric_cells=filled,
        formulas=dict(ttft_speedup='mean baseline TTFT / mean method TTFT',accuracy_improvement_percent='100 * (mean method score - mean baseline score) / mean baseline score'),
        policy='Require source rows0–9 and valid receipts for each populated group. No estimates. Negative values retained. Baseline-zero relative accuracy improvement stays blank.',
        chunk_size=512,corrected_template_headers='Final duplicated VT column at32K/64K is CWE, consistent with8K.',groups=metadata,pending_groups=pending))
    lines=['# KV repair CSV provenance','',f'Updated Unix time: {time.time():.3f}. Filled {filled}/192 metric cells; {len(pending)} method/task/context groups pending.','',
        'Model: Qwen3-4B-Instruct-2507. BF16, TP1, eager, greedy128-token cap, non-thinking chat. Nominal chunk size512; ten source rows0–9/task/method. TTFT seconds; accuracy0–100%. Blank means incomplete/unavailable; zero-baseline relative accuracy change is undefined.','',
        'TTFT speedup =mean baseline TTFT/mean method TTFT; baseline1.00x, larger is faster. Accuracy improvement =100*(mean method accuracy−mean baseline accuracy)/mean baseline accuracy. This is relative percent, not percentage points. Signed negative values are regressions. TTFT speedup is the ratio of arithmetic means, not the geometric mean of per-sample ratios.','',
        'Full prefill means no precomputed KV reuse. Online query probing/transfers/selection/recomputation are included in TTFT. Loading, offline cache construction and warmup are excluded. Storage measurements are buffered local warm cache. All source artifacts and hashes are listed in report-metadata.json.','',
        'The historical8K/32K pilot supplies completed cells; supplemental cases run in a later phase on GPUs3/4. GPU identity may differ from the historical baseline for some sample pairs; phase/device provenance is retained. Scores are official RULER substring scores; inspect length-limit counts. Larger chunk results from the active sweep are not mixed into this512-token CSV.','',
        'The repeated final VT header in32K/64K is interpreted as CWE, matching the8K template.','',
        '| Context | Task | Method | n | Length-limited |','|---|---|---|---:|---:|']
    for e in metadata:lines.append(f"| {e['context_target']} | {e['task']} | {e['case']} | {e['n']} | {e['length_limited']} |")
    atomic(ROOT/'README.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(filled_numeric_cells=filled,pending_groups=len(pending),complete=complete)),flush=True)
    return complete


def watch():
    with (ROOT/'report.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dump(ROOT/'report-supervisor.json',dict(pid=os.getpid(),started_at=time.time(),state='running',cpu_only=True))
        while True:
            complete=refresh()
            if complete:
                dump(ROOT/'report-supervisor.json',dict(pid=os.getpid(),ended_at=time.time(),state='complete',cpu_only=True));return
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('refresh','watch'),nargs='?',default='refresh');a=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if a.command=='watch':watch()
    else:refresh()
