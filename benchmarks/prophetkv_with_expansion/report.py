"""CPU-only reporting after 400 validated measurements and owned-engine exit."""
import argparse
import csv
import fcntl
import json
import os
import statistics
import time
from pathlib import Path
from prophetkv_common import dump, sha
from prophetkv_gpu import ROOT, REPO, identity
from cacheblend_prophetkv import CASES
import validation as val
from run import verify, ensure_no_workers, lock, hashes


def finalize():
    with lock():
        p=verify();ensure_no_workers()
        supervisor=json.loads((ROOT/'supervisor.json').read_text())
        if identity(supervisor['pid'])==supervisor['identity']:
            raise RuntimeError('Supervisor has not exited')
        cleanup=json.loads((ROOT/'cleanup.json').read_text())
        if supervisor['state']!='complete' or not cleanup['complete'] or not cleanup['owned_workers_exited']:
            raise RuntimeError('Experiment incomplete or engines not retired')
        rows=val.all_records(p)
        if len(rows)!=400:raise ValueError('Expected 400 validated measurements')
        directory=ROOT/'final';directory.mkdir(exist_ok=True)
        dump(directory/'records.json',rows)
        summaries=[]
        for task in sorted({row['label'] for row in rows})+['aggregate']:
            subset=[r for r in rows if task=='aggregate' or r['label']==task]
            baseline={r['sample_id']:r for r in subset if r['case']=='baseline'}
            for case in CASES:
                selected=[r for r in subset if r['case']==case]
                if {r['sample_id'] for r in selected}!=set(baseline):raise ValueError('Unpaired rows')
                mean=statistics.mean(r['ttft_seconds'] for r in selected)
                paired_baseline=statistics.mean(baseline[r['sample_id']]['ttft_seconds'] for r in selected)
                summaries.append(dict(task=task,method=case,samples=len(selected),
                    accuracy_percent=100*statistics.mean(r['score'] for r in selected),
                    mean_ttft_seconds=mean,median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in selected),
                    paired_mean_ttft_speedup=paired_baseline/mean,
                    prompt_tokens_min=min(r['prompt_tokens'] for r in selected),
                    prompt_tokens_max=max(r['prompt_tokens'] for r in selected),
                    length_limited_outputs=sum(r['finish_reason']=='length' for r in selected),
                    mean_cache_build_seconds=statistics.mean(r['cache_build_seconds'] for r in selected),
                    mean_cache_readiness_seconds=statistics.mean(r['cache_readiness_seconds'] for r in selected),
                    mean_artifact_export_seconds=statistics.mean(r['artifact_export_seconds'] for r in selected)))
        with (directory/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        fields=('sample_id','label','source_row','case','physical_gpu','prompt_tokens','score','ttft_seconds',
                'generation_seconds','cache_build_seconds','cache_readiness_seconds','prime_seconds',
                'artifact_export_seconds','retirement_seconds','output_tokens','finish_reason')
        with (directory/'measurements.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        lines=['Qwen3-4B-Instruct-2507: baseline vs prophetkv_with_expansion',
               '200 distinct prompts, 50/task; 400 fresh validated measurements.',
               'Parameters: '+json.dumps(p['expansion_parameters'],sort_keys=True),
               'BF16 TP1 eager; GPUs 1–4 by UUID; native RoPE; thinking off; greedy 128-token cap.',
               'Fully formatted prompts <=65536 tokens, no truncation; engine allocation 65792.',
               '4096-token chunks and 256 fresh suffix tokens; same inputs/GPU for both methods.',
               'No separate model qualification or smoke. Normal initialization and priming retained.',
               'TTFT: '+p['timing'],
               'Warm buffered local cache; cache construction is excluded from TTFT and reported separately.',
               'Speedup: paired arithmetic-mean baseline TTFT / arithmetic-mean method TTFT.',
               'Method phases are counterbalanced across GPUs; concurrent workloads can affect timings.',
               'Task scores use official local RULER substring metrics. Length-limited outputs remain included.',
               'Every measured expansion mask matches the saved-score float64 reference; every layer uses that set.',
               '', 'task | method | n | accuracy % | mean TTFT s | median TTFT s | speedup | prompt tokens | length limited']
        for row in summaries:
            lines.append(f"{row['task']} | {row['method']} | {row['samples']} | {row['accuracy_percent']:.3f} | "
                         f"{row['mean_ttft_seconds']:.6f} | {row['median_ttft_seconds']:.6f} | "
                         f"{row['paired_mean_ttft_speedup']:.4f}x | {row['prompt_tokens_min']}–{row['prompt_tokens_max']} | "
                         f"{row['length_limited_outputs']}")
        text='\n'.join(lines)+'\n'
        (directory/'results.txt').write_text(text)
        (REPO/'prophetkv_with_expansion_results.txt').write_text(text)
        dump(directory/'validation.json',dict(complete=True,validated=400,
             protocol_sha256=sha(ROOT/'protocol.json'),cleanup_sha256=sha(ROOT/'cleanup.json'),
             artifacts={str(f.relative_to(directory)):sha(f) for f in directory.iterdir()
                        if f.is_file() and f.name!='validation.json'},completed_at=time.time()))
        return summaries


def watch():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Reporter must be CPU-only')
    with (ROOT/'report.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),pgid=os.getpgrp(),sid=os.getsid(0),
                   state='waiting',started_at=time.time())
        dump(ROOT/'reporter.json',state)
        try:
            while True:
                path=ROOT/'supervisor.json'
                if path.exists():
                    supervisor=json.loads(path.read_text())
                    live=identity(supervisor['pid'])==supervisor['identity']
                    if not live and supervisor['state']=='complete':
                        finalize();break
                    if not live and supervisor['state'] in ('stopped','failed'):
                        raise RuntimeError('Supervisor ended before completion')
                time.sleep(10)
            state['state']='complete'
        except BaseException as error:
            state.update(state='failed',error=repr(error));raise
        finally:
            dump(ROOT/'reporter.json',{**state,'ended_at':time.time()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('watch','finalize'))
    args=parser.parse_args()
    watch() if args.command=='watch' else finalize()
