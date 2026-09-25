"""CPU-only final report after every fresh measurement validates and engines exit."""
import argparse,csv,fcntl,json,os,statistics,time
from pathlib import Path
from prophetkv_common import dump,sha
from prophetkv_gpu import ROOT,REPO,identity
from settings import CASES,PARAMETERS
import validation as val
from run import verify,ensure_no_workers,lock
from protocol import verify_preserved


def finalize():
    with lock():
        p=verify();ensure_no_workers()
        supervisor=json.loads((ROOT/'supervisor.json').read_text())
        if identity(supervisor['pid'])==supervisor['identity']:raise RuntimeError('Supervisor has not exited')
        cleanup=json.loads((ROOT/'cleanup.json').read_text())
        if supervisor['state']!='complete' or not cleanup['complete'] or not cleanup['owned_workers_exited']:
            raise RuntimeError('Incomplete experiment or engines not retired')
        preserved=verify_preserved()
        rows=val.all_records(p)
        if len(rows)!=p['measured_requests']:raise ValueError('Incomplete validated matrix')
        directory=ROOT/'final';directory.mkdir(exist_ok=True);dump(directory/'records.json',rows)
        summaries=[]
        for task in sorted({r['label'] for r in rows})+['ruler_aggregate','aggregate']:
            subset=[r for r in rows if task=='aggregate' or r['label']==task or (task=='ruler_aggregate' and r['dataset']=='ruler')]
            baseline={r['sample_id']:r for r in subset if r['case']=='baseline'}
            for case in CASES:
                selected=[r for r in subset if r['case']==case]
                if {r['sample_id'] for r in selected}!=set(baseline):raise ValueError('Unpaired rows')
                mean=statistics.mean(r['ttft_seconds'] for r in selected)
                summaries.append(dict(task=task,method=case,samples=len(selected),
                    total_ratio=PARAMETERS[case].get('total_ratio',''),anchor_ratio=PARAMETERS[case].get('anchor_ratio',''),
                    max_gap=PARAMETERS[case].get('max_gap',''),
                    accuracy_percent=100*statistics.mean(r['score'] for r in selected),mean_ttft_seconds=mean,
                    median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in selected),
                    paired_mean_ttft_speedup=statistics.mean(baseline[r['sample_id']]['ttft_seconds'] for r in selected)/mean,
                    prompt_tokens_min=min(r['prompt_tokens'] for r in selected),prompt_tokens_max=max(r['prompt_tokens'] for r in selected),
                    length_limited_outputs=sum(r['finish_reason']=='length' for r in selected),
                    mean_cache_build_seconds=statistics.mean(r['cache_build_seconds'] for r in selected),
                    mean_cache_readiness_seconds=statistics.mean(r['cache_readiness_seconds'] for r in selected),
                    mean_artifact_export_seconds=statistics.mean(r['artifact_export_seconds'] for r in selected)))
        with (directory/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        fields=('sample_id','dataset','label','source_row','case','physical_gpu','prompt_tokens','score','ttft_seconds',
                'generation_seconds','cache_build_seconds','cache_readiness_seconds','prime_seconds',
                'artifact_export_seconds','retirement_seconds','output_tokens','finish_reason')
        with (directory/'measurements.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        lines=['Qwen3-4B-Instruct-2507 expansion half-budget gap4 output256',
               f"{len(p['samples'])} distinct prompts, {len(rows)} validated fresh measurements.",
               f"RULER: 100 prompts each of seven tasks. LongBench v2: all {p['longbench_count']} qualifying rows.",
               'Parameters: '+json.dumps(PARAMETERS,sort_keys=True),
               'Baseline: no UCM connector; prefix caching disabled. Expansion: half total ratio allocated to anchors; remaining budget uses right windows then ranked fallback.',
               'BF16 TP1 eager, native RoPE, non-thinking, greedy output cap256, GPUs1–4 by UUID.',
               f"4096-token chunks; common engine allocation {p['max_model_len']} tokens.",
               p['ruler_selection'],p['longbench_eligibility'],p['fresh_suffix_policy'],
               'No truncation. No separate model qualification/smoke; normal loading, priming and readiness checks retained.',
               'Warmup cleanup requires full-size committed disk blocks, after transfer retirement; excluded from TTFT.',
               'TTFT: '+p['timing'],p['timing_cohort_note'],
               'RULER: official local substring scoring; QA accepts any reference. LongBench v2: exact extracted multiple-choice letter.',
               'Length-limited outputs remain included. Every measured expansion mask matches its saved-score reference; every layer uses the same set.',
               'Accuracy aggregation is sample-weighted. Speedup is paired arithmetic-mean baseline TTFT divided by method mean TTFT.',
               '', 'task | method | n | accuracy % | mean TTFT s | median TTFT s | speedup | prompt tokens | length limited']
        for row in summaries:
            lines.append(f"{row['task']} | {row['method']} | {row['samples']} | {row['accuracy_percent']:.3f} | "
                         f"{row['mean_ttft_seconds']:.6f} | {row['median_ttft_seconds']:.6f} | "
                         f"{row['paired_mean_ttft_speedup']:.4f}x | {row['prompt_tokens_min']}–{row['prompt_tokens_max']} | {row['length_limited_outputs']}")
        text='\n'.join(lines)+'\n';(directory/'results.txt').write_text(text)
        (REPO/'prophetkv_expansion_half_gap4_results.txt').write_text(text)
        dump(directory/'validation.json',dict(complete=True,validated=len(rows),preserved_prior_artifacts=preserved,
             protocol_sha256=sha(ROOT/'protocol.json'),cleanup_sha256=sha(ROOT/'cleanup.json'),
             preserved_prior_sha256=sha(ROOT/'preserved-prior.json'),
             artifacts={str(f.relative_to(directory)):sha(f) for f in directory.iterdir() if f.is_file() and f.name!='validation.json'},
             completed_at=time.time()))
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
