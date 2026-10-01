#!/usr/bin/env python3
"""User-requested original-input controls; fresh outputs, no policy training."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runner.setups import atomic_json,file_hash,runtime_identity
from runner.corpus_records import accepted,publish,record_dir
from runner.router_process import identity,group_alive
from scripts.router_control import environment,terminate_owned,cleanup_caches
from scripts.corpus_control import check_hardware


def read(path):return json.loads(Path(path).read_text())


def progress(root,message):
    atomic_json(root/'progress.json',dict(at=time.time(),message=message))
    print(message,flush=True)


def verify(root):
    protocol=read(root/'protocol.json');rows=read(root/'rows.json')
    current=runtime_identity()
    if current!=protocol['runtime']:raise ValueError('Runtime changed; retain this attempt before an explicit migration')
    for row in rows:
        if file_hash(Path(protocol['prepared'])/row['prepared'])!=row['sha256']:raise ValueError('Input changed')
    return protocol,rows


def report(root,final=False):
    from runner.reporting import metrics
    protocol,rows=verify(root);result={};tasks=sorted({r['subtask'] for r in rows})
    for action in protocol['actions']:
        records=[accepted(root,action['id'],row,protocol) for row in rows]
        records=[r for r in records if r is not None]
        result[action['id']]=dict(overall=metrics(records,len(rows)),tasks={t:metrics([r for r in records if r['subtask']==t],sum(r['subtask']==t for r in rows)) for t in tasks})
        if final and len(records)!=len(rows):raise ValueError('Cannot finalize incomplete answers')
    probes=sum(accepted(root,'probe',row,protocol) is not None for row in rows)
    if final and probes!=len(rows):raise ValueError('Cannot finalize incomplete probes')
    value=dict(dataset=protocol['dataset'],final=final,actions=result,probes=probes,
        definition='TTFT includes request metadata and TP action synchronization, excludes separate cache construction/readiness/priming and independent diagnostic probe; lengths are generated content tokens, controls separate.')
    atomic_json(root/'report.json',value)
    lines=['# rpkv GPU controls', '',value['definition'],'','| Task | Method | N | Accuracy % | TTFT s | Thinking tokens | Answer tokens | Capped |','|---|---|---:|---:|---:|---:|---:|---:|']
    csvrows=[]
    def fmt(v):return '-' if v is None else f'{v:.4f}'
    for task in ['overall']+tasks:
        for action,data in result.items():
            m=data['overall'] if task=='overall' else data['tasks'][task]
            lines.append(f"| {task} | {action} | {m['completed']}/{m['expected']} | {fmt(m['accuracy_percent'])} | {fmt(m['mean_ttft_seconds'])} | {fmt(m['mean_thinking_tokens'])} | {fmt(m['mean_answer_tokens'])} | {m['output_cap_reached']} |")
            csvrows.append(dict(task=task,method=action,**m))
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    with (root/'report.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    return value


def worker(root,phase,attempt):
    from runner.corpus_runtime import Engine
    from runner.corpus import load_attention
    from runner.setups import check_environment
    protocol,rows=verify(root);check_environment(4)
    inventory=protocol['actions'][:1] if phase=='baseline' else protocol['actions'][1:]
    remaining=[r for r in rows if any(accepted(root,a['id'],r,protocol) is None for a in inventory)]
    if not remaining:return
    engine=Engine(root,protocol,0,phase,attempt,remaining)
    try:
        for row in remaining:
            progress(root,f"{phase} starting {row['id']}");engine.begin(row);attention=None
            if phase=='cached':
                folder=record_dir(root,'probe',row['id']);probe=accepted(root,'probe',row,protocol)
                if probe is None:
                    probe,ds,attention=engine.probe(folder);publish(root,'probe',row,protocol,probe,ds)
                else:attention=load_attention(folder,probe,engine.sample,protocol['actions'])
            for action in inventory:
                if accepted(root,action['id'],row,protocol) is not None:continue
                result,ds=engine.answer(action,action['id'],attention)
                if protocol['dataset']=='ruler' and result['thinking_tokens']!=0:raise ValueError('RULER unexpectedly generated thinking')
                publish(root,action['id'],row,protocol,result,ds)
                progress(root,f"{action['id']} accepted {row['id']}")
            deletion=engine.end();atomic_json(root/'deletions'/f'{phase}-{row["id"]}.json',deletion)
            report(root)
    finally:engine.close()


def supervise(root):
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol,rows=verify(root)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',started_at=time.time())
        atomic_json(root/'supervisor.json',state)
        try:
            for phase in protocol.get('phase_order',['cached','baseline']):
                inventory=protocol['actions'][:1] if phase=='baseline' else protocol['actions'][1:]
                if all(accepted(root,a['id'],r,protocol) is not None for r in rows for a in inventory):continue
                check_hardware(protocol);attempt=uuid.uuid4().hex
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--root',str(root),'--phase',phase,'--attempt',attempt]
                with (root/f'{phase}-{attempt}.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=environment(','.join(protocol['groups'][0])),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                receipt=dict(pid=child.pid,identity=identity(child.pid),phase=phase,attempt=attempt,command=command)
                atomic_json(root/'processes'/f'{attempt}.json',receipt);state.update(worker=receipt);atomic_json(root/'supervisor.json',state)
                last=time.monotonic();mtime=0
                try:
                    while child.poll() is None:
                        p=root/'progress.json'
                        if p.exists() and p.stat().st_mtime>mtime:mtime=p.stat().st_mtime;last=time.monotonic()
                        if time.monotonic()-last>1800:raise RuntimeError('1800-second progress watchdog expired')
                        time.sleep(2)
                except BaseException:
                    terminate_owned(receipt);child.wait(timeout=10);raise
                if group_alive(child.pid):terminate_owned(receipt)
                atomic_json(root/'processes'/f'{attempt}-exit.json',dict(**receipt,exit_code=child.returncode,owned_engines_exited=not group_alive(child.pid)))
                cleanup_caches(root,protocol)
                if child.returncode:raise RuntimeError(f'{phase} failed ({child.returncode}); inspect preserved log')
            report(root,True);state.update(state='complete',finished_at=time.time())
            atomic_json(root/'complete.json',dict(answers=3*len(rows),probes=len(rows),owned_engines_exited=True))
        except BaseException as error:
            state.update(state='failed',error=str(error),finished_at=time.time());raise
        finally:atomic_json(root/'supervisor.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('worker','supervise','report'))
    p.add_argument('--root',required=True,type=Path);p.add_argument('--phase',choices=('baseline','cached'));p.add_argument('--attempt')
    a=p.parse_args();root=a.root.resolve()
    if a.command=='worker':worker(root,a.phase,a.attempt)
    elif a.command=='supervise':supervise(root)
    else:report(root,(root/'complete.json').exists())

if __name__=='__main__':main()
