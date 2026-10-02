#!/usr/bin/env python3
"""Finish the authorized LongBench controls after the independent RULER group exits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runner.setups import atomic_json,file_hash
from runner.router_process import identity,alive,group_alive
from scripts.router_control import environment,terminate_owned


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();base=a.root.resolve()
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (base/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),state='waiting-for-ruler',reporter_sha256=file_hash(ROOT/'scripts/rpkv_gpu_results.py'))
        atomic_json(base/'sequence.json',state)
        try:
            ruler=base/'results/ruler'
            while True:
                parent=json.loads((ruler/'supervisor.json').read_text())
                if parent['state']=='failed':raise RuntimeError('RULER failed; dependent GPU work halted')
                if parent['state']=='complete' and not alive(parent):break
                if parent['state']!='complete' and not alive(parent):raise RuntimeError('RULER supervisor exited without completion')
                time.sleep(5)
            for path in (ruler/'processes').glob('*-exit.json'):
                if group_alive(json.loads(path.read_text())['pid']):raise RuntimeError('RULER process group still alive')
            target=base/'results/longbench-v2';command=[sys.executable,'-u',str(base/'frozen-code/scripts/rpkv_gpu_validation.py'),'supervise','--root',str(target)]
            with (target/'supervisor.log').open('w') as log:
                child=subprocess.Popen(command,cwd=base/'frozen-code',env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt=dict(pid=child.pid,identity=identity(child.pid),command=command)
            atomic_json(target/'launch.json',receipt);state.update(state='longbench',child=receipt);atomic_json(base/'sequence.json',state)
            code=child.wait()
            if code:raise RuntimeError(f'LongBench failed ({code}); preserved logs')
            if file_hash(ROOT/'scripts/rpkv_gpu_results.py')!=state['reporter_sha256']:raise ValueError('Reporter source changed')
            state.update(state='reporting');atomic_json(base/'sequence.json',state)
            with (base/'final-report.log').open('w') as log:
                subprocess.run([sys.executable,'-u',str(ROOT/'scripts/rpkv_gpu_results.py'),'--root',str(base)],env=environment(),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            state.update(state='complete',finished_at=time.time())
        except BaseException as error:state.update(state='failed',error=str(error),finished_at=time.time());raise
        finally:atomic_json(base/'sequence.json',state)

if __name__=='__main__':main()
