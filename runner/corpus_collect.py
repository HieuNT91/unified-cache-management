"""One TP4 worker, baseline tranche first then prompt-major cached collection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import uuid
from runner.corpus_records import accepted,publish,record_dir
from runner.corpus_runtime import Engine,infer_prompt
from runner.corpus import load_attention
from runner.setups import atomic_json,check_environment
from runner.tree_policy import actions,load


def run_group(root,group,phase,attempt):
    from scripts.corpus_control import load_run
    root=Path(root);protocol,rows=load_run(root,hardware=False)
    selected=[r for r in rows if r['ordinal']%len(protocol['groups'])==group]
    cases=['nocache'] if phase=='baseline' else ['router'] if protocol['kind']=='inference' else ['probe']+[a for a in actions(protocol['actions']) if a!='nocache']
    pending=[r for r in selected if any(accepted(root,c,r,protocol) is None for c in cases)]
    if not pending:return
    if check_environment(4)!=protocol['groups'][group]:raise ValueError('GPU group changed')
    with (root/f'group{group}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine=Engine(root,protocol,group,phase,attempt,pending)
        tree=load(root/'tree.json') if protocol['kind']=='inference' else None
        try:
            for row in pending:
                engine.begin(row);staged=[]
                for case in cases:
                    if accepted(root,case,row,protocol) is not None:continue
                    folder=record_dir(root,case,row['id'])
                    if folder.exists():
                        history=root/'incomplete'/f'{case}-{row["id"]}-{uuid.uuid4().hex}'
                        history.parent.mkdir(exist_ok=True);folder.rename(history)
                    folder.mkdir(parents=True)
                    if case=='probe':record,ds,attention=engine.probe(folder)
                    elif case=='router':
                        baseline=accepted(root,'nocache',row,protocol)
                        if baseline is None:raise ValueError('Missing connector-free baseline')
                        baseline_ds=json.loads((record_dir(root,'nocache',row['id'])/'diagnostics.json').read_text())
                        record,ds=infer_prompt(engine,tree,folder,baseline,baseline_ds)
                    else:
                        if case!='nocache' and not any(c=='probe' for c,_,_ in staged):
                            probe=accepted(root,'probe',row,protocol)
                            if probe is None:raise ValueError('Missing accepted independent probe')
                            attention=load_attention(record_dir(root,'probe',row['id']),probe,engine.sample,protocol['actions'])
                        record,ds=engine.answer(actions(protocol['actions'])[case],case,None if case=='nocache' else attention)
                    staged.append((case,record,ds))
                deletion=engine.end()
                for case,record,ds in staged:
                    record['cache_deletion']=deletion
                    publish(root,case,row,protocol,record,ds)
                    atomic_json(root/f'progress-group{group}.json',dict(phase=phase,case=case,prompt_id=row['id'],accepted_at=time.time()))
                    print(f'{case} {row["id"]}: accepted',flush=True)
        finally:engine.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--group',type=int,required=True)
    p.add_argument('--phase',choices=('baseline','cached'),required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args()
    try:run_group(a.root,a.group,a.phase,a.attempt)
    except (TimeoutError,ConnectionError,OSError):
        import traceback;traceback.print_exc();raise SystemExit(75)
    except Exception:
        import traceback;traceback.print_exc();raise SystemExit(76)
if __name__=='__main__':main()
