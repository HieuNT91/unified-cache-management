#!/usr/bin/env python3
"""Opt-in TP4 original-input probe/control/dense-bypass acceptance. Never auto-run."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','input','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu-uuids',nargs=4,required=True)
    parser.add_argument('--hardware-profile',choices=('server','rtx4500ada'),default='server')
    args=parser.parse_args()
    if len(set(args.gpu_uuids))!=4 or any(not x.startswith('GPU-') for x in args.gpu_uuids):
        parser.error('Four explicitly assigned UUIDs required')
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(args.gpu_uuids),ENABLE_SPARSE='TRUE',VLLM_USE_V1='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_ALLOW_INSECURE_SERIALIZATION='1',
        VLLM_ATTENTION_BACKEND='FLASH_ATTN',PLATFORM='cuda',VLLM_USE_REROPE='0')
    from runner.config import validate_sample
    from runner.layout import token_hash
    from runner.setups import atomic_json,file_hash,check_environment
    from runner.corpus_runtime import Engine
    sample=validate_sample(json.loads(args.input.read_text()))
    if not sample.get('evaluation_protocol') or len(sample['token_ids'])<32768:
        parser.error('Use a prepared original RULER sample spanning multiple prefill steps')
    check_environment(4)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'input.json',sample)
    row=dict(id='acceptance',prepared='input.json',sha256=file_hash(root/'input.json'),
        subtask=sample['task'],references=sample['references'],scoring=sample['scoring'])
    actions=[dict(id='nocache',method='baseline',ratio=None),
             dict(id='prophetkv-1',method='prophetkv',ratio=.01),
             dict(id='prophetkv-100',method='prophetkv',ratio=1.)]
    protocol=dict(model=str(args.model.resolve()),prepared=str(root),cache_root=str(root/'cache'),
        groups=[args.gpu_uuids],dataset='ruler',hardware_profile=args.hardware_profile,actions=actions)
    records={}
    for phase in ('baseline','cached'):
        engine=Engine(root,protocol,0,phase,phase,[row])
        try:
            engine.begin(row)
            if phase=='baseline':
                baseline,ds=engine.answer(actions[0],'baseline')
                records['baseline']=baseline;atomic_json(root/'baseline-diagnostics.json',ds)
            else:
                probe,probe_ds,attention=engine.probe(root/'probe')
                records['probe']=probe;atomic_json(root/'probe-diagnostics.json',probe_ds)
                for action in actions[1:]:
                    record,ds=engine.answer(action,action['id'],attention)
                    records[action['id']]=record;atomic_json(root/(action['id']+'-diagnostics.json'),ds)
                # Explicit dense control tests the fallback path without inventing
                # a learned policy or invoking a historical tree.
                dense,ds=engine.answer(actions[0],'dense-fallback')
                records['dense-fallback']=dense;atomic_json(root/'dense-diagnostics.json',ds)
                if dense['output_token_ids']!=baseline['output_token_ids']:
                    raise RuntimeError('Dense fallback differs from connector-free baseline')
                if records['prophetkv-100']['output_token_ids']!=baseline['output_token_ids']:
                    raise RuntimeError('Full repair differs from baseline; investigate numerical drift')
                if any(r['store_operations']!=dict(lookup=0,load=0,store=0) for r in dense['dense_receipts']):
                    raise RuntimeError('Dense fallback accessed UCM')
            records[phase+'-deletion']=engine.end()
        finally:
            engine.close()
    atomic_json(root/'validation.json',dict(passed=True,token_sha256=token_hash(sample['token_ids']),
        max_output_tokens=sample['max_output_tokens'],records=records))

if __name__=='__main__':main()
