#!/usr/bin/env python3
"""Opt-in: compare prompt-major policy switching against fresh single-input runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runner.setups import atomic_json
from runner.sweep import configurations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuids',nargs='+',required=True)
    args=parser.parse_args()
    if len(args.inputs)<2 or len(args.gpu_uuids) not in (1,2,4,8) or len(set(args.gpu_uuids))!=len(args.gpu_uuids) or any(
            not s.startswith('GPU-') for s in args.gpu_uuids):
        parser.error('Provide at least two prepared inputs and distinct assigned GPU UUIDs')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for index,path in enumerate(args.inputs):
        sample=json.loads(path.read_text())
        sample.update(thinking=sample['thinking'])
        prepared=args.output/f'input-{index}.json';atomic_json(prepared,sample)
        rows.append(dict(id=f'prompt-{index}',prepared=str(prepared)))
    manifest=args.output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    env=dict(os.environ,PYTHON_BIN=sys.executable,CUDA_VISIBLE_DEVICES=','.join(args.gpu_uuids))
    def call(*flags):
        subprocess.run(['bash',str(ROOT/'run.sh'),*map(str,flags)],cwd=ROOT,env=env,check=True)
    out=args.output/'sweep';cache=args.output/'cache'
    call('sweep','--model',args.model.resolve(),'--manifest',manifest,'--output',out,
         '--cache-root',cache,'--tp',len(args.gpu_uuids))
    if list(cache.iterdir()):raise RuntimeError('Temporary KV survived successful sweep')
    comparisons=[]
    for config in configurations():
        for index,row in enumerate(rows):
            fresh=args.output/f'fresh-{config["name"]}-{index}'
            flags=['--ratio',str(config['ratio'])] if config['ratio'] is not None else []
            if config['layers']:flags+=['--layers',*map(str,config['layers'])]
            call('run','--model',args.model.resolve(),'--input',row['prepared'],'--method',config['method'],
                 '--output',fresh,'--tp',len(args.gpu_uuids),*flags)
            measured=out/config['name']/f'{index:06d}'
            a=json.loads((measured/'result.json').read_text())
            b=json.loads((fresh/'result.json').read_text())
            if a['output_token_ids']!=b['output_token_ids']:
                raise RuntimeError(f'Output mismatch: {config["name"]}/{index}')
            def selections(directory):
                return [[{k:e[k] for k in ('scores','selected_positions','scoring_layers','alignment_count')}
                         for e in worker['diagnostics'] if e['kind']=='prophetkv_selection']
                        for worker in json.loads((directory/'diagnostics.json').read_text())]
            if selections(measured)!=selections(fresh):
                raise RuntimeError(f'Score/mask mismatch: {config["name"]}/{index}')
            comparisons.append(dict(config=config['name'],prompt_id=row['id'],equal=True))
    atomic_json(args.output/'validation.json',dict(comparisons=comparisons,temporary_kv_deleted=True))

if __name__=='__main__':main()
