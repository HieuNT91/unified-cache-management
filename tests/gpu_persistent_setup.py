#!/usr/bin/env python3
"""Opt-in GPU collection reuse/equivalence check; never runs during CPU discovery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.setups import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True,
                        help='At least two prepared JSON prompts, each with chunks <=4096')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuids', nargs='+', required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2 or len(args.gpu_uuids) not in (1,2,4,8) or len(set(args.gpu_uuids)) != len(args.gpu_uuids) or any(
            not value.startswith('GPU-') for value in args.gpu_uuids):
        parser.error('Supply at least two prompts and distinct, explicitly assigned GPU UUIDs')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # Preserve every input token; make generation deterministic for exact comparisons.
    entries = []
    for i, path in enumerate(args.inputs):
        sample = json.loads(path.read_text())
        sample.update(thinking=sample['thinking'])
        prepared = args.output / f'input-{i}.json'
        atomic_json(prepared, sample)
        entries.append(dict(id=f'prompt-{i}', prepared=str(prepared)))
    manifest = args.output / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(e) for e in entries)+'\n')
    env = dict(os.environ, PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES=','.join(args.gpu_uuids))

    def call(*arguments):
        subprocess.run(['bash',str(ROOT/'run.sh'),*map(str,arguments)],cwd=ROOT,env=env,check=True)

    setup = args.output / 'setup'
    command = ('setup','--model',args.model.resolve(),'--manifest',manifest,'--output',setup,
               '--tp',len(args.gpu_uuids))
    call(*command)

    def artifacts():
        return {str(p.relative_to(setup)):file_hash(p) for p in setup.rglob('*')
                if p.is_file() and p.name != '.lock'}

    original = artifacts()
    call(*command)  # Must return without loading another engine or changing receipts.
    if original != artifacts():
        raise RuntimeError('Completed setup changed on repeated setup command')
    matrix = [ ('prophet10','prophetkv',['--ratio','.1']),
               ('prophet20','prophetkv',['--ratio','.2']),
               ('last5','selective_prophetkv',['--num-layers','5','--ratio','.2']),
               ('explicit5','selective_prophetkv',['--layers','45','48','50','56','58','--ratio','.2']),
               ('baseline','baseline',[]) ]
    comparisons = []
    for name, method, flags in matrix:
        output = args.output/name
        call('run','--setup',setup,'--method',method,'--output',output,*flags)
        if original != artifacts():
            raise RuntimeError('Persistent setup changed during experiment/retirement')
        for i, entry in enumerate(entries):
            fresh = args.output/f'fresh-{name}-{i}'
            call('run','--model',args.model.resolve(),'--input',entry['prepared'],
                 '--method',method,'--output',fresh,'--tp',len(args.gpu_uuids),*flags)
            retained = json.loads((output/f'{i:06d}'/'result.json').read_text())
            rebuilt = json.loads((fresh/'result.json').read_text())
            if retained['output_token_ids'] != rebuilt['output_token_ids']:
                raise RuntimeError(f'Fresh/persistent output mismatch: {name}/{i}')
            def selection(path):
                return [[e for e in worker['diagnostics'] if e['kind']=='prophetkv_selection']
                        for worker in json.loads(path.read_text())]
            if selection(output/f'{i:06d}'/'diagnostics.json') != selection(fresh/'diagnostics.json'):
                raise RuntimeError(f'Fresh/persistent score/mask mismatch: {name}/{i}')
            comparisons.append(dict(method=name,prompt_id=entry['id'],equal=True))
    call('run','--setup',setup,'--method','prophetkv','--ratio','.1',
         '--prompt-ids','prompt-1','--output',args.output/'subset')
    if original != artifacts():
        raise RuntimeError('Subset run mutated setup')
    atomic_json(args.output/'validation.json',dict(comparisons=comparisons,
        immutable_setup_artifacts=original, gpu_uuids=args.gpu_uuids, passed=True))


if __name__ == '__main__':
    main()
