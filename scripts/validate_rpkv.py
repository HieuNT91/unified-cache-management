#!/usr/bin/env python3
"""CPU acceptance against the pinned upstream generators and evaluator; no inference."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ruler_64000 import TASKS,CAPS,REVISION,prepare,definitions,model_template,generator_command
from runner.setups import atomic_json,file_hash,fingerprint
from runner.layout import token_hash,request_metadata,validate_request_metadata
from runner.reporting import score_answer,ruler_preprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ruler','model','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--samples',type=int,default=2)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only: set CUDA_VISIBLE_DEVICES empty')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    for batch in ('first','repeat'):
        prepare(SimpleNamespace(ruler=args.ruler,model=args.model,output=args.output/batch,samples=args.samples,seed=42))
    # Execute upstream evaluator functions without executing its CLI/import side effects.
    source=ast.parse((args.ruler/'scripts/eval/evaluate.py').read_text())
    fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_pred')
    namespace={'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<upstream-postprocess>','exec'),namespace)
    spec=importlib.util.spec_from_file_location('rpkv_official_metrics',args.ruler/'scripts/eval/synthetic/constants.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    custom,constants=definitions(args.ruler)
    evidence=[]
    for task in TASKS:
        first=args.output/'first/raw'/task/'validation.jsonl';repeat=args.output/'repeat/raw'/task/'validation.jsonl'
        if first.read_bytes()!=repeat.read_bytes():raise ValueError('Original seeded task batch is not byte-identical')
        rows=[json.loads(line) for line in first.read_text().splitlines()]
        command=generator_command(SimpleNamespace(ruler=args.ruler,model=args.model,samples=500,seed=42),task,Path('/unused'),tokenizer)
        assert command[command.index('--num_samples')+1]=='500'
        assert command[command.index('--max_seq_length')+1]=='65536'
        assert command[command.index('--tokens_to_generate')+1]==str(CAPS[task])
        predictions,refs=[],[]
        for i,row in enumerate(rows):
            path=args.output/'first/samples'/f'{task}-{i:03d}.json';sample=json.loads(path.read_text())
            repeat_sample=json.loads((args.output/'repeat/samples'/path.name).read_text())
            assert sample==repeat_sample
            original=row['input']+row['answer_prefix']
            # Independently tokenize the exact upstream string, without formatter.
            ids=tokenizer.encode(original,add_special_tokens=False)
            assert ids==sample['token_ids'] and token_hash(ids)==sample['token_sha256']
            assert len(ids)+CAPS[task]<=65536
            assert tokenizer.decode(ids,skip_special_tokens=False)==original
            assert sample['formatted_text']==original and not sample['thinking']
            assert sample['template_sha256']==__import__('hashlib').sha256(model_template(tokenizer,constants[custom[task]['task']]).encode()).hexdigest()
            for method in ('baseline','prophetkv','probe','dense-fallback'):
                rid='a'*32+':'+method
                metadata=request_metadata(rid,ids,'read',sample)
                validate_request_metadata(metadata,rid,ids,sparse=method not in ('baseline','dense-fallback'))
                assert metadata['token_sha256']==sample['token_sha256']
            encoded=tokenizer(original,add_special_tokens=False,return_offsets_mapping=True)
            start=original.rfind(sample['question'])
            positions=[j for j,(a,b) in enumerate(encoded['offset_mapping']) if b>start and a<start+len(sample['question'])]
            assert positions==sample['question_positions']
            b=sample['boundaries'];assert b[-2]==64*(min(positions[0],len(ids)-256)//64)
            evaluation=dict(scoring=sample['scoring'],references=row['outputs'])
            probes=['', ' \t\x00 ', row['outputs'][0].upper(), '\x01'+row['outputs'][0]+'\r\n', 'irrelevant']
            for prediction in probes:
                processed=namespace['postprocess_pred'](prediction,{})
                assert processed==ruler_preprocess(prediction)
                expected=metrics.TASKS[custom[task]['task']]['metric_fn']([processed],[row['outputs']])
                assert round(score_answer(prediction,evaluation)*100,2)==expected
                predictions.append(prediction);refs.append(row['outputs'])
            evidence.append(dict(task=task,row=i,input_tokens=len(ids),cap=CAPS[task],
                token_sha256=token_hash(ids),template_sha256=sample['template_sha256'],raw_sha256=file_hash(first),
                context_chunks=len(b)-2,suffix_tokens=b[-1]-b[-2],source_prefix_once=True))
        expected=metrics.TASKS[custom[task]['task']]['metric_fn']([ruler_preprocess(p) for p in predictions],refs)
        actual=round(sum(score_answer(p,dict(scoring='ruler_any' if task.startswith('qa_') else 'ruler_all',references=r)) for p,r in zip(predictions,refs))/len(predictions)*100,2)
        assert actual==expected
    atomic_json(args.output/'validation.json',dict(passed=True,revision=REVISION,tasks=list(TASKS),
        samples_per_task=args.samples,default_batch_samples=500,seed=42,rows=evidence,
        generator_repeat_byte_equal=True,original_evaluator_equal=True,gpu_executed=False))
    print(f'Validated {len(evidence)} original 64K samples across all 13 tasks; no GPU execution.')

if __name__=='__main__':main()
