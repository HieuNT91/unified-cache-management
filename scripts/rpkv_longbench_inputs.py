#!/usr/bin/env python3
"""Select full-text LongBench inputs fitting the explicitly approved local window."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runner.config import validate_sample
from runner.setups import atomic_json,file_hash,fingerprint
from scripts.longbench_v2 import prepare_prompt,render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--samples',type=int,required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation must be CPU-only')
    if a.samples<1 or a.output.exists():raise ValueError('Use a positive sample count and fresh output directory')
    from transformers import AutoTokenizer
    rows=json.loads(a.source.read_text());tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    template=(ROOT/'scripts/longbench_0shot.txt').read_text();eligible=[];lengths={}
    for index,row in enumerate(rows):
        content,_,_=render(row,template)
        ids=tok.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        lengths[index]=len(ids)
        if len(ids)+16384<=65920 and len(ids)>8192:eligible.append(index)
        if index%50==0:print(f'Scanned {index}; eligible {len(eligible)}',flush=True)
    indices=random.Random(a.seed).sample(eligible,a.samples);manifest=[]
    for index in indices:
        row=rows[index];sample=prepare_prompt(tok,row,template)
        if sample['truncation']['truncated'] or len(sample['token_ids'])!=lengths[index]:raise ValueError('Full-text reconstruction mismatch')
        sample.update(question_positions=sample['query']['positions'],thinking=True,max_output_tokens=16384,
            model_config_sha256=file_hash(a.model/'config.json'),source_row=index,source_id=row['_id'],
            source_row_sha256=fingerprint(row),source_metadata={k:row[k] for k in ('domain','sub_domain','difficulty','length')})
        validate_sample(sample);name=f'samples/{index:03d}.json';atomic_json(a.output/name,sample)
        manifest.append(dict(id=f'longbench-v2-{index:03d}',prepared=name,sha256=file_hash(a.output/name),
            subtask=row['domain']+' / '+row['sub_domain'],references=[row['answer']],scoring='longbench_v2'))
        print(f'Selected {index}: {len(sample["token_ids"])} tokens',flush=True)
    atomic_json(a.output/'selection.json',dict(seed=a.seed,samples=a.samples,source=str(a.source.resolve()),
        source_sha256=file_hash(a.source),eligible=eligible,all_formatted_lengths=lengths,indices=indices,rows=manifest,
        template_sha256=fingerprint(template),tokenizer_hashes={p.name:file_hash(p) for p in a.model.iterdir() if p.is_file() and p.suffix in ('.json','.jinja','.txt','.model','.tiktoken')},
        preparation_sha256=file_hash(Path(__file__)),formatter_sha256=file_hash(ROOT/'scripts/longbench_v2.py'),
        eligibility='Full native formatted input + 16384 output <= 65920; input >8192 supports sparse context; no truncation',user_authorized_fitting_pool=True))
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))


if __name__=='__main__':main()
