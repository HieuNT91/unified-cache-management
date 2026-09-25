"""Freeze 100 RULER rows/task and every fully formatted LongBench v2 prompt <64K."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
from prophetkv_gpu import ROOT
from prophetkv_common import sha, dump, load_sample
from cacheblend_ruler import cacheblend_prompt, prompt_digest
from settings import TASKS, CASES, MAX_OUTPUT_TOKENS

MODEL=Path('/home/thnguyen/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554')
OLD=Path('/home/thnguyen/unified-cache-management')
FIRST=OLD/'.results/prophetkv-qwen3-ruler100-longbenchv2-64k-c4096-20260918'
SECOND=OLD/'.results/prophetkv-expanded-ruler13x100-longbenchv2-20260918'


def encode(tokenizer, content, begin, end, answer_prefix='', limit=None):
    text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)+answer_prefix
    offset=text.index(content)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids']
    query=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>offset+begin and a<offset+end]
    if not query:raise ValueError('Empty query span')
    suffix=max(256,len(ids)-query[0])
    chunks,tokens=cacheblend_prompt(ids,tokenizer,tokenizer.pad_token_id,4095,suffix)
    bounds=[0]
    for chunk in chunks:bounds.append(bounds[-1]+len(chunk))
    bounds.append(len(tokens))
    if len(chunks)<2 or any(len(c)!=4096 for c in chunks[:-1]):raise ValueError('Invalid chunk layout')
    positions=[i+len(tokens)-len(ids) for i in query]
    if tokens[-suffix:]!=ids[-suffix:] or positions[0]<bounds[-2]:raise ValueError('Query/suffix changed')
    result=dict(original_tokens=len(ids),tokens=len(tokens),fresh_suffix_tokens=suffix,
        thinking_enabled=False,query=dict(text=content[begin:end],positions=positions))
    if limit is None or len(tokens)<limit:
        result.update(token_ids=tokens,boundaries=bounds,formatted_text=text,prompt_sha256=prompt_digest(tokens))
    return result


def verify():
    m=json.loads((ROOT/'input-manifest.json').read_text())
    for name,digest in m['datasets'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Dataset changed: '+name)
    if len(m['samples'])!=700+m['longbench_count']:raise ValueError('Sample count changed')
    for task in TASKS:
        if [s['source_row'] for s in m['samples'] if s['label']==task]!=list(range(100)):raise ValueError('RULER rows changed')
    if {r['row'] for r in m['longbench_census'] if r['included']}!={s['source_row'] for s in m['samples'] if s['dataset']=='longbench_v2'}:raise ValueError('LongBench selection changed')
    if any(r['included']!=(r['tokens']<65536) for r in m['longbench_census']):raise ValueError('Length eligibility changed')
    seen=set()
    for i,meta in enumerate(m['samples']):
        p=ROOT/meta['input_file']
        if sha(p)!=meta['input_sha256']:raise ValueError('Sample changed')
        sample=load_sample(p)
        if sample['physical_gpu']!=1+i%4:raise ValueError('GPU affinity changed')
        if sample['dataset']=='longbench_v2' and sample['tokens']>=65536:raise ValueError('LongBench over limit')
        if sample['tokens']+MAX_OUTPUT_TOKENS>m['max_model_len']:raise ValueError('Engine capacity too small')
        if sample['prompt_sha256'] in seen:raise ValueError('Duplicate prompt')
        seen.add(sample['prompt_sha256'])
    return m


def prepare():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only preparation required')
    if (ROOT/'input-manifest.json').exists():return verify()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    samples=[];datasets={};census=[]
    def copy(source,name):
        target=ROOT/'datasets'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=sha(source):raise ValueError('Dataset copy mismatch')
        if not target.exists():shutil.copy2(source,target)
        datasets[str(target.relative_to(ROOT))]=sha(target)
        return target
    def save(sample):
        sample.update(chunk_size=4096,aligned_chunk_size=4096,context_target=65536,physical_gpu=1+len(samples)%4)
        relative=f"samples/{sample['id']}/input.json";path=ROOT/relative;dump(path,sample)
        samples.append({k:sample[k] for k in ('id','dataset','label','source_row','tokens','prompt_sha256','physical_gpu')}
            |dict(input_file=relative,input_sha256=sha(path)))
    for task in TASKS:
        original=(FIRST if (FIRST/f'datasets/65536/{task}/validation.jsonl').exists() else SECOND)/f'datasets/65536/{task}/validation.jsonl'
        source=copy(original,f'ruler/{task}/validation.jsonl');digest=sha(source)
        rows=[json.loads(line) for line in source.read_text().splitlines()]
        if len(rows)!=100:raise ValueError('Expected 100 source rows')
        for ordinal,row in enumerate(rows):
            content=row['input'];marker='\nWhat ' if task.startswith('niah_') else '\nQuestion:'
            begin=content.rfind(marker)+1
            if begin==0:raise ValueError('Question missing')
            separator=' The special magic' if task.startswith('niah_') else ' Answer:'
            question=content[begin:].split(separator,1)[0]
            save(dict(id=f'{task}-65536-{ordinal:03d}-c4096',dataset='ruler',label=task,
                source_row=ordinal,source_index=row.get('source_index',row.get('index')),source_path=str(source),
                source_metadata=dict(task=task,references=row['outputs'],source_sha256=digest),
                **encode(tokenizer,content,begin,begin+len(question),row.get('answer_prefix',''))))
        print('Frozen RULER',task,100,flush=True)
    source=copy(OLD/'.data/LongBench-v2/data.json','longbench_v2/data.json');digest=sha(source)
    template=copy(OLD/'.data/LongBench-v2/0shot.txt','longbench_v2/0shot.txt').read_text().replace('\\n','\n')
    before,sep,after=template.partition('$DOC$')
    if not sep:raise ValueError('Missing document placeholder')
    markers={'$Q$':'question',**{f'$C_{c}$':f'choice_{c}' for c in 'ABCD'}}
    for ordinal,row in enumerate(json.loads(source.read_text())):
        tail=re.sub(r'\$(?:Q|C_[ABCD])\$',lambda m:row[markers[m[0]]],after)
        content=before+row['context']+tail
        begin=len(before)+len(row['context'])+tail.index('What is the correct answer to this question:')
        end=len(before)+len(row['context'])+tail.rindex('\n\nFormat your response')
        encoded=encode(tokenizer,content,begin,end,limit=65536);included=encoded['tokens']<65536
        census.append(dict(row=ordinal,id=row['_id'],tokens=encoded['tokens'],included=included))
        if included:
            save(dict(id=f'longbench-v2-{ordinal:03d}-c4096',dataset='longbench_v2',label='longbench_v2',
                source_row=ordinal,source_index=row['_id'],source_path=str(source),
                source_metadata=dict(answer=row['answer'],references=[row['answer']],domain=row['domain'],
                    sub_domain=row['sub_domain'],difficulty=row['difficulty'],length_category=row['length'],source_sha256=digest),
                **encoded))
        if (ordinal+1)%20==0:print('LongBench scanned',ordinal+1,'included',len(samples)-700,flush=True)
    maximum=max(m['tokens'] for m in samples)
    manifest=dict(model=str(MODEL),samples=samples,datasets=datasets,longbench_census=census,
        longbench_count=len(samples)-700,methods=list(CASES),measured_requests=len(samples)*len(CASES),
        max_output_tokens=MAX_OUTPUT_TOKENS,max_model_len=max(65792,((maximum+MAX_OUTPUT_TOKENS+63)//64)*64),
        truncated=False,longbench_eligibility='all fully formatted chat/chunked prompts strictly below 65536 tokens',
        ruler_selection='all 100 source rows/task, unchanged 64K-target content',
        fresh_suffix_policy='max(256, complete question/choices plus trailing instructions and assistant prefix)')
    dump(ROOT/'input-manifest.json',manifest)
    return verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','verify'));args=parser.parse_args()
    m=prepare() if args.command=='prepare' else verify()
    print('Verified',len(m['samples']),'prompts;',m['measured_requests'],'requests; engine',m['max_model_len'],flush=True)
