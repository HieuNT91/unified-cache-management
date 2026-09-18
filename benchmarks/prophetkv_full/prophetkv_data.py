"""Freeze 100 RULER rows/task and every LongBench v2 context below 65536 tokens."""
import json
import re
from pathlib import Path
from cacheblend_ruler import cacheblend_prompt,prompt_digest
from prophetkv_gpu import REPO,ROOT
MODEL=Path('/home/thnguyen/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554')
TASKS=('vt','cwe','niah_multivalue','niah_single_1','niah_multikey_3','qa_1','qa_2')
LENGTHS=(65536,)
CHUNKS=(4096,)

def generate_sources():
    for task in TASKS:
        path=ROOT/f'datasets/65536/{task}/validation.jsonl'
        if not path.exists() or len(path.read_text().splitlines())!=100:
            raise ValueError(f'Missing 100-row generated dataset: {path}')

def encode(tokenizer,content,begin,end,answer_prefix=''):
    text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)+answer_prefix
    offset=text.index(content)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    query=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>offset+begin and a<offset+end]
    if not query:raise ValueError('Empty query span')
    ids=encoded['input_ids'];suffix=max(256,len(ids)-query[0])
    chunks,tokens=cacheblend_prompt(ids,tokenizer,tokenizer.pad_token_id,4095,suffix)
    if len(chunks)<2:raise ValueError('Fewer than two context chunks')
    if any(len(c)!=4096 for c in chunks[:-1]) or len(chunks[-1])>4096:raise ValueError('Invalid chunk sizes')
    bounds=[0]
    for chunk in chunks:bounds.append(bounds[-1]+len(chunk))
    bounds.append(len(tokens))
    positions=[i+len(tokens)-len(ids) for i in query]
    if positions[0]<bounds[-2] or ids[-suffix:]!=tokens[-suffix:]:raise ValueError('Query/suffix changed')
    return dict(original_tokens=len(ids),tokens=len(tokens),token_ids=tokens,boundaries=bounds,
        fresh_suffix_tokens=suffix,prompt_sha256=prompt_digest(tokens),
        query=dict(text=content[begin:end],positions=positions,extraction='question text and MCQ choices where applicable; excludes reference answer, output instructions and assistant prefix'))

def prepare_samples(sha,dump):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    samples=[];datasets={}
    def save(sample):
        sample.update(chunk_size=4096,aligned_chunk_size=4096,context_target=65536,physical_gpu=1+len(samples)%4)
        path=ROOT/'samples'/sample['id']/'input.json';dump(path,sample)
        samples.append({k:v for k,v in sample.items() if k!='token_ids'}|dict(input_path=str(path),input_sha256=sha(path)))
    for task in TASKS:
        source=ROOT/f'datasets/65536/{task}/validation.jsonl';digest=sha(source);datasets[str(source)]=digest
        for ordinal,line in enumerate(source.read_text().splitlines()):
            row=json.loads(line);content=row['input']
            marker='\nWhat ' if task.startswith('niah_') else '\nQuestion:'
            start=content.rfind(marker)+1
            if start==0:raise ValueError('Question missing')
            separator=' The special magic' if task.startswith('niah_') else ' Answer:'
            value=content[start:].split(separator,1)[0]
            save(dict(id=f'{task}-65536-{ordinal:03d}-c4096',dataset='ruler',label=task,source_row=ordinal,
                source_index=row.get('source_index',row.get('index')),source_path=str(source),
                source_metadata=dict(task=task,references=row['outputs'],source_sha256=digest),
                **encode(tokenizer,content,start,start+len(value),row.get('answer_prefix',''))))
        print(f'frozen RULER {task} n=100',flush=True)
    source=REPO/'.data/LongBench-v2/data.json';template_path=REPO/'.data/LongBench-v2/0shot.txt'
    for p in (source,template_path):
        target=ROOT/'datasets/longbench_v2'/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        datasets[str(target)]=sha(target)
    source=ROOT/'datasets/longbench_v2/data.json';digest=sha(source)
    template=(ROOT/'datasets/longbench_v2/0shot.txt').read_text().replace('\\n','\n')
    markers={'$Q$':'question',**{f'$C_{c}$':f'choice_{c}' for c in 'ABCD'}}
    census=[]
    cached=ROOT/'preparation/longbench-census.json'
    counts={x['row']:x['context_tokens'] for x in json.loads(cached.read_text())} if cached.exists() else {}
    for ordinal,row in enumerate(json.loads(source.read_text())):
        context_tokens=counts.get(ordinal)
        if context_tokens is None:context_tokens=len(tokenizer.encode(row['context'],add_special_tokens=False))
        census.append(dict(row=ordinal,id=row['_id'],context_tokens=context_tokens,included=context_tokens<65536,source_category=row['length']))
        if context_tokens>=65536:continue
        before,sep,after=template.partition('$DOC$')
        if not sep:raise ValueError('Missing document placeholder')
        tail=re.sub(r'\$(?:Q|C_[ABCD])\$',lambda m:row[markers[m[0]]],after)
        content=before+row['context']+tail
        begin=len(before)+len(row['context'])+tail.index('What is the correct answer to this question:')
        end=len(before)+len(row['context'])+tail.rindex('\n\nFormat your response')
        label=re.sub('[^a-z0-9]+','_',row['domain'].lower()).strip('_')
        save(dict(id=f'longbench-v2-{ordinal:03d}-c4096',dataset='longbench_v2',label=label,
            source_row=ordinal,source_index=row['_id'],source_path=str(source),context_tokens=context_tokens,
            source_metadata=dict(answer=row['answer'],references=[row['answer']],domain=row['domain'],
                sub_domain=row['sub_domain'],difficulty=row['difficulty'],length_category=row['length'],source_sha256=digest),
            **encode(tokenizer,content,begin,end)))
        if len(samples)%20==0:print(f'frozen LongBench included={len(samples)-700} scanned={ordinal+1}',flush=True)
    dump(ROOT/'longbench-selection.json',dict(threshold=65536,comparison='strictly less than',unit='Qwen3 pinned tokenizer context tokens without chat/question/markers',source_sha256=digest,rows=census,included=sum(r['included'] for r in census),truncated=False))
    return samples,datasets
