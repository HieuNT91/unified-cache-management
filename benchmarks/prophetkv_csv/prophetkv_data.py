"""Frozen RULER sources and native chat layouts for ProphetKV."""
import json
import os
from pathlib import Path
import subprocess
from cacheblend_ruler import cacheblend_prompt,prompt_digest
from prophetkv_gpu import REPO,ROOT,PYTHON

MODEL=Path('/home/thnguyen/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554')
TASKS=('niah_single_1','niah_multivalue','vt','cwe')
LENGTHS=(8192,32768)
CHUNKS=(512,)

def question(tokenizer,task,row,text,tokens):
    marker='\nWhat ' if task.startswith('niah_') else '\nQuestion:'
    at=row['input'].rfind(marker)
    if at<0:raise ValueError('Missing trailing question')
    value=row['input'][at+1:]
    # Local RULER embeds the answer preamble into input, rather than a separate field.
    separator=' Answer:' if not task.startswith('niah_') else ' The special magic'
    value=value.split(separator,1)[0]
    # Only question text, excluding assistant role markers and answer prefix.
    begin=text.index(row['input'])+at+1
    end=begin+len(value)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    original=encoded['input_ids']
    if original[-256:]!=tokens[-256:]:raise ValueError('Changed fresh suffix')
    positions=[len(tokens)-len(original)+i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>begin and a<end]
    if not positions or positions[0]<len(tokens)-256:raise ValueError('Question exceeds fresh suffix')
    return dict(text=value,positions=positions,extraction='question text only, without references or assistant prefix')

def generate_sources():
    import shutil
    for length in LENGTHS:
        for task in TASKS:
            target=ROOT/f'datasets/{length}/{task}/validation.jsonl'
            target.parent.mkdir(parents=True,exist_ok=True)
            if task!='niah_single_1':
                old=REPO/f'.results/prophetkv-qwen3-ruler30-20260918/datasets/{length}/{task}/validation.jsonl'
                shutil.copy2(old,target)
                continue
            if target.exists() and len(target.read_text().splitlines())==10:continue
            log=ROOT/f'preparation/{length}-{task}.log';log.parent.mkdir(parents=True,exist_ok=True)
            cmd=[str(PYTHON),str(REPO/'.downloads/RULER/scripts/data/prepare.py'),
                 '--save_dir',str(ROOT/f'datasets/{length}'),'--benchmark','synthetic','--task',task,
                 '--tokenizer_path',str(MODEL),'--tokenizer_type','hf','--max_seq_length',str(length),
                 '--model_template_type','base','--num_samples','10','--random_seed','42']
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            env.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PATH=str(PYTHON.parent)+os.pathsep+env['PATH'])
            with log.open('w') as out:
                subprocess.run(cmd,cwd=REPO/'.downloads/RULER/scripts/data',env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
            if len(target.read_text().splitlines())!=10:raise RuntimeError('Incomplete NIAH dataset')

def prepare_samples(sha,dump):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    samples=[];datasets={}
    for length in LENGTHS:
        for task_index,task in enumerate(TASKS):
            source=(REPO/f'.data/RULER/qwen3-50/65536/{task}/validation.jsonl' if length==65536
                    else ROOT/f'datasets/{length}/{task}/validation.jsonl')
            datasets[str(source)]=sha(source)
            snapshot=ROOT/f'datasets/{length}/{task}/validation.jsonl'
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes(source.read_bytes())
            datasets[str(snapshot)]=sha(snapshot)
            source=snapshot
            rows=[json.loads(line) for line in source.read_text().splitlines()]
            if len(rows)<10:raise ValueError('Too few source rows')
            for ordinal,row in enumerate(rows[:10]):
                text=tokenizer.apply_chat_template([{'role':'user','content':row['input']}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)+row.get('answer_prefix','')
                ids=tokenizer.encode(text,add_special_tokens=False)
                for size in CHUNKS:
                    chunks,tokens=cacheblend_prompt(ids,tokenizer,tokenizer.pad_token_id,size-1,256)
                    aligned_size=((size+63)//64)*64
                    if any(len(c)>aligned_size for c in chunks) or any(len(c)!=aligned_size for c in chunks[:-1]):
                        raise ValueError('Chunk size includes markers and padding')
                    bounds=[0]
                    for c in chunks:bounds.append(bounds[-1]+len(c))
                    bounds.append(len(tokens))
                    sid=f'{task}-{length}-{ordinal:03d}-c{size}'
                    sample=dict(id=sid,dataset='ruler',label=task,context_target=length,chunk_size=size,aligned_chunk_size=aligned_size,
                        source_row=ordinal,source_index=row.get('source_index',row['index']),source_path=str(source),
                        original_tokens=len(ids),tokens=len(tokens),token_ids=tokens,boundaries=bounds,
                        prompt_sha256=prompt_digest(tokens),query=question(tokenizer,task,row,text,tokens),
                        source_metadata=dict(task=task,references=row['outputs'],source_sha256=sha(source)),
                        physical_gpu=3+(task_index*10+ordinal)%2)
                    sample['required_cases']=(['baseline','prophetkv-20','prophetkv-30','prophetkv-40'] if task=='niah_single_1' else ['prophetkv-30','prophetkv-40'])
                    if task!='niah_single_1':
                        old_path=REPO/f'.results/prophetkv-qwen3-ruler30-20260918/samples/{sid}/input.json'
                        old=json.loads(old_path.read_text())
                        if tokens!=old['token_ids'] or sample['source_metadata']['references']!=old['source_metadata']['references']:
                            raise ValueError('Supplemental prompt differs from historical comparator')
                        sample['baseline_source_input']=str(old_path)
                        sample['baseline_source_input_sha256']=sha(old_path)
                    path=ROOT/'samples'/sid/'input.json';dump(path,sample)
                    samples.append({k:v for k,v in sample.items() if k!='token_ids'}|dict(input_path=str(path),input_sha256=sha(path)))
            print(f'frozen length={length} task={task}',flush=True)
    return samples,datasets

if __name__=='__main__':generate_sources()
