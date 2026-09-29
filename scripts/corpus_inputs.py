"""New, unsplit 13 x 200 RULER cohort; CPU only, pinned official generators."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from runner.setups import atomic_json, file_hash, fingerprint, setup_lock
from scripts.ruler_64000 import format_sample, definitions, query_span

SOURCES = Path(__file__).with_name('corpus_sources.json')
TASKS = tuple(json.loads(SOURCES.read_text())['definitions'])
SAMPLES = 200
SEED_DOMAIN = 'ruler-attention-outcomes-unsplit-v1-20260929'


def spec():
    return json.loads(SOURCES.read_text())


def question_hash(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',text).casefold().split()).encode()).hexdigest()


def qa_inventory(ruler):
    root=Path(ruler)/'scripts/data/synthetic/json'
    squad=json.loads((root/'squad.json').read_text())
    hotpot=json.loads((root/'hotpotqa.json').read_text())
    questions={'qa_1':[q['question'] for d in squad['data'] for p in d['paragraphs']
                       for q in p['qas'] if not q['is_impossible']],
               'qa_2':[q['question'] for q in hotpot]}
    seen=set();result={}
    for task,rows in questions.items():
        # Hash order avoids relying on global RNG state and freezes source identities.
        order=sorted(range(len(rows)),key=lambda i:fingerprint([SEED_DOMAIN,task,i]))
        selected=[]
        for index in order:
            digest=question_hash(rows[index])
            if digest in seen:continue
            seen.add(digest);selected.append(dict(source_index=index,question_sha256=digest,question=rows[index]))
            if len(selected)==SAMPLES:break
        if len(selected)!=SAMPLES:raise ValueError('Insufficient unique QA questions')
        result[task]=selected
    return result


def inventory(qa):
    return [dict(id=f'{task}-{i:03d}',task=task,ordinal=i,
                 seeds=[int(fingerprint([SEED_DOMAIN,task,i,a])[:8],16) for a in range(8)],
                 **(qa[task][i] if task.startswith('qa_') else {}))
            for i in range(SAMPLES) for task in TASKS]


def verify_sources(ruler,model):
    from importlib.metadata import version
    data=spec()
    for base,key in ((ruler,'source_files'),(model,'model_fingerprints')):
        for name,digest in data[key].items():
            if file_hash(Path(base)/name)!=digest:raise ValueError('Pinned source/model mismatch: '+name)
    for name,wanted in data['versions'].items():
        if version(name)!=wanted:raise ValueError(f'Preparation requires {name}=={wanted}')
    return data


def format_corpus(tokenizer,row,task):
    sample=format_sample(tokenizer,row,task,thinking=False,output=256)
    content=row['input']
    rendered=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,
                                            add_generation_prompt=True,enable_thinking=False)
    original=tokenizer.encode(rendered,add_special_tokens=False)
    pad=sample['padding'];at=pad['original_insertion_position'];count=pad['count']
    expanded=original[:at]+[pad['token_id']]*count+original[at:]
    suffix=sample['boundaries'][-1]-sample['boundaries'][-2]
    mapping=[];cursor=0
    for i,(a,b) in enumerate(zip(sample['boundaries'][:-2],sample['boundaries'][1:-1])):
        marker=len(tokenizer.encode(f'\n[Context chunk {i+1}]\n',add_special_tokens=False))
        length=min(4095-marker,len(expanded)-suffix-cursor)
        mapping.extend(range(a+marker,a+marker+length));cursor+=length
    mapping.extend(range(sample['boundaries'][-2],len(sample['token_ids'])))
    if [sample['token_ids'][p] for p in mapping]!=expanded:raise ValueError('Full token reconstruction failed')
    original_mapping=mapping[:at]+mapping[at+count:]
    if [sample['token_ids'][p] for p in original_mapping]!=original:raise ValueError('Source token reconstruction failed')
    sample.update(original_to_formatted=original_mapping,original_token_sha256=fingerprint(original),
                  prompt_sha256=fingerprint(sample['token_ids']),full_reconstruction=True)
    return sample


_TOKENIZER=None
_WORK=None

def initialize(ruler,model,output):
    global _TOKENIZER,_WORK
    from transformers import AutoTokenizer
    _TOKENIZER=AutoTokenizer.from_pretrained(model,local_files_only=True)
    _WORK=Path(ruler),Path(model),Path(output)


def generate(entry):
    ruler,model,output=_WORK;task=entry['task'];data=spec()
    folder=output/'generation'/entry['id'];receipt=folder/'receipt.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['identity']!=entry:raise ValueError('Generation identity changed')
        for name,digest in saved['files'].items():
            if file_hash(output/name)!=digest:raise ValueError('Accepted preparation changed')
        return saved['row']
    definition=data['definitions'][task];_,constants=definitions(ruler);base=constants[definition['task']]
    folder.mkdir(parents=True,exist_ok=True)
    for attempt,seed in enumerate(entry['seeds']):
        attempt_dir=folder/f'attempt-{attempt}'
        attempt_dir.mkdir(exist_ok=True)
        command=[sys.executable,str(ruler/'scripts/data/synthetic'/f"{definition['task']}.py"),
            '--save_dir',str(attempt_dir),'--save_name',task,'--subset','validation',
            '--tokenizer_path',str(model),'--tokenizer_type','hf','--max_seq_length',str(63000-attempt*512),
            '--tokens_to_generate',str(data['output_reserve'][task]),'--num_samples','1',
            '--random_seed',str(seed),'--template',base['template']+base.get('answer_prefix','')]
        if 'source_index' in entry:command+=['--pre_samples',str(entry['source_index'])]
        for key,value in definition['args'].items():command+=['--'+key,str(value)]
        with (attempt_dir/'generator.log').open('w') as log:
            subprocess.run(command,check=True,timeout=900,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0',HF_HUB_OFFLINE='1',
                         OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
        raw_path=attempt_dir/task/'validation.jsonl'
        raw=[json.loads(line) for line in raw_path.read_text().splitlines()]
        if len(raw)!=1 or not raw[0].get('outputs'):raise ValueError('Invalid official generator output')
        row=raw[0]
        if task.startswith('qa_'):
            question=query_span(row['input'],task)[1].removeprefix('Question:').strip()
            if question_hash(question)!=entry['question_sha256']:raise ValueError('QA identity mismatch')
        try:sample=format_corpus(_TOKENIZER,row,task)
        except ValueError as error:
            if 'exceeds exact input target' not in str(error):raise
            continue
        # Independent reconstruction from source before freezing any accepted input.
        if format_corpus(_TOKENIZER,json.loads(json.dumps(row)),task)!=sample:
            raise ValueError('Non-deterministic formatter')
        sample['model_config_sha256']=data['model_fingerprints']['config.json']
        relative=f"samples/{entry['id']}.json";atomic_json(output/relative,sample)
        record=dict(id=entry['id'],ordinal=entry['ordinal'],subtask=task,prepared=relative,
            sha256=file_hash(output/relative),prompt_sha256=sample['prompt_sha256'],identity=entry,
            references=row['outputs'],scoring='ruler_any' if task.startswith('qa_') else 'ruler_all',
            raw=str(raw_path.relative_to(output)),raw_sha256=file_hash(raw_path),attempt=attempt,seed=seed)
        atomic_json(receipt,dict(identity=entry,command=command,row=record,
                    files={relative:record['sha256'],record['raw']:record['raw_sha256']}))
        print('Prepared '+entry['id'],flush=True)
        return record
    raise ValueError('All deterministic generation attempts exceeded 64000 tokens')


def validate_inventory(rows):
    if (len(rows)!=2600 or len({r['id'] for r in rows})!=2600 or
            Counter(r['subtask'] for r in rows)!=Counter({t:200 for t in TASKS}) or
            any({r['ordinal'] for r in rows if r['subtask']==t}!=set(range(200)) for t in TASKS)):
        raise ValueError('Expected unsplit 13 x 200 inventory')
    if len({r['prompt_sha256'] for r in rows})!=2600:raise ValueError('Duplicate prompts')
    qa=[r['identity']['question_sha256'] for r in rows if r['subtask'].startswith('qa_')]
    if len(set(qa))!=400:raise ValueError('Duplicate QA question identity')


def prepared_rows(output):
    """Read immutable per-sample publications, including during preparation."""
    output=Path(output);plan=json.loads((output/'plan.json').read_text())
    if plan['sources_sha256']!=file_hash(SOURCES):raise ValueError('Source pins changed')
    if len(plan['entries'])!=2600 or len({e['id'] for e in plan['entries']})!=2600:
        raise ValueError('Invalid frozen generation plan')
    rows=[];hashes=set();questions=set()
    for entry in plan['entries']:
        receipt=output/'generation'/entry['id']/'receipt.json'
        if not receipt.exists():continue
        saved=json.loads(receipt.read_text());row=saved['row']
        if saved['identity']!=entry or row['identity']!=entry or row['id']!=entry['id'] or row['ordinal']!=entry['ordinal'] or row['subtask']!=entry['task']:
            raise ValueError('Preparation identity changed')
        if saved['files'].get(row['prepared'])!=row['sha256'] or saved['files'].get(row['raw'])!=row['raw_sha256']:
            raise ValueError('Invalid preparation receipt')
        if row['prompt_sha256'] in hashes:raise ValueError('Duplicate prompts')
        hashes.add(row['prompt_sha256'])
        if row['subtask'].startswith('qa_'):
            q=row['identity']['question_sha256']
            if q in questions:raise ValueError('Duplicate QA question')
            questions.add(q)
        rows.append(row)
    return rows


def verify_prepared(output,rebuild=False,model=None,limit=200):
    from runner.router_measure import validate_profile
    from runner.corpus import relative
    output=Path(output);rows=prepared_rows(output)
    if rebuild:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    for row in rows:
        for name,expected in ((row['prepared'],row['sha256']),(row['raw'],row['raw_sha256'])):
            if file_hash(relative(output,name))!=expected:raise ValueError('Prepared artifact changed: '+name)
        sample=json.loads(relative(output,row['prepared']).read_text());validate_profile(sample,4)
        if len(sample['token_ids'])!=64000 or fingerprint(sample['token_ids'])!=row['prompt_sha256']:
            raise ValueError('Input length or token hash mismatch')
        if any(k in sample for k in ('outputs','references','answer','gold')):raise ValueError('Reference leaked into inference')
        original=[sample['token_ids'][p] for p in sample['original_to_formatted']]
        if fingerprint(original)!=sample['original_token_sha256']:raise ValueError('Source reconstruction mismatch')
        if rebuild:
            raw=json.loads(relative(output,row['raw']).read_text())
            rebuilt=format_corpus(tokenizer,raw,row['subtask'])
            rebuilt['model_config_sha256']=spec()['model_fingerprints']['config.json']
            if rebuilt!=sample:raise ValueError('Independent source/span rebuild mismatch')
    selected=[r for r in rows if r['ordinal']<limit]
    if Counter(r['subtask'] for r in selected)!=Counter({t:limit for t in TASKS}):
        raise ValueError('Requested tranche is not fully prepared')
    if len(rows)==2600:validate_inventory(rows)
    return rows


def prepare(ruler,model,output,workers=4,limit=200):
    if type(limit) is not int or not 1<=limit<=200:raise ValueError('limit-per-task must be 1..200')
    ruler,model,output=map(lambda p:Path(p).resolve(),(ruler,model,output))
    verify_sources(ruler,model)
    with setup_lock(output,build=True):
        entries=inventory(qa_inventory(ruler))
        plan=dict(schema='ruler-corpus-plan-v1',sources_sha256=file_hash(SOURCES),entries=entries,
                  formatter_sha256=file_hash(Path(__file__)),base_formatter_sha256=file_hash(Path(__file__).with_name('ruler_64000.py')))
        if (output/'plan.json').exists():
            if json.loads((output/'plan.json').read_text())!=plan:raise ValueError('Preparation identity changed; use a new directory')
        else:atomic_json(output/'plan.json',plan)
        selected=[e for e in entries if e['ordinal']<limit]
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(ruler,model,output)) as pool:
            list(pool.map(generate,selected))
        rows=prepared_rows(output)
        temp=output/'manifest.jsonl.tmp';temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(output/'manifest.jsonl')
        atomic_json(output/'preparation-progress.json',dict(planned=2600,prepared=len(rows),limit_per_task=limit,complete=len(rows)==2600))
        if len(rows)==2600:
            validate_inventory(rows)
            paths=['plan.json','manifest.jsonl']+[name for r in rows for name in (r['prepared'],r['raw'])]
            if not (output/'prepared.json').exists():
                atomic_json(output/'prepared.json',dict(schema='ruler-corpus-prepared-v1',sources_sha256=file_hash(SOURCES),
                    files={name:file_hash(output/name) for name in paths},prompts=2600))
    return verify_prepared(output,True,model,limit)
