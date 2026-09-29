#!/usr/bin/env python3
"""CPU regeneration of the frozen 1300-row router cohort using official RULER."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runner.cache import cacheblend_prompt
from runner.config import validate_sample
from runner.setups import atomic_json, file_hash, setup_lock
COHORT = ROOT/'scripts/router_cohort.json'


def load_spec():
    spec=json.loads(COHORT.read_text())
    if spec['schema'] != 'clean-router-cohort-v1' or len(spec['entries']) != 1300:
        raise ValueError('Wrong frozen cohort')
    return spec


def verify_sources(ruler, model):
    from importlib.metadata import version
    spec=load_spec()
    for name,expected in spec['source_files'].items():
        if file_hash(Path(ruler)/name) != expected:
            raise ValueError('Official RULER asset differs from frozen training source: '+name)
    for name,expected in spec['model_fingerprints'].items():
        if file_hash(Path(model)/name) != expected:
            raise ValueError('Frozen model/tokenizer mismatch: '+name)
    for name,expected in spec['versions'].items():
        if version(name) != expected:
            raise ValueError(f'Frozen generator requires {name}=={expected}')
    return spec


def format_frozen(tokenizer, row, entry):
    """Rebuild token layout and its historical data envelope for hash comparison."""
    ident, expected = entry['identity'],entry['expected']
    task, ordinal = ident['task'],ident['ordinal']
    content = row['input']
    marker = '\nWhat ' if task.startswith('niah_') else '\nQuestion:'
    start = content.rfind(marker)
    if start < 0:
        raise ValueError('Missing complete RULER question')
    begin = start+1
    if task == 'fwe':
        begin = content.index('What are the three most frequently appeared',begin)
    question = content[begin:]
    if task.startswith('niah_'):
        question = question.split(' The special magic',1)[0]
    rendered = tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)+row.get('answer_prefix','')
    offset = rendered.index(content)+begin
    encoded = tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    original = encoded['input_ids']
    positions = [i for i,(a,b) in enumerate(encoded['offset_mapping']) if b > offset and a < offset+len(question)]
    if not positions:
        raise ValueError('Empty question span')
    suffix = max(256,len(original)-positions[0])
    chunks,tokens = cacheblend_prompt(original,tokenizer,tokenizer.pad_token_id,4095,suffix)
    bounds=[0]
    mapping=[]
    cursor=0
    for i,chunk in enumerate(chunks):
        count_marker=len(tokenizer.encode(f'\n[Context chunk {i+1}]\n',add_special_tokens=False))
        count=min(4095-count_marker,len(original)-suffix-cursor)
        mapping.extend(range(bounds[-1]+count_marker,bounds[-1]+count_marker+count))
        cursor+=count
        bounds.append(bounds[-1]+len(chunk))
    mapping.extend(range(bounds[-1],len(tokens)))
    bounds.append(len(tokens))
    query=dict(text=question,positions=[p+len(tokens)-len(original) for p in positions])
    if [tokens[p] for p in mapping] != original or tokens[-suffix:] != original[-suffix:]:
        raise ValueError('Full token reconstruction failed')
    # JSON insertion order is part of the frozen historical checksum contract.
    identity=dict(task=task,split='test',ordinal=ordinal,seeds=ident['seeds'])
    if 'source_index' in ident:
        identity.update(source_index=ident['source_index'],question_sha256=ident['question_sha256'])
        normalized=' '.join(unicodedata.normalize('NFKC',question.removeprefix('Question:').strip()).casefold().split())
        if hashlib.sha256(normalized.encode()).hexdigest() != ident['question_sha256']:
            raise ValueError('QA source question identity changed')
    frozen=dict(id=expected['id'],dataset='ruler',label=task,context_target=64000,source_row=ordinal,
        source_index=identity.get('source_index',row.get('index')),chunk_size=4096,
        original_tokens=len(original),tokens=len(tokens),token_ids=tokens,boundaries=bounds,
        fresh_suffix_tokens=suffix,thinking_enabled=False,source_metadata=dict(references=row['outputs']),query=query,
        split='test',seed=expected['seed'],generator_source=row,generation_attempt=expected['generation_attempt'],source_identity=identity)
    meta=dict(original_token_sha256=hashlib.sha256(json.dumps(original).encode()).hexdigest(),
        mapping_sha256=hashlib.sha256(json.dumps(mapping).encode()).hexdigest(),query=query,full_reconstruction=True)
    checks=dict(input_sha256=hashlib.sha256((json.dumps(frozen,indent=2)+'\n').encode()).hexdigest(),
        metadata_sha256=hashlib.sha256((json.dumps(meta,indent=2)+'\n').encode()).hexdigest(),
        prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),tokens=len(tokens))
    if any(checks[k] != expected[k] for k in checks):
        raise ValueError('Frozen input/token/span hash mismatch: '+expected['id']+' '+str(checks))
    sample=dict(token_ids=tokens,boundaries=bounds,question_positions=query['positions'],thinking=False,
                max_output_tokens=256,model_config_sha256=load_spec()['model_fingerprints']['config.json'])
    validate_sample(sample,4096,64000)
    return sample,dict(id=expected['id'],subtask=task,ordinal=ordinal,references=row['outputs'],
        scoring='ruler_any' if task.startswith('qa_') else 'ruler_all',frozen=checks,identity=identity)


_TOKENIZER=None
_WORK=None

def initialize(ruler,model,output):
    global _TOKENIZER,_WORK
    from transformers import AutoTokenizer
    _TOKENIZER=AutoTokenizer.from_pretrained(model,local_files_only=True)
    _WORK=(Path(ruler),Path(model),Path(output))


def regenerate(entry):
    ruler,model,output=_WORK
    spec=load_spec();ident=entry['identity'];task=ident['task'];definition=spec['definitions'][task]
    from scripts.ruler_64000 import definitions
    _,constants=definitions(ruler)
    base=constants[definition['task']]
    # The accepted attempt and its seed were frozen before training.
    attempt=entry['expected']['generation_attempt']
    folder=output/'generation'/entry['expected']['id']
    folder.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ruler/'scripts/data/synthetic'/f"{definition['task']}.py"),
        '--save_dir',str(folder),'--save_name',task,'--subset','validation','--tokenizer_path',str(model),
        '--tokenizer_type','hf','--max_seq_length',str(63000-attempt*512),
        '--tokens_to_generate',str(spec['output_reserve'][task]),'--num_samples','1',
        '--random_seed',str(ident['seeds'][attempt]),'--template',base['template']+base.get('answer_prefix','')]
    if 'source_index' in ident:
        cmd += ['--pre_samples',str(ident['source_index'])]
    for key,value in definition['args'].items():
        cmd += ['--'+key,str(value)]
    with (folder/'generator.log').open('w') as log:
        subprocess.run(cmd,check=True,timeout=600,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1'))
    rows=[json.loads(line) for line in (folder/task/'validation.jsonl').read_text().splitlines()]
    if len(rows) != 1:
        raise ValueError('Generator returned wrong number of samples')
    sample,record=format_frozen(_TOKENIZER,rows[0],entry)
    relative=f"samples/{record['id']}.json"
    atomic_json(output/relative,sample)
    record.update(prepared=relative,sha256=file_hash(output/relative))
    atomic_json(folder/'receipt.json',dict(command=cmd,entry=entry,record=record))
    print('Rebuilt '+record['id'],flush=True)
    return record


def verify_prepared(output):
    output=Path(output);receipt=json.loads((output/'prepared.json').read_text())
    if receipt['cohort_sha256'] != file_hash(COHORT) or receipt['manifest_sha256'] != file_hash(output/'manifest.jsonl'):
        raise ValueError('Prepared cohort identity changed')
    rows=json.loads((output/'cohort-index.json').read_text())
    if receipt['index_sha256'] != file_hash(output/'cohort-index.json'):
        raise ValueError('Prepared cohort index changed')
    spec=load_spec();expected={e['expected']['id']:e for e in spec['entries']}
    if len(rows) != 1300 or len({r['id'] for r in rows}) != 1300:
        raise ValueError('Incomplete/duplicate frozen inputs')
    for row in rows:
        e=expected[row['id']]
        if row['identity'] != e['identity'] or any(v != e['expected'][k] for k,v in row['frozen'].items()):
            raise ValueError('Prepared input identity mismatch')
        path=output/row['prepared']
        if file_hash(path) != row['sha256']:
            raise ValueError('Prepared input changed: '+row['id'])
        sample=json.loads(path.read_text());validate_sample(sample,4096,64000)
        if hashlib.sha256(json.dumps(sample['token_ids']).encode()).hexdigest() != e['expected']['prompt_sha256']:
            raise ValueError('Prepared tokens changed')
    return rows


def prepare(ruler,model,output,workers=8):
    spec=verify_sources(ruler,model)
    with setup_lock(output,build=True):
        if (output/'prepared.json').exists():
            return verify_prepared(output)
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(ruler),str(model),str(output))) as pool:
            rows=list(pool.map(regenerate,spec['entries']))
        # Task-major order retains ordinal parity, independent of generator scheduling.
        rows.sort(key=lambda r:(r['subtask'],r['ordinal']))
        atomic_json(output/'cohort-index.json',rows)
        (output/'manifest.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ('id','prepared','subtask','references','scoring')})+'\n' for r in rows))
        atomic_json(output/'prepared.json',dict(cohort_sha256=file_hash(COHORT),
            manifest_sha256=file_hash(output/'manifest.jsonl'),index_sha256=file_hash(output/'cohort-index.json'),prompts=1300,
            tasks=dict(Counter(r['subtask'] for r in rows)),full_token_span_reconstruction=True))
    return verify_prepared(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ruler',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):
        raise ValueError('Preparation must be CPU only')
    prepare(a.ruler.resolve(),a.model.resolve(),a.output.resolve(),a.workers)
