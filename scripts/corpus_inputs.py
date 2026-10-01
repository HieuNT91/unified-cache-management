"""Original task-batch preparation plus read-only historical corpus inventories."""
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
    sample=format_sample(tokenizer,row,task)
    sample.pop('references',None)
    sample.update(original_to_formatted=list(range(len(sample['token_ids']))),
        original_token_sha256=fingerprint(sample['token_ids']),
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
    raise ValueError('Per-sample seed generation is retired; use original task-batch prepare')


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
    output=Path(output)
    if (output/'preparation.json').exists():return original_rows(output)
    plan=json.loads((output/'plan.json').read_text())
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


def verify_prepared(output,rebuild=False,model=None,limit=500):
    output=Path(output)
    if not (output/'preparation.json').exists():
        raise ValueError('Legacy RULER preparation is incompatible; rebuild original task batches')
    rows=original_rows(output)
    if any(sum(r['subtask']==t and r['ordinal']<limit for r in rows)!=limit for t in TASKS):
        raise ValueError('Requested tranche is not fully prepared')
    if rebuild:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        spec=json.loads((output/'spec.json').read_text())
        batches={t:[json.loads(line) for line in (output/'raw'/t/'validation.jsonl').read_text().splitlines()] for t in TASKS}
        for row in rows:
            sample=json.loads((output/row['prepared']).read_text())
            rebuilt=format_sample(tokenizer,batches[row['subtask']][row['ordinal']],row['subtask'],template=spec['templates'][row['subtask']])
            rebuilt.update(source_row=row['ordinal'],model_config_sha256=spec['tokenizer_hashes']['config.json'],generator_config_sha256=fingerprint(spec))
            if rebuilt!=sample:raise ValueError('Original token/span rebuild mismatch')
    return rows


def prepare(ruler,model,output,workers=4,limit=500):
    from types import SimpleNamespace
    from scripts.ruler_64000 import prepare as prepare_original
    # Always generate the original 500-row task batch before selecting a tranche.
    prepare_original(SimpleNamespace(ruler=Path(ruler),model=Path(model),output=Path(output),samples=500,seed=42))
    return original_rows(Path(output))


def original_rows(output):
    from scripts.ruler_64000 import EVALUATION_PROTOCOL
    output=Path(output);receipt=json.loads((output/'preparation.json').read_text())
    if receipt['spec']['protocol'] != EVALUATION_PROTOCOL:
        raise ValueError('Incompatible RULER protocol')
    for name,digest in receipt['files'].items():
        if file_hash(output/name)!=digest:raise ValueError('Prepared original batch changed')
    rows=[]
    for row in map(json.loads,(output/'manifest.jsonl').read_text().splitlines()):
        sample=json.loads((output/row['prepared']).read_text())
        from runner.config import validate_sample
        validate_sample(sample)
        raw=f"raw/{row['subtask']}/validation.jsonl"
        rows.append(dict(row,ordinal=sample['source_row'],sha256=file_hash(output/row['prepared']),
            prompt_sha256=fingerprint(sample['token_ids']),raw=raw,raw_sha256=file_hash(output/raw),
            raw_row=sample['source_row'],identity=dict(task=row['subtask'],ordinal=sample['source_row'],seed=42)))
    plan=dict(schema='rpkv-original-ruler-plan-v1',protocol=EVALUATION_PROTOCOL,
        sources_sha256=file_hash(output/'spec.json'),entries=[r['identity'] for r in rows])
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=plan:
        raise ValueError('Incompatible collection plan; use a new directory')
    if not (output/'plan.json').exists():atomic_json(output/'plan.json',plan)
    return rows
