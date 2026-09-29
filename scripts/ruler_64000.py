#!/usr/bin/env python3
"""Official RULER assets, CPU generation, and exact 64000-token thinking inputs."""
import argparse
from collections import Counter
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.cache import cacheblend_prompt
from runner.config import validate_sample
from runner.setups import atomic_json, file_hash, fingerprint, setup_lock

REVISION = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
TASKS = ('cwe','fwe','vt','qa_1','qa_2','niah_multivalue','niah_multikey_2','niah_multikey_3')
PERCENTAGES = (1,5,10,15,20,30,40,60,80)
INPUT = 64000
OUTPUT = 16384
GENERATOR_LIMIT = 63360  # Leave room for chat/chunk markers; never truncate source input.


def source_check(root):
    head = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head != REVISION:
        raise ValueError(f'RULER must be pinned to {REVISION}; use a dedicated source directory')
    wordlist = 'scripts/data/synthetic/json/english_words.json'
    changed = subprocess.check_output(['git','-C',str(root),'diff','--name-only','HEAD'],text=True).splitlines()
    if set(changed)-{wordlist}:
        raise ValueError('Tracked official RULER sources were modified')
    pointer = subprocess.check_output(['git','-C',str(root),'show',f'HEAD:{wordlist}'],text=True)
    expected = next(line.split('sha256:')[1] for line in pointer.splitlines() if line.startswith('oid sha256:'))
    path = root/wordlist
    if path.read_bytes() != pointer.encode() and file_hash(path) != expected:
        raise ValueError('CWE word list does not match its official Git LFS digest')


def valid_asset(name, data):
    if name == 'english_words.json':
        return isinstance(data,dict) and len(data)>8000 and all(isinstance(v,str) for v in data.values())
    if name == 'squad.json':
        return isinstance(data,dict) and bool(data.get('data'))
    if name == 'hotpotqa.json':
        return isinstance(data,list) and bool(data) and all(
            {'_id','question','answer','context'} <= row.keys() for row in data)
    return isinstance(data,dict) and isinstance(data.get('text'),str) and len(data['text']) > 10000


def fetch(url, destination):
    with urllib.request.urlopen(url,timeout=60) as source, destination.open('wb') as target:
        shutil.copyfileobj(source,target)


def download_json(path, urls):
    if path.exists():
        if not valid_asset(path.name,json.loads(path.read_text())):
            raise ValueError(f'Invalid existing asset: {path}')
        return 'existing local file'
    errors = []
    for url in urls:
        temporary = path.with_suffix('.download')
        try:
            fetch(url, temporary)
            if not valid_asset(path.name,json.loads(temporary.read_text())):
                raise ValueError('Unexpected dataset schema')
            temporary.replace(path)
            return url
        except Exception as error:
            errors.append(f'{url}: {error}')
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError('\n'.join(errors))


def assets(args):
    args.ruler = args.ruler.resolve()
    with setup_lock(args.ruler.parent/'ruler-assets-lock',build=True):
        if not args.ruler.exists():
            subprocess.run(['git','clone','https://github.com/NVIDIA/RULER.git',str(args.ruler)],check=True,
                           env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'))
            subprocess.run(['git','-C',str(args.ruler),'checkout','--detach',REVISION],check=True)
        source_check(args.ruler)
        root = args.ruler/'scripts/data/synthetic/json'
        sources = {}
        wordlist = root/'english_words.json'
        if wordlist.stat().st_size < 1000:
            pointer = wordlist.read_text()
            expected = next(line.split('sha256:')[1] for line in pointer.splitlines() if line.startswith('oid sha256:'))
            url = f'https://media.githubusercontent.com/media/NVIDIA/RULER/{REVISION}/scripts/data/synthetic/json/english_words.json'
            temp = wordlist.with_suffix('.download')
            try:
                fetch(url,temp)
                if file_hash(temp) != expected or not valid_asset(wordlist.name,json.loads(temp.read_text())):
                    raise ValueError('Downloaded CWE word list does not match its official LFS digest')
                temp.replace(wordlist)
            finally:
                temp.unlink(missing_ok=True)
            sources[wordlist.name] = url
        else:
            sources[wordlist.name] = 'existing local file; official LFS digest verified'
        sources['squad.json'] = download_json(root/'squad.json',[
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'])
        sources['hotpotqa.json'] = download_json(root/'hotpotqa.json',[
            'https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
            'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
            # This pinned fallback is provided by NVIDIA's own download script.
            'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json'])
        essay = root/'PaulGrahamEssays.json'
        if not essay.exists():
            with (root/'download-essays.log').open('w') as log:
                subprocess.run([sys.executable,str(root/'download_paulgraham_essay.py')],
                    cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not valid_asset(essay.name,json.loads(essay.read_text())):
            raise ValueError('Essay download failed; inspect download-essays.log')
        sources[essay.name] = 'NVIDIA download_paulgraham_essay.py; see download-essays.log for any skipped URLs'
        import nltk
        for resource, location in [('punkt','tokenizers/punkt'),('punkt_tab','tokenizers/punkt_tab')]:
            try:
                nltk.data.find(location)
            except LookupError:
                if not nltk.download(resource,raise_on_error=True):
                    raise RuntimeError(f'Failed to download NLTK {resource}')
        atomic_json(args.ruler/'asset-downloads.json',dict(revision=REVISION,sources=sources,
                    hashes={name:file_hash(root/name) for name in sources}))
        print('RULER sources and assets ready. No Python packages installed; no GPU use.',flush=True)


def query_span(content, task):
    marker = '\nWhat ' if task.startswith('niah_') else '\nQuestion:'
    start = content.rfind(marker)
    if start < 0:
        raise ValueError(f'Cannot locate complete final question for {task}')
    return start+1, content[start+1:]


def format_sample(tokenizer, row, task, target=INPUT, thinking=True, output=OUTPUT):
    content = row['input']
    begin, question = query_span(content,task)
    rendered = tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,
                add_generation_prompt=True,enable_thinking=thinking)
    offset = rendered.index(content)
    encoded = tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'],encoded['offset_mapping']
    positions = [i for i,(a,b) in enumerate(offsets) if b > offset+begin and a < offset+len(content)]
    if not positions:
        raise ValueError('Question token span is empty')
    # Round the fresh suffix to a block so exact formatted lengths are attainable.
    suffix = ((max(256,len(ids)-positions[0])+63)//64)*64
    insertion = next(i for i,(a,b) in enumerate(offsets) if b > offset)
    if offsets[insertion][0] != offset or insertion >= len(ids)-suffix:
        raise ValueError('Cannot pad safely before the user text')
    newline = tokenizer.encode('\n',add_special_tokens=False)
    marker = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    if len(newline) != 1 or marker is None or newline[0] == marker:
        raise ValueError('Expected Qwen3 newline token and cache delimiter')
    def layout(count):
        expanded = ids[:insertion]+newline*count+ids[insertion:]
        chunks,tokens = cacheblend_prompt(expanded,tokenizer,marker,4095,suffix)
        return chunks,tokens
    chunks,tokens = layout(0)
    if len(tokens) > target:
        raise ValueError('Generated prompt exceeds exact input target; source truncation is forbidden')
    low,high = 0,max(0,target-len(ids))
    while low < high:
        mid = (low+high)//2
        if len(layout(mid)[1]) < target:
            low = mid+1
        else:
            high = mid
    chunks,tokens = layout(low)
    if len(tokens) != target:
        raise ValueError('Cannot attain exact formatted target without truncating source input')
    bounds = [0]
    for chunk in chunks:
        bounds.append(bounds[-1]+len(chunk))
    bounds.append(len(tokens))
    shift = len(tokens)-len(ids)
    query = [p+shift for p in positions]
    if tokens[-suffix:] != ids[-suffix:] or [tokens[p] for p in query] != [ids[p] for p in positions]:
        raise RuntimeError('Question or fresh suffix changed')
    sample = dict(token_ids=tokens,boundaries=bounds,question_positions=query,
        thinking=thinking,max_output_tokens=output,task=task,question=question,
        original_chat_tokens=len(ids),source_index=row['index'],source_row_sha256=fingerprint(row),
        padding=dict(kind='newline tokens before user text',token_id=newline[0],count=low,
                     original_insertion_position=insertion),truncated=False,
        answer_prefix_policy=('native thinking template; no forced assistant answer prefix' if thinking else
                              'native non-thinking template; no forced assistant answer prefix'))
    validate_sample(sample,4096,target)
    return sample


def definitions(root):
    import yaml
    path = root/'scripts/data/synthetic/constants.py'
    spec = importlib.util.spec_from_file_location('official_ruler_constants',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return yaml.safe_load((root/'scripts/synthetic.yaml').read_text()),module.TASKS


def generator_command(args, task, target):
    customized,constants = definitions(args.ruler)
    definition = customized[task]
    base = constants[definition['task']]
    command = [sys.executable,str(args.ruler/'scripts/data/synthetic'/f"{definition['task']}.py"),
        '--save_dir',str(target),'--save_name',task,'--subset','validation',
        '--tokenizer_path',str(args.model),'--tokenizer_type','hf',
        '--max_seq_length',str(GENERATOR_LIMIT),'--tokens_to_generate','128',
        '--num_samples',str(args.samples),'--random_seed','42',
        '--template',base['template']+base.get('answer_prefix','')]
    for key,value in definition['args'].items():
        command += ['--'+key,str(value)]
    return command


def prepare(args):
    from run import check_model
    from transformers import AutoTokenizer
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU preparation requires CUDA_VISIBLE_DEVICES=''")
    args.ruler,args.model,args.output = args.ruler.resolve(),args.model.resolve(),args.output.resolve()
    source_check(args.ruler);check_model(args.model)
    if not 1 <= args.samples <= 100:
        raise ValueError('Use 1..100 samples per task (full experiment: 100)')
    assets_root = args.ruler/'scripts/data/synthetic/json'
    asset_hashes = {name:file_hash(assets_root/name) for name in
        ('squad.json','hotpotqa.json','PaulGrahamEssays.json','english_words.json')}
    tokenizer_hashes = {p.name:file_hash(p) for p in args.model.iterdir() if p.is_file() and
                       (p.name.startswith('tokenizer') or p.name in ('vocab.json','merges.txt','config.json'))}
    from importlib.metadata import version
    versions = {name:version(name) for name in ('transformers','numpy','wonderwords','nltk','scipy','PyYAML')}
    spec = dict(revision=REVISION,tasks=TASKS,samples=args.samples,input_tokens=INPUT,output_tokens=OUTPUT,
        raw_generator_limit=GENERATOR_LIMIT,raw_generator_output_reserve=128,seed=42,
        tokenizer_hashes=tokenizer_hashes,assets=asset_hashes,versions=versions,
        adapter=file_hash(Path(__file__)),chunker=file_hash(ROOT/'runner/cache.py'))
    # JSON roundtrip gives consistent tuple/list comparison after a restart.
    spec = json.loads(json.dumps(spec))
    with setup_lock(args.output,build=True):
        spec_path = args.output/'spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError('Preparation settings changed; use a new PREPARED_DIR')
        atomic_json(spec_path,spec)
        complete = args.output/'preparation.json'
        if complete.exists():
            receipt = json.loads(complete.read_text())
            for name,digest in receipt['files'].items():
                if file_hash(args.output/name) != digest:
                    raise ValueError(f'Prepared artifact changed: {name}')
            print(json.dumps(receipt['summary'],indent=2));return
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        rows,files = {},{}
        for task in TASKS:
            raw = args.output/'raw'/task/'validation.jsonl'
            raw_receipt = raw.with_suffix('.receipt.json')
            command = generator_command(args,task,args.output/'raw')
            if raw_receipt.exists():
                if json.loads(raw_receipt.read_text())['sha256'] != file_hash(raw):
                    raise ValueError(f'Raw source changed: {raw}')
            else:
                raw.parent.mkdir(parents=True,exist_ok=True)
                with (raw.parent/'generation.log').open('w') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0',
                                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
                atomic_json(raw_receipt,dict(command=command,sha256=file_hash(raw)))
            rows[task] = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
            if len(rows[task]) != args.samples:
                raise ValueError(f'Expected {args.samples} source rows for {task}')
            # Upstream NIAH stores answer character offset in `index`, not row ID.
            if not task.startswith('niah_') and [r['index'] for r in rows[task]] != list(range(args.samples)):
                raise ValueError(f'Unexpected source row order for {task}')
            if any(not r.get('outputs') or not all(isinstance(v,str) and v for v in r['outputs']) for r in rows[task]):
                raise ValueError(f'Missing references in {task}')
            for path in (raw,raw_receipt):files[str(path.relative_to(args.output))]=file_hash(path)
        manifest,padding_counts = [],[]
        # Task-major order gives each even/odd shard 50 rows of EVERY task.
        for task in TASKS:
            for ordinal,row in enumerate(rows[task]):
                sample = format_sample(tokenizer,row,task)
                sample['source_row'] = ordinal
                sample['model_config_sha256'] = tokenizer_hashes['config.json']
                relative = f'samples/{task}-{ordinal:03d}.json'
                atomic_json(args.output/relative,sample)
                files[relative] = file_hash(args.output/relative)
                manifest.append(dict(id=f'{task}-{ordinal:03d}',prepared=relative,subtask=task,
                    references=row['outputs'],scoring='ruler_any' if task.startswith('qa_') else 'ruler_all'))
                padding_counts.append(sample['padding']['count'])
                print(f'Prepared {len(manifest)}/{len(TASKS)*args.samples}: {task}/{ordinal}, '
                      f'64000 tokens, {padding_counts[-1]} newline padding tokens',flush=True)
        path=args.output/'manifest.jsonl'
        tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(row)+'\n' for row in manifest));tmp.replace(path)
        files['manifest.jsonl']=file_hash(path)
        summary=dict(prompts=len(manifest),subtasks=dict(Counter(r['subtask'] for r in manifest)),
            measurements=len(manifest)*19,input_tokens=INPUT,max_output_tokens=OUTPUT,thinking=True,
            min_padding_tokens=min(padding_counts),max_padding_tokens=max(padding_counts),
            padding_policy='prepend newline tokens inside user message; source tokens untruncated',
            maximum_temporary_kv_GiB_two_groups=2*(INPUT-256)*262144/2**30)
        atomic_json(complete,dict(spec=spec,files=files,summary=summary))
        print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    download=commands.add_parser('download')
    download.add_argument('--ruler',type=Path,required=True)
    download.set_defaults(func=assets)
    prep=commands.add_parser('prepare')
    prep.add_argument('--ruler',type=Path,required=True)
    prep.add_argument('--model',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--samples',type=int,default=100)
    prep.set_defaults(func=prepare)
    args=parser.parse_args();args.func(args)

if __name__=='__main__':main()
