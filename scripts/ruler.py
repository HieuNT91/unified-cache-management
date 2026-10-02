#!/usr/bin/env python3
"""Official RULER assets, CPU generation, and original 64K protocol with native Qwen3 non-thinking adaptation."""
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
from runner.config import validate_sample
from runner.setups import atomic_json, file_hash, fingerprint, setup_lock

REVISION = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
TASKS = ('niah_single_1','niah_single_2','niah_single_3','niah_multikey_1',
         'niah_multikey_2','niah_multikey_3','niah_multivalue','niah_multiquery',
         'vt','cwe','fwe','qa_1','qa_2')
PERCENTAGES = (1,5,10,15,20,30,40,60,80)
INPUT = 65536  # Entire formatted input plus the task's generation reserve.
EVALUATION_PROTOCOL = 'nvidia-ruler-c3f5e3b-qwen3-nonthinking-v1'
CAPS = {task: 128 if task.startswith('niah_') else
        32 if task.startswith('qa_') else {'vt':30,'cwe':120,'fwe':50}[task] for task in TASKS}


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
        raise ValueError('Cannot locate original final question')
    return start+1, content[start+1:]


def definitions(root):
    import yaml
    path = root/'scripts/data/synthetic/constants.py'
    spec = importlib.util.spec_from_file_location('official_ruler_constants',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return yaml.safe_load((root/'scripts/synthetic.yaml').read_text()),module.TASKS


def model_template(tokenizer, base):
    """Upstream generator formats this whole template before counting tokens."""
    return tokenizer.apply_chat_template([dict(role='user',content=base['template'])],
        tokenize=False,add_generation_prompt=True,enable_thinking=False) + base['answer_prefix']


def format_sample(tokenizer, row, task, *, template=None):
    from runner.layout import boundaries_for, stamp_sample
    if task not in TASKS or not isinstance(row.get('answer_prefix'), str) or not row['answer_prefix']:
        raise ValueError('Expected original RULER row with a separate answer_prefix')
    # Upstream splits its fully formatted string into input + answer_prefix.
    # Rejoin once, without another chat template or any text normalization.
    text = row['input'] + row['answer_prefix']
    sentinel = 'RPKV_TEMPLATE_CONTENT_9fd17'
    rendered = tokenizer.apply_chat_template([dict(role='user',content=sentinel)],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    head, tail = rendered.split(sentinel)
    if not row['input'].startswith(head) or not row['input'].endswith(tail):
        raise ValueError('Row must be generated with the complete native Qwen3 template')
    end = len(row['input']) - len(tail)
    marker = '\nWhat ' if task.startswith('niah_') else '\nQuestion:'
    begin = row['input'].rfind(marker, 0, end)
    if begin < 0:
        raise ValueError('Cannot locate original final question')
    begin += 1
    encoded = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids = encoded['input_ids']
    positions = [i for i,(a,b) in enumerate(encoded['offset_mapping']) if b > begin and a < end]
    if len(ids) + CAPS[task] > INPUT:
        raise ValueError(f'{task}: original row exceeds 65536 including reserve; retain raw and halt')
    sample = dict(token_ids=ids,boundaries=boundaries_for(len(ids),positions[0]),
        question_positions=positions,thinking=False,max_output_tokens=CAPS[task],
        task=task,question=text[begin:end],formatted_text=text,answer_prefix=row['answer_prefix'],
        source_index=row['index'],source_row_sha256=fingerprint(row),
        references=row['outputs'],scoring='ruler_any' if task.startswith('qa_') else 'ruler_all',
        evaluation_protocol=EVALUATION_PROTOCOL,original_chat_tokens=len(ids),truncated=False,
        answer_prefix_policy='original prefix once in open assistant; input/prefill',
        generator_revision=REVISION,max_seq_length=INPUT)
    stamp_sample(sample,template if template is not None else tokenizer.chat_template)
    validate_sample(sample,4096,INPUT)
    return sample


def generator_command(args, task, target, tokenizer=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    customized, constants = definitions(args.ruler)
    definition = customized[task]; base = constants[definition['task']]
    if base['tokens_to_generate'] != CAPS[task]:
        raise ValueError('Original task output reserve changed')
    command = [sys.executable,str(args.ruler/'scripts/data/synthetic'/f"{definition['task']}.py"),
        '--save_dir',str(target),'--save_name',task,'--subset','validation',
        '--tokenizer_path',str(args.model),'--tokenizer_type','hf',
        '--max_seq_length',str(INPUT),'--tokens_to_generate',str(CAPS[task]),
        '--num_samples',str(args.samples),'--random_seed',str(getattr(args,'seed',42)),
        '--template',model_template(tokenizer,base)]
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
    if args.samples < 1:
        raise ValueError('samples must be positive; original default is 500/task')
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    custom, constants = definitions(args.ruler)
    templates = {t:model_template(tokenizer,constants[custom[t]['task']]) for t in TASKS}
    asset_root = args.ruler/'scripts/data/synthetic/json'
    from importlib.metadata import version
    spec = dict(protocol=EVALUATION_PROTOCOL,revision=REVISION,tasks=list(TASKS),samples=args.samples,
        max_seq_length=INPUT,output_caps=CAPS,seed=getattr(args,'seed',42),
        task_parameters=custom,templates=templates,
        tokenizer_hashes={p.name:file_hash(p) for p in args.model.iterdir() if p.is_file() and
            p.suffix in ('.json','.jinja','.txt','.model','.tiktoken')},
        source_hashes={str(p.relative_to(args.ruler)):file_hash(p) for p in (args.ruler/'scripts').rglob('*')
            if p.is_file() and p.suffix in ('.py','.yaml','.sh')},
        assets={name:file_hash(asset_root/name) for name in
            ('squad.json','hotpotqa.json','PaulGrahamEssays.json','english_words.json')},
        versions={n:version(n) for n in ('transformers','numpy','wonderwords','nltk','scipy','PyYAML')},
        adapter=file_hash(Path(__file__)),chunker=file_hash(ROOT/'runner/layout.py'))
    spec = json.loads(json.dumps(spec))
    with setup_lock(args.output,build=True):
        sp = args.output/'spec.json'
        if sp.exists() and json.loads(sp.read_text()) != spec:
            raise ValueError('Incompatible preparation; use a new output directory')
        atomic_json(sp,spec)
        complete = args.output/'preparation.json'
        if complete.exists():
            receipt=json.loads(complete.read_text())
            for name,digest in receipt['files'].items():
                if file_hash(args.output/name) != digest:
                    raise ValueError(f'Prepared artifact changed: {name}')
            print(json.dumps(receipt['summary'],indent=2));return
        files,manifest,lengths = {},[],[]
        for task in TASKS:
            raw=args.output/'raw'/task/'validation.jsonl'
            receipt=raw.with_suffix('.receipt.json')
            command=generator_command(args,task,args.output/'raw',tokenizer)
            if receipt.exists():
                if json.loads(receipt.read_text()) != dict(command=command,sha256=file_hash(raw)):
                    raise ValueError('Raw batch identity changed')
            else:
                if raw.exists():
                    raise ValueError(f'Unreceipted raw batch retained at {raw}; inspect before using a new directory')
                raw.parent.mkdir(parents=True,exist_ok=True)
                with (raw.parent/'generation.log').open('w') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0',
                                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
                atomic_json(receipt,dict(command=command,sha256=file_hash(raw)))
            rows=[json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
            if len(rows) != args.samples:
                raise ValueError('Original task batch has an unexpected row count')
            for path in (raw,receipt): files[str(path.relative_to(args.output))]=file_hash(path)
            # Sharding occurs only downstream, after a complete task batch.
            for ordinal,row in enumerate(rows):
                try:
                    sample=format_sample(tokenizer,row,task,template=templates[task])
                except Exception as error:
                    atomic_json(args.output/'validation-error.json',dict(task=task,row=ordinal,
                        raw=str(raw),raw_sha256=file_hash(raw),error=str(error)))
                    raise
                sample.update(source_row=ordinal,model_config_sha256=spec['tokenizer_hashes']['config.json'],
                    generator_config_sha256=fingerprint(spec))
                name=f'samples/{task}-{ordinal:03d}.json'
                atomic_json(args.output/name,sample);files[name]=file_hash(args.output/name)
                manifest.append(dict(id=f'{task}-{ordinal:03d}',prepared=name,subtask=task,
                    references=row['outputs'],scoring=sample['scoring']))
                lengths.append(len(sample['token_ids']))
            print(f'{task}: {len(rows)} original batch rows validated',flush=True)
        path=args.output/'manifest.jsonl';temp=path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(r)+'\n' for r in manifest));temp.replace(path)
        files['manifest.jsonl']=file_hash(path)
        summary=dict(prompts=len(manifest),subtasks=dict(Counter(r['subtask'] for r in manifest)),
            min_input_tokens=min(lengths),max_input_tokens=max(lengths),output_caps=CAPS,
            max_seq_length=INPUT,thinking=False,protocol=EVALUATION_PROTOCOL)
        atomic_json(complete,dict(spec=spec,files=files,summary=summary))
        print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    download=commands.add_parser('download');download.add_argument('--ruler',type=Path,required=True)
    download.set_defaults(func=assets)
    prep=commands.add_parser('prepare')
    for name in ('ruler','model','output'):prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--samples',type=int,default=500);prep.add_argument('--seed',type=int,default=42)
    prep.set_defaults(func=prepare)
    args=parser.parse_args();args.func(args)

if __name__=='__main__':main()
