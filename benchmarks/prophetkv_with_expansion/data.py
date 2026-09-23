"""CPU-only, immutable input preparation for the expansion comparison.

This module does not import a scheduler or start an inference engine.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '.results/prophetkv-with-expansion-ruler4x50-20260923'
MODEL = Path('/home/thnguyen/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554')
TASKS = ('niah_multikey_2', 'niah_multikey_3', 'cwe', 'qa_1')
PARENT = Path('/home/thnguyen/unified-cache-management/.results')
SOURCES = {
    task: PARENT / ('prophetkv-expanded-ruler13x100-longbenchv2-20260918'
                    if task == 'niah_multikey_2' else
                    'prophetkv-qwen3-ruler100-longbenchv2-64k-c4096-20260918')
    / 'datasets/65536' / task / 'validation.jsonl' for task in TASKS
}
FORMATTER = REPO / 'benchmarks/prophetkv_full/cacheblend_ruler.py'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def formatter():
    spec = importlib.util.spec_from_file_location('_expansion_prompt_format', FORMATTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(tokenizer, row, task, helpers):
    content = row['input']
    marker = '\nWhat ' if task.startswith('niah_') else '\nQuestion:'
    begin = content.rfind(marker) + 1
    if begin == 0:
        raise ValueError(f'Missing question: {task}')
    separator = ' The special magic' if task.startswith('niah_') else ' Answer:'
    question = content[begin:].split(separator, 1)[0]
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) + row.get('answer_prefix', '')
    offset = text.index(content)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    query = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
             if b > offset + begin and a < offset + begin + len(question)]
    ids = encoded['input_ids']
    if not query or len(ids) - query[0] > 256:
        raise ValueError('Complete query must fit the fixed 256-token fresh suffix')
    chunks, tokens = helpers.cacheblend_prompt(ids, tokenizer, tokenizer.pad_token_id, 4095, 256)
    if len(chunks) < 2 or any(len(c) != 4096 for c in chunks[:-1]) or len(chunks[-1]) > 4096:
        raise ValueError('Invalid chunk layout')
    bounds = [0]
    for chunk in chunks:
        bounds.append(bounds[-1] + len(chunk))
    bounds.append(len(tokens))
    positions = [i + len(tokens) - len(ids) for i in query]
    if tokens[-256:] != ids[-256:] or positions[0] < bounds[-2]:
        raise ValueError('Query or suffix changed')
    return dict(formatted_text=text, original_tokens=len(ids), tokens=len(tokens),
                token_ids=tokens, boundaries=bounds, fresh_suffix_tokens=256,
                prompt_sha256=helpers.prompt_digest(tokens), thinking_enabled=False,
                query=dict(text=question, positions=positions))


def qualifying(rows, encode_row, count=50):
    """First qualifying source rows; reject oversize prompts without truncation."""
    selected, census = [], []
    for ordinal, row in enumerate(rows):
        encoded = encode_row(row)
        accepted = encoded['tokens'] <= 65536
        census.append(dict(source_row=ordinal, tokens=encoded['tokens'], included=accepted))
        if accepted:
            selected.append((ordinal, row, encoded))
        if len(selected) == count:
            return selected, census
    raise ValueError(f'Only {len(selected)} qualifying rows; need {count}')


def verify(root=ROOT):
    manifest = json.loads((root / 'input-manifest.json').read_text())
    helpers = formatter()
    for name, digest in manifest['preparation_sources'].items():
        if sha(name) != digest:
            raise ValueError(f'Preparation source changed: {name}')
    for name, digest in manifest['tokenizer_hashes'].items():
        if sha(MODEL / name) != digest:
            raise ValueError(f'Tokenizer changed: {name}')
    for task, info in manifest['datasets'].items():
        if sha(info['source_path']) != info['sha256'] or sha(root / info['copy']) != info['sha256']:
            raise ValueError(f'Dataset changed: {task}')
        census = manifest['selection'][task]
        expected = [r['source_row'] for r in census if r['included']]
        actual = [r['source_row'] for r in manifest['samples'] if r['label'] == task]
        if expected != actual or len(actual) != 50:
            raise ValueError('First-50 selection changed')
        if any(r['included'] != (r['tokens'] <= 65536) for r in census):
            raise ValueError('Length eligibility changed')
    hashes = set()
    for i, meta in enumerate(manifest['samples']):
        path = root / meta['input_file']
        if sha(path) != meta['input_sha256']:
            raise ValueError('Frozen input changed')
        sample = json.loads(path.read_text())
        if not (sample['tokens'] == len(sample['token_ids']) == meta['tokens'] <= 65536):
            raise ValueError('Invalid formatted prompt length')
        if helpers.prompt_digest(sample['token_ids']) != sample['prompt_sha256']:
            raise ValueError('Prompt digest changed')
        if sample['physical_gpu'] != 1 + i % 4 or meta['physical_gpu'] != sample['physical_gpu']:
            raise ValueError('GPU affinity changed')
        if sample['fresh_suffix_tokens'] != 256 or sample['boundaries'][-1] - sample['boundaries'][-2] != 256:
            raise ValueError('Fresh suffix changed')
        hashes.add(sample['prompt_sha256'])
    if len(hashes) != 200 or len(manifest['samples']) != 200:
        raise ValueError('Expected 200 distinct prompts')
    return manifest


def prepare(root=ROOT):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Preparation requires CUDA_VISIBLE_DEVICES empty')
    if root.exists():
        return verify(root)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    helpers = formatter()
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=root.name + '-prepare-', dir=root.parent))
    try:
        manifest = dict(state='inputs_prepared', model=str(MODEL), model_revision=MODEL.name,
                        methods=['baseline', 'prophetkv_with_expansion'], measured_requests=400,
                        max_model_len=65792, max_output_tokens=128, truncated=False,
                        selection_policy='first 50 source rows/task with fully formatted tokens <= 65536',
                        datasets={}, samples=[], selection={},
                        preparation_sources={str(p): sha(p) for p in (Path(__file__), FORMATTER)},
                        tokenizer_hashes={p.name: sha(p) for p in MODEL.iterdir()
                                          if p.is_file() and p.suffix in ('.json', '.jinja')})
        for task in TASKS:
            source = SOURCES[task]
            raw = source.read_bytes()
            rows = [json.loads(line) for line in raw.splitlines()]
            if len(rows) != 100:
                raise ValueError(f'Expected 100 source rows: {source}')
            relative = f'datasets/{task}/validation.jsonl'
            target = staging / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(raw)
            manifest['datasets'][task] = dict(source_path=str(source), copy=relative, sha256=sha(source))
            selected, census = qualifying(rows, lambda row: encode(tokenizer, row, task, helpers))
            manifest['selection'][task] = census
            for ordinal, row, encoded in selected:
                sample = dict(id=f'{task}-65536-{ordinal:03d}-c4096', dataset='ruler', label=task,
                              source_row=ordinal, source_index=row.get('source_index', row.get('index')),
                              source_path=str(source), chunk_size=4096, aligned_chunk_size=4096,
                              context_target=65536, physical_gpu=1 + len(manifest['samples']) % 4,
                              source_metadata=dict(task=task, references=row['outputs'], source_sha256=sha(target)),
                              **encoded)
                relative = f"samples/{sample['id']}/input.json"
                dump(staging / relative, sample)
                manifest['samples'].append({k: sample[k] for k in
                    ('id', 'label', 'source_row', 'tokens', 'prompt_sha256', 'physical_gpu')}
                    | dict(input_file=relative, input_sha256=sha(staging / relative)))
            print(f'{task}: selected 50 from {len(census)} rows; '
                  f"tokens {min(e['tokens'] for _, _, e in selected)}–{max(e['tokens'] for _, _, e in selected)}", flush=True)
        dump(staging / 'input-manifest.json', manifest)
        verify(staging)
        staging.rename(root)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify'))
    args = parser.parse_args()
    result = prepare() if args.command == 'prepare' else verify()
    print(f"Verified {len(result['samples'])} distinct inputs at {ROOT}")
