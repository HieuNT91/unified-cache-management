"""Read-only adoption of complete prepared cohorts for the zero-repair control."""
import json
from pathlib import Path

from runner.corpus import relative
from runner.setups import file_hash
from runner.tree_profiles import validate


def protect_paths(root, prepared, cache, model):
    root, prepared, cache, model = [Path(p).resolve() for p in (root, prepared, cache, model)]
    for new in (root, cache):
        for old in (prepared, model):
            if new == old or new in old.parents or old in new.parents:
                raise ValueError('Naive reuse needs separate result/cache and existing prepared/model paths')
    if root == cache or root in cache.parents or cache in root.parents:
        raise ValueError('Result and cache paths must be separate')
    if cache.exists() and any(cache.iterdir()):
        raise ValueError('Configure naive reuse with a fresh empty cache directory')


def validate_existing(settings):
    root = Path(settings['prepared']); model = Path(settings['model'])
    receipt = json.loads((root/'preparation.json').read_text())
    spec = receipt['spec']; dataset = settings['dataset']
    if dataset == 'ruler' and (spec.get('samples') != settings['samples_per_task'] or spec.get('seed') != 42):
        raise ValueError('Prepared RULER count/seed differs; full cohort required')
    hashes = spec.get('tokenizer_hashes' if dataset == 'ruler' else 'tokenizer_files', {})
    if not hashes:
        raise ValueError('Prepared tokenizer identity is missing')
    for name, digest in hashes.items():
        if Path(name).name != name or file_hash(model/name) != digest:
            raise ValueError('Prepared tokenizer/model identity differs')
    config_hash = file_hash(model/'config.json')
    if dataset == 'longbench-v2' and spec.get('model_config_sha256') != config_hash:
        raise ValueError('Prepared LongBench model identity differs')
    for name, digest in receipt['files'].items():
        if file_hash(relative(root, name)) != digest:
            raise ValueError(f'Prepared artifact changed: {name}')
    manifest = [json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines() if s.strip()]
    for row in manifest:
        path = relative(root, row['prepared'])
        if receipt['files'].get(row['prepared']) != file_hash(path):
            raise ValueError('Unpinned prepared sample')
        sample = json.loads(path.read_text()); validate(sample, dataset)
        if sample.get('model_config_sha256') != config_hash:
            raise ValueError('Prepared sample belongs to another model')
        if dataset == 'ruler' and sample.get('task') != row['subtask']:
            raise ValueError('RULER prepared task identity differs')
    return receipt
