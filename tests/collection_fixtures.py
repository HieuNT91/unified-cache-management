"""Synthetic prepared RULER inputs shared by collection and memory tests."""
import json
from types import SimpleNamespace as NS
from runner.inputs.layout import stamp_sample
from runner.cache.setups import atomic_json, file_hash, fingerprint
from runner.runtime import output_budget as budget
from runner.runtime.thinking_budget import KEY
from scripts.ruler import TASKS
from test_ruler_thinking import Tokenizer

GPUS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(10)]


def fixture(root, n=5):
    model = root/'model'; prepared = root/'prepared'
    atomic_json(model/'config.json', {'model_type': 'qwen3'})
    atomic_json(model/'tokenizer.json', {'test': True})
    spec = dict(samples=n, seed=42, protocol=budget.RULER_PROTOCOL, execution_profile='ruler-thinking',
        tasks=list(TASKS), tokenizer_hashes={name: file_hash(model/name) for name in ('config.json','tokenizer.json')})
    files = {}; manifest = []
    for task in TASKS:
        for i in range(n):
            name = f'samples/{task}-{i:03d}.json'
            sample = dict(token_ids=list(range(576)), boundaries=[0,64,320,576], question_positions=[320],
                thinking=True, max_output_tokens=16384, task=task, source_row=i,
                model_config_sha256=spec['tokenizer_hashes']['config.json'], generator_config_sha256=fingerprint(spec),
                execution_profile='ruler-thinking', evaluation_protocol=budget.RULER_PROTOCOL,
                **{KEY: budget.make_policy(Tokenizer(), ' Answer:')})
            stamp_sample(sample, 'native-template'); atomic_json(prepared/name, sample)
            files[name] = file_hash(prepared/name)
            manifest.append(dict(id=f'{task}-{i:03d}', prepared=name, subtask=task, references=['A'], scoring='ruler_all'))
    (prepared/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    files['manifest.jsonl'] = file_hash(prepared/'manifest.jsonl')
    atomic_json(prepared/'spec.json', spec)
    atomic_json(prepared/'preparation.json', dict(spec=spec, files=files))
    return NS(root=root/'result', prepared=prepared, model=model, cache_root=root/'cache',
              devices=','.join(GPUS), tp=2, seed=42, samples_per_task=n, percentages=[5, 20, 40, 60])
