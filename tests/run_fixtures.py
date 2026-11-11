"""Shared synthetic inputs for run, generation and status tests."""
import json
from types import SimpleNamespace as NS

from runner.cache.setups import atomic_json, file_hash, fingerprint
from runner.inputs.layout import stamp_sample
from runner.runtime.output_budget import make_policy, RULER_PROTOCOL, LONGBENCH_PROTOCOL
from runner.runtime.thinking_budget import KEY
from scripts.collection.collection_v3 import VERSION, GENERATION_PROTOCOL
from scripts.data.ruler import TASKS

GPUS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(4)]


class Tokenizer:
    all_special_ids = [3, 4]
    eos_token_id = 3
    def convert_tokens_to_ids(self, text): return {'<think>': 1, '</think>': 2}[text]
    def convert_ids_to_tokens(self, token): return {1: '<think>', 2: '</think>'}[token]
    def encode(self, text, **kwargs): return list(map(ord, text))


def fixture(base, dataset, chunk, *, samples=500, seed=42):
    model, prepared = base/'model', base/'prepared'
    atomic_json(model/'config.json', {'model_type': 'qwen3'})
    atomic_json(model/'tokenizer.json', {'synthetic': True})
    hashes = {name: file_hash(model/name) for name in ('config.json', 'tokenizer.json')}
    ruler = dataset == 'ruler'
    spec = (dict(samples=samples, seed=seed, protocol=RULER_PROTOCOL, execution_profile='ruler-thinking',
                 tasks=list(TASKS), tokenizer_hashes=hashes) if ruler else
            dict(model_config_sha256=hashes['config.json'], tokenizer_files=hashes,
                 collection_protocol=VERSION, generation_protocol=GENERATION_PROTOCOL))
    rows, files = [], {}
    for i in range(2):
        name = f'samples/{i}.json'
        sample = dict(token_ids=list(range(10048)), boundaries=[0,4096,8192,9728,10048],
                      question_positions=[9728,9729], thinking=True, max_output_tokens=16384,
                      source_row=i, model_config_sha256=hashes['config.json'],
                      evaluation_protocol=RULER_PROTOCOL if ruler else LONGBENCH_PROTOCOL,
                      **{KEY: make_policy(Tokenizer(), ' Answer:' if ruler else '')})
        if ruler:
            sample.update(task=TASKS[0], execution_profile='ruler-thinking', generator_config_sha256=fingerprint(spec))
        else:
            sample.update(source_id=f'source-{i}', source_metadata=dict(domain='d', sub_domain='s', difficulty='easy', length='short'))
        stamp_sample(sample, 'native-template'); atomic_json(prepared/name, sample)
        files[name] = file_hash(prepared/name)
        rows.append(dict(id=f'input-{i}', prepared=name, subtask=TASKS[0] if ruler else 'd / s',
                         references=['A'], scoring='ruler_all' if ruler else 'longbench_v2'))
    (prepared/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    files['manifest.jsonl'] = file_hash(prepared/'manifest.jsonl')
    atomic_json(prepared/'preparation.json', dict(spec=spec, files=files))
    return NS(root=base/'results', prepared=prepared, model=model, cache_root=base/'cache',
              dataset=dataset, adatree_v1=True, baseline_source=None,
              local_validation=True, validation_prompt_ids=['input-0','input-1'],
              tp=4, number_gpus=4, devices=','.join(GPUS), gpu_memory_utilization=.95,
              samples_per_task=samples, seed=seed, answer_seeds=None if ruler else [0],
              kv_chunk_size=chunk, percentages=[5,20,40] if chunk==4096 else [5,20,60])
