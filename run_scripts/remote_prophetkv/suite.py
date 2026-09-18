#!/usr/bin/env python3
"""Remote-only, locked Qwen3-32B RULER launcher; never imports old schedulers."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
from importlib.metadata import version
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import statistics
import subprocess
import sys
import time

from common import (HERE, REPO, LENGTHS, TASKS, JOBS, CASES, CONTROLS, TIMING, dump,
                    load, settings, selected_devices, busy_devices,
                    identity, group_alive, rope_for_length)
from cacheblend_ruler import cacheblend_prompt
from prophetkv_common import load_sample, score
from ruler_assets import materialize_hotpot

STOP = False
FATAL = re.compile(r'Traceback|\[UC\]\[E\]|load kv cache failed|dump kv cache failed|'
                   r'CUDA out of memory|EngineDeadError|ERROR\s|\b(?:RuntimeError|AssertionError|ValueError):')


def cpu_env():
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', PYTHONHASHSEED='0')
    return env


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another preparation/smoke/run/report process holds run.lock') from None
        yield


def preflight(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'PYTHONPATH' in os.environ:
        raise RuntimeError('Invoke via the shell entry points, which hide GPUs from the supervisor')
    from typing_extensions import Self  # noqa: F401: required by UCM on Python 3.10
    model = Path(cfg['model'])
    config = load(model / 'config.json')
    expected = dict(model_type='qwen3', num_hidden_layers=64, hidden_size=5120,
                    num_attention_heads=64, num_key_value_heads=8)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('MODEL_PATH must be the unquantized Qwen/Qwen3-32B checkpoint')
    if config.get('quantization_config') or config.get('rope_scaling'):
        raise ValueError('Use the original BF16 checkpoint; RoPE is set per length by the launcher')
    weights = list(model.glob('*.safetensors'))
    if not weights:
        raise ValueError('MODEL_PATH has no safetensors weights')
    index = model / 'model.safetensors.index.json'
    if index.exists() and any(not (model / name).is_file() for name in set(load(index)['weight_map'].values())):
        raise ValueError('Checkpoint shards are incomplete')
    runtime = {name: version(name) for name in ('uc-manager', 'vllm', 'torch', 'transformers')}
    for name, wanted in [('uc-manager', '0.3.0'), ('vllm', '0.9.2'), ('torch', '2.7.0'), ('transformers', '4.53.2')]:
        if runtime[name].split('+')[0] != wanted:
            raise RuntimeError(f'{name}={runtime[name]}; this port requires the coupled runtime {wanted}')
    ucm = Path(find_spec('ucm').origin).parent
    vllm = Path(find_spec('vllm').origin).parent
    if ucm.is_relative_to(REPO / 'ucm'):
        raise RuntimeError('Development UCM was imported; use the prepared installed runtime')
    qwen = (vllm / 'model_executor/models/qwen3.py').read_text()
    runner = (vllm / 'v1/worker/gpu_model_runner.py').read_text()
    if 'ucm' not in qwen.lower() or 'ucm' not in runner.lower():
        raise RuntimeError('vLLM is missing the UCM/Qwen3 hooks; an unpatched pip vLLM is insufficient')
    ruler = Path(cfg['ruler'])
    tasks = {unit['task'] for unit in cfg['scope']}
    if 'qa_2' in tasks:
        materialize_hotpot(ruler / 'scripts/data/synthetic/json')
    required = ['scripts/synthetic.yaml', 'scripts/data/synthetic/constants.py',
                'scripts/data/synthetic/json/PaulGrahamEssays.json']
    if 'qa_1' in tasks: required.append('scripts/data/synthetic/json/squad.json')
    if 'qa_2' in tasks: required.append('scripts/data/synthetic/json/hotpotqa.json')
    if 'cwe' in tasks: required.append('scripts/data/synthetic/json/english_words.json')
    for name in required:
        if not (ruler / name).is_file():
            raise ValueError(f'Missing RULER asset: {ruler / name}. See run_scripts/README.md')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError('A fast tokenizer and pad token are required')
    off = tokenizer.apply_chat_template([dict(role='user', content='Test')], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    on = tokenizer.apply_chat_template([dict(role='user', content='Test')], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
    if on == off or '<think>\n\n</think>' not in off:
        raise ValueError('Tokenizer did not apply Qwen3 non-thinking chat mode')
    import nltk
    import wonderwords  # noqa: F401: fail before a lengthy preparation
    nltk.sent_tokenize('This is the RULER tokenizer check.')
    import yaml
    definitions = yaml.safe_load((ruler / 'scripts/synthetic.yaml').read_text())
    for task in tasks:
        if not (ruler / 'scripts/data/synthetic' / (definitions[task]['task'] + '.py')).is_file():
            raise ValueError(f'Missing RULER generator for {task}')
    listing, devices = selected_devices(cfg['indices'])
    # Lower bound only; eager activations/allocator overhead need further space.
    max_target = max(unit['length'] for unit in cfg['scope'])
    weight_bytes = sum(path.stat().st_size for path in weights) / len(devices)
    kv_bytes = max_target * 64 * 8 * 128 * 4 / len(devices)
    if any((weight_bytes + kv_bytes) > d['memory_mib'] * 2**20 * cfg['memory'] for d in devices):
        raise RuntimeError('TP=2 weights plus KV exceed the configured memory budget; 64K BF16 needs A800 80GB-class capacity')
    print(listing, end='', flush=True)
    print(json.dumps(dict(runtime=runtime, gpu_devices=devices,
                          tensor_parallel_size=len(devices), measured_requests=len(cfg['scope']) * cfg['samples'] * len(CASES)), indent=2), flush=True)
    return runtime, ucm, vllm, devices, tokenizer


def query_span(content, task):
    if task.startswith('niah_'):
        begin = content.rfind('\nWhat ') + 1
        if begin == 0:
            raise ValueError('RULER NIAH question not found')
        question = content[begin:].split(' The special magic', 1)[0]
    else:
        marker = '\nQuestion:'
        start = content.rfind(marker)
        if start < 0:
            raise ValueError(f'RULER question not found for {task}')
        begin = start + len(marker)
        while begin < len(content) and content[begin].isspace():
            begin += 1
        question = content[begin:]
        # fwe includes output instructions immediately before its actual question.
        if task == 'fwe':
            skip = question.index('What are the three most frequently appeared')
            begin += skip
            question = question[skip:]
    if not question.strip():
        raise ValueError('Empty RULER question')
    return begin, question


def encode(tokenizer, row, cfg, length, ordinal, task='niah_multivalue'):
    content = row['input']
    begin, question = query_span(content, task)
    text = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    text += row.get('answer_prefix', '')
    offset = text.index(content)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    query = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
             if b > offset + begin and a < offset + begin + len(question)]
    if not query:
        raise ValueError('Empty question token span')
    ids = encoded['input_ids']
    suffix = max(256, len(ids) - query[0])
    chunks, tokens = cacheblend_prompt(ids, tokenizer, tokenizer.pad_token_id, cfg['chunk'] - 1, suffix)
    boundaries = [0]
    for chunk in chunks:
        boundaries.append(boundaries[-1] + len(chunk))
    boundaries.append(len(tokens))
    if len(chunks) < 2 or tokens[-suffix:] != ids[-suffix:]:
        raise ValueError('Expected at least two intact context chunks and an unchanged suffix')
    return dict(id=f'{task}-{length}-{ordinal:04d}', dataset='ruler', label=task,
                context_target=length, source_row=ordinal, source_index=row.get('index'),
                chunk_size=cfg['chunk'], original_tokens=len(ids), tokens=len(tokens),
                token_ids=tokens, boundaries=boundaries, fresh_suffix_tokens=suffix,
                thinking_enabled=False,
                source_metadata=dict(references=row['outputs']),
                query=dict(text=question, positions=[i + len(tokens) - len(ids) for i in query]))


def generator_command(cfg, length, task, source, definitions, constants):
    definition = definitions[task]
    base = constants[definition['task']]
    command = [sys.executable, str(Path(cfg['ruler']) / 'scripts/data/synthetic' / (definition['task'] + '.py')),
               '--save_dir', str(source.parent.parent), '--save_name', task,
               '--subset', 'validation', '--tokenizer_path', cfg['model'], '--tokenizer_type', 'hf',
               '--max_seq_length', str(length), '--tokens_to_generate', '128',
               '--num_samples', str(cfg['samples']), '--random_seed', '42',
               '--template', base['template'] + base.get('answer_prefix', '')]
    for key, value in definition['args'].items():
        command += ['--' + key, str(value)]
    return command


def copy_private_ucm(ucm, private):
    """Keep the installed UCM version, with the Python 3.10 import backport."""
    if private.exists():
        shutil.rmtree(private)
    shutil.copytree(ucm, private, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    connector = private / 'integration/vllm/blend_connector.py'
    source = connector.read_text()
    fixed = compatible_connector_source(source)
    compile(fixed, str(connector), 'exec')
    if fixed != source:
        connector.write_text(fixed)


def compatible_connector_source(source):
    fixed = source.replace('from typing import TYPE_CHECKING, List, Self, Tuple',
                           'from typing import TYPE_CHECKING, List, Tuple\n'
                           'from typing_extensions import Self')
    return fixed


def refresh_runtime(cfg, root):
    """Repair a stopped, prepared job without regenerating data or repinning it."""
    check_orphan(root)
    if live_supervisor(root):
        raise RuntimeError('Stop the supervisor before refreshing its private runtime')
    p = load(root / 'protocol.json')
    if p['settings'] != cfg:
        raise ValueError('Configuration changed: use the original preparation settings')
    private = root / 'private_ucm/ucm'
    connector = private / 'integration/vllm/blend_connector.py'
    runtime = private / 'sparse/prophetkv/runtime.py'
    updates = {connector: compatible_connector_source(connector.read_text()),
               runtime: (HERE / 'method/prophetkv/runtime.py').read_text()}
    updates = {path: text for path, text in updates.items() if path.read_text() != text}
    if not updates:
        print('Private runtime is already current; no changes made.', flush=True)
        return
    if any((root / 'records').rglob('*.validated.json')):
        raise RuntimeError('Private runtime refresh requires a job with no validated measurements')
    for path, text in updates.items():
        compile(text, str(path), 'exec')
    history = root / 'runtime-refresh' / str(time.time_ns())
    for path in updates:
        backup = history / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    # Old smoke checks used the previous implementation and must run again.
    if (root / 'smoke').exists():
        (root / 'smoke').rename(history / 'smoke')
    for path, text in updates.items():
        temporary = path.with_suffix('.py.tmp')
        temporary.write_text(text)
        temporary.replace(path)
    dump(history / 'receipt.json', dict(changed=[str(path) for path in updates],
         reason='Initialize connector RoPE before normalization; support Python 3.10 Self import',
         updated_at=time.time()))
    progress(root, p, 'prepared')
    print(f'Private runtime updated. Prior files/smoke logs: {history}. Prompts retained.', flush=True)


def prepare(cfg, root):
    if (root / 'protocol.json').exists():
        return verify(cfg, root)
    runtime, ucm, _, devices, tokenizer = preflight(cfg)
    if (root / 'preparation.json').exists() and load(root / 'preparation.json') != cfg:
        raise ValueError('Partially prepared output has different settings; choose a new RESULT_ROOT')
    dump(root / 'preparation.json', cfg)
    samples = []
    datasets = []
    ruler = Path(cfg['ruler'])
    constants = runpy.run_path(str(ruler / 'scripts/data/synthetic/constants.py'))['TASKS']
    import yaml
    definitions = yaml.safe_load((ruler / 'scripts/synthetic.yaml').read_text())
    for unit in cfg['scope']:
        length, task = unit['length'], unit['task']
        source = root / f'datasets/{length}/{task}/validation.jsonl'
        if not source.exists():
            command = generator_command(cfg, length, task, source, definitions, constants)
            log_path = root / f'logs/prepare-{length}-{task}.log'
            log_path.parent.mkdir(exist_ok=True)
            print(f'Generating {length}/{task}: {log_path}', flush=True)
            with log_path.open('w') as log:
                subprocess.run(command, env=cpu_env(), cwd='/tmp', stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(rows) != cfg['samples']:
            raise ValueError(f'Expected {cfg["samples"]} rows in {source}, got {len(rows)}')
        datasets.append(str(source))
        for ordinal, row in enumerate(rows):
            sample = encode(tokenizer, row, cfg, length, ordinal, task)
            path = root / 'samples' / f'{sample["id"]}.json'
            sample['source_path'] = str(source)
            dump(path, sample)
            load_sample(path)
            samples.append({k: v for k, v in sample.items() if k != 'token_ids'} |
                           dict(input_path=str(path)))
        print(f'Frozen {length}/{task}: {len(rows)} prompts', flush=True)
    private = root / 'private_ucm/ucm'
    copy_private_ucm(ucm, private)
    for name in ('blend.py', 'selection.py'):
        shutil.copy2(REPO / 'ucm/sparse/blend' / name, private / 'sparse/blend' / name)
    method = private / 'sparse/prophetkv'
    if method.exists():
        shutil.rmtree(method)
    shutil.copytree(HERE / 'method/prophetkv', method, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    factory = private / 'sparse/factory.py'
    with factory.open('a') as stream:
        stream.write('\nUcmSparseFactory.register_sparse_method("ProphetKV", "ucm.sparse.prophetkv.prophetkv", "ProphetKV")\n')
    snapshot = root / 'source'
    shutil.copytree(HERE, snapshot, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    print('Writing protocol (file checksums disabled)...', flush=True)
    protocol = dict(schema_version=2, study='ProphetKV UCM/vLLM port, remote Qwen3-32B', settings=cfg,
                    model=cfg['model'], runtime_versions=runtime, artifact_checksums=False,
                    ruler_task_definitions=definitions,
                    gpu_devices=devices, tensor_parallel_size=len(devices), num_layers=64,
                    gpu_memory_utilization=cfg['memory'], max_output_tokens=128, thinking_enabled=False,
                    decoding=dict(temperature=0, top_p=1, seed=0), dtype='bfloat16', eager=True,
                    cases=list(CASES), lengths=sorted({u['length'] for u in cfg['scope']}),
                    scope=cfg['scope'], samples_per_task_length=cfg['samples'], dataset_output_reserve=128,
                    samples=samples, datasets=datasets, measured_requests=len(samples) * len(CASES),
                    rope_scaling_64k=rope_for_length(65536), rope_scaling_other_lengths=None,
                    prompt_layout='native non-thinking chat, numbered block-padded chunks, fresh query suffix >=256',
                    rate_denominator='cached region after exact first chunk, before fresh suffix',
                    selection='question-only context-softmax scores; all-layer sum; TP head mean; shared global top-k',
                    timing_source=TIMING, timing='includes probe, transfer, selection, fusion; excludes load/build/warmup',
                    storage='buffered local warm cache', engine_lifetime='fresh per sample and configuration',
                    kernel_reference='uYanJX/QCFuse@38795d91d900debb5f2df23bd7296b3ff97960c8', created_at=time.time())
    dump(root / 'prompt_manifest.json', samples)
    dump(root / 'protocol.json', protocol)
    progress(root, protocol, 'prepared')
    print(f'PREPARED {len(samples)} prompts; {protocol["measured_requests"]} measured requests', flush=True)
    return protocol


def verify(cfg, root):
    p = load(root / 'protocol.json')
    if p['settings'] != cfg:
        raise ValueError('Configuration changed: use the original settings or a new RESULT_ROOT')
    if {n: version(n) for n in p['runtime_versions']} != p['runtime_versions']:
        raise ValueError('Runtime versions changed')
    _, devices = selected_devices(cfg['indices'])
    if devices != p['gpu_devices']:
        raise ValueError('GPU identities changed')
    return p


def validate(root, p, meta, case, path, smoke=False):
    r = load(path)
    log = path.with_suffix('.log').read_text(errors='replace')
    if FATAL.search(log) or 'WORKER_COMPLETE' not in log:
        raise ValueError(f'Worker failed: {path.with_suffix(".log")}')
    expected = dict(sample_id=meta['id'], case=case,
                    prompt_tokens=meta['tokens'], label=meta['label'], context_target=meta['context_target'],
                    max_output_tokens=16 if smoke else 128, smoke=smoke,
                    timing_source=TIMING, cache_unchanged=True, online_mask_reused=False,
                    gpu_devices=p['gpu_devices'], tensor_parallel_size=p['tensor_parallel_size'])
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError('Record identity/protocol mismatch')
    if not (0 < r['ttft_seconds'] <= r['generation_seconds']) or not all(
            math.isfinite(r[k]) for k in ('ttft_seconds', 'generation_seconds')):
        raise ValueError('Invalid TTFT/generation timing')
    if not 0 < r['output_tokens'] <= r['max_output_tokens'] or r['output_tokens'] != len(r['output_token_ids']):
        raise ValueError('Invalid output budget/token IDs')
    sample = load_sample(meta['input_path'])
    if (r['references'] != sample['source_metadata']['references'] or
            r['score'] != score(sample, r['prediction'])[0] or r['finish_reason'] not in ('stop', 'length')):
        raise ValueError('Invalid scoring or termination')
    imports = r['worker_imports']
    if sorted(w['rank'] for w in imports) != list(range(p['tensor_parallel_size'])):
        raise ValueError('Missing TP rank provenance')
    for w in imports:
        if (w['visible_uuid'] != p['gpu_devices'][w['rank']]['uuid'] or
                not Path(w['ucm_path']).resolve().is_relative_to(root / 'private_ucm')):
            raise ValueError('Wrong worker GPU or UCM import')
    dp = path.with_suffix('.diagnostics.json')
    workers = load(dp)
    if sorted(w['rank'] for w in workers) != list(range(p['tensor_parallel_size'])):
        raise ValueError('Missing TP rank diagnostics')
    if case == 'baseline':
        if r['num_cached_tokens'] != 0 or any(w['diagnostics'] for w in workers):
            raise ValueError('No-cache baseline reused prior KV')
        return r
    verification = r['cache_verification']
    if not verification['complete'] or verification['verified_shards'] != verification['expected_unique_blocks'] * p['tensor_parallel_size']:
        raise ValueError('Incomplete cache shards')
    b = meta['boundaries']
    eligible = b[-2] - b[1]
    count = math.floor(eligible * int(case.split('-')[-1]) / 100)
    measured = log.split('MEASURE_BEGIN', 1)[1].split('MEASURE_END', 1)[0]
    hits = re.findall(r'request_id: measured-1-0,.*?req_stage: BlendStage\.(\w+), first chunk prefix hit: (\d+), chunks cache total hit: (\d+)', measured)
    if not hits or any(h != ('CACHE_BLEND', str(b[1] // 64), str(eligible // 64)) for h in hits):
        raise ValueError('Missing complete measured cache-hit evidence')
    previous = None
    for w in sorted(workers, key=lambda x: x['rank']):
        selection = [d for d in w['diagnostics'] if d['kind'] == 'prophetkv_selection']
        layers = [d for d in w['diagnostics'] if d['kind'] in ('layer_counts', 'fusion_audit')]
        if len(selection) != 1 or len(layers) != p['num_layers'] or len(w['diagnostics']) != 1 + len(layers):
            raise ValueError('Missing selection/layer diagnostics')
        d = selection[0]
        if any(d[k] != v for k, v in dict(request_id='measured-1-0', eligible_count=eligible,
                selected_count=count, probe_layers=p['num_layers'], alignment_count=p['num_layers'],
                question_positions=meta['query']['positions']).items()):
            raise ValueError('Incorrect probe or selection budget')
        scores = d['scores']
        if len(scores) != eligible or any(not math.isfinite(x) or x < 0 for x in scores):
            raise ValueError('Invalid importance scores')
        chosen = sorted(sorted(range(eligible), key=lambda i: (-scores[i], i))[:count])
        if d['selected_positions'] != [i + b[1] for i in chosen]:
            raise ValueError('Selection is not stable global top-k')
        current = (d['selected_positions'], scores)
        if previous is not None and previous != current:
            raise ValueError('TP ranks disagree on scores or selected positions')
        previous = current
        for i, event in enumerate(layers):
            if event['layer'] != f'model.layers.{i}.self_attn.attn' or any(
                    event[k] != count + meta['fresh_suffix_tokens']
                    for k in ('projection_tokens', 'attention_tokens', 'ffn_tokens')):
                raise ValueError('Unexpected layer compute counts')
            if smoke and not all(event.get(k) is True for k in
                    ('k_write_verified', 'v_write_verified', 'skipped_preserved', 'prefix_preserved',
                     'suffix_verified', 'causal_attention_verified')):
                raise ValueError('Per-layer tensor/causality audit failed')
    return r


def accept(root, p, meta, case, path, smoke=False):
    r = validate(root, p, meta, case, path, smoke)
    dump(path.with_suffix('.validated.json'), dict(complete=True, sample_id=meta['id'], case=case))
    return r


def valid(root, p, meta, case, path, smoke=False):
    if not path.with_suffix('.validated.json').exists():
        return False
    receipt = load(path.with_suffix('.validated.json'))
    if receipt.get('complete') is False:
        return False
    validate(root, p, meta, case, path, smoke)
    return True


def progress(root, p, state, **extra):
    counts = {f"{u['length']}/{u['task']}": {
        case: sum((root / 'records' / m['id'] / f'{case}.validated.json').exists()
                  for m in p['samples'] if m['context_target'] == u['length'] and m['label'] == u['task'])
        for case in CASES} for u in p['scope']}
    dump(root / 'progress.json', dict(state=state, validated=sum(sum(c.values()) for c in counts.values()),
         target=p['measured_requests'], by_task_length=counts, updated_at=time.time(), **extra))



def terminate(child):
    if group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        until = time.monotonic() + 15
        while group_alive(child.pid) and time.monotonic() < until:
            time.sleep(.2)
        if group_alive(child.pid):
            os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=20)
    until = time.monotonic() + 20
    while group_alive(child.pid) and time.monotonic() < until:
        time.sleep(.2)
    if group_alive(child.pid):
        raise RuntimeError('Owned engine group still exists; cache preserved')


def check_orphan(root):
    active = root / 'active.json'
    if active.exists():
        a = load(active)
        if a.get('pgid') and group_alive(a['pgid']):
            raise RuntimeError(f'Previous engine group {a["pgid"]} is still alive. Cache preserved; do not start a duplicate.')


def cleanup(cfg, root):
    check_orphan(root)
    cache = Path(cfg['cache']) / 'owned-sample-cache'
    owner = cache / 'owner.json'
    if cache.exists():
        if not owner.is_file() or load(owner) != dict(result_root=str(root)):
            raise RuntimeError(f'Refusing to remove unrecognized cache directory: {cache}')
        shutil.rmtree(cache)
    return cache


def launch(cfg, root, p, meta, case, path, smoke=False):
    global STOP
    check_orphan(root)
    if STOP:
        raise InterruptedError('Stop requested')
    cache = Path(cfg['cache']) / 'owned-sample-cache'
    cache.mkdir(parents=True, exist_ok=True)
    dump(cache / 'owner.json', dict(result_root=str(root)))
    while busy_devices(p['gpu_devices']):
        dump(root / 'active.json', dict(state='waiting_for_free_gpus', case=case, sample_id=meta['id']))
        print('Waiting for occupied selected GPUs; no process will be displaced', flush=True)
        for _ in range(10):
            if STOP:
                raise InterruptedError('Stop requested')
            time.sleep(1)
    _, devices = selected_devices(cfg['indices'])
    if devices != p['gpu_devices']:
        raise RuntimeError('GPU identities changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.log').exists():
        if path.with_suffix('.validated.json').exists():
            raise RuntimeError('Refusing to overwrite accepted measurement')
        archive = path.parent / 'attempts' / str(time.time_ns())
        archive.mkdir(parents=True)
        for artifact in path.parent.glob(path.stem + '.*'):
            if artifact.is_file():
                artifact.rename(archive / artifact.name)
    script = 'reference.py' if case == 'reference' else 'worker.py'
    command = [sys.executable, '-u', str(HERE / script), '--protocol', str(root / 'protocol.json'),
               '--sample', meta['input_path'], '--output', str(path)]
    if case != 'reference':
        command += ['--case', case, '--cache-dir', str(cache)]
        if smoke:
            command.append('--smoke')
    env = cpu_env()
    env.update(CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in devices),
               REMOTE_GPU_DEVICES=json.dumps(devices), SELECTOR_UCM_ROOT=str(root / 'private_ucm'),
               ENABLE_SPARSE='TRUE', VLLM_USE_V1='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_ALLOW_INSECURE_SERIALIZATION='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    print(f'LAUNCH {meta["id"]} {case} smoke={smoke} log={path.with_suffix(".log")}', flush=True)
    with path.with_suffix('.log').open('w') as log:
        child = subprocess.Popen(command, cwd='/tmp', env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        dump(root / 'active.json', dict(state='running', pid=child.pid, pgid=child.pid,
             identity=identity(child.pid), command=command, sample_id=meta['id'], case=case,
             log=str(path.with_suffix('.log')), gpu_devices=devices, started_at=time.time()))
        try:
            while child.poll() is None:
                if STOP:
                    raise InterruptedError('Stop requested')
                age = time.time() - path.with_suffix('.log').stat().st_mtime
                if age > int(os.environ.get('WATCHDOG_SECONDS', '1800')):
                    raise RuntimeError(f'Worker made no logged progress for {age:.0f}s')
                with path.with_suffix('.log').open(errors='replace') as reader:
                    reader.seek(max(0, path.with_suffix('.log').stat().st_size - 32768))
                    if FATAL.search(reader.read()):
                        raise RuntimeError(f'Fatal engine log: {path.with_suffix(".log")}')
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f'Worker exit={child.returncode}: {path.with_suffix(".log")}')
        finally:
            terminate(child)
            dump(root / 'active.json', dict(state='engine_exited', pgid=child.pid, case=case,
                 sample_id=meta['id'], log=str(path.with_suffix('.log')), ended_at=time.time()))
    text = path.with_suffix('.log').read_text(errors='replace')
    if FATAL.search(text):
        raise ValueError('Fatal error in completed worker log')
    if case == 'populate':
        if 'WORKER_COMPLETE' not in text or not load(path)['cache_verification']['complete']:
            raise ValueError('Incomplete cache population')
    elif case == 'reference':
        if 'REFERENCE_COMPLETE' not in text:
            raise ValueError('Incomplete native reference')
    else:
        accept(root, p, meta, case, path, smoke)


def valid_gate(root, p, length, task):
    gate = root / 'smoke' / str(length) / task
    if (gate / 'validation.json').exists():
        receipt = load(gate / 'validation.json')
        if not receipt['complete']:
            return False
        if receipt['sample_id'] not in {m['id'] for m in p['samples']
                if m['context_target'] == length and m['label'] == task}:
            raise ValueError('Gate sample does not belong to this task/length')
        # Older receipts store a path -> checksum dict; only paths are needed.
        for path in receipt['artifacts']:
            if not Path(path).is_file():
                raise ValueError(f'Missing gate artifact: {path}')
        return True
    return False


def smoke_gate(cfg, root, p, length, task):
    gate = root / 'smoke' / str(length) / task
    if valid_gate(root, p, length, task):
        return
    meta = max((m for m in p['samples'] if m['context_target'] == length and m['label'] == task), key=lambda x: x['tokens'])
    cleanup(cfg, root)
    for case in ('reference', 'populate', *CASES, *CONTROLS):
        path = gate / f'{case}.json'
        if case not in ('reference', 'populate') and valid(root, p, meta, case, path, True):
            continue
        launch(cfg, root, p, meta, case, path, True)
    # Match the no-custom-attention reference with EXACT first-chunk reuse.
    # Full no-cache prefill has different numerical scheduling and is not the oracle.
    import torch
    comparisons = []
    for rank in range(p['tensor_parallel_size']):
        ref = torch.load(str(gate / 'reference.model-audit.pt') + f'.rank{rank}.pt', weights_only=True)[0]
        full = torch.load(str(gate / 'prophetkv-100.model-audit.pt') + f'.rank{rank}.pt', weights_only=True)[0]
        if set(ref) != set(range(p['num_layers'])) or set(full) != set(ref):
            raise ValueError('Native reference is missing decoder layers')
        for layer in ref:
            if not torch.equal(ref[layer], full[layer]):
                raise ValueError(f'ProphetKV100/native-prefix mismatch on TP rank {rank}, layer {layer}')
        comparisons.append(dict(rank=rank, bitwise_equal_layers=len(ref)))
    cleanup(cfg, root)
    artifacts = [str(path) for path in gate.iterdir() if path.is_file() and path.name != 'validation.json']
    dump(gate / 'validation.json', dict(complete=True, sample_id=meta['id'],
         comparisons=comparisons, artifacts=artifacts, smoke_output_tokens=16, measured=False))
    print(f'SMOKE_GATE_PASSED {length}/{task}', flush=True)


def run(cfg, root, args):
    global STOP
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    check_orphan(root)
    dump(root / 'supervisor.json', dict(pid=os.getpid(), identity=identity(os.getpid()),
         pgid=os.getpgrp(), sid=os.getsid(0), command=sys.argv, started_at=time.time(), state='running'))
    p = load(root / 'protocol.json')
    try:
        progress(root, p, 'verifying')
        print('Checking settings, runtime versions and GPU identities (no file hashing)...', flush=True)
        p = verify(cfg, root)
        units = [u for u in p['scope'] if (args.length is None or u['length'] == args.length)
                 and (args.task is None or u['task'] == args.task)]
        if not units:
            raise ValueError('Requested task/length is not assigned to this job')
        for unit in units:
            length, task = unit['length'], unit['task']
            progress(root, p, 'smoke', context_target=length, task=task)
            smoke_gate(cfg, root, p, length, task)
            if args.command == 'smoke':
                continue
            for meta in (m for m in p['samples'] if m['context_target'] == length and m['label'] == task):
                # Rotate percentage order across prompts; baseline always measures full prefill.
                methods = list(CASES[1:])
                offset = meta['source_row'] % len(methods)
                methods = [CASES[0], *methods[offset:], *methods[:offset]]
                if args.method:
                    methods = [args.method]
                missing = [c for c in methods if not valid(root, p, meta, c, root / 'records' / meta['id'] / f'{c}.json')]
                if not missing:
                    continue
                cleanup(cfg, root)
                if any(c != 'baseline' for c in missing):
                    launch(cfg, root, p, meta, 'populate', root / 'cache-builds' / f'{meta["id"]}-{time.time_ns()}.json')
                for case in missing:
                    progress(root, p, 'running', sample_id=meta['id'], method=case)
                    launch(cfg, root, p, meta, case, root / 'records' / meta['id'] / f'{case}.json')
                    progress(root, p, 'running')
                cleanup(cfg, root)
        progress(root, p, 'smoke_complete' if args.command == 'smoke' else 'partial')
        if args.command != 'smoke':
            report(root, p)
    except BaseException as error:
        progress(root, p, 'stopped' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed', error=str(error))
        raise
    finally:
        check_orphan(root)
        cleanup(cfg, root)
        supervisor = load(root / 'supervisor.json')
        dump(root / 'supervisor.json', supervisor | dict(state='exited', ended_at=time.time()))
        dump(root / 'cleanup.json', dict(owned_engine_groups_exited=True,
             cache_removed=True, remaining_selected_gpu_processes=busy_devices(p['gpu_devices']), time=time.time()))


def report(root, p):
    check_orphan(root)
    rows = []
    for meta in p['samples']:
        for case in CASES:
            path = root / 'records' / meta['id'] / f'{case}.json'
            if valid(root, p, meta, case, path):
                rows.append(load(path))
    directory = root / ('final' if len(rows) == p['measured_requests'] else 'partial')
    directory.mkdir(exist_ok=True)
    (directory / 'raw_records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    comparison = []
    for unit in p['scope']:
        length, task = unit['length'], unit['task']
        baseline = {r['sample_id']: r for r in rows if r['context_target'] == length and r['label'] == task and r['case'] == 'baseline'}
        for case in CASES:
            subset = [r for r in rows if r['context_target'] == length and r['label'] == task and r['case'] == case]
            paired = [r for r in subset if r['sample_id'] in baseline]
            avg = lambda key: statistics.mean(r[key] for r in subset) if subset else ''
            comparison.append(dict(context_target=length, task=task, method=case, samples=len(subset),
                accuracy_percent=100 * avg('score') if subset else '', mean_ttft_seconds=avg('ttft_seconds'),
                mean_generation_seconds=avg('generation_seconds'), mean_output_tokens=avg('output_tokens'),
                mean_prompt_tokens=avg('prompt_tokens'), length_limited=sum(r['finish_reason'] == 'length' for r in subset),
                paired_samples=len(paired), ttft_speedup=(statistics.mean(baseline[r['sample_id']]['ttft_seconds'] for r in paired)
                    / statistics.mean(r['ttft_seconds'] for r in paired)) if paired else ''))
    with (directory / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    costs = [load(path) for path in sorted((root / 'cache-builds').glob('*.json'))]
    dump(directory / 'cache_build_costs.json', costs)
    (directory / 'REPORT.md').write_text(
        f'# Qwen3-32B ProphetKV UCM/vLLM port\n\nValidated {len(rows)}/{p["measured_requests"]} requests. '
        f'{p["samples_per_task_length"]} independent prompts per task/length; one timing per prompt/method.\n\n'
        'See comparison.csv for official RULER reference-substring accuracy (percent), TTFT, total generation time, '
        'length-limited outputs, and paired TTFT speedup. Speedup is mean paired baseline TTFT / mean paired method TTFT; >1 is faster.\n\n'
        'Targets 8K/16K/32K/64K are dataset targets, not exact token counts. Raw records give actual lengths. '
        'Native non-thinking chat and identical chunk markers/padding modify original RULER inputs for all methods. '
        'Ratios select cached tokens after the exact first chunk, excluding the fresh suffix. '
        'All 64 layers probe and repair; TP ranks share the globally averaged head scores and selected positions.\n\n'
        'TTFT includes online probing, transfers, ranking and inference. Loading, offline chunk population, cache readiness '
        'and warmup are excluded and recorded separately. Storage is buffered local warm cache. '
        'Fresh engines isolate each sample/method; this adds substantial wall-clock loading cost. '
        '64K alone uses YaRN factor 4; other lengths use original RoPE. Compare methods within each length.\n\n'
        'Smoke controls (0/100% plus native dense exact-prefix reference) are excluded from measured results. '
        'This is the local ProphetKV UCM/vLLM port, not a claim of an upstream pipeline reproduction.\n')
    complete = len(rows) == p['measured_requests']
    if complete:
        for unit in p['scope']:
            length, task = unit['length'], unit['task']
            if not valid_gate(root, p, length, task):
                raise ValueError('Missing smoke gate')
        dump(directory / 'validation.json', dict(complete=True, validated=len(rows), target=p['measured_requests'],
             artifacts=[str(path.relative_to(directory)) for path in directory.rglob('*')
                        if path.is_file() and path.name != 'validation.json']))
    progress(root, p, 'complete' if complete else 'partial')
    print(f'REPORT {directory} ({len(rows)}/{p["measured_requests"]})', flush=True)


def live_supervisor(root):
    path = root / 'supervisor.json'
    if not path.exists():
        return None
    saved = load(path)
    return saved if identity(saved['pid']) == saved['identity'] else None


def detach(cfg, root, args):
    with locked(root):
        if live_supervisor(root):
            raise RuntimeError('Supervisor already running')
        check_orphan(root)
        if not (root / 'protocol.json').exists():
            raise ValueError('Run prepare.sh before detach.sh')
    command = ['nohup', sys.executable, '-u', str(HERE / 'suite.py'), 'run']
    if args.length:
        command += ['--length', str(args.length)]
    if args.method:
        command += ['--method', args.method]
    if args.task:
        command += ['--task', args.task]
    with (root / 'supervisor.log').open('a') as log:
        child = subprocess.Popen(command, cwd='/tmp', env=cpu_env(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f'Detached launcher exited ({child.returncode}); read {root / "supervisor.log"}')
        live = live_supervisor(root)
        if live and live['pid'] == child.pid:
            if os.getsid(child.pid) != child.pid or os.getpgid(child.pid) != child.pid:
                raise RuntimeError('Detached supervisor did not acquire an independent session')
            dump(root / 'detachment.json', dict(pid=child.pid, identity=identity(child.pid),
                 command=command, sid=os.getsid(child.pid), pgid=os.getpgid(child.pid),
                 stdin='/dev/null', stdout=str(root / 'supervisor.log'), nohup=True))
            print(f'Detached PID {child.pid}. Log: {root / "supervisor.log"}\nRun status.sh after this command exits to confirm survival.')
            return
        time.sleep(.2)
    raise RuntimeError('Supervisor did not publish startup state; inspect supervisor.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('jobs', 'plan', 'preflight', 'prepare', 'refresh', 'smoke', 'run', 'detach', 'status', 'stop', 'report', 'logs'))
    parser.add_argument('--length', type=int, choices=LENGTHS)
    parser.add_argument('--method', choices=CASES)
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    cfg = settings()
    root = Path(cfg['root'])
    if args.command == 'jobs':
        print('Job | Physical GPUs | Task/length units | Measured requests | Detached command')
        print('--- | --- | --- | --- | ---')
        for job, units in JOBS.items():
            description = '; '.join(f'{task}@{length // 1024}K' for length, task in units)
            print(f'{job} | {int(job)*2},{int(job)*2+1} | {description} | '
                  f'{len(units)*cfg["samples"]*len(CASES)} | bash run_scripts/job_{job}.sh detach')
    elif args.command == 'plan':
        print(json.dumps(dict(settings=cfg, scope=cfg['scope'], methods=CASES,
              requests=len(cfg['scope']) * cfg['samples'] * len(CASES), thinking=False, max_new_tokens=128,
              tensor_parallel_size=len(cfg['indices'])), indent=2))
    elif args.command == 'status':
        live = live_supervisor(root)
        print(json.dumps(dict(supervisor_live=bool(live), supervisor=live,
             progress=load(root / 'progress.json') if (root / 'progress.json').exists() else None,
             active=load(root / 'active.json') if (root / 'active.json').exists() else None), indent=2))
    elif args.command == 'logs':
        active = load(root / 'active.json') if (root / 'active.json').exists() else {}
        path = active.get('log', str(root / 'supervisor.log'))
        print(f'Following {path} (Ctrl-C stops the viewer only)', flush=True)
        os.execvp('tail', ['tail', '-n', '80', '-F', path])
    elif args.command == 'stop':
        live = live_supervisor(root)
        if not live:
            print('No verified live supervisor; no signal sent.')
            check_orphan(root)
            return
        os.kill(live['pid'], signal.SIGTERM)
        print(f'SIGTERM sent to verified supervisor {live["pid"]}. Use status.sh to confirm shutdown.')
    elif args.command == 'detach':
        detach(cfg, root, args)
    elif args.command == 'preflight':
        preflight(cfg)
    else:
        with locked(root):
            if args.command == 'prepare':
                prepare(cfg, root)
            elif args.command == 'refresh':
                refresh_runtime(cfg, root)
            elif args.command == 'report':
                report(root, verify(cfg, root))
            else:
                run(cfg, root, args)


if __name__ == '__main__':
    main()
