#!/usr/bin/env python3
"""Remote-only, locked Qwen3-32B LongBench thinking launcher; never imports old schedulers."""
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
import uuid

from common import (HERE, REPO, LENGTHS, TASKS, JOBS, CASES, CONTROLS, TIMING, dump,
                    load, settings, selected_devices, busy_devices,
                    identity, group_alive, rope_for_length, ENGINE_POLICY, engine_group, model_limit)
from cacheblend_ruler import cacheblend_prompt
from prophetkv_common import load_sample, score

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
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError('A fast tokenizer and pad token are required')
    off = tokenizer.apply_chat_template([dict(role='user', content='Test')], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    on = tokenizer.apply_chat_template([dict(role='user', content='Test')], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
    if on == off or not on.rstrip().endswith('<think>'):
        raise ValueError('Tokenizer did not apply Qwen3 thinking chat mode')
    listing, devices = selected_devices(cfg['indices'])
    # Lower bound only; eager activations/allocator overhead need further space.
    max_target = cfg['max_model_len']
    weight_bytes = sum(path.stat().st_size for path in weights) / len(devices)
    kv_bytes = max_target * 64 * 8 * 128 * 4 / len(devices)
    if any((weight_bytes + kv_bytes) > d['memory_mib'] * 2**20 * cfg['memory'] for d in devices):
        raise RuntimeError('TP=2 weights plus KV exceed the configured memory budget; 64K BF16 needs A800 80GB-class capacity')
    print(listing, end='', flush=True)
    print(json.dumps(dict(runtime=runtime, gpu_devices=devices,
                          tensor_parallel_size=len(devices), longbench_requests='determined by full tokenizer census'), indent=2), flush=True)
    return runtime, ucm, vllm, devices, tokenizer


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


def persistent_sources(private):
    return {private / 'sparse/prophetkv/lifecycle.py': (HERE / 'lifecycle.py').read_text(),
            private / 'integration/vllm/persistent_connector.py': (HERE / 'persistent_connector.py').read_text()}


def prepare(cfg, root):
    from prepare import prepare_experiment
    return prepare_experiment(cfg, root)


def verify(cfg, root):
    p = load(root / 'protocol.json')
    if p['settings'] != cfg:
        raise ValueError('Configuration changed: use the original settings or a new RESULT_ROOT')
    if {n: version(n) for n in p['runtime_versions']} != p['runtime_versions']:
        raise ValueError('Runtime versions changed')
    _, devices = selected_devices(cfg['indices'])
    if devices != p['gpu_devices']:
        raise ValueError('GPU identities changed')
    from prepare import verify_frozen
    verify_frozen(root, p)
    return p


def validate(root, p, meta, case, path, smoke=False):
    r = load(path)
    from prepare import sha
    if r.get('input_sha256') != meta['input_sha256'] or r.get('protocol_sha256') != sha(root/'protocol.json'):
        raise ValueError('Record input/protocol provenance changed')
    log = path.with_suffix('.log').read_text(errors='replace')
    if FATAL.search(log) or 'WORKER_COMPLETE' not in log:
        raise ValueError(f'Worker failed: {path.with_suffix(".log")}')
    expected = dict(sample_id=meta['id'], case=case,
                    prompt_tokens=meta['tokens'], label=meta['label'], context_target=meta['context_target'],
                    max_output_tokens=meta['max_output_tokens'], smoke=False, thinking_enabled=meta['thinking_enabled'],
                    method='baseline' if case=='baseline' else 'prophetkv_with_expansion',
                    parameters=p['case_parameters'].get(case),
                    timing_source=TIMING, cache_unchanged=True, online_mask_reused=False,
                    gpu_devices=p['gpu_devices'], tensor_parallel_size=p['tensor_parallel_size'])
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError('Record identity/protocol mismatch')
    if not (0 < r['ttft_seconds'] <= r['generation_seconds']) or not all(
            math.isfinite(r[k]) for k in ('ttft_seconds', 'generation_seconds')):
        raise ValueError('Invalid TTFT/generation timing')
    for key in ('cache_build_seconds','prime_seconds','cache_readiness_seconds','artifact_export_seconds','retirement_seconds'):
        if not math.isfinite(r.get(key, float('nan'))) or r[key] < 0:
            raise ValueError('Invalid separate duration: '+key)
    session = load(Path(r['session_path']))
    if len(session.get('setup_receipts',[])) != p['tensor_parallel_size']:
        raise ValueError('Missing RoPE setup receipts')
    for receipt in session['setup_receipts']:
        audits = receipt['rope_windows']
        if len(audits) != 64 or not all(a['table_positions']==81920 and a['factor']==2.0 and
                a['original_entries_bitwise_equal'] and a['extended_formula_bitwise_equal'] for a in audits):
            raise ValueError('Invalid RoPE table audit')
    if case != 'baseline':
        readiness = session.get('warmup_cache_readiness') or {}
        if readiness.get('complete') is not True or readiness.get('verified_shards') != 2*p['tensor_parallel_size']:
            raise ValueError('Missing committed warmup cache shards')
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
    request_id = r.get('request_id', 'measured-1-0')
    if r.get('engine_policy') != ENGINE_POLICY:
        raise ValueError('Wrong engine lifecycle')
    validate_retirement(r, p, meta)
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
    count = math.floor(eligible * p['case_parameters'][case]['total_ratio'])
    measured = log.split('MEASURE_BEGIN', 1)[1].split('MEASURE_END', 1)[0]
    hits = re.findall(r'request_id: ' + re.escape(request_id) + r',.*?req_stage: BlendStage\.(\w+), first chunk prefix hit: (\d+), chunks cache total hit: (\d+)', measured)
    if not hits or any(h != ('CACHE_BLEND', str(b[1] // 64), str(eligible // 64)) for h in hits):
        raise ValueError('Missing complete measured cache-hit evidence')
    previous = None
    for w in sorted(workers, key=lambda x: x['rank']):
        selection = [d for d in w['diagnostics'] if d['kind'] == 'prophetkv_selection']
        layers = [d for d in w['diagnostics'] if d['kind'] in ('layer_counts', 'fusion_audit')]
        if len(selection) != 1 or len(layers) != p['num_layers'] or len(w['diagnostics']) != 1 + len(layers):
            raise ValueError('Missing selection/layer diagnostics')
        d = selection[0]
        if any(d[k] != v for k, v in dict(request_id=request_id, eligible_count=eligible,
                selected_count=count, probe_layers=p['num_layers'], alignment_count=p['num_layers'],
                question_positions=meta['query']['positions']).items()):
            raise ValueError('Incorrect probe or selection budget')
        scores = d['scores']
        if len(scores) != eligible or any(not math.isfinite(x) or x < 0 for x in scores):
            raise ValueError('Invalid importance scores')
        from selection_reference import expansion_reference
        chosen, details = expansion_reference(scores, b[1], p['case_parameters'][case])
        if d['selected_positions'] != chosen or any(d.get(k) != v for k,v in details.items()):
            raise ValueError('Expansion differs from saved-score CPU reference')
        if d.get('method') != 'prophetkv_with_expansion' or d.get('parameters') != p['case_parameters'][case]:
            raise ValueError('Wrong expansion configuration')
        current = (d['selected_positions'], scores)
        if previous is not None and previous != current:
            raise ValueError('TP ranks disagree on scores or selected positions')
        previous = current
        for i, event in enumerate(layers):
            if event.get('request_id') != request_id or event.get('selected_positions') != chosen:
                raise ValueError('Layer selected positions differ')
            if event['layer'] != f'model.layers.{i}.self_attn.attn' or any(
                    event[k] != count + meta['fresh_suffix_tokens']
                    for k in ('projection_tokens', 'attention_tokens', 'ffn_tokens')):
                raise ValueError('Unexpected layer compute counts')
            if smoke and not all(event.get(k) is True for k in
                    ('k_write_verified', 'v_write_verified', 'skipped_preserved', 'prefix_preserved',
                     'suffix_verified', 'causal_attention_verified')):
                raise ValueError('Per-layer tensor/causality audit failed')
    return r


def validate_retirement(record, protocol, meta):
    from lifecycle import namespace_from_id
    if namespace_from_id(record['request_id']) != record['namespace']:
        raise ValueError('Wrong request namespace')
    if (record['max_model_len'] != model_limit(protocol, meta) or
            record['rope_group'] != engine_group(meta['context_target']) or
            record['cache_files_after_retirement'] != 0):
        raise ValueError('Invalid persistent engine configuration/cleanup')
    workers = record['retirement']['workers']
    if sorted(w['rank'] for w in workers) != list(range(protocol['tensor_parallel_size'])) or not all(
            w['quiescent'] and w['transfers']['pending'] == 0 and w['request_bookkeeping'] == 0 for w in workers):
        raise ValueError('Missing TP retirement acknowledgement')
    if record['case'] != 'baseline':
        if record['retirement']['scheduler'] != dict(request_id=record['request_id'], requests_blend_meta=0, requests_meta=0):
            raise ValueError('Stale scheduler retirement')
        if record['cache_verification']['namespace'] != record['namespace']:
            raise ValueError('Cache verified under a different namespace')


def accept(root, p, meta, case, path, smoke=False):
    r = validate(root, p, meta, case, path, smoke)
    from prepare import sha
    dump(path.with_suffix('.validated.json'), dict(complete=True, sample_id=meta['id'], case=case,
        hashes={suffix:sha(path.with_suffix(suffix)) for suffix in ('.json','.log','.diagnostics.json')}))
    return r


def valid(root, p, meta, case, path, smoke=False):
    if not path.with_suffix('.validated.json').exists():
        return False
    receipt = load(path.with_suffix('.validated.json'))
    if receipt.get('complete') is not True:
        return False
    from prepare import sha
    if set(receipt.get('hashes',{})) != {'.json','.log','.diagnostics.json'} or any(
            sha(path.with_suffix(suffix)) != digest for suffix,digest in receipt['hashes'].items()):
        raise ValueError('Accepted artifacts changed')
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


def session_plan(root, p, samples, method=None):
    """Keep each configuration resident across every compatible missing prompt."""
    for group in dict.fromkeys(engine_group(m['context_target']) for m in samples):
        for case in ([method] if method else p['methods_order']):
            entries = []
            for meta in samples:
                if engine_group(meta['context_target']) != group:
                    continue
                path = root / 'records' / meta['id'] / f'{case}.json'
                if not valid(root, p, meta, case, path):
                    entries.append(meta | dict(output=str(path)))
            if entries:
                yield group, case, entries


def collect_completed(root, p, case, pending, text, smoke):
    """Accept only complete request intervals; a later failure cannot spoil them."""
    while pending:
        entry = pending[0]
        path = Path(entry['output'])
        if not path.exists():
            break
        record = load(path)
        rid = record['request_id']
        marker = 'REQUEST_COMPLETE ' + rid + ' ' + str(path) + '\n'
        if marker not in text:
            break
        before, text = text.split(marker, 1)
        begin = 'REQUEST_BEGIN ' + rid + '\n'
        if begin not in before or FATAL.search(before):
            raise RuntimeError('Invalid request completion log')
        path.with_suffix('.log').write_text(begin + before.split(begin, 1)[1] + marker + 'WORKER_COMPLETE\n')
        accept(root, p, entry, case, path, smoke)
        pending.pop(0)
        print(f'VALIDATED {entry["id"]} {case} remaining_in_session={len(pending)}', flush=True)
        if not smoke:
            progress(root, p, 'running', sample_id=entry['id'], method=case,
                     engine_policy=ENGINE_POLICY, session_id=record['session_id'])
    return text


def launch_session(cfg, root, p, case, entries, smoke=False, retries=1):
    pending = list(entries)
    for attempt in range(retries + 1):
        if not pending:
            return
        check_orphan(root)
        if STOP:
            raise InterruptedError('Stop requested')
        cache = cleanup(cfg, root)
        cache.mkdir(parents=True, exist_ok=True)
        dump(cache / 'owner.json', dict(result_root=str(root)))
        while busy_devices(p['gpu_devices']):
            dump(root / 'active.json', dict(state='waiting_for_free_gpus', case=case))
            print('Waiting for selected GPUs', flush=True)
            for _ in range(10):
                if STOP:
                    raise InterruptedError('Stop requested')
                time.sleep(1)
        _, devices = selected_devices(cfg['indices'])
        if devices != p['gpu_devices']:
            raise RuntimeError('GPU identities changed')
        for entry in pending:
            path = Path(entry['output'])
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.with_suffix('.validated.json').exists():
                raise RuntimeError('Refusing to overwrite accepted request')
            artifacts = [a for a in path.parent.glob(path.stem + '.*') if a.is_file()]
            if artifacts:
                archive = path.parent / 'attempts' / str(time.time_ns())
                archive.mkdir(parents=True)
                for artifact in artifacts:
                    artifact.rename(archive / artifact.name)
        sid = uuid.uuid4().hex
        session = root / 'sessions' / (sid + '.json')
        manifest = session.with_suffix('.manifest.json')
        log_path = session.with_suffix('.log')
        dump(manifest, pending)
        dump(session, dict(session_id=sid, state='starting', case=case, engine_policy=ENGINE_POLICY,
                           requested_samples=[m['id'] for m in pending], started_at=time.time()))
        command = [sys.executable, '-u', str(HERE / 'persistent_worker.py'),
            '--protocol', str(root / 'protocol.json'), '--manifest', str(manifest),
            '--case', case, '--cache-dir', str(cache), '--session', str(session)]
        if smoke:
            raise ValueError('Separate qualification/smoke is disabled')
        env = cpu_env()
        env.update(CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in devices),
            REMOTE_GPU_DEVICES=json.dumps(devices), SELECTOR_UCM_ROOT=str(root / 'private_ucm'),
            ENABLE_SPARSE='TRUE', VLLM_USE_V1='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            VLLM_ALLOW_INSECURE_SERIALIZATION='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            VLLM_ALLOW_LONG_MAX_MODEL_LEN='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            PROPHETKV_SCHEDULER_RECEIPT=str(session.with_suffix('.scheduler.json')))
        print(f'LAUNCH_SESSION {case} prompts={len(pending)} smoke={smoke} log={log_path}', flush=True)
        try:
            with log_path.open('w') as log:
                child = subprocess.Popen(command, cwd='/tmp', env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                dump(root / 'active.json', dict(state='running', pid=child.pid, pgid=child.pid,
                    identity=identity(child.pid), command=command, case=case, session_id=sid,
                    engine_policy=ENGINE_POLICY, samples=len(pending), log=str(log_path),
                    gpu_devices=devices, started_at=time.time()))
                buffer = ''
                try:
                    with log_path.open(errors='replace') as reader:
                        while True:
                            returncode = child.poll()
                            buffer += reader.read()
                            buffer = collect_completed(root, p, case, pending, buffer, smoke)
                            if STOP:
                                raise InterruptedError('Stop requested')
                            if FATAL.search(buffer):
                                raise RuntimeError(f'Fatal engine log: {log_path}')
                            if returncode is not None:
                                if returncode or pending or 'SESSION_COMPLETE' not in buffer:
                                    raise RuntimeError(f'Incomplete session exit={returncode}: {log_path}')
                                break
                            if time.time() - log_path.stat().st_mtime > int(os.environ.get('WATCHDOG_SECONDS', '1800')):
                                raise RuntimeError(f'Session made no logged progress: {log_path}')
                            time.sleep(1)
                finally:
                    terminate(child)
                    dump(root / 'active.json', dict(state='engine_exited', pgid=child.pid,
                        case=case, session_id=sid, log=str(log_path), ended_at=time.time()))
        except RuntimeError as error:
            dump(session, load(session) | dict(state='failed', error=str(error), ended_at=time.time()))
            if not pending or attempt == retries:
                raise
            print(f'RETRY_SESSION {case} missing={len(pending)}: {error}', flush=True)
        finally:
            cleanup(cfg, root)


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
        print('Checking frozen sources, inputs, runtime versions and GPU identities...', flush=True)
        p = verify(cfg, root)
        if any(not path.is_file() or path.read_text() != text for path, text in
               persistent_sources(root / 'private_ucm/ucm').items()):
            raise RuntimeError('Private persistence modules changed; use a new result directory')
        units = [u for u in p['scope'] if (args.length is None or u['length'] == args.length)
                 and (args.task is None or u['task'] == args.task)]
        if not units:
            raise ValueError('Requested task/length is not assigned to this job')
        if args.command == 'smoke':
            raise ValueError('Separate qualification/smoke is disabled')
        if args.command != 'smoke':
            selected = [m for m in p['samples'] if any(
                m['context_target'] == u['length'] and m['label'] == u['task'] for u in units)]
            dump(root / 'execution-persistent.json', dict(engine_policy=ENGINE_POLICY,
                 source=str(HERE), artifact_checksums=True, updated_at=time.time(),
                 note='Resume retains validated measurements; only missing requests run.'))
            for group, case, entries in session_plan(root, p, selected, args.method):
                progress(root, p, 'running', method=case, rope_group=group)
                launch_session(cfg, root, p, case, entries)
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
    from report import report_job
    return report_job(root, p)


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
    parser.add_argument('command', choices=('plan', 'preflight', 'prepare', 'run', 'detach', 'status', 'stop', 'report', 'logs'))
    parser.add_argument('--length', type=int, choices=LENGTHS)
    parser.add_argument('--method', choices=CASES)
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    cfg = settings()
    root = Path(cfg['root'])
    if args.command == 'plan':
        print(json.dumps(dict(settings=cfg, methods=CASES,
              longbench_requests='4 * eligible thinking inputs assigned to this pair',
              output_tokens={'thinking':16384}, thinking_enabled=True,
              tensor_parallel_size=2, max_model_len=81920, rope=rope_for_length(65536)), indent=2))
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
            elif args.command == 'report':
                report(root, verify(cfg, root))
            else:
                run(cfg, root, args)


if __name__ == '__main__':
    main()
