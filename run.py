#!/usr/bin/env python3
"""Prepare one immutable prompt, then run any of three methods on it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from runner.config import METHODS, VERSIONS, engine_config, validate_sample


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_model(path):
    cfg = json.loads((path / 'config.json').read_text())
    if (cfg.get('model_type'), cfg.get('num_hidden_layers'), cfg.get('hidden_size')) != ('qwen3', 64, 5120):
        raise ValueError('Use a local Qwen3-32B checkpoint')
    if cfg.get('quantization_config'):
        raise ValueError('Use the unquantized BF16 checkpoint')


def prepare(args):
    from transformers import AutoTokenizer
    from runner.cache import cacheblend_prompt
    check_model(args.model)
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    question = args.question.read_text().strip()
    if not question:
        raise ValueError('Empty question')
    messages = [{'role': 'user', 'content': args.context.read_text() + '\n\n' + question}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=args.thinking)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    start = rendered.rfind(question)
    if start < 0:
        raise ValueError('Question not preserved by chat template')
    question_indices = [i for i, (a, b) in enumerate(offsets)
                        if b > start and a < start + len(question)]
    if not question_indices:
        raise ValueError('Question token mapping is empty')
    suffix = max(256, len(ids) - question_indices[0])
    marker = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    if marker is None or tokenizer.convert_ids_to_tokens(marker) != '<|endoftext|>':
        raise ValueError('Expected Qwen3 end-of-text delimiter')
    chunks, formatted = cacheblend_prompt(ids, tokenizer, marker, 4095, suffix)
    bounds = [0]
    for chunk in chunks:
        bounds.append(bounds[-1] + len(chunk))
    shift = bounds[-1] - (len(ids) - suffix)
    bounds.append(len(formatted))
    sample = dict(token_ids=formatted, boundaries=bounds,
        question_positions=[i + shift for i in question_indices],
        thinking=args.thinking, max_output_tokens=args.max_output_tokens,
        model_config_sha256=digest(args.model / 'config.json'),
        context_sha256=digest(args.context), question_sha256=digest(args.question))
    validate_sample(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, sample)
    print(f'Prepared {len(formatted)} tokens, {len(chunks)} chunks, {suffix} fresh suffix tokens')


def generate(engine, tokens, budget, request_id, thinking=False):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    if engine.has_unfinished_requests():
        raise RuntimeError('Concurrent requests are unsupported')
    params = SamplingParams(temperature=.6 if thinking else 0., top_p=.95 if thinking else 1.,
                            top_k=20 if thinking else -1, min_p=0., seed=0, max_tokens=budget)
    started = time.perf_counter()
    engine.add_request(request_id, TokensPrompt(prompt_token_ids=tokens), params)
    first, result = None, None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.request_id != request_id:
                raise RuntimeError('Unexpected request')
            if output.outputs and output.outputs[0].token_ids and first is None:
                first = time.perf_counter() - started
            if output.finished:
                result = output
    if result is None or first is None:
        raise RuntimeError('Generation did not return a token-bearing result')
    return result, first, time.perf_counter() - started


def verify_diagnostics(workers, sample, method, ratio, tp, expected_layers):
    """Independent CPU replay of each TP rank's global scores and layer sets."""
    import math
    if sorted(w['rank'] for w in workers) != list(range(tp)):
        raise RuntimeError('Missing tensor-parallel rank')
    common = None
    for worker in workers:
        events = worker['diagnostics']
        if method == 'baseline':
            if events:
                raise RuntimeError('Baseline used a sparse method')
            continue
        selections = [e for e in events if e['kind'] == 'prophetkv_selection']
        if len(selections) != 1 or any(e['kind'] == 'cache_miss' for e in events):
            raise RuntimeError('Expected exactly one successful cache selection')
        event = selections[0]
        scores = event['scores']
        prefix, end = sample['boundaries'][1], sample['boundaries'][-2]
        if len(scores) != end - prefix or not all(math.isfinite(s) for s in scores):
            raise RuntimeError('Invalid selection scores')
        count = math.floor(len(scores) * ratio)
        reference = sorted(i + prefix for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:count])
        if (event['selected_positions'] != reference or event['scoring_layers'] != list(expected_layers)
                or event['fusion'] != 'mean_layers_fp32' or event['alignment_count'] != 64):
            raise RuntimeError('Selection replay, mean or cache alignment mismatch')
        if common is not None and common != reference:
            raise RuntimeError('TP ranks disagree on selected positions')
        common = reference
        layers = [e for e in events if e['kind'] == 'layer_counts']
        expected = {f'model.layers.{i}.self_attn.attn' for i in range(64)}
        if len(layers) != 64 or {e['layer'] for e in layers} != expected or not all(
                e['selected_set_verified'] for e in layers):
            raise RuntimeError('Missing all-layer selected-set verification')


def run(args):
    from importlib.metadata import version
    check_model(args.model)
    sample = validate_sample(json.loads(args.input.read_text()))
    if sample['model_config_sha256'] != digest(args.model / 'config.json'):
        raise ValueError('Input was prepared with a different model configuration')
    cfg = engine_config(args.model, args.method, args.ratio, args.layers, args.num_layers,
        args.tp, args.memory, args.output.resolve() / 'cache',
        sample['token_ids'][sample['boundaries'][1] - 1])
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != args.tp or len(set(devices)) != args.tp or any(
            not d.startswith('GPU-') for d in devices):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly --tp distinct GPU UUIDs')
    for package, expected in VERSIONS.items():
        if version(package).split('+')[0] != expected:
            raise RuntimeError(f'{package} must be {expected}')
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output.resolve() / 'cache'
    cache.mkdir()
    os.environ['PROPHETKV_SCHEDULER_RECEIPT'] = str(args.output.resolve() / 'scheduler.json')
    from ucm.integration.vllm.patch.apply_patch import apply_all_patches
    apply_all_patches()
    from vllm import LLM
    from vllm.config import KVTransferConfig
    from runner.worker import setup, arm, drain, retire
    from runner.cache import wait_for_cache
    from ucm.sparse.prophetkv.lifecycle import seed_value, delete_retired_files
    from ucm.sparse.prophetkv.layers import resolve_layers
    dump(args.output / 'config.json', cfg)
    if 'kv_transfer_config' in cfg:
        cfg['kv_transfer_config'] = KVTransferConfig(**cfg['kv_transfer_config'])
    started = time.perf_counter()
    llm = LLM(**cfg)
    timings = dict(model_load_seconds=time.perf_counter() - started)
    cached = args.method != 'baseline'
    namespace = uuid.uuid4().hex
    bounds, ids = sample['boundaries'], sample['token_ids']
    chunks = [ids[a:b] for a, b in zip(bounds[:-2], bounds[1:-1])]
    readiness_args = SimpleNamespace(model_path=args.model, tensor_parallel_size=args.tp,
        cache_dir=cache, cache_ready_timeout_seconds=600, hash_seed=seed_value(namespace))
    try:
        setup_receipts = llm.collective_rpc(setup)
        started = time.perf_counter()
        warmup = [100, 200, 300, 400] * 32
        generate(llm.llm_engine, warmup, 1, namespace + ':warmup')
        if cached:
            wait_for_cache(readiness_args, [warmup])
        llm.collective_rpc(retire)
        timings['warmup_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        readiness = None
        if cached:
            for i, chunk in enumerate(chunks):
                generate(llm.llm_engine, chunk, 1, namespace + f':populate-{i}')
            readiness = wait_for_cache(readiness_args, chunks)
        timings['cache_build_and_readiness_seconds'] = time.perf_counter() - started

        def metadata(rid):
            if cached:
                llm.collective_rpc(arm, kwargs=dict(request_id=rid, boundaries=bounds,
                    question_positions=sample['question_positions']))

        started = time.perf_counter()
        metadata(namespace + ':prime')
        generate(llm.llm_engine, ids, 1, namespace + ':prime')
        llm.collective_rpc(drain)
        timings['priming_seconds'] = time.perf_counter() - started
        before = {str(p.relative_to(cache)): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in (cache / 'kv').glob('*/*') if p.is_file()}
        metadata(namespace + ':measured')
        result, ttft, elapsed = generate(llm.llm_engine, ids, sample['max_output_tokens'],
                                         namespace + ':measured', sample['thinking'])
        timings.update(ttft_seconds=ttft, generation_seconds=elapsed)
        started = time.perf_counter()
        diagnostics = llm.collective_rpc(drain)
        scoring_layers = resolve_layers(args.method, args.layers, args.num_layers) if cached else ()
        verify_diagnostics(diagnostics, sample, args.method, args.ratio, args.tp, scoring_layers)
        if not cached and result.num_cached_tokens != 0:
            raise RuntimeError('Baseline reused cached tokens')
        after = {str(p.relative_to(cache)): (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in (cache / 'kv').glob('*/*') if p.is_file()}
        if before != after:
            raise RuntimeError('Offline cache changed during measured generation')
        dump(args.output / 'diagnostics.json', diagnostics)
        timings['validation_and_export_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        retirement = llm.collective_rpc(retire)
        if len(retirement) != args.tp or not all(r['quiescent'] and r['transfers']['pending'] == 0
                and r['request_bookkeeping'] == 0 for r in retirement):
            raise RuntimeError('Request retirement failed')
        if cached:
            receipt = json.loads((args.output / 'scheduler.json').read_text())
            if receipt != dict(request_id=namespace + ':measured', requests_blend_meta=0, requests_meta=0):
                raise RuntimeError('Scheduler request not retired')
            delete_retired_files(cache, before)
        timings['retirement_seconds'] = time.perf_counter() - started
        output = result.outputs[0]
        record = dict(method=args.method, ratio=args.ratio if cached else None,
            scoring_layers=list(scoring_layers), input_sha256=digest(args.input),
            model=str(args.model), gpu_uuids=devices, runtime_versions=VERSIONS,
            prompt_tokens=len(ids), output_token_ids=list(output.token_ids), prediction=output.text,
            max_output_tokens=sample['max_output_tokens'], thinking=sample['thinking'],
            finish_reason=output.finish_reason, num_cached_tokens=result.num_cached_tokens,
            timings=timings, setup=setup_receipts, readiness=readiness, retirement=retirement)
    finally:
        llm.llm_engine.engine_core.shutdown()
    dump(args.output / 'result.json', record)
    print(f'Wrote {args.output / "result.json"}; TTFT {ttft:.3f}s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='CPU tokenization; no truncation')
    prep.add_argument('--model', type=Path, required=True)
    prep.add_argument('--context', type=Path, required=True)
    prep.add_argument('--question', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument('--max-output-tokens', type=int, default=16384)
    prep.set_defaults(func=prepare)
    launch = sub.add_parser('run')
    launch.add_argument('--model', type=Path, required=True)
    launch.add_argument('--input', type=Path, required=True)
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('--method', choices=METHODS, required=True)
    launch.add_argument('--ratio', type=float, default=.2)
    layers = launch.add_mutually_exclusive_group()
    layers.add_argument('--layers', nargs='+', type=int)
    layers.add_argument('--num-layers', type=int)
    launch.add_argument('--tp', type=int, default=4)
    launch.add_argument('--memory', type=float, default=.9)
    launch.add_argument('--dry-run', action='store_true', help='Print config without loading GPU libraries')
    launch.set_defaults(func=run)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.func(args)


if __name__ == '__main__':
    main()
