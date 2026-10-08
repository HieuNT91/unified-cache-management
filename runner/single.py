"""Single-input baseline, ProphetKV and router execution."""
import json
import os
import time
from types import SimpleNamespace
import uuid

from runner.config import VERSIONS, engine_config, validate_sample
from runner.layout import sample_provenance
from runner.preparation import check_model, digest, dump
from runner.generation import generate, verify_diagnostics


def run(args):
    from importlib.metadata import version
    check_model(args.model)
    sample = json.loads(args.input.read_text())
    if args.max_output_tokens is not None:
        sample['max_output_tokens'] = args.max_output_tokens
    validate_sample(sample)
    if args.method != 'baseline':
        from runner.layout import validate_layout
        validate_layout(sample['token_ids'], sample['boundaries'], sample['question_positions'], sparse=True)
    if sample['model_config_sha256'] != digest(args.model / 'config.json'):
        raise ValueError('Input was prepared with a different model configuration')
    cfg = engine_config(args.model, args.method, args.ratio, args.layers, args.num_layers,
        args.tp, args.memory, args.output.resolve() / 'cache',
        None,
        router_policy=args.router_policy, router_id=args.router_id)
    if args.method == 'router':
        from runner.router_measure import validate_profile
        validate_profile(sample,args.tp)
        from runner.layout import validate_policy_protocol
        from runner.router_policy import load_policy
        validate_policy_protocol(load_policy(args.router_policy),sample)
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
        generate(llm.llm_engine, warmup, 1, namespace + ':warmup', phase='populate' if cached else None)
        if cached:
            wait_for_cache(readiness_args, [warmup])
        llm.collective_rpc(retire)
        timings['warmup_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        readiness = None
        if cached:
            for i, chunk in enumerate(chunks):
                generate(llm.llm_engine, chunk, 1, namespace + f':populate-{i}', phase='populate')
                llm.collective_rpc(retire)
            readiness = wait_for_cache(readiness_args, chunks)
        timings['cache_build_and_readiness_seconds'] = time.perf_counter() - started

        def metadata(rid):
            if cached:
                llm.collective_rpc(arm, kwargs=dict(request_id=rid, boundaries=bounds,
                    question_positions=sample['question_positions']))

        started = time.perf_counter()
        metadata(namespace + ':prime')
        generate(llm.llm_engine, ids, 1, namespace + ':prime', sample=sample)
        llm.collective_rpc(drain)
        llm.collective_rpc(retire)
        timings['priming_seconds'] = time.perf_counter() - started
        before = {str(p.relative_to(cache)): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in (cache / 'kv').glob('*/*') if p.is_file()}
        routing = None
        if args.method == 'router':
            from runner.router_measure import measure
            from runner.router_policy import load_policy
            def unchanged():
                current = {str(p.relative_to(cache)): (p.stat().st_size,p.stat().st_mtime_ns)
                           for p in (cache/'kv').glob('*/*') if p.is_file()}
                if current != before:
                    raise RuntimeError('Offline KV changed during routing')
            result, ttft, elapsed, routing = measure(llm,sample,namespace+':measured',
                load_policy(args.router_policy),args.router_id,args.output/'routing',args.tp,unchanged,
                scheduler_path=args.output/'scheduler.json')
        else:
            metadata(namespace + ':measured')
            result, ttft, elapsed = generate(llm.llm_engine, ids, sample['max_output_tokens'],
                                             namespace + ':measured', sample['thinking'], sample=sample)
        timings.update(ttft_seconds=ttft, generation_seconds=elapsed)
        started = time.perf_counter()
        diagnostics = llm.collective_rpc(drain)
        scoring_layers = resolve_layers(args.method, args.layers, args.num_layers) if cached else ()
        if routing:
            from runner.router_measure import validate_answer
            validate_answer(llm,diagnostics,result,sample,routing,args.tp)
        else:
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
            if receipt != dict(request_id=routing['answer_request_id'] if routing else namespace + ':measured', requests_blend_meta=0, requests_meta=0):
                raise RuntimeError('Scheduler request not retired')
            delete_retired_files(cache, before)
        timings['retirement_seconds'] = time.perf_counter() - started
        output = result.outputs[0]
        from runner.reporting import OutputAnalyzer, evaluation_metadata, score_answer, scoring_text
        evaluation=evaluation_metadata(sample)
        output_metrics=OutputAnalyzer(llm.get_tokenizer()).analyze(list(output.token_ids),ids,sample['thinking'])
        accuracy=score_answer(scoring_text(output.text,output_metrics,evaluation),evaluation)
        record = dict(**sample_provenance(sample),method=args.method, ratio=args.ratio if cached else None,
            scoring_layers=list(scoring_layers), input_sha256=digest(args.input),
            model=str(args.model), gpu_uuids=devices, runtime_versions=VERSIONS,
            prompt_tokens=len(ids), output_token_ids=list(output.token_ids), prediction=output.text,
            max_output_tokens=sample['max_output_tokens'], thinking=sample['thinking'],
            finish_reason=output.finish_reason, num_cached_tokens=result.num_cached_tokens,
            timings=timings, setup=setup_receipts, readiness=readiness, retirement=retirement, routing=routing,
            **evaluation, **output_metrics, accuracy=accuracy, output_cap_reached=len(output.token_ids)>=sample['max_output_tokens'])
    finally:
        llm.llm_engine.engine_core.shutdown()
    dump(args.output / 'result.json', record)
    print(f'Wrote {args.output / "result.json"}; TTFT {ttft:.3f}s')
