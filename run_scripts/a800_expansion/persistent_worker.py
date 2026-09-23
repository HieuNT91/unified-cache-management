#!/usr/bin/env python3
"""One TP=2 engine for every prompt in one method/RoPE configuration.

Adapted from benchmarks/prophetkv_persistent in the user's original checkout.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

if os.environ.get('SELECTOR_UCM_ROOT'):
    sys.path.insert(0, os.environ['SELECTOR_UCM_ROOT'])
from common import CASES, CONTROLS, ENGINE_POLICY, engine_group, model_limit, verify_gpu_visibility
from prophetkv_common import dump, load_sample, cache_snapshot, generate, score
from worker import build, setup, set_request, worker_probe, audit_capture, TIMING
from cacheblend_ruler import wait_for_cache
from lifecycle import seed_value, delete_retired_files


def retire_rpc(worker):
    from ucm.sparse.prophetkv.lifecycle import retire
    return retire(worker)


def barrier(llm, request_id, cached, tp):
    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('Attempt retirement while requests are live')
    workers = llm.collective_rpc(retire_rpc)
    if sorted(w['rank'] for w in workers) != list(range(tp)) or not all(
            w['quiescent'] and w['transfers']['pending'] == 0 and w['request_bookkeeping'] == 0
            for w in workers):
        raise RuntimeError('Every TP rank must acknowledge quiescence')
    scheduler = None
    if cached:
        scheduler = json.loads(Path(os.environ['PROPHETKV_SCHEDULER_RECEIPT']).read_text())
        if scheduler != dict(request_id=request_id, requests_blend_meta=0, requests_meta=0):
            raise RuntimeError('Scheduler did not acknowledge request retirement')
    return dict(workers=workers, scheduler=scheduler)


def run_session(args, p, manifest):
    if not manifest:
        raise ValueError('Empty manifest')
    cached = args.case != 'baseline'
    args.persistent = True
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if list((args.cache_dir / 'kv').glob('*/*')):
        raise RuntimeError('Session cache not empty')
    first = load_sample(manifest[0]['input_path'])
    group = engine_group(first['context_target'])
    limit = model_limit(p, first)
    t = time.perf_counter()
    llm = build(args, first, p)
    load_seconds = time.perf_counter() - t
    try:
        setup_receipts = llm.collective_rpc(setup)
        imports = llm.collective_rpc(worker_probe)
        print('verified_imports ' + json.dumps(imports), flush=True)
        rid = uuid.uuid4().hex + ':engine-warmup'
        t = time.perf_counter()
        generate(llm.llm_engine, [100,200,300,400,500,600,700,800] * 16, 4, rid)
        warmup_seconds = time.perf_counter() - t
        warmup_barrier = barrier(llm, rid, cached, p['tensor_parallel_size'])
        warmup_readiness = None
        if cached:
            warmup_readiness = wait_for_cache(SimpleNamespace(model_path=Path(p['model']),
                tensor_parallel_size=p['tensor_parallel_size'], cache_dir=args.cache_dir,
                cache_ready_timeout_seconds=600, hash_seed=seed_value(rid.split(':')[0])),
                [[100,200,300,400,500,600,700,800] * 16])
        delete_retired_files(args.cache_dir, cache_snapshot(args.cache_dir))
        session = dict(session_id=args.session.stem, case=args.case, engine_policy=ENGINE_POLICY,
            gpu_devices=p['gpu_devices'], tensor_parallel_size=p['tensor_parallel_size'],
            engine_initializations=1, max_model_len=limit, rope_group=group,
            model_load_seconds=load_seconds, warmup_seconds=warmup_seconds,
            warmup_barrier=warmup_barrier, warmup_cache_readiness=warmup_readiness,
            setup_receipts=setup_receipts, started_at=time.time(), state='running')
        dump(args.session, session)
        print('SESSION_READY ' + args.session.stem, flush=True)
        for index, entry in enumerate(manifest):
            sample = load_sample(entry['input_path'])
            if engine_group(sample['context_target']) != group or sample['tokens'] + sample['max_output_tokens'] > limit:
                raise ValueError('Sample is incompatible with resident engine configuration')
            ids, bounds = sample['token_ids'], sample['boundaries']
            if ids[bounds[1] - 1] != first['token_ids'][first['boundaries'][1] - 1]:
                raise ValueError('Chunk delimiter changed within session')
            output_path = Path(entry['output'])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                raise RuntimeError('Refusing to overwrite completed request')
            namespace = uuid.uuid4().hex
            rid = namespace + ':measured'
            print('REQUEST_BEGIN ' + rid, flush=True)
            chunks = [ids[a:z] for a, z in zip(bounds[:-2], bounds[1:-1])]
            verification = None
            t = time.perf_counter()
            if cached:
                for i, chunk in enumerate(chunks):
                    generate(llm.llm_engine, chunk, 1, namespace + f':populate-{i}')
                    print(f'cache_population namespace={namespace} chunk={i+1}/{len(chunks)}', flush=True)
                verification = wait_for_cache(SimpleNamespace(model_path=Path(p['model']),
                    tensor_parallel_size=p['tensor_parallel_size'], cache_dir=args.cache_dir,
                    cache_ready_timeout_seconds=600, hash_seed=seed_value(namespace)), chunks)
                verification['namespace'] = namespace
            build_seconds = time.perf_counter() - t

            def metadata(request_id):
                if cached:
                    llm.collective_rpc(set_request, kwargs=dict(request_id=request_id,
                        boundaries=bounds, question_positions=sample['query']['positions']))

            prime_id = namespace + ':prime'
            t = time.perf_counter()
            metadata(prime_id)
            generate(llm.llm_engine, ids, 1, prime_id)
            prime_seconds = time.perf_counter() - t
            llm.collective_rpc(worker_probe, kwargs={'drain': True})
            before = cache_snapshot(args.cache_dir)
            metadata(rid)
            if args.smoke and args.case in ('baseline', 'prophetkv-100'):
                llm.collective_rpc(audit_capture, kwargs=dict(enable=True, suffix_tokens=sample['fresh_suffix_tokens']))
            print('MEASURE_BEGIN', flush=True)
            result, ttft, total = generate(llm.llm_engine, ids, sample['max_output_tokens'], rid, thinking=sample['thinking_enabled'])
            print('MEASURE_END', flush=True)
            export_started = time.perf_counter()
            diagnostics = llm.collective_rpc(worker_probe, kwargs={'drain': True})
            if args.smoke and args.case in ('baseline', 'prophetkv-100'):
                receipts = llm.collective_rpc(audit_capture, kwargs=dict(enable=False,
                    output_path=str(output_path.with_suffix('.model-audit.pt'))))
                if any(r['layers'] != p['num_layers'] or not Path(r['path']).is_file() for r in receipts):
                    raise RuntimeError('Incomplete model audit')
            if cache_snapshot(args.cache_dir) != before:
                raise RuntimeError('Offline cache changed during measurement')
            if not cached and result.num_cached_tokens != 0:
                raise RuntimeError('Baseline reused tokens')
            out = result.outputs[0]
            value, extracted = score(sample, out.text)
            dump(output_path.with_suffix('.diagnostics.json'), diagnostics)
            artifact_export_seconds = time.perf_counter() - export_started
            retirement_started = time.perf_counter()
            retirement = barrier(llm, rid, cached, p['tensor_parallel_size'])
            delete_retired_files(args.cache_dir, before)
            if cache_snapshot(args.cache_dir):
                raise RuntimeError('Cache files remain after retirement')
            retirement_seconds = time.perf_counter() - retirement_started
            from prepare import sha
            record = dict(protocol_sha256=sha(args.protocol), input_sha256=sha(entry['input_path']),
                thinking_enabled=sample['thinking_enabled'],
                artifact_export_seconds=artifact_export_seconds, retirement_seconds=retirement_seconds,
                cache_readiness_seconds=verification['readiness_wait_seconds'] if verification else 0.,
                method='baseline' if not cached else 'prophetkv_with_expansion',
                parameters=p['case_parameters'].get(args.case), sample_id=sample['id'], gpu_devices=p['gpu_devices'],
                tensor_parallel_size=p['tensor_parallel_size'], worker_imports=imports,
                engine_policy=ENGINE_POLICY, session_id=args.session.stem, session_path=str(args.session),
                session_request_index=index, request_id=rid, namespace=namespace,
                max_model_len=limit, rope_group=group, retirement=retirement,
                cache_disk_bytes=sum(x[0] for x in before.values()), cache_files_after_retirement=0,
                cache_build_seconds=build_seconds, schema_version=2, case=args.case, smoke=args.smoke,
                dataset=sample['dataset'], label=sample['label'], context_target=sample['context_target'],
                chunk_size=sample['chunk_size'], source_row=sample['source_row'],
                prompt_tokens=len(ids), max_output_tokens=sample['max_output_tokens'],
                timing_source=TIMING, ttft_seconds=ttft, generation_seconds=total,
                prime_seconds=prime_seconds, prediction=out.text, output_token_ids=list(out.token_ids),
                output_tokens=len(out.token_ids), finish_reason=out.finish_reason, score=value,
                extracted_answer=extracted, references=sample['source_metadata']['references'],
                num_cached_tokens=result.num_cached_tokens, cache_verification=verification,
                warm_cache=True, cache_unchanged=True, online_mask_reused=False)
            dump(output_path, record)
            print('REQUEST_COMPLETE ' + rid + ' ' + str(output_path), flush=True)
            print(f'result case={args.case} sample={sample["id"]} score={value} ttft={ttft:.3f}', flush=True)
            fail_after = os.environ.get('PROPHETKV_FAIL_AFTER')
            if fail_after and index + 1 == int(fail_after):
                raise RuntimeError('Injected persistent worker failure')
    finally:
        llm.llm_engine.engine_core.shutdown()
    dump(args.session, session | dict(state='complete', completed_requests=len(manifest), ended_at=time.time()))
    print('SESSION_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--case', choices=(*CASES, *CONTROLS), required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        raise ValueError('Separate qualification/smoke is disabled')
    verify_gpu_visibility()
    if Path.cwd() != Path('/tmp') or 'PYTHONPATH' in os.environ:
        raise RuntimeError('Unsafe launch environment')
    p = json.loads(args.protocol.read_text())
    if p.get('thinking_enabled') is not False or p.get('output_tokens') != {'non_thinking':256}:
        raise ValueError('Protocol must be non-thinking only, with a 256-token output cap')
    if {n: version(n) for n in p['runtime_versions']} != p['runtime_versions']:
        raise RuntimeError('Runtime changed')
    from prepare import verify_frozen
    verify_frozen(args.protocol.parent, p)
    run_session(args, p, json.loads(args.manifest.read_text()))


if __name__ == '__main__':
    main()
