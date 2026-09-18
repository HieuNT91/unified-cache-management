#!/usr/bin/env python3
"""One resident engine for an ordered, single-configuration sample manifest."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import uuid
if os.environ.get('SELECTOR_UCM_ROOT'):
    sys.path.insert(0, os.environ['SELECTOR_UCM_ROOT'])
from prophetkv_common import sha, dump, load_sample, cache_snapshot, generate, score
from prophetkv_gpu import check, GPU_UUIDS
from cacheblend_prophetkv import build, setup, set_request, worker_probe, audit_capture, TIMING
from cacheblend_ruler import wait_for_cache
from lifecycle import seed_value, delete_retired_files


def retire_rpc(worker):
    from ucm.sparse.prophetkv.lifecycle import retire
    return retire(worker)


def barrier(llm, request_id, cached):
    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('Attempt retirement while requests are live')
    workers = llm.collective_rpc(retire_rpc)
    if not all(w['quiescent'] and w['transfers']['pending'] == 0 for w in workers):
        raise RuntimeError('Worker not quiescent')
    scheduler = None
    if cached:
        scheduler = json.loads(Path(os.environ['PROPHETKV_SCHEDULER_RECEIPT']).read_text())
        if scheduler != dict(request_id=request_id, requests_blend_meta=0, requests_meta=0):
            raise RuntimeError('Scheduler did not acknowledge request retirement')
    return dict(workers=workers, scheduler=scheduler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--engine-start-count', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    gpu = int(os.environ['SELECTOR_PHYSICAL_GPU']); check(gpu)
    if Path.cwd() != Path('/tmp') or 'PYTHONPATH' in os.environ or os.environ['CUDA_VISIBLE_DEVICES'] != GPU_UUIDS[gpu]:
        raise RuntimeError('Unsafe launch environment')
    p = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    if not manifest: raise ValueError('Empty manifest')
    if {n:version(n) for n in p['runtime_versions']} != p['runtime_versions']:
        raise RuntimeError('Runtime changed')
    for path, digest in p['runtime_file_hashes'].items():
        if sha(path) != digest: raise RuntimeError('Patched runtime changed')
    for name, digest in p['sources'].items():
        if sha(Path(__file__).parent / name) != digest: raise RuntimeError('Worker sources changed')
    cached = args.case != 'baseline'
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if list((args.cache_dir/'kv').glob('*/*')): raise RuntimeError('Session cache not empty')
    sample = load_sample(manifest[0]['input_path'])
    t = time.perf_counter(); llm = build(args, sample, p); load_seconds = time.perf_counter()-t
    llm.collective_rpc(setup)
    imports = llm.collective_rpc(worker_probe)
    session_id = args.session.stem
    namespace = uuid.uuid4().hex
    rid = namespace + ':engine-warmup'
    t = time.perf_counter()
    generate(llm.llm_engine, [100,200,300,400,500,600,700,800]*16, 4, rid)
    warmup_seconds = time.perf_counter()-t
    warmup_barrier = barrier(llm, rid, cached)
    # The same store stays open for the session; only closed operations' files retire.
    delete_retired_files(args.cache_dir, cache_snapshot(args.cache_dir))
    session = dict(session_id=session_id, case=args.case, physical_gpu=gpu,
        engine_start_count=args.engine_start_count, configuration_phase=args.phase,
        max_model_len=p['max_model_len'], model_load_seconds=load_seconds,
        warmup_seconds=warmup_seconds, engine_initializations=1,
        warmup_barrier=warmup_barrier, started_at=time.time(), worker_imports=imports,
        protocol_sha256=sha(args.protocol), manifest_sha256=sha(args.manifest), state='running')
    dump(args.session, session)
    print('SESSION_READY '+session_id, flush=True)
    for index, entry in enumerate(manifest):
        sample = load_sample(entry['input_path'])
        if sample['physical_gpu'] != gpu: raise RuntimeError('Sample affinity changed')
        output_path = Path(entry['output']); output_path.parent.mkdir(parents=True,exist_ok=True)
        if output_path.exists(): raise RuntimeError('Refusing to overwrite completed request')
        ids, bounds = sample['token_ids'], sample['boundaries']
        chunks = [ids[a:z] for a,z in zip(bounds[:-2],bounds[1:-1])]
        namespace = uuid.uuid4().hex
        rid = namespace + ':measured'
        verification = None
        print('REQUEST_BEGIN '+rid, flush=True)
        t = time.perf_counter()
        if cached:
            for chunk_index, chunk in enumerate(chunks):
                generate(llm.llm_engine, chunk, 1, namespace+f':populate-{chunk_index}')
                print(f'cache_population namespace={namespace} chunk={chunk_index+1}/{len(chunks)}',flush=True)
            verification = wait_for_cache(SimpleNamespace(model_path=Path(p['model']),
                tensor_parallel_size=1, cache_dir=args.cache_dir,
                cache_ready_timeout_seconds=600, hash_seed=seed_value(namespace)), chunks)
            verification['namespace'] = namespace
        cache_build_seconds = time.perf_counter()-t
        def metadata(request_id):
            if cached:
                llm.collective_rpc(set_request, kwargs=dict(request_id=request_id,
                    boundaries=bounds, question_positions=sample['query']['positions']))
        prime_id = namespace + ':prime'
        t = time.perf_counter(); metadata(prime_id)
        generate(llm.llm_engine, ids, 1, prime_id)
        prime_seconds = time.perf_counter()-t
        llm.collective_rpc(worker_probe, kwargs={'drain':True})
        before = cache_snapshot(args.cache_dir)
        metadata(rid)
        if args.smoke and args.case in ('baseline','prophetkv-100'):
            llm.collective_rpc(audit_capture, kwargs={'enable':True, 'suffix_tokens':sample['fresh_suffix_tokens']})
        print('MEASURE_BEGIN', flush=True)
        result, ttft, total = generate(llm.llm_engine, ids, 16 if args.smoke else 128, rid)
        print('MEASURE_END', flush=True)
        diagnostics = llm.collective_rpc(worker_probe, kwargs={'drain':True})
        if args.smoke and args.case in ('baseline','prophetkv-100'):
            llm.collective_rpc(audit_capture, kwargs={'enable':False,
                'output_path':str(output_path.with_suffix('.model-audit.pt'))})
        if cache_snapshot(args.cache_dir) != before: raise RuntimeError('Offline cache changed during measurement')
        if not cached and result.num_cached_tokens != 0: raise RuntimeError('Baseline reused tokens')
        output = result.outputs[0]; value, extracted = score(sample, output.text)
        dp = output_path.with_suffix('.diagnostics.json'); dump(dp, diagnostics)
        retirement = barrier(llm, rid, cached)
        disk_bytes = sum(x[0] for x in before.values())
        # No active requests, scheduler metadata, transfers or worker tensors remain.
        delete_retired_files(args.cache_dir, before)
        if list(args.cache_dir.rglob('*')):
            files = [f for f in args.cache_dir.rglob('*') if f.is_file()]
            if files: raise RuntimeError('Unexpected cache files after retirement')
        record = dict(protocol_sha256=sha(args.protocol), input_sha256=sha(entry['input_path']),
            sample_id=sample['id'], prompt_sha256=sample['prompt_sha256'], physical_gpu=gpu,
            gpu_uuid=GPU_UUIDS[gpu], worker_imports=imports, session_id=session_id,
            session_path=str(args.session), engine_start_count=args.engine_start_count,
            request_id=rid, namespace=namespace, configuration_phase=args.phase,
            session_request_index=index, max_model_len=p['max_model_len'], retirement=retirement,
            cache_disk_bytes=disk_bytes, cache_files_after_retirement=0,
            cache_build_seconds=cache_build_seconds, schema_version=2, case=args.case,
            smoke=args.smoke, dataset=sample['dataset'], label=sample['label'],
            context_target=sample['context_target'], chunk_size=sample['chunk_size'], source_row=sample['source_row'],
            prompt_tokens=len(ids), max_output_tokens=16 if args.smoke else 128,
            timing_source=TIMING, ttft_seconds=ttft, generation_seconds=total, prime_seconds=prime_seconds,
            prediction=output.text, output_token_ids=list(output.token_ids), output_tokens=len(output.token_ids),
            finish_reason=output.finish_reason, score=value, extracted_answer=extracted,
            references=sample['source_metadata']['references'], num_cached_tokens=result.num_cached_tokens,
            cache_verification=verification, warm_cache=True, cache_unchanged=True,
            online_mask_reused=False, diagnostics_sha256=sha(dp))
        dump(output_path, record)
        print('REQUEST_COMPLETE '+rid+' '+str(output_path), flush=True)
        # Optional real failure injection, after an accepted request; validation only.
        fail_after = os.environ.get('PROPHETKV_FAIL_AFTER')
        if fail_after and index+1 == int(fail_after):
            raise RuntimeError('Injected persistent worker failure')
    llm.llm_engine.engine_core.shutdown()
    dump(args.session, {**session, 'state':'complete', 'ended_at':time.time(), 'completed_requests':len(manifest)})
    print('SESSION_COMPLETE', flush=True)

if __name__ == '__main__': main()
