"""Prompt-major temporary KV sweeps. No collection-wide KV preparation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import uuid

from runner.config import VERSIONS, engine_config, validate_sample
from runner.reporting import AggregationReporter, OutputAnalyzer, evaluation_metadata, score_answer
from runner.setups import (atomic_json, file_hash, fingerprint, manifest_entries,
                           check_environment, start_engine, retirement, runtime_identity)


def configurations(percentages=(1,5,10,15,20,30), layers=(11,12,13,14,15)):
    if not percentages or len(set(percentages)) != len(percentages) or any(
            type(p) is not int or not 0 < p <= 100 for p in percentages):
        raise ValueError("Ratios must be distinct integer percentages in [1,100]")
    from ucm.sparse.prophetkv.layers import resolve_layers
    selected = list(resolve_layers("selective_prophetkv", layers))
    return [dict(name='baseline', method='baseline', ratio=None, layers=None)] + [
        dict(name=f'{name}-{pct}', method=method, ratio=pct/100, layers=layers)
        for name, method, layers in [('prophetkv', 'prophetkv', None),
            ('selective', 'selective_prophetkv', selected)]
        for pct in percentages]


class SharedReporter:
    """Serialize two CPU writers; finalize only after all engines have exited."""
    def __init__(self, output, expected, context, shard, shards):
        self.output, self.expected, self.context = Path(output), expected, context
        self.shard, self.shards = shard, shards
        self.output.mkdir(parents=True, exist_ok=True)
        self.update()

    def update(self, record=None, finished=False, error=None):
        with (self.output / '.report.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.output / 'aggregation_state.json'
            identity = fingerprint(dict(expected=self.expected, context=self.context, shards=self.shards))
            state = json.loads(path.read_text()) if path.exists() else dict(
                identity=identity, records=[], finished=[], errors={})
            if state['identity'] != identity:
                raise RuntimeError('Sweep report identity changed; use a new output directory')
            if record is not None:
                # Keep the ledger small; raw output/diagnostics live in per-prompt files.
                keys = ('prompt_id','subtask','accuracy','thinking_tokens','answer_tokens','control_tokens',
                        'output_tokens','output_cap_reached','unfinished_thinking','timings')
                state['records'].append({key:record[key] for key in keys})
            if finished:
                if self.shard in state['finished']:
                    raise RuntimeError('Duplicate engine completion')
                state['finished'].append(self.shard)
            if error is not None and self.shard not in state['finished']:
                state['errors'][str(self.shard)] = str(error)
            reporter = object.__new__(AggregationReporter)
            reporter.output, reporter.expected, reporter.context = self.output, self.expected, self.context
            reporter.records = state['records']
            complete = sorted(state['finished']) == list(range(self.shards)) and not state['errors']
            if complete:
                report = reporter.finish()  # rejects missing/duplicate/unexpected results
            else:
                report = reporter.publish('live', 'failed' if state['errors'] else 'running',
                                          state['errors'] or None)
            atomic_json(path, state)
            return report

    def accept(self, record):
        return self.update(record=record)

    def finish(self):
        return self.update(finished=True)


class PromptCache:
    """One owned namespace; verify committed TP shards before reusing or deleting."""
    def __init__(self, cache, model, tp, namespace, sample):
        from runner.identity import BlockHasher, block_keys, shard_name
        from ucm.sparse.prophetkv.lifecycle import seed_value
        self.cache, self.tp = Path(cache), tp
        self.chunks = list(dict.fromkeys(tuple(sample['token_ids'][a:b]) for a,b in
                           zip(sample['boundaries'][:-2], sample['boundaries'][1:-1])))
        self.args = SimpleNamespace(model_path=model, tensor_parallel_size=tp, cache_dir=self.cache,
            cache_ready_timeout_seconds=600, hash_seed=seed_value(namespace))
        hasher = BlockHasher(str(model), tp, 0)
        keys = {key for chunk in self.chunks for key in block_keys(hasher, chunk, self.args.hash_seed)}
        names = [shard_name(key, str(model), tp, rank) for key in keys for rank in range(tp)]
        self.files = {f'kv/{name[:8]}/{name}' for name in names}
        self.before = None

    def ready(self):
        from runner.cache import wait_for_cache
        result = wait_for_cache(self.args, self.chunks)
        self.before = self.snapshot()
        actual = {str(p.relative_to(self.cache)) for p in (self.cache/'kv').rglob('*') if p.is_file()}
        if actual != self.files:
            raise RuntimeError('Unexpected files in temporary prompt cache')
        return result

    def snapshot(self):
        result = {}
        for name in self.files:
            path = self.cache/name
            if path.is_symlink() or path.parent.is_symlink():
                raise RuntimeError('Symlink in temporary cache')
            stat = path.stat()
            result[name] = [stat.st_size, stat.st_mtime_ns]
        return result

    def unchanged(self):
        if self.before is None or self.before != self.snapshot():
            raise RuntimeError('Original KV changed during inference')

    def delete(self):
        from ucm.sparse.prophetkv.lifecycle import delete_retired_files
        self.unchanged()
        delete_retired_files(self.cache, self.files)
        return dict(deleted_shards=len(self.files), deleted_bytes=sum(s[0] for s in self.before.values()),
                    remaining_shards=0)


def load_inputs(args):
    from run import check_model
    check_model(args.model)
    rows = manifest_entries(args.manifest)
    expected, selected, hashes = [], [], []
    config_hash = file_hash(args.model/'config.json')
    for index, row in enumerate(rows):
        if 'prepared' not in row:
            raise ValueError('Sweep requires frozen prepared JSON inputs; run CPU preparation first')
        path = Path(row['prepared'])
        hashes.append(file_hash(path))
        # Evaluation metadata belongs to the manifest, not the inference input.
        evaluation = evaluation_metadata({}, row)
        expected.append(dict(prompt_id=row['id'], subtask=evaluation['subtask']))
        if index % args.shards != args.shard:
            continue
        sample = json.loads(path.read_text())
        validate_sample(sample, 4096, getattr(args, 'context_length', 114688))
        exact = getattr(args, 'exact_input_tokens', None)
        if exact is not None and len(sample['token_ids']) != exact:
            raise ValueError(f'Expected exactly {exact} formatted input tokens')
        if sample['model_config_sha256'] != config_hash:
            raise ValueError('Prepared input model configuration changed')
        selected.append((index, dict(id=row['id'], evaluation=evaluation, sha256=hashes[-1]), sample))
    if not selected:
        raise ValueError('Empty prompt shard')
    identity = fingerprint(dict(inputs=hashes, manifest=file_hash(args.manifest), model=config_hash,
                                runtime=runtime_identity(), tp=args.tp, configs=configurations(getattr(args, 'percentages', (1,5,10,15,20,30)),
                                getattr(args, 'layers', (11,12,13,14,15))),
                                context_length=getattr(args, 'context_length', 114688),
                                exact_input_tokens=getattr(args, 'exact_input_tokens', None)))
    return expected, selected, identity


def execute_phase(args, selected, configs, reporters, group, cache, devices):
    from run import generate, verify_diagnostics
    from runner.worker import setup, arm, drain, configure
    from ucm.sparse.prophetkv.layers import resolve_layers
    cached = configs[0]['method'] != 'baseline'
    sample = selected[0][2]
    cfg = engine_config(args.model, configs[0]['method'], configs[0]['ratio'] or .2,
        tp=args.tp, memory=args.memory, cache_dir=cache,
        end_token=sample['token_ids'][sample['boundaries'][1]-1])
    if cached:
        cfg['kv_transfer_config']['kv_connector_extra_config']['temporary_layouts'] = {
            f'prompt-{index}': sample['boundaries'] for index, _, sample in selected}
    phase = 'cached' if cached else 'baseline'
    atomic_json(group/f'{phase}-config.json', cfg)
    started = time.perf_counter()
    llm = start_engine(cfg)
    session = dict(model_load_seconds=time.perf_counter()-started)
    try:
        session['setup'] = llm.collective_rpc(setup)
        analyzer = OutputAnalyzer(llm.get_tokenizer())
        if not cached:
            started = time.perf_counter()
            generate(llm.llm_engine, [100,200,300,400]*32, 1, uuid.uuid4().hex+':warmup')
            retirement(llm,args.tp)
            session['warmup_seconds'] = time.perf_counter()-started
        for index, entry, sample in selected:
            namespace, tag = uuid.uuid4().hex, f'prompt-{index}'
            bounds, ids = sample['boundaries'], sample['token_ids']
            prompt_cache, construction = None, None
            if cached:
                started = time.perf_counter()
                prompt_cache = PromptCache(cache, args.model, args.tp, namespace, sample)
                for chunk_index, chunk in enumerate(prompt_cache.chunks):
                    generate(llm.llm_engine, list(chunk), 1, f'{namespace}:{tag}:populate:{chunk_index}')
                    retirement(llm,args.tp)
                readiness = prompt_cache.ready()
                construction = dict(prompt_id=entry['id'], namespace=namespace,
                    seconds=time.perf_counter()-started, readiness=readiness,
                    unique_chunks=len(prompt_cache.chunks), reused_by=[c['name'] for c in configs])
                atomic_json(group/'construction'/f'{index:06d}.json', construction)
            for config in configs:
                method, ratio = config['method'], config['ratio']
                scoring = resolve_layers(method,config['layers']) if cached else ()
                if cached:
                    replies = llm.collective_rpc(configure, kwargs=dict(method=method, ratio=ratio, layers=config['layers']))
                    wanted = dict(method=method, ratio=ratio, scoring_layers=list(scoring))
                    if sorted(r['rank'] for r in replies) != list(range(args.tp)) or any(
                            {k:v for k,v in r.items() if k != 'rank'} != wanted for r in replies):
                        raise RuntimeError('TP workers disagree on sweep configuration')

                def infer(purpose, budget, thinking=False):
                    rid = f'{namespace}:{tag}:read:{config["name"]}-{purpose}'
                    if cached:
                        llm.collective_rpc(arm, kwargs=dict(request_id=rid, boundaries=bounds,
                                                          question_positions=sample['question_positions']))
                    return rid, generate(llm.llm_engine, ids, budget, rid, thinking)

                timings = {}
                # Warm each policy once, and prime every measured request separately.
                if cached and index == selected[0][0]:
                    started = time.perf_counter()
                    infer('warmup',1)
                    retirement(llm,args.tp)
                    session.setdefault('policy_warmup_seconds',{})[config['name']] = time.perf_counter()-started
                started = time.perf_counter()
                if cached:
                    prompt_cache.unchanged()
                timings['readiness_seconds'] = time.perf_counter()-started
                started = time.perf_counter()
                infer('prime',1)
                llm.collective_rpc(drain)
                retirement(llm,args.tp)
                timings['priming_seconds'] = time.perf_counter()-started
                rid, (result,ttft,elapsed) = infer('measured',sample['max_output_tokens'],sample['thinking'])
                timings.update(ttft_seconds=ttft,generation_seconds=elapsed)
                started = time.perf_counter()
                diagnostics = llm.collective_rpc(drain)
                verify_diagnostics(diagnostics,sample,method,ratio,args.tp,scoring)
                if not cached and result.num_cached_tokens != 0:
                    raise RuntimeError('Baseline reused cached tokens')
                output_dir = args.output/config['name']/f'{index:06d}'
                output_dir.mkdir(exist_ok=False)
                atomic_json(output_dir/'diagnostics.json',diagnostics)
                timings['validation_and_export_seconds'] = time.perf_counter()-started
                started = time.perf_counter()
                retired = retirement(llm,args.tp)
                if cached:
                    receipt = json.loads((group/'scheduler.json').read_text())
                    if receipt != dict(request_id=rid,requests_blend_meta=0,requests_meta=0):
                        raise RuntimeError('Scheduler request not retired')
                    prompt_cache.unchanged()
                timings['retirement_seconds'] = time.perf_counter()-started
                output = result.outputs[0]
                started = time.perf_counter()
                lengths = analyzer.analyze(list(output.token_ids),ids,sample['thinking'])
                evaluation = entry['evaluation']
                accuracy = score_answer(lengths['answer_text'],evaluation)
                timings['scoring_seconds'] = time.perf_counter()-started
                record = dict(prompt_id=entry['id'], method=method, ratio=ratio, scoring_layers=list(scoring),
                    request_id=rid, input_sha256=entry['sha256'], model=str(args.model), gpu_uuids=devices,
                    runtime_versions=VERSIONS, prompt_tokens=len(ids), output_token_ids=list(output.token_ids),
                    prediction=output.text, max_output_tokens=sample['max_output_tokens'], thinking=sample['thinking'],
                    finish_reason=output.finish_reason, num_cached_tokens=result.num_cached_tokens,
                    timings=timings, retirement=retired, construction=construction, **lengths, **evaluation,
                    accuracy=accuracy, output_cap_reached=len(output.token_ids)>=sample['max_output_tokens'])
                atomic_json(output_dir/'result.json',record)
                reporters[config['name']].accept(record)
                print(f"{entry['id']} {config['name']}: TTFT {ttft:.3f}s",flush=True)
            if cached:
                # All policies have passed their measured validation and retirement.
                started = time.perf_counter()
                cleanup = prompt_cache.delete()
                cleanup['seconds'] = time.perf_counter()-started
                atomic_json(group/'cleanup'/f'{index:06d}.json',cleanup)
    finally:
        # Never remove cache files while an owned engine can still be using them.
        llm.llm_engine.engine_core.shutdown()
        if cached:
            shutil.rmtree(cache)
    atomic_json(group/f'{phase}-session.json',dict(**session,engine_shutdown=True))
    for config in configs:
        reporters[config['name']].finish()


def run_sweep(args):
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError('Invalid prompt shard')
    args.model, args.output = args.model.resolve(), args.output.resolve()
    expected, selected, identity = load_inputs(args)
    cache = args.cache_root.resolve()/f'sweep-{uuid.uuid4().hex}'
    configs = configurations(args.percentages, args.layers)
    if args.dry_run:
        print(json.dumps(dict(prompts=len(expected), shard_prompts=len(selected),
            shard=args.shard, configs=configs, measurements=len(selected)*len(configs),
            cache_policy='construct one prompt, reuse all cached methods, retire and delete',
            max_retained_context_tokens=max(s['boundaries'][-2] for _,_,s in selected),
            baseline=engine_config(args.model,'baseline',tp=args.tp,memory=args.memory),
            cached=engine_config(args.model,'prophetkv',tp=args.tp,memory=args.memory,
                cache_dir=cache,end_token=selected[0][2]['token_ids'][selected[0][2]['boundaries'][1]-1])),indent=2))
        return
    devices = check_environment(args.tp)
    group = args.output/f'group-{args.shard}'
    group.mkdir(parents=True,exist_ok=False)  # no ambiguous resume or duplicate group
    cache.mkdir(parents=True,exist_ok=False)
    os.environ['PROPHETKV_SCHEDULER_RECEIPT'] = str(group/'scheduler.json')
    atomic_json(group/'plan.json',dict(identity=identity,gpu_uuids=devices,cache=str(cache),
        prompt_ids=[e['id'] for _,e,_ in selected],configurations=configs,shard=args.shard,shards=args.shards))
    reporters = {c['name']:SharedReporter(args.output/c['name'],expected,
        dict(method=c['method'],ratio=c['ratio'],
             scoring_layers=(c['layers'] or list(range(64))) if c['method'] != 'baseline' else [],
             sweep_fingerprint=identity,
             cache_policy='temporary per prompt; construction excluded from TTFT'),args.shard,args.shards)
        for c in configs}
    try:
        execute_phase(args,selected,configs[:1],reporters,group,cache,devices)
        execute_phase(args,selected,configs[1:],reporters,group,cache,devices)
        atomic_json(group/'complete.json',dict(engine_shutdown=True,cache_deleted=not cache.exists(),
                    measurements=len(selected)*len(configs)))
    except BaseException as error:
        atomic_json(group/'failure.json',dict(error=str(error),cache=str(cache)))
        for reporter in reporters.values():
            # Preserve any earlier phase's already-completed report.
            if not (reporter.output/'final_aggregation.json').exists():
                reporter.update(error=error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--tp',type=int,default=4)
    parser.add_argument('--memory',type=float,default=.9)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--percentages',type=int,nargs='+',default=[1,5,10,15,20,30])
    parser.add_argument('--layers',type=int,nargs='+',default=[11,12,13,14,15])
    parser.add_argument('--context-length',type=int,default=114688)
    parser.add_argument('--exact-input-tokens',type=int)
    parser.add_argument('--dry-run',action='store_true')
    run_sweep(parser.parse_args())


if __name__ == '__main__':
    main()
