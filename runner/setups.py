"""Persistent prompt collections. CPU metadata code never imports a GPU runtime."""
from runner.layout import sample_provenance
from runner.reporting import scoring_text
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from runner.config import ROPE, VERSIONS, WINDOW, engine_config, validate_preparation, validate_sample
from runner.identity import CACHE_FORMAT, HASH_PROTOCOL, BlockHasher, block_keys, setup_seed, shard_name
from runner.reporting import EVALUATION_FIELDS, evaluation_metadata

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def setup_lock(root, build=False):
    root = Path(root)
    if build:
        root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if build else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Setup is in use; retry after its builder/readers exit') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def assert_no_build_workers(root):
    for path in (root / 'owners').glob('*.json'):
        owner = json.loads(path.read_text())
        try:
            # starttime survives PID reuse checks; zombies cannot touch files.
            tail = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        if tail[0] != 'Z' and tail[19] == owner['starttime']:
            raise RuntimeError('A prior setup worker is still alive; wait for its exit before reusing setup')


def runtime_identity():
    paths = [ROOT / 'run.py', ROOT / 'run.sh']
    for directory in ('runner', 'ucm'):
        paths.extend(sorted((ROOT / directory).rglob('*.py')))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def model_files(model):
    return sorted(p for p in model.rglob('*') if p.is_file() and p.suffix in
                  ('.json', '.safetensors', '.bin', '.model', '.tiktoken', '.jinja', '.txt'))


def model_stamp(model):
    result = {}
    for path in model_files(model):
        stat = path.stat()
        result[str(path.relative_to(model))] = [stat.st_size, stat.st_mtime_ns,
                                              stat.st_ctime_ns, stat.st_ino, stat.st_dev]
    return result


def model_identity(model, cached=None, stamp=None):
    from runner.preparation import check_model
    check_model(model)
    files = model_files(model)
    if not any(p.suffix in ('.safetensors', '.bin') for p in files):
        raise ValueError('Checkpoint has no local weights')
    if not any(p.name in ('tokenizer.json', 'tokenizer.model') for p in files):
        raise ValueError('Checkpoint has no local tokenizer')
    before = model_stamp(model)
    if cached is not None and stamp == before:
        return cached
    result = {str(p.relative_to(model)): file_hash(p) for p in files}
    if before != model_stamp(model):
        raise RuntimeError('Checkpoint changed while fingerprinting')
    return result


def manifest_entries(path):
    """Resolve paths once; IDs need not be valid filesystem names."""
    entries, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_id = row.get('id')
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in seen:
            raise ValueError('Manifest requires unique, nonempty string IDs')
        seen.add(prompt_id)
        input_fields = set(row) - EVALUATION_FIELDS
        if input_fields == {'id', 'prepared'}:
            fields = ('prepared',)
        elif input_fields == {'id', 'context', 'question'}:
            fields = ('context', 'question')
        else:
            raise ValueError('Each row needs id and either prepared or context/question paths')
        entries.append(dict(id=prompt_id, **{key: (path.parent / row[key]).resolve() for key in fields},
                            **{key: row[key] for key in EVALUATION_FIELDS if key in row}))
    if not entries:
        raise ValueError('Manifest is empty')
    return entries


def source_identity(entries):
    result = []
    for row in entries:
        if 'prepared' in row:
            sample = json.loads(row['prepared'].read_text())
            # Generation caps never affect context KV identity.
            value = {k: v for k, v in sample.items() if k not in
                     {'max_output_tokens', 'subtask', 'task', 'references', 'answers', 'scoring', 'evaluation'}}
            result.append(dict(id=row['id'], prepared=fingerprint(value)))
        else:
            result.append(dict(id=row['id'], context=file_hash(row['context']),
                               question=file_hash(row['question'])))
    return result


def preparation_identity(args, entries, previous=None):
    validate_preparation(args.kv_chunk_size, args.context_length)
    # Validate TP/memory with the same public config path before any engine.
    engine_config(args.model, 'baseline', tp=args.tp, memory=args.memory)
    return dict(model=model_identity(args.model, previous['preparation']['model'] if previous else None,
                                    previous.get('model_stamp') if previous else None), sources=source_identity(entries),
        runtime=runtime_identity(), versions=VERSIONS, cache_format=CACHE_FORMAT,
        hash_protocol=HASH_PROTOCOL, tp=args.tp, dtype='bfloat16', rope=ROPE,
        window=WINDOW, kv_chunk_size=args.kv_chunk_size, context_length=args.context_length,
        suffix_policy='rpkv-original-question-min256-v1', thinking=args.thinking,
        block_size=64, prefill_budget=16384)


def sample_identity(sample):
    return {k: sample[k] for k in ('token_ids', 'boundaries', 'question_positions', 'thinking', 'prompt_protocol', 'token_sha256')}


def chunks_of(sample):
    ids, bounds = sample['token_ids'], sample['boundaries']
    return [ids[a:b] for a, b in zip(bounds[:-2], bounds[1:-1])]


def bytes_per_shard(model, tp):
    cfg = json.loads((model / 'config.json').read_text())
    heads = cfg.get('num_key_value_heads', cfg['num_attention_heads'])
    dim = cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])
    return 64 * max(1, heads // tp) * dim * 2 * 2 * cfg['num_hidden_layers']


def inventory_for(samples, identity, tp, size):
    inventory = dict(bytes_per_shard=size, chunks={}, blocks={})
    hasher = BlockHasher(identity, tp, 0, True)
    for sample in samples:
        for tokens in chunks_of(sample):
            cid = fingerprint(tokens)
            if cid in inventory['chunks']:
                continue
            keys = block_keys(hasher, tokens, setup_seed(identity))
            inventory['chunks'][cid] = dict(block_keys=[key.hex() for key in keys])
            for key in keys:
                names = [shard_name(key, identity, tp, rank, True) for rank in range(tp)]
                inventory['blocks'][key.hex()] = [f'kv/{name[:8]}/{name}' for name in names]
    inventory['expected_bytes'] = len(inventory['blocks']) * tp * size
    return inventory


def required_files(inventory, chunk_ids=None):
    if chunk_ids is None:
        keys = inventory['blocks']
    else:
        keys = {key for cid in chunk_ids for key in inventory['chunks'][cid]['block_keys']}
    return sorted({name for key in keys for name in inventory['blocks'][key]})


def snapshot(cache, inventory, chunk_ids=None):
    state = {}
    for name in required_files(inventory, chunk_ids):
        path = cache / name
        if path.is_symlink() or path.parent.is_symlink():
            raise RuntimeError('Symlinked KV shard; rerun setup in a new directory')
        try:
            stat = path.stat()
        except FileNotFoundError:
            raise RuntimeError(f'Missing KV shard {name}; rerun setup') from None
        if not path.is_file() or stat.st_size != inventory['bytes_per_shard']:
            raise RuntimeError(f'Incomplete KV shard {name}; rerun setup')
        state[name] = [stat.st_size, stat.st_mtime_ns]
    return state


def incomplete_chunks(root, inventory):
    result = []
    for cid in inventory['chunks']:
        try:
            snapshot(root / 'cache', inventory, [cid])
        except RuntimeError:
            result.append(cid)
    return result


def load_frozen(root, descriptor, prompt_ids=None, max_output_tokens=None):
    indexed = {entry['id']: entry for entry in descriptor['samples']}
    if prompt_ids is None:
        prompt_ids = list(indexed)
    if not prompt_ids or len(prompt_ids) != len(set(prompt_ids)) or set(prompt_ids) - set(indexed):
        raise ValueError('prompt-ids must be distinct IDs present in the setup')
    samples = []
    for pid in prompt_ids:
        entry = indexed[pid]
        path = root / entry['file']
        if file_hash(path) != entry['sha256']:
            raise RuntimeError('Frozen sample changed; use a new setup directory')
        sample = json.loads(path.read_text())
        if fingerprint(sample_identity(sample)) != entry['identity'] or [fingerprint(t) for t in chunks_of(sample)] != entry['chunks']:
            raise RuntimeError('Frozen token identity or chunk inventory changed')
        if max_output_tokens is not None:
            sample['max_output_tokens'] = max_output_tokens
        validate_sample(sample, descriptor['preparation']['kv_chunk_size'],
                        descriptor['preparation']['context_length'])
        samples.append((entry, sample))
    return samples


def read_descriptor(root):
    descriptor = json.loads((root / 'setup.json').read_text())
    if fingerprint(descriptor['identity']) != descriptor['fingerprint']:
        raise RuntimeError('Setup fingerprint mismatch')
    if descriptor['preparation'] != descriptor['identity']['preparation']:
        raise RuntimeError('Setup preparation metadata changed')
    if descriptor['preparation']['runtime'] != runtime_identity():
        raise RuntimeError('Setup runtime is incompatible; prepare another setup directory')
    if descriptor['inventory_sha256'] != file_hash(root / 'inventory.json'):
        raise RuntimeError('Setup inventory changed')
    # Bind the on-disk sample manifest to the preparation identity.
    if [(s['id'], s['identity']) for s in descriptor['samples']] != [
            (s['id'], s['sample']) for s in descriptor['identity']['prompts']]:
        raise RuntimeError('Setup sample manifest changed')
    if 'evaluation_sha256' in descriptor and descriptor['evaluation_sha256'] != fingerprint(
            [entry['evaluation'] for entry in descriptor['samples']]):
        raise RuntimeError('Frozen evaluation metadata changed')
    return descriptor


def check_receipt(root, descriptor):
    path = root / 'complete.json'
    if not path.exists():
        raise RuntimeError('Setup is incomplete; rerun setup before experiments')
    receipt = json.loads(path.read_text())
    if receipt['fingerprint'] != descriptor['fingerprint'] or receipt['inventory_sha256'] != descriptor['inventory_sha256']:
        raise RuntimeError('Setup completion receipt is incompatible; rerun setup')
    return receipt


def check_environment(tp):
    from importlib.metadata import version
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != tp or len(set(devices)) != tp or any(not d.startswith('GPU-') for d in devices):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly --tp distinct GPU UUIDs')
    for package, expected in VERSIONS.items():
        if version(package).split('+')[0] != expected:
            raise RuntimeError(f'{package} must be {expected}')
    return devices


def start_engine(cfg):
    from ucm.integration.vllm.patch.apply_patch import apply_all_patches
    apply_all_patches()
    from vllm import LLM
    from vllm.config import KVTransferConfig
    options = copy.deepcopy(cfg)
    if 'kv_transfer_config' in options:
        options['kv_transfer_config'] = KVTransferConfig(**options['kv_transfer_config'])
    return LLM(**options)


def retirement(llm, tp):
    from runner.worker import retire
    receipts = llm.collective_rpc(retire)
    if sorted(r['rank'] for r in receipts) != list(range(tp)) or not all(
            r['quiescent'] and r['transfers']['pending'] == 0 and r['request_bookkeeping'] == 0 for r in receipts):
        raise RuntimeError('Request retirement failed')
    return receipts


def persistent_config(root, descriptor, readonly, layouts=None):
    return dict(fingerprint=descriptor['fingerprint'], tp=descriptor['preparation']['tp'],
        cache_dir=str(root / 'cache'), bytes_per_shard=descriptor['inventory']['bytes_per_shard'],
        readonly=readonly, layouts=layouts or {}, setup_root=str(root))


def construct_missing(root, descriptor, samples, missing, args, progress):
    """One engine, only incomplete independent chunks; completion follows retirement."""
    from runner.generation import generate
    from runner.cache import wait_for_cache
    check_environment(args.tp)
    from runner.worker import setup
    namespace = uuid.uuid4().hex
    cache = root / 'cache'
    cache.mkdir(exist_ok=True)
    cfg = engine_config(args.model, 'prophetkv', tp=args.tp, memory=args.memory,
        cache_dir=cache, end_token=None,
        persistent=persistent_config(root, descriptor, False))
    os.environ['PROPHETKV_SCHEDULER_RECEIPT'] = str(root / 'scheduler.json')
    started = time.perf_counter()
    llm = start_engine(cfg)
    attempt = dict(id=namespace, model_load_seconds=time.perf_counter()-started, chunks=[])
    progress['attempts'].append(attempt)
    atomic_json(root / 'progress.json', progress)
    tokens_by_id = {fingerprint(tokens): tokens for _, sample in samples for tokens in chunks_of(sample)}
    readiness = SimpleNamespace(model_path=args.model, tensor_parallel_size=args.tp,
        cache_dir=cache, cache_ready_timeout_seconds=600,
        setup_fingerprint=descriptor['fingerprint'], hash_seed=setup_seed(descriptor['fingerprint']))
    try:
        llm.collective_rpc(setup)
        for cid in missing:
            # A preceding chunk can also complete shared blocks for this chunk.
            try:
                snapshot(cache, descriptor['inventory'], [cid])
                continue
            except RuntimeError:
                pass
            started = time.perf_counter()
            generate(llm.llm_engine, tokens_by_id[cid], 1, namespace + ':populate-' + cid, phase='populate')
            generation_seconds = time.perf_counter()-started
            ready = wait_for_cache(readiness, [tokens_by_id[cid]])
            started = time.perf_counter()
            retired = retirement(llm, args.tp)
            entry = dict(chunk=cid, generation_seconds=generation_seconds, readiness=ready,
                         retirement=retired, retirement_seconds=time.perf_counter()-started)
            attempt['chunks'].append(entry)
            atomic_json(root / 'progress.json', progress)
    finally:
        llm.llm_engine.engine_core.shutdown()
    attempt['engine_exited'] = True
    atomic_json(root / 'progress.json', progress)


def build_setup(args):
    root = args.output.resolve()
    entries = manifest_entries(args.manifest.resolve())
    prepare_started = time.perf_counter()
    if args.dry_run:
        preparation = preparation_identity(args, entries)
        # Same validation and inventory, with no setup mutations or GPU imports.
        return prepare_collection(args, entries, preparation, write=False)
    with setup_lock(root, build=True):
        assert_no_build_workers(root)
        if (root / 'setup.json').exists():
            descriptor = read_descriptor(root)
            preparation = preparation_identity(args, entries, descriptor)
            if descriptor['preparation'] != preparation:
                raise ValueError('Incompatible preparation settings or inputs; use another setup directory')
            samples = load_frozen(root, descriptor, max_output_tokens=args.max_output_tokens)
            for row, (entry, sample) in zip(entries, samples):
                source = json.loads(row['prepared'].read_text()) if 'prepared' in row else {}
                if evaluation_metadata(source, row) != entry.get('evaluation', evaluation_metadata(sample)):
                    raise ValueError('Evaluation metadata changed; use another setup directory')
            descriptor['inventory'] = json.loads((root / 'inventory.json').read_text())
        else:
            preparation = preparation_identity(args, entries)
            descriptor, samples = prepare_collection(args, entries, preparation, write=True)
        inventory = descriptor['inventory']
        missing = incomplete_chunks(root, inventory)
        print(f"Persistent KV: {inventory['expected_bytes']} bytes, {len(inventory['chunks'])} unique chunks, "
              f"{len(missing)} incomplete", flush=True)
        if not missing and (root / 'complete.json').exists():
            check_receipt(root, descriptor)
            print(f"Setup reusable: {root} ({descriptor['fingerprint']})")
            return
        # A failed rebuild must never leave a stale completion marker.
        (root / 'complete.json').unlink(missing_ok=True)
        progress_path = root / 'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(attempts=[])
        progress.setdefault('cpu_preparation_seconds', time.perf_counter()-prepare_started)
        atomic_json(progress_path, progress)
        if missing:
            construct_missing(root, descriptor, samples, missing, args, progress)
        state = snapshot(root / 'cache', inventory)
        assert_no_build_workers(root)
        for attempt in progress['attempts']:
            if not attempt.get('engine_exited'):
                attempt['recovered_after_verified_worker_exit'] = True
        atomic_json(progress_path, progress)
        # File-size readiness is followed by fsync before publishing completion.
        for name in state:
            with (root / 'cache' / name).open('rb') as stream:
                os.fsync(stream.fileno())
        atomic_json(root / 'complete.json', dict(fingerprint=descriptor['fingerprint'],
            inventory_sha256=descriptor['inventory_sha256'], committed_shards=len(state),
            preparation_timings='progress.json', retirement_verified=True, completed_at=time.time()))
        print(f'Setup complete: {root}')


def prepare_collection(args, entries, preparation, write):
    from runner.preparation import tokenize_sample
    tokenizer = None
    samples = []
    config_sha = file_hash(args.model / 'config.json')
    for row in entries:
        if 'prepared' in row:
            sample = json.loads(row['prepared'].read_text())
            if args.max_output_tokens is not None:
                sample['max_output_tokens'] = args.max_output_tokens
        else:
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            options = SimpleNamespace(**{**vars(args), **row})
            options.max_output_tokens = args.max_output_tokens if args.max_output_tokens is not None else 16384
            sample = tokenize_sample(options, tokenizer)
        validate_sample(sample, args.kv_chunk_size, args.context_length)
        if sample.get('model_config_sha256') != config_sha:
            raise ValueError('Prepared input uses another model configuration')
        samples.append(sample)
    identity = dict(preparation=preparation, prompts=[dict(id=row['id'], sample=fingerprint(sample_identity(sample)))
        for row, sample in zip(entries, samples)])
    fp = fingerprint(identity)
    inventory = inventory_for(samples, fp, args.tp, bytes_per_shard(args.model, args.tp))
    descriptor = dict(fingerprint=fp, identity=identity, preparation=preparation,
                      model=str(args.model), model_stamp=model_stamp(args.model), samples=[], inventory=inventory)
    for index, (row, sample) in enumerate(zip(entries, samples)):
        descriptor['samples'].append(dict(id=row['id'], file=f'samples/{index:06d}.json',
            identity=fingerprint(sample_identity(sample)), chunks=[fingerprint(t) for t in chunks_of(sample)],
            evaluation=evaluation_metadata(sample, row)))
    descriptor['evaluation_sha256'] = fingerprint([e['evaluation'] for e in descriptor['samples']])
    root = args.output.resolve()
    if not write:
        cfg = engine_config(args.model, 'prophetkv', tp=args.tp, memory=args.memory,
            cache_dir=root / 'cache', end_token=None,
            persistent=persistent_config(root, descriptor, False))
        print(json.dumps(dict(fingerprint=fp, prompts=len(samples), chunks=len(inventory['chunks']),
            expected_kv_bytes=inventory['expected_bytes'], engine=cfg), indent=2))
        return descriptor, list(zip(descriptor['samples'], samples))
    for entry, sample in zip(descriptor['samples'], samples):
        atomic_json(root / entry['file'], sample)
        entry['sha256'] = file_hash(root / entry['file'])
    atomic_json(root / 'inventory.json', inventory)
    descriptor['inventory_sha256'] = file_hash(root / 'inventory.json')
    atomic_json(root / 'setup.json', {k: v for k, v in descriptor.items() if k != 'inventory'})
    return descriptor, list(zip(descriptor['samples'], samples))


def run_collection(args):
    root = args.setup.resolve()
    with setup_lock(root):
        assert_no_build_workers(root)
        descriptor = read_descriptor(root)
        receipt = check_receipt(root, descriptor)
        model = args.model or Path(descriptor['model'])
        if model_identity(model, descriptor['preparation']['model'], descriptor.get('model_stamp')) != descriptor['preparation']['model']:
            raise ValueError('Model/tokenizer changed; prepare another setup directory')
        tp = descriptor['preparation']['tp']
        if args.tp is not None and args.tp != tp:
            raise ValueError('TP differs from setup; prepare another setup directory')
        samples = load_frozen(root, descriptor, args.prompt_ids, args.max_output_tokens)
        inventory = json.loads((root / 'inventory.json').read_text())
        descriptor['inventory'] = inventory
        cached = args.method != 'baseline'
        if cached:
            from runner.layout import validate_layout
            for _,sample in samples:
                validate_layout(sample['token_ids'],sample['boundaries'],sample['question_positions'],sparse=True)
        layouts = {f'prompt-{i}': sample['boundaries'] for i, (_, sample) in enumerate(samples)}
        cfg = engine_config(model, args.method, args.ratio, args.layers, args.num_layers,
            tp, args.memory, root / 'cache',
            None,
            persistent=persistent_config(root, descriptor, True, layouts) if cached else None,
            router_policy=getattr(args,'router_policy',None),router_id=getattr(args,'router_id','router1'))
        if args.method == 'router':
            from runner.router_measure import validate_profile
            from runner.layout import validate_policy_protocol
            from runner.router_policy import load_policy
            policy=load_policy(args.router_policy)
            for _,sample in samples:
                validate_profile(sample,tp)
                validate_policy_protocol(policy,sample)
        if args.dry_run:
            print(json.dumps(dict(setup_fingerprint=descriptor['fingerprint'],
                prompt_ids=[entry['id'] for entry, _ in samples], engine=cfg), indent=2))
            return
        # Baseline deliberately does not touch any KV shard or instantiate a store.
        if cached:
            for entry, _ in samples:
                snapshot(root / 'cache', inventory, entry['chunks'])
        devices = check_environment(tp)
        args.output.mkdir(parents=True, exist_ok=False)
        os.environ['PROPHETKV_SCHEDULER_RECEIPT'] = str(args.output.resolve() / 'scheduler.json')
        atomic_json(args.output / 'config.json', cfg)
        from runner.reporting import AggregationReporter
        evaluations = [entry.get('evaluation', evaluation_metadata(sample)) for entry, sample in samples]
        from ucm.sparse.prophetkv.layers import resolve_layers
        context = dict(setup_fingerprint=descriptor['fingerprint'], method=args.method,
                       evaluation_sha256=descriptor.get('evaluation_sha256'), evaluation_file='evaluation.json',
                       ratio=args.ratio if cached else None,
                       scoring_layers=list(resolve_layers(args.method, args.layers, args.num_layers)) if cached else [])
        reporter = AggregationReporter(args.output,
            [dict(prompt_id=entry['id'], subtask=evaluation['subtask'])
             for (entry, _), evaluation in zip(samples, evaluations)], context)
        atomic_json(args.output / 'evaluation.json',
                    [dict(prompt_id=entry['id'], **evaluation)
                     for (entry, _), evaluation in zip(samples, evaluations)])
        try:
            execute_collection(args, root, descriptor, samples, cfg, devices, receipt, reporter)
            reporter.finish()
            print(f'Wrote {args.output / "final_aggregation.md"}', flush=True)
        except BaseException as error:
            reporter.publish('live', 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error)
            raise


def execute_collection(args, root, descriptor, samples, cfg, devices, receipt, reporter=None):
    from runner.generation import generate, verify_diagnostics
    from runner.worker import setup, arm, drain
    from ucm.sparse.prophetkv.layers import resolve_layers
    cached = args.method != 'baseline'
    tp = descriptor['preparation']['tp']
    inventory = descriptor['inventory']
    namespace = uuid.uuid4().hex
    scoring = resolve_layers(args.method, args.layers, args.num_layers) if cached else ()
    started = time.perf_counter()
    llm = start_engine(cfg)
    session = dict(model_load_seconds=time.perf_counter()-started)
    records = []

    def infer(sample, tag, purpose, budget, thinking=False):
        rid = f'{namespace}:{tag}:{purpose}'
        if cached:
            if args.method == 'router':
                from runner.router_measure import sync_mode
                sync_mode(llm,rid,'prophetkv-all64-1',tp)
            llm.collective_rpc(arm, kwargs=dict(request_id=rid, boundaries=sample['boundaries'],
                                               question_positions=sample['question_positions']))
        return generate(llm.llm_engine, sample['token_ids'], budget, rid, thinking, sample=sample)

    try:
        from runner.reporting import OutputAnalyzer, score_answer
        analyzer = OutputAnalyzer(llm.get_tokenizer())
        if reporter is not None:
            reporter.publish('live', 'running')
        setup_receipts = llm.collective_rpc(setup)
        started = time.perf_counter()
        if cached:
            # Warm the actual read path using frozen KV, with no warmup population.
            infer(samples[0][1], 'prompt-0', 'warmup', 1)
        else:
            generate(llm.llm_engine, [100, 200, 300, 400]*32, 1, namespace+':warmup')
        retirement(llm, tp)
        session['warmup_seconds'] = time.perf_counter()-started
        for index, (entry, sample) in enumerate(samples):
            tag = f'prompt-{index}'
            output_dir = args.output / f'{index:06d}'
            output_dir.mkdir()
            timings = {}
            started = time.perf_counter()
            before = snapshot(root / 'cache', inventory, entry['chunks']) if cached else None
            timings['readiness_seconds'] = time.perf_counter()-started
            started = time.perf_counter()
            infer(sample, tag, 'prime', 1)
            llm.collective_rpc(drain)
            retirement(llm, tp)
            timings['priming_seconds'] = time.perf_counter()-started
            routing = None
            if args.method == 'router':
                from runner.router_measure import measure
                from runner.router_policy import load_policy
                def unchanged():
                    if before != snapshot(root/'cache', inventory, entry['chunks']):
                        raise RuntimeError('Persistent KV changed during routing')
                result,ttft,elapsed,routing = measure(llm,sample,f'{namespace}:{tag}:measured',
                    load_policy(args.router_policy),args.router_id,output_dir/'routing',tp,unchanged,
                    args.output/'scheduler.json')
            else:
                result, ttft, elapsed = infer(sample, tag, 'measured', sample['max_output_tokens'], sample['thinking'])
            timings.update(ttft_seconds=ttft, generation_seconds=elapsed)
            started = time.perf_counter()
            diagnostics = llm.collective_rpc(drain)
            if routing:
                from runner.router_measure import validate_answer
                validate_answer(llm,diagnostics,result,sample,routing,tp)
            else:
                verify_diagnostics(diagnostics, sample, args.method, args.ratio, tp, scoring)
            if not cached and result.num_cached_tokens != 0:
                raise RuntimeError('Baseline reused cached tokens')
            atomic_json(output_dir / 'diagnostics.json', diagnostics)
            timings['validation_and_export_seconds'] = time.perf_counter()-started
            started = time.perf_counter()
            retired = retirement(llm, tp)
            if cached:
                scheduler = json.loads((args.output / 'scheduler.json').read_text())
                if scheduler != dict(request_id=routing['answer_request_id'] if routing else f'{namespace}:{tag}:measured', requests_blend_meta=0, requests_meta=0):
                    raise RuntimeError('Scheduler request not retired')
                if before != snapshot(root / 'cache', inventory, entry['chunks']):
                    raise RuntimeError('Persistent KV changed during inference or retirement')
            timings['retirement_seconds'] = time.perf_counter()-started
            output = result.outputs[0]
            evaluation = entry.get('evaluation', evaluation_metadata(sample))
            reporting_started = time.perf_counter()
            output_metrics = analyzer.analyze(list(output.token_ids), sample['token_ids'], sample['thinking'])
            accuracy = score_answer(scoring_text(output.text,output_metrics,evaluation), evaluation)
            timings['scoring_seconds'] = time.perf_counter()-reporting_started
            record = dict(**sample_provenance(sample),prompt_id=entry['id'], setup_fingerprint=descriptor['fingerprint'],
                setup=str(root), construction_timings=str(root / receipt['preparation_timings']),
                method=args.method, ratio=args.ratio if cached else None, scoring_layers=list(scoring),
                input_sha256=entry['sha256'], model=cfg['model'], gpu_uuids=devices,
                runtime_versions=VERSIONS, prompt_tokens=len(sample['token_ids']),
                output_token_ids=list(output.token_ids), prediction=output.text,
                max_output_tokens=sample['max_output_tokens'], thinking=sample['thinking'],
                finish_reason=output.finish_reason, num_cached_tokens=result.num_cached_tokens,
                timings=timings, retirement=retired, routing=routing, **output_metrics, **evaluation, accuracy=accuracy,
                output_cap_reached=len(output.token_ids) >= sample['max_output_tokens'])
            atomic_json(output_dir / 'result.json', record)
            records.append(record)
            if reporter is not None:
                reporter.accept(record)
            print(f"{entry['id']}: TTFT {ttft:.3f}s", flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()
    atomic_json(args.output / 'summary.json', dict(setup_fingerprint=descriptor['fingerprint'],
        setup=str(root), construction_timings=str(root / receipt['preparation_timings']),
        method=args.method, prompt_ids=[r['prompt_id'] for r in records],
        completed=len(records), session_timings=session, setup_receipts=setup_receipts,
        mean_ttft_seconds=sum(r['timings']['ttft_seconds'] for r in records)/len(records),
        aggregations=dict(live='live_aggregation.json', final='final_aggregation.json'),
        results=[f'{i:06d}/result.json' for i in range(len(records))]))
