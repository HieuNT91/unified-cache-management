"""Audited ProphetKV-only continuation of an interrupted two-worker sweep."""
import argparse
import json
from pathlib import Path
import re
import shutil
import time
import uuid

from runner.config import engine_config, VERSIONS
from runner.reporting import aggregate
from runner.setups import atomic_json, file_hash, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def resolve_scope(output):
    output = Path(output)
    marker = output/'continuation.json'
    if not marker.exists():
        return output, output, None
    scope = json.loads(marker.read_text())
    attempt = output/scope['attempt']
    if attempt.resolve().parent != (output/'continuation').resolve():
        raise RuntimeError('Invalid continuation path')
    return attempt/'reports', attempt, scope


def compatible_identity(spec, original):
    if fingerprint(spec) == original:
        return 'current runtime'
    for version in json.loads((ROOT/'runner/legacy_sweep_runtimes.json').read_text()):
        candidate = dict(spec, runtime=version['runtime'])
        if version['legacy_lengths']:
            from runner.sweep import configurations
            if spec['configs'] != configurations():
                continue
            if spec['context_length'] != 114688 or spec['exact_input_tokens'] is not None:
                continue
            candidate.pop('context_length'); candidate.pop('exact_input_tokens')
        if fingerprint(candidate) == original:
            return version['commit']
    raise RuntimeError('Original sweep fingerprint does not match these inputs/settings or a supported runtime')


def validate_record(path, config, entry, sample, model, tp, devices):
    from run import verify_diagnostics
    from ucm.sparse.prophetkv.layers import resolve_layers
    record = json.loads(path.read_text())
    layers = list(resolve_layers(config['method'], config['layers'])) if config['method'] != 'baseline' else []
    required = dict(prompt_id=entry['id'], input_sha256=entry['sha256'], model=str(model),
                    method=config['method'], ratio=config['ratio'], scoring_layers=layers,
                    prompt_tokens=len(sample['token_ids']), thinking=sample['thinking'],
                    max_output_tokens=sample['max_output_tokens'], runtime_versions=VERSIONS,
                    gpu_uuids=devices, **entry['evaluation'])
    if any(record.get(k) != v for k, v in required.items()):
        raise RuntimeError(f'Retained result has incompatible metadata: {path}')
    retired = record['retirement']
    if (sorted(r['rank'] for r in retired) != list(range(tp)) or not all(
            r['quiescent'] and r['transfers']['pending'] == 0 and r['request_bookkeeping'] == 0 for r in retired)):
        raise RuntimeError(f'Retained result was not retired: {path}')
    if len(record['output_token_ids']) != record['output_tokens']:
        raise RuntimeError(f'Output length mismatch: {path}')
    if config['method'] == 'baseline' and record['num_cached_tokens'] != 0:
        raise RuntimeError(f'Baseline reused cache: {path}')
    aggregate([record], [dict(prompt_id=entry['id'], subtask=entry['evaluation']['subtask'])], {}, 'completed')
    verify_diagnostics(json.loads(path.with_name('diagnostics.json').read_text()), sample,
                       config['method'], config['ratio'], tp, layers)
    return record


def prepare(args):
    import fcntl
    import os
    # The launcher already holds fd 9. A direct CLI caller must take the same
    # lock; never open a second conflicting flock on behalf of the coordinator.
    path = args.output.resolve()/'sweep.lock'
    try:
        inherited = os.fstat(9)
        stat = path.stat()
        borrowed = ((inherited.st_dev, inherited.st_ino) == (stat.st_dev, stat.st_ino)
                    and os.environ.get('UCM_SWEEP_COORDINATOR_PID') == str(os.getppid()))
    except OSError:
        borrowed = False
    if borrowed:
        fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(args)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(args)


def _prepare(args):
    from runner.sweep import load_inputs, configurations, SharedReporter
    from scripts.sweep_control import assert_stopped
    args.output = args.output.resolve(); args.model = args.model.resolve()
    assert_stopped(args.output)
    if args.shards != 2:
        raise ValueError('This continuation supports the existing two-shard launchers')
    all_configs = configurations(args.percentages, args.layers)
    configs = configurations(args.percentages, args.layers, skip_selective=True)
    report_root, old_attempt, previous = resolve_scope(args.output)
    if previous:
        for relative, digest in previous['retained_files'].items():
            if file_hash(args.output/relative) != digest:
                raise RuntimeError(f'Retained artifact changed: {relative}')
    entries = []
    plans = []
    for shard in range(args.shards):
        args.shard = shard
        expected, selected, current = load_inputs(args)
        plan = json.loads((args.output/f'group-{shard}/plan.json').read_text())
        if (plan['prompt_ids'] != [e['id'] for _, e, _ in selected] or plan['shard'] != shard
                or plan['shards'] != args.shards or plan['configurations'] != all_configs):
            raise RuntimeError('Original shard/configuration layout differs')
        plans.append(plan); entries.extend(selected)
        baseline = json.loads((args.output/f'group-{shard}/baseline-session.json').read_text())
        if not baseline['engine_shutdown']:
            raise RuntimeError('Baseline engine shutdown is not recorded')
        # Reject changes to model, TP, RoPE, dtype, allocation, or scheduler budget.
        old_cfg = json.loads((args.output/f'group-{shard}/baseline-config.json').read_text())
        if old_cfg != engine_config(args.model, 'baseline', tp=args.tp, memory=args.memory):
            raise RuntimeError('Original engine configuration differs')
    if len({p['identity'] for p in plans}) != 1:
        raise RuntimeError('Original shards have different fingerprints')
    source = compatible_identity(args.identity_spec, plans[0]['identity'])
    if previous and previous['current_identity'] != current:
        raise RuntimeError('Continuation code or inputs changed; use the same committed runtime to resume')
    attempt = args.output/'continuation'/('attempt-'+uuid.uuid4().hex)
    pending = {c['name']: [] for c in configs}
    records = {c['name']: [] for c in configs}
    retained = {}
    incomplete = []
    for index, entry, sample in sorted(entries):
        shard = index % args.shards
        for config in configs:
            folder = args.output/config['name']/f'{index:06d}'
            path = folder/'result.json'
            if not path.exists():
                if config['method'] == 'baseline':
                    raise RuntimeError(f'Baseline must be complete before continuation: {path}')
                if folder.exists():
                    incomplete.append(folder)
                pending[config['name']].append(index)
                continue
            record = validate_record(path, config, entry, sample, args.model, args.tp, plans[shard]['gpu_uuids'])
            cohort = record.get('continuation_fingerprint')
            if cohort and (not previous or cohort != previous['fingerprint']):
                raise RuntimeError(f'Result belongs to a different continuation: {path}')
            keys = ('prompt_id','subtask','accuracy','thinking_tokens','answer_tokens','control_tokens',
                    'output_tokens','output_cap_reached','unfinished_thinking','timings')
            records[config['name']].append({key:record[key] for key in keys})
            for artifact in (path, path.with_name('diagnostics.json')):
                retained[str(artifact.relative_to(args.output))] = file_hash(artifact)
        if (index+1) % 50 == 0:
            print(f'Validated retained results through prompt {index+1}/{len(entries)}', flush=True)
    # Do not touch results until all completed artifacts pass validation.
    attempt.mkdir(parents=True)
    for folder in incomplete:
        target = attempt/'incomplete'/folder.relative_to(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(target)
    # Stop verification precedes deletion. Only uniquely owned cache roots from
    # saved plans, under this exact CACHE_ROOT, may be removed.
    cache_paths = {p['cache'] for p in plans}
    if previous:
        for plan_path in old_attempt.glob('group-*/plan.json'):
            cache_paths.add(json.loads(plan_path.read_text())['cache'])
    for name in cache_paths:
        path = Path(name)
        if path.parent.resolve() != args.cache_root.resolve() or not re.fullmatch(r'sweep-[0-9a-f]{32}', path.name):
            raise RuntimeError(f'Cache ownership path mismatch: {path}')
        if path.exists():
            if path.is_symlink() or path.absolute() != path.resolve():
                raise RuntimeError(f'Symlink in owned cache path: {path}')
            shutil.rmtree(path)
    identity = fingerprint(dict(original=plans[0]['identity'], current=current, configs=configs))
    scope = dict(schema_version=1, fingerprint=identity, original_identity=plans[0]['identity'],
                 source_revision=source, current_identity=current, attempt=str(attempt.relative_to(args.output)),
                 configurations=configs, discontinued=[c['name'] for c in all_configs if c not in configs],
                 retained_files=retained, pending=pending, gpu_uuids=[p['gpu_uuids'] for p in plans],
                 created_at=time.time(), prior_attempt=previous['attempt'] if previous else None,
                 cache_cleanup=list(cache_paths), expected=len(entries)*len(configs))
    # Rebuild reporting from committed, validated result files; this also handles
    # a crash between atomic result publication and acceptance by the old ledger.
    for config in configs:
        context = dict(method=config['method'], ratio=config['ratio'],
                       scoring_layers=list(range(64)) if config['method'] != 'baseline' else [],
                       sweep_fingerprint=identity,
                       cache_policy='temporary per prompt; construction excluded from TTFT')
        dest = attempt/'reports'/config['name']
        reporter = SharedReporter(dest, expected, context, 0, args.shards)
        state = dict(identity=fingerprint(dict(expected=expected, context=context, shards=args.shards)),
                     records=records[config['name']], finished=[], errors={})
        atomic_json(dest/'aggregation_state.json', state)
        reporter.update()
    atomic_json(attempt/'scope.json', scope)
    atomic_json(args.output/'continuation.json', scope)
    print(f'Continuation ready: {sum(map(len, pending.values()))} missing ProphetKV measurements; '
          f'{len(entries)} baselines reused; selective discontinued.', flush=True)
    return scope


def attach(args, current):
    report_root, attempt, scope = resolve_scope(args.output)
    if scope is None or scope['current_identity'] != current:
        raise RuntimeError('Prepare the continuation using the launcher resume command first')
    from runner.sweep import configurations
    configs = configurations(args.percentages, args.layers, skip_selective=True)
    if configs != scope['configurations']:
        raise RuntimeError('Continuation configurations differ')
    import os
    if not args.dry_run and os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') != scope['gpu_uuids'][args.shard]:
        raise RuntimeError('Resume must use the original shard GPU UUIDs')
    all_indices = set(range(len(args.identity_spec['inputs'])))
    args.completed = {name: all_indices-set(indices) for name, indices in scope['pending'].items()}
    args.continuation_fingerprint = scope['fingerprint']
    return configs, scope['fingerprint'], report_root, attempt/f'group-{args.shard}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'manifest', 'output', 'cache-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--tp', type=int, default=4)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--memory', type=float, default=.9)
    parser.add_argument('--percentages', type=int, nargs='+', default=[1,5,10,15,20,30])
    parser.add_argument('--layers', type=int, nargs='+', default=[11,12,13,14,15])
    parser.add_argument('--context-length', type=int, default=114688)
    parser.add_argument('--exact-input-tokens', type=int)
    prepare(parser.parse_args())
