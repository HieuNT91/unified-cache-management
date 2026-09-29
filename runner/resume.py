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


def active_configurations(percentages, layers=(11,12,13,14,15), scope=None):
    """Keep original input fingerprints while narrowing only the execution pool."""
    from runner.sweep import configurations
    excluded = scope.get('excluded_percentages', []) if scope else []
    if len(set(excluded)) != len(excluded) or not set(excluded) <= set(percentages):
        raise ValueError('Excluded ratios must be distinct members of the original percentages')
    remaining = [p for p in percentages if p not in excluded]
    if not remaining:
        raise ValueError('At least one ProphetKV ratio must remain active')
    configs = configurations(remaining, layers, skip_selective=True)
    if scope and configs != scope['configurations']:
        raise RuntimeError('Continuation configurations differ')
    return configs


def compatible_identity(spec, original, diagnostic_path=None):
    if fingerprint(spec) == original:
        return 'current runtime'
    candidates = []
    for version in json.loads((ROOT/'runner/legacy_sweep_runtimes.json').read_text()):
        candidate = dict(spec, runtime=version['runtime'])
        if version['legacy_lengths']:
            from runner.sweep import configurations
            if spec['configs'] != configurations():
                continue
            if spec['context_length'] != 114688 or spec['exact_input_tokens'] is not None:
                continue
            candidate.pop('context_length'); candidate.pop('exact_input_tokens')
        variants = [(version['commit'], candidate)]
        # Historical runtime_identity() recursively included local vendor checkouts.
        # Reconstruct that exact payload; never discard their hashes or overwrite
        # pinned release files. Changed/missing vendor files still fail the hash.
        vendor = {name: digest for name, digest in spec['runtime'].items()
                  if name.startswith(('ucm/vendor/', 'ucm/.cache/vendor/')) and name not in version['runtime']}
        if vendor:
            variants.append((version['commit'] + ' with preserved ucm/vendor files',
                             dict(candidate, runtime={**version['runtime'], **vendor})))
        for source, payload in variants:
            digest = fingerprint(payload)
            candidates.append(dict(commit=source, fingerprint=digest,
                runtime_differences=[name for name in sorted(spec['runtime'].keys() | payload['runtime'].keys())
                                     if spec['runtime'].get(name) != payload['runtime'].get(name)]))
            if digest == original:
                return source
    message = 'Original sweep fingerprint does not match these inputs/settings or a supported runtime'
    if diagnostic_path is not None:
        atomic_json(diagnostic_path, dict(original_fingerprint=original,
            current_fingerprint=fingerprint(spec), current_spec=spec, candidates=candidates,
            explanation='The original receipt stores a combined hash, so the differing original field '
                        'cannot be recovered from it. Compare source revision, local Python files, '
                        'manifest and prepared inputs; no compatibility check has been bypassed.'))
        message += f'. Diagnostic report: {diagnostic_path}'
    raise RuntimeError(message)


def validate_record(path, config, entry, sample, model, tp, devices, validation='full'):
    if validation not in ('full', 'fast'):
        raise ValueError('Resume validation must be full or fast')
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
    diagnostics = path.with_name('diagnostics.json')
    if not diagnostics.is_file() or diagnostics.stat().st_size == 0:
        raise RuntimeError(f'Missing retained diagnostics: {diagnostics}')
    if validation == 'full':
        verify_diagnostics(json.loads(diagnostics.read_text()), sample,
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
    report_root, old_attempt, previous = resolve_scope(args.output)
    excluded = sorted(set(previous.get('excluded_percentages', []) if previous else []) |
                      set(getattr(args, 'exclude_percentages', None) or []))
    # Validate the requested subset before any mutation or GPU work.
    configs = active_configurations(args.percentages, args.layers)
    if not set(excluded) <= set(args.percentages) or len(excluded) == len(args.percentages):
        raise ValueError('Exclude only original ratios and retain at least one ProphetKV ratio')
    configs = [c for c in configs if c['name'] not in {f'prophetkv-{p}' for p in excluded}]
    from scripts.sweep_counts import saved_counts
    counts = saved_counts(args.output, args.manifest, args.percentages, args.shards, excluded)
    validation = getattr(args, 'validation', 'full')
    if validation not in ('full', 'fast'):
        raise ValueError('Resume validation must be full or fast')
    print(f'Resume validation: {validation}. '
          + ('Saved diagnostic replay and hashing are skipped; new measurements remain fully validated.'
             if validation == 'fast' else 'Saved diagnostics will be replayed and hashed.'), flush=True)
    assert_stopped(args.output)
    if args.shards != 2:
        raise ValueError('This continuation supports the existing two-shard launchers')
    all_configs = configurations(args.percentages, args.layers)
    if previous:
        for relative, digest in previous['retained_files'].items():
            if validation == 'fast' and Path(relative).name == 'diagnostics.json':
                continue
            if file_hash(args.output/relative) != digest:
                raise RuntimeError(f'Retained artifact changed: {relative}')
    entries = []
    plans = []
    for shard in range(args.shards):
        args.shard = shard
        print(f'Checking prepared inputs and original settings for group {shard}...', flush=True)
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
    single_group = getattr(args, 'single_group_gpus', None)
    if single_group is not None:
        devices = single_group.split(',')
        if len(devices) != args.tp or len(set(devices)) != args.tp or any(
                not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', d) for d in devices):
            raise ValueError('Single-group resume requires exactly TP distinct full GPU UUIDs')
        gpu_uuids = [devices for _ in plans]
        execution_mode = 'sequential'
    else:
        gpu_uuids = previous['gpu_uuids'] if previous else [p['gpu_uuids'] for p in plans]
        execution_mode = previous.get('execution_mode', 'parallel') if previous else 'parallel'
    source = compatible_identity(args.identity_spec, plans[0]['identity'],
                                 args.output/'resume-compatibility.json')
    previous_source = None
    if previous and previous['current_identity'] != current:
        previous_source = compatible_identity(args.identity_spec, previous['current_identity'],
                                              args.output/'resume-compatibility.json')
    accepted_fingerprints = set()
    if previous:
        accepted_fingerprints.update(previous.get('accepted_fingerprints', []))
        accepted_fingerprints.add(previous['fingerprint'])
    attempt = args.output/'continuation'/('attempt-'+uuid.uuid4().hex)
    pending = {c['name']: [] for c in configs}
    records = {c['name']: [] for c in configs}
    # Keep prior diagnostic pins for a later full audit, but do not read those
    # large files in fast mode. The receipt explicitly records that distinction.
    retained = dict(previous['retained_files']) if previous else {}
    # Bind each continuation fingerprint to the device assignments it authorized.
    cohort_devices = dict(previous.get('cohort_gpu_uuids', {})) if previous else {}
    if previous:
        cohort_devices.setdefault(previous['fingerprint'], previous['gpu_uuids'])
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
            cohort = json.loads(path.read_text()).get('continuation_fingerprint')
            if cohort and cohort not in accepted_fingerprints:
                raise RuntimeError(f'Result belongs to a different continuation: {path}')
            devices = cohort_devices.get(cohort, [p['gpu_uuids'] for p in plans])[shard]
            record = validate_record(path, config, entry, sample, args.model, args.tp, devices, validation)
            keys = ('prompt_id','subtask','accuracy','thinking_tokens','answer_tokens','control_tokens',
                    'output_tokens','output_cap_reached','unfinished_thinking','timings')
            records[config['name']].append({key:record[key] for key in keys})
            artifacts = (path, path.with_name('diagnostics.json')) if validation == 'full' else (path,)
            for artifact in artifacts:
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
    identity = fingerprint(dict(original=plans[0]['identity'], current=current, configs=configs,
                                gpu_uuids=gpu_uuids, execution_mode=execution_mode))
    accepted_fingerprints.add(identity)
    cohort_devices[identity] = gpu_uuids
    scope = dict(schema_version=1, fingerprint=identity, original_identity=plans[0]['identity'],
                 source_revision=source, current_identity=current, attempt=str(attempt.relative_to(args.output)),
                 configurations=configs, discontinued=[c['name'] for c in all_configs if c not in configs],
                 retained_files=retained, pending=pending, gpu_uuids=gpu_uuids,
                 cohort_gpu_uuids=cohort_devices, execution_mode=execution_mode, excluded_percentages=excluded,
                 original_gpu_uuids=[p['gpu_uuids'] for p in plans],
                 created_at=time.time(), prior_attempt=previous['attempt'] if previous else None,
                 cache_cleanup=list(cache_paths), expected=len(entries)*len(configs),
                 validation=dict(mode=validation, diagnostic_replay=validation == 'full',
                                 diagnostic_hashes_verified=validation == 'full',
                                 result_hashes_verified=True, new_measurements='full'),
                 saved_counts=counts, accepted_fingerprints=sorted(accepted_fingerprints),
                 previous_source_revision=previous_source)
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
    configs = active_configurations(args.percentages, args.layers, scope)
    import os
    if not args.dry_run and os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') != scope['gpu_uuids'][args.shard]:
        raise RuntimeError('Resume must use the continuation shard GPU UUIDs')
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
    parser.add_argument('--validation', choices=['full', 'fast'], default='full')
    parser.add_argument('--exclude-percentages', type=int, nargs='+')
    parser.add_argument('--single-group-gpus', help='Run both original shards sequentially on this comma-separated TP group')
    prepare(parser.parse_args())
