#!/usr/bin/env python3
"""Adopt prepared thinking RULER, then collect features and last_traj_4f on L20."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.corpus import relative
from runner.corpus_records import accepted, NATIVE_ANSWER_VALIDATION, PROBE_ANSWER_VALIDATION
from runner.last_traj_4 import COLLECTION_SCHEMA, FIXED_METHOD, control_actions, inventory
from runner.setups import atomic_json, file_hash, fingerprint
from runner.tree_profiles import validate
from runner.thinking_budget import KEY
from runner.output_budget import RULER_PROTOCOL
from scripts import longbench_a800_control as control
from scripts.longbench_a800_data import read, freeze
from scripts.ruler import TASKS

KERNEL = '_ragged_positions_fa2_kernel'
RATIOS = (1, 5, 10, 20, 30, 40, 60, 80)
PHASES = ('features', 'cached')


def validate_rows(settings, rows):
    n = settings['samples_per_task']
    if (len(rows) != len(TASKS)*n or len({r['id'] for r in rows}) != len(rows)
            or set(r['subtask'] for r in rows) != set(TASKS)
            or any(sorted(r['ordinal'] for r in rows if r['subtask'] == task) != list(range(n)) for task in TASKS)):
        raise ValueError('Expected complete original RULER task membership')


def protocol_for(settings, plan, phase):
    features = phase == 'features'
    # This is a mask archive inventory, not a scheduled baseline/answer phase.
    from scripts.collection_v3 import ACTIONS as PROBE_MASKS
    from runner.longbench_features import DEFINITIONS
    actions = PROBE_MASKS if features else control_actions(settings['percentages'], FIXED_METHOD)[1:]
    result = dict(schema=COLLECTION_SCHEMA, dataset='ruler', execution_profile='ruler-thinking',
        evaluation_protocol=RULER_PROTOCOL, tp=2, hardware_profile='l20-tp2',
        groups=settings['groups'], model=settings['model'], prepared=settings['prepared'],
        cache_root=settings['cache_root'], percentages=settings['percentages'], actions=actions,
        scheduled_actions=['probe'] if features else [a['id'] for a in actions],
        answer_validation=PROBE_ANSWER_VALIDATION if features else NATIVE_ANSWER_VALIDATION,
        position_attention_kernel=KERNEL, plan_sha256=fingerprint(plan), generation_seed=0,
        prepared_receipt_sha256=plan['preparation_sha256'])
    if features:
        result['feature_profile'] = DEFINITIONS
    else:
        inventory(result)
    return result


def adopt(args):
    """First launch writes run metadata only; prepared files are never modified."""
    from scripts.router_control import code_hashes
    from scripts.naive_reuse_inputs import protect_paths
    if args.tp != 2 or args.seed != 42 or args.samples_per_task < 1:
        raise ValueError('Expected TP2, preparation seed42 and a positive sample count')
    groups = control.resolve_groups(args.devices, 10, tp=2, query_inventory=False)
    actions = control_actions(args.percentages, FIXED_METHOD)[1:]
    settings = dict(schema=COLLECTION_SCHEMA, dataset='ruler', execution_profile='ruler-thinking',
        model=str(args.model), prepared=str(args.prepared), cache_root=str(args.cache_root), tp=2,
        hardware_profile='l20-tp2', groups=groups, samples_per_task=args.samples_per_task,
        seed=42, generation_seed=0, percentages=args.percentages, position_attention_kernel=KERNEL,
        phase_order=list(PHASES), watchdog_seconds=7200)
    if (args.root/'settings.json').exists():
        old, _, _ = load(args.root)
        if any(old[k] != value for k, value in settings.items()):
            raise ValueError('Existing run parameters differ; keep its env or use fresh paths')
        return
    protect_paths(args.root, args.prepared, args.cache_root, args.model)
    if args.root.exists() and any(args.root.iterdir()):
        raise ValueError('Use an empty result directory for adoption')
    receipt_path = args.prepared/'preparation.json'
    receipt_hash = file_hash(receipt_path)
    receipt = read(receipt_path); spec = receipt['spec']
    if (spec.get('protocol') != RULER_PROTOCOL or spec.get('execution_profile') != 'ruler-thinking'
            or spec.get('samples') != args.samples_per_task or spec.get('seed') != 42
            or set(spec.get('tasks', ())) != set(TASKS)):
        raise ValueError('Expected the original full thinking RULER v3 prepared cohort')
    model_hashes = spec.get('tokenizer_hashes', {})
    if 'config.json' not in model_hashes or not any(k.startswith('tokenizer') for k in model_hashes):
        raise ValueError('Prepared model/tokenizer hashes are missing')
    for name, digest in model_hashes.items():
        if Path(name).name != name or file_hash(args.model/name) != digest:
            raise ValueError('L20 model/tokenizer differs from prepared inputs: ' + name)
    manifest_hash = file_hash(args.prepared/'manifest.jsonl')
    if receipt['files'].get('manifest.jsonl') != manifest_hash:
        raise ValueError('Prepared manifest changed')
    rows = []
    print('Adopting existing prepared inputs once; no generation or tokenization.', flush=True)
    for line in (args.prepared/'manifest.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line); path = relative(args.prepared, row['prepared'])
        digest = file_hash(path)
        if receipt['files'].get(row['prepared']) != digest:
            raise ValueError('Prepared sample changed: ' + row['prepared'])
        sample = validate(read(path), 'ruler-thinking')
        if (sample.get('model_config_sha256') != model_hashes['config.json']
                or sample.get('generator_config_sha256') != fingerprint(spec)
                or sample.get('task') != row['subtask'] or sample.get('evaluation_protocol') != RULER_PROTOCOL):
            raise ValueError('Prepared sample model/task/generation provenance differs')
        rows.append(dict(row, ordinal=sample['source_row'], sha256=digest, dataset='ruler',
                         input_tokens=len(sample['token_ids']), evaluation_protocol=RULER_PROTOCOL,
                         **{KEY: sample[KEY]}))
    validate_rows(settings, rows)
    rows.sort(key=lambda r: (r['ordinal'], TASKS.index(r['subtask'])))
    if file_hash(receipt_path) != receipt_hash or file_hash(args.prepared/'manifest.jsonl') != manifest_hash:
        raise ValueError('Prepared receipts changed during adoption')
    settings['code'] = code_hashes()  # provenance, never a resume source guard
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'adopt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(args.root/'settings.json', settings)
        freeze(args.root/'rows.json', rows)
        plan = dict(schema=COLLECTION_SCHEMA, settings_sha256=file_hash(args.root/'settings.json'),
            rows_sha256=file_hash(args.root/'rows.json'), preparation_sha256=receipt_hash,
            manifest_sha256=manifest_hash, samples=len(rows), probes=len(rows),
            answers=len(rows)*len(actions), actions=actions, phase_order=list(PHASES), baseline_answers=0)
        freeze(args.root/'plan.json', plan)
        for phase in ('features', 'ruler'):
            freeze(args.root/phase/'protocol.json', protocol_for(settings, plan, phase))
    print(f'Adopted {len(rows)} prompts: {len(rows)} probes, {plan["answers"]} trajectory answers, 5 TP2 pairs.', flush=True)


def load(base):
    base = Path(base); settings = read(base/'settings.json'); plan = read(base/'plan.json')
    if settings.get('schema') != COLLECTION_SCHEMA or plan.get('schema') != COLLECTION_SCHEMA:
        raise ValueError('Not an L20 thinking trajectory collection')
    if (file_hash(base/'settings.json') != plan['settings_sha256']
            or file_hash(base/'rows.json') != plan['rows_sha256']
            or file_hash(Path(settings['prepared'])/'preparation.json') != plan['preparation_sha256']
            or file_hash(Path(settings['prepared'])/'manifest.jsonl') != plan['manifest_sha256']):
        raise ValueError('Run metadata or prepared receipt changed')
    rows = read(base/'rows.json'); validate_rows(settings, rows)
    actions = control_actions(settings['percentages'], FIXED_METHOD)[1:]
    if (plan['actions'] != actions or plan['answers'] != len(rows)*len(actions)
            or plan['probes'] != len(rows) or plan['samples'] != len(rows) or plan['baseline_answers'] != 0
            or plan['phase_order'] != list(PHASES) or settings['position_attention_kernel'] != KERNEL):
        raise ValueError('Collection scope changed')
    return settings, plan, rows


def stage(base, role):
    if role not in ('ruler', 'features'):
        raise ValueError('Unknown collection stage')
    settings, plan, rows = load(base)
    protocol = read(Path(base)/role/'protocol.json')
    if protocol != protocol_for(settings, plan, role):
        raise ValueError('Stage metadata changed')
    return protocol, rows


def report(base, final=False):
    from runner.reporting import aggregate
    base = Path(base)
    settings, plan, rows = load(base)
    methods = {}; probes = 0
    expected = [dict(prompt_id=r['id'], subtask=r['subtask']) for r in rows]
    for role in ('features', 'ruler'):
        protocol, _ = stage(base, role)
        for case in protocol['scheduled_actions']:
            values = [value for row in rows if (value := accepted(base/role, case, row, protocol)) is not None]
            if case == 'probe':
                probes = len(values)
            else:
                methods[case] = aggregate(values, expected, {}, 'completed' if final else 'running')
    answers = sum(m['overall']['completed'] for m in methods.values())
    if final and (answers != plan['answers'] or probes != plan['probes']):
        raise ValueError('Cannot finalize incomplete collection')
    result = dict(final=final, expected_probes=plan['probes'], feature_probes=probes,
        expected_answers=plan['answers'], available_answers=answers, baseline_answers=0, methods=methods,
        report_validation='committed-results-only-v1', position_attention_kernel=KERNEL)
    supervisor = base/'ruler/supervisor.json'
    if supervisor.exists():
        saved = read(supervisor)
        result['supervisor'] = dict(saved, alive=control.alive(saved))
    atomic_json(base/('final.json' if final else 'status.json'), result)
    lines = [f'Features {probes}/{plan["probes"]}; last_traj_4f answers {answers}/{plan["answers"]}; no-cache 0.',
             '', '| Method | Done | Accuracy % | TTFT s |', '|---|---:|---:|---:|']
    for name, summary in methods.items():
        value = summary['overall']
        fmt = lambda v: 'N/A' if v is None else f'{v:.3f}'
        lines.append(f'| {name} | {value["completed"]}/{value["expected"]} | {fmt(value["accuracy_percent"])} | {fmt(value["mean_ttft_seconds"])} |')
    text = '\n'.join(lines)+'\n'
    (base/('final.md' if final else 'status.md')).write_text(text)
    print(text, flush=True)
    return result


def supervise(base):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (base/'ruler/run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), identity=control.identity(os.getpid()), state='running', role='ruler', started_at=time.time())
        atomic_json(base/'ruler/supervisor.json', state)
        try:
            for phase in PHASES:
                control.execute_phase(base, 'ruler', phase, state, run_loader=load, stage_loader=stage, entrypoint=__file__)
            report(base, final=True)
            freeze(base/'ruler/complete.json', dict(complete=True, owned_engines_exited=True,
                   plan_sha256=file_hash(base/'plan.json'), baseline_answers=0))
            state.update(state='complete', finished_at=time.time())
        except BaseException as error:
            state.update(state='failed', error=str(error), finished_at=time.time())
            raise
        finally:
            atomic_json(base/'ruler/supervisor.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('start', 'detach', 'resume', 'stop', 'status', 'report', 'supervise', 'worker'))
    p.add_argument('--root', type=Path, required=True)
    for name in ('model', 'prepared', 'cache-root'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--devices', default='')
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples-per-task', type=int, default=500)
    p.add_argument('--percentages', type=int, nargs='+', default=list(RATIOS))
    p.add_argument('--role', choices=('ruler',), default='ruler')
    p.add_argument('--phase', choices=PHASES)
    p.add_argument('--attempt')
    p.add_argument('--group', type=int, default=0)
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command != 'worker' and os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise ValueError('Coordinator must be CPU-only')
    if args.command in ('start', 'detach', 'resume'):
        if any(getattr(args, k) is None for k in ('model', 'prepared', 'cache_root')):
            p.error('Launch requires model/prepared/cache-root paths (loaded from the launcher env)')
        adopt(args)
        control.detach(args.root, 'ruler', resume=args.command == 'resume', stage_loader=stage, entrypoint=__file__)
    elif args.command == 'worker':
        if args.phase is None or args.attempt is None:
            p.error('worker requires --phase and --attempt')
        control.worker(args.root, 'ruler', args.phase, args.attempt, args.group, stage_loader=stage)
    elif args.command == 'supervise':
        supervise(args.root)
    elif args.command == 'stop':
        from scripts.launcher_stop import stop
        stop(args.root, Path(__file__).name, args.root/'ruler')
    else:
        report(args.root)


if __name__ == '__main__':
    main()
