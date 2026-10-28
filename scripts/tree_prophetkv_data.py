#!/usr/bin/env python3
"""Run only tree-prophetkv on manually copied, immutable thinking inputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.corpus import relative
from runner.corpus_records import accepted, NATIVE_ANSWER_VALIDATION
from runner.setups import atomic_json, file_hash, fingerprint
from runner.tree_collection import SCHEMA, RATIOS, actions, inventory
from runner.tree_prophetkv import PROTOCOL, budget
from runner.tree_profiles import validate
from runner.thinking_budget import KEY
from runner.output_budget import RULER_PROTOCOL, LONGBENCH_PROTOCOL, validate_sample_policy, validate_policy
from scripts.collection_v3 import declaration, seeds, answer_metrics
from scripts import longbench_a800_control as control
from scripts.longbench_a800_data import read, freeze
from scripts.ruler import TASKS

ROLE = 'tree'


def validate_rows(settings, rows):
    ruler = settings['dataset'] == 'ruler'
    local = settings.get('local_validation', False)
    selected = settings.get('validation_prompt_ids', [])
    count = len(TASKS)*settings['samples_per_task'] if ruler else 503
    if local:
        if (settings['tp'] != 4 or len(settings['groups']) != 1
                or settings['hardware_profile'] != 'rtx4500ada-tree-validation'
                or not selected or len(set(selected)) != len(selected)
                or {r['id'] for r in rows} != set(selected)):
            raise ValueError('Invalid explicit local validation selection/profile')
        count = len(selected)
    if len(rows) != count or len({r['id'] for r in rows}) != count:
        raise ValueError('Incomplete or duplicate prepared cohort')
    if ruler and not local:
        if (set(r['subtask'] for r in rows) != set(TASKS)
                or any(sorted(r['ordinal'] for r in rows if r['subtask'] == task)
                       != list(range(settings['samples_per_task'])) for task in TASKS)):
            raise ValueError('Expected all original RULER task ordinals')
    elif not ruler and not local and (len({r['source_id'] for r in rows}) != 503
          or sorted(r['ordinal'] for r in rows) != list(range(503))):
        raise ValueError('Expected all 503 distinct LongBench v2 source rows')
    for row in rows:
        if (not re.fullmatch('[a-zA-Z0-9_-]+', row['id'])
                or row['evaluation_protocol'] != settings['evaluation_protocol']):
            raise ValueError('Invalid prompt identity or thinking protocol')
        validate_policy(row[KEY])


def protect_paths(root, prepared, cache, model):
    """Allow a manually copied root/prepared, but never write inside inputs."""
    def overlaps(a, b):
        return a == b or a in b.parents or b in a.parents
    if (any(overlaps(cache, p) for p in (root, prepared, model))
            or overlaps(root, model) or overlaps(prepared, model)
            or root == prepared or prepared in root.parents):
        raise ValueError('Result/cache/model paths must be separate from prepared inputs')
    if root in prepared.parents and prepared != root/'prepared':
        raise ValueError('Nested inputs must live at EXPERIMENT_DIR/prepared')
    if root.exists() and any(p != prepared for p in root.iterdir()):
        raise ValueError('Use a fresh experiment directory containing only your copied prepared folder')
    if cache.exists() and any(cache.iterdir()):
        raise ValueError('Use a fresh empty tree cache root')


def protocol_for(settings, plan):
    result = dict(settings, actions=actions(settings['percentages']),
                  scheduled_actions=[a['id'] for a in actions(settings['percentages'])],
                  answer_validation=NATIVE_ANSWER_VALIDATION, plan_sha256=fingerprint(plan),
                  prepared_receipt_sha256=plan['preparation_sha256'])
    result.pop('code', None)
    inventory(result)
    return result


def adopt(args):
    from scripts.router_control import code_hashes
    ruler = args.dataset == 'ruler'
    local = getattr(args, 'local_validation', False)
    selected = getattr(args, 'validation_prompt_ids', None)
    if local != bool(selected) or (selected and len(set(selected)) != len(selected)):
        raise ValueError('Local validation requires explicit distinct --validation-prompt-ids')
    tp = 4 if local else 2
    if args.dataset not in ('ruler', 'longbench-v2') or args.tp != tp or args.seed != 42:
        raise ValueError('Expected RULER/LongBench v2, TP2 and preparation seed42')
    if type(args.samples_per_task) is not int or args.samples_per_task < 1:
        raise ValueError('Expected a positive RULER sample count')
    definitions = actions(args.percentages)
    groups = control.resolve_groups(args.devices, 4 if local else 10 if ruler else 4, tp=tp, query_inventory=False)
    settings = dict(schema=SCHEMA, dataset=args.dataset, tree_protocol=PROTOCOL,
        execution_profile='ruler-thinking' if ruler else 'longbench-v2',
        evaluation_protocol=RULER_PROTOCOL if ruler else LONGBENCH_PROTOCOL,
        model=str(args.model), prepared=str(args.prepared), cache_root=str(args.cache_root),
        tp=tp, hardware_profile='rtx4500ada-tree-validation' if local else 'l20-tp2' if ruler else 'server', groups=groups,
        samples_per_task=args.samples_per_task if ruler else None, seed=42,
        percentages=args.percentages, watchdog_seconds=7200,
        **declaration(args.dataset, args.answers_per_question, thinking=True))
    selected_seeds = getattr(args, 'answer_seeds', None)
    if selected_seeds is not None:
        if ruler:
            raise ValueError('Explicit answer seeds are supported only for LongBench v2 tree runs')
        settings['answer_seeds'] = list(selected_seeds)
    seeds(settings)  # Validate before adopting inputs or creating any run files.
    settings['phase_order'] = ['cached']
    if local:
        settings.update(local_validation=True, validation_prompt_ids=selected)
    if (args.root/'settings.json').exists():
        old, _, _ = load(args.root)
        if any(old.get(k) != v for k, v in settings.items()):
            raise ValueError('Existing run parameters differ; use its original env or fresh paths')
        return
    protect_paths(args.root, args.prepared, args.cache_root, args.model)
    receipt_path = args.prepared/'preparation.json'
    if not receipt_path.is_file():
        raise ValueError('Missing prepared/preparation.json; manually copy the complete prepared folder first')
    receipt_hash = file_hash(receipt_path)
    receipt = read(receipt_path); spec = receipt['spec']
    if ruler:
        if (spec.get('protocol') != RULER_PROTOCOL or spec.get('execution_profile') != 'ruler-thinking'
                or spec.get('samples') != args.samples_per_task or spec.get('seed') != 42
                or set(spec.get('tasks', ())) != set(TASKS)):
            raise ValueError('Expected original complete thinking RULER v3 preparation')
        model_hashes = spec.get('tokenizer_hashes', {})
    else:
        if (spec.get('collection_protocol') != settings['collection_protocol']
                or spec.get('generation_protocol') != settings['generation_protocol']
                or type(spec.get('answers_per_question')) is not int
                or spec['answers_per_question'] < args.answers_per_question
                or spec.get('answer_seeds') != list(range(spec['answers_per_question']))
                or not set(seeds(settings)).issubset(spec['answer_seeds'])):
            raise ValueError('LongBench prepared generation/answer seeds differ from this run')
        model_hashes = spec.get('tokenizer_files', {})
        if spec.get('model_config_sha256') != file_hash(args.model/'config.json'):
            raise ValueError('Prepared LongBench model differs')
    if 'config.json' not in model_hashes or not any(k.startswith('tokenizer') for k in model_hashes):
        raise ValueError('Prepared model/tokenizer hashes are missing')
    for name, digest in model_hashes.items():
        if Path(name).name != name or file_hash(args.model/name) != digest:
            raise ValueError('Model/tokenizer differs from prepared inputs: ' + name)
    manifest_hash = file_hash(args.prepared/'manifest.jsonl')
    if receipt['files'].get('manifest.jsonl') != manifest_hash:
        raise ValueError('Prepared manifest changed')
    rows = []
    print('Adopting manually copied prepared inputs; no copying, tokenization or preparation.', flush=True)
    for line in (args.prepared/'manifest.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if local and row['id'] not in selected:
            continue
        path = relative(args.prepared, row['prepared'])
        digest = file_hash(path)
        if receipt['files'].get(row['prepared']) != digest:
            raise ValueError('Prepared sample changed: ' + row['prepared'])
        sample = validate(read(path), settings['execution_profile'])
        validate_sample_policy(sample)
        if (sample.get('model_config_sha256') != model_hashes['config.json']
                or sample.get('evaluation_protocol') != settings['evaluation_protocol']):
            raise ValueError('Prepared sample model or thinking policy differs')
        if ruler and (sample.get('generator_config_sha256') != fingerprint(spec)
                      or sample.get('task') != row['subtask']):
            raise ValueError('RULER generation/task provenance differs')
        for definition in definitions:
            try:
                budget(sample['boundaries'], definition['ratio'])
            except ValueError as error:
                raise ValueError(f'{row["id"]} at {definition["id"]}: {error}') from error
        value = dict(row, ordinal=sample['source_row'], sha256=digest, dataset=args.dataset,
                     input_tokens=len(sample['token_ids']), evaluation_protocol=sample['evaluation_protocol'],
                     **{KEY: sample[KEY]})
        if not ruler:
            metadata = sample['source_metadata']
            if (metadata['length'] not in ('short', 'medium', 'long')
                    or row['subtask'] != metadata['domain']+' / '+metadata['sub_domain']):
                raise ValueError('LongBench source metadata differs')
            value.update(source_id=sample['source_id'], **metadata)
        rows.append(value)
    validate_rows(settings, rows)
    rows.sort(key=lambda r: (r['ordinal'], r['subtask']))
    if file_hash(receipt_path) != receipt_hash or file_hash(args.prepared/'manifest.jsonl') != manifest_hash:
        raise ValueError('Prepared receipts changed during adoption')
    settings['code'] = code_hashes()  # provenance; resume accepts source updates
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'adopt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(args.root/'settings.json', settings)
        freeze(args.root/'rows.json', rows)
        plan = dict(schema=SCHEMA, settings_sha256=file_hash(args.root/'settings.json'),
            rows_sha256=file_hash(args.root/'rows.json'), preparation_sha256=receipt_hash,
            manifest_sha256=manifest_hash, samples=len(rows), probes=0, baseline_answers=0,
            answers=len(rows)*len(definitions)*args.answers_per_question,
            actions=definitions, phase_order=['cached'])
        freeze(args.root/'plan.json', plan)
        freeze(args.root/ROLE/'protocol.json', protocol_for(settings, plan))
        freeze(args.root/ROLE/'devices.json', dict(tp=tp, groups=groups))
    print(f'Adopted {len(rows)} prompts; {plan["answers"]} tree answers; zero baseline/feature probes.', flush=True)


def load(base):
    base = Path(base); settings = read(base/'settings.json'); plan = read(base/'plan.json')
    if settings.get('schema') != SCHEMA or plan.get('schema') != SCHEMA:
        raise ValueError('Not a tree-prophetkv collection')
    if (file_hash(base/'settings.json') != plan['settings_sha256']
            or file_hash(base/'rows.json') != plan['rows_sha256']
            or file_hash(Path(settings['prepared'])/'preparation.json') != plan['preparation_sha256']
            or file_hash(Path(settings['prepared'])/'manifest.jsonl') != plan['manifest_sha256']):
        raise ValueError('Run metadata or prepared receipts changed')
    rows = read(base/'rows.json'); validate_rows(settings, rows)
    expected = actions(settings['percentages'])
    if (plan['actions'] != expected or plan['answers'] != len(rows)*len(expected)*len(seeds(settings))
            or plan['probes'] != 0 or plan['baseline_answers'] != 0 or plan['samples'] != len(rows)
            or plan['phase_order'] != ['cached'] or settings['phase_order'] != ['cached']):
        raise ValueError('Tree collection scope changed')
    return settings, plan, rows


def stage(base, role):
    if role != ROLE:
        raise ValueError('Tree collection has no baseline or feature stage')
    settings, plan, rows = load(base)
    protocol = read(Path(base)/ROLE/'protocol.json')
    if (protocol != protocol_for(settings, plan)
            or read(Path(base)/ROLE/'devices.json') != dict(tp=settings['tp'], groups=settings['groups'])):
        raise ValueError('Stage/device metadata changed')
    return protocol, rows


def report(base, final=False):
    from runner.reporting import aggregate
    base = Path(base); settings, plan, _ = load(base); protocol, rows = stage(base, ROLE)
    methods = {}
    expected = [dict(prompt_id=f'{r["id"]}/seed-{seed}', subtask=r['subtask'])
                for r in rows for seed in seeds(protocol)]
    for case in protocol['scheduled_actions']:
        values = []
        for row in rows:
            for seed in seeds(protocol):
                result = accepted(base/ROLE, case, row, protocol, seed=seed)
                if result is not None:
                    values.append(dict(answer_metrics(result), prompt_id=f'{row["id"]}/seed-{seed}'))
        methods[case] = aggregate(values, expected, {}, 'completed' if final else 'running')
    answers = sum(m['overall']['completed'] for m in methods.values())
    if final and answers != plan['answers']:
        raise ValueError('Cannot finalize incomplete tree collection')
    result = dict(final=final, samples=plan['samples'], expected_answers=plan['answers'],
        available_answers=answers, baseline_answers=0, feature_probes=0, methods=methods,
        answer_seeds=settings['answer_seeds'], report_validation='committed-results-only-v1')
    supervisor = base/ROLE/'supervisor.json'
    if supervisor.exists():
        saved = read(supervisor); result['supervisor'] = dict(saved, alive=control.alive(saved))
    atomic_json(base/('final.json' if final else 'status.json'), result)
    lines = [f'Tree answers {answers}/{plan["answers"]}; no-cache 0; feature probes 0.', '',
             '| Method | Done | Accuracy % | TTFT s |', '|---|---:|---:|---:|']
    for name, summary in methods.items():
        value = summary['overall']; fmt = lambda v: 'N/A' if v is None else f'{v:.3f}'
        lines.append(f'| {name} | {value["completed"]}/{value["expected"]} | {fmt(value["accuracy_percent"])} | {fmt(value["mean_ttft_seconds"])} |')
    text = '\n'.join(lines)+'\n'; (base/('final.md' if final else 'status.md')).write_text(text)
    print(text, flush=True)
    return result


def supervise(base):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (base/ROLE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), identity=control.identity(os.getpid()), state='running', role=ROLE, started_at=time.time())
        atomic_json(base/ROLE/'supervisor.json', state)
        try:
            control.execute_phase(base, ROLE, 'cached', state, run_loader=load, stage_loader=stage, entrypoint=__file__)
            report(base, final=True)
            freeze(base/ROLE/'complete.json', dict(complete=True, owned_engines_exited=True,
                   plan_sha256=file_hash(base/'plan.json'), baseline_answers=0, feature_probes=0))
            state.update(state='complete', finished_at=time.time())
        except BaseException as error:
            state.update(state='failed', error=str(error), finished_at=time.time()); raise
        finally:
            atomic_json(base/ROLE/'supervisor.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('configure', 'start', 'detach', 'resume', 'stop', 'status', 'report', 'supervise', 'worker'))
    p.add_argument('--root', type=Path, required=True)
    for name in ('model', 'prepared', 'cache-root'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--dataset', choices=('ruler', 'longbench-v2'))
    p.add_argument('--devices', default='')
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--local-validation', action='store_true', help='Explicit TP4 RTX 4500 Ada test only; never a server default')
    p.add_argument('--validation-prompt-ids', nargs='+')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples-per-task', type=int, default=500)
    p.add_argument('--answers-per-question', type=int, default=1)
    p.add_argument('--answer-seeds', type=int, nargs='+',
                   help='LongBench tree only: distinct ascending seeds from prepared provenance; count must match answers-per-question')
    p.add_argument('--percentages', type=int, nargs='+', default=list(RATIOS))
    p.add_argument('--role', choices=(ROLE,), default=ROLE)
    p.add_argument('--phase', choices=('cached',))
    p.add_argument('--attempt'); p.add_argument('--group', type=int, default=0)
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command != 'worker' and os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise ValueError('Coordinator must be CPU-only')
    if args.command in ('configure', 'start', 'detach', 'resume'):
        if any(getattr(args, k) is None for k in ('dataset', 'model', 'prepared', 'cache_root')):
            p.error('Launch requires dataset/model/prepared/cache-root (loaded from the launcher env)')
        adopt(args)
        if args.command != 'configure':
            control.detach(args.root, ROLE, resume=args.command == 'resume', stage_loader=stage, entrypoint=__file__)
    elif args.command == 'worker':
        if args.phase != 'cached' or args.attempt is None:
            p.error('worker requires --phase cached and --attempt')
        control.worker(args.root, ROLE, 'cached', args.attempt, args.group, stage_loader=stage)
    elif args.command == 'supervise':
        supervise(args.root)
    elif args.command == 'stop':
        from scripts.launcher_stop import stop
        stage(args.root, ROLE)
        stop(args.root, Path(__file__).name, args.root/ROLE)
    else:
        report(args.root)


if __name__ == '__main__':
    main()
