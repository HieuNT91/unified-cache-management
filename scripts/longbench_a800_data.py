"""Frozen two-group LongBench v2 scope; CPU preparation and small metadata reads."""
import json
from pathlib import Path
from types import SimpleNamespace
from runner.setups import atomic_json, file_hash, fingerprint
from runner.longbench_features import DEFINITIONS
from runner.corpus_records import NATIVE_ANSWER_VALIDATION, PROBE_ANSWER_VALIDATION

RATIOS = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90)
ACTIONS = [dict(id='nocache', method='baseline', ratio=None)] + [
    dict(id=f'prophetkv-{p}', method='prophetkv', ratio=p/100) for p in RATIOS]
SCHEDULE = dict(primary=['nocache', *[f'prophetkv-{p}' for p in (1, 5, 20, 40, 70, 90)]],
                extra=[f'prophetkv-{p}' for p in (10, 30, 50, 60, 80)], features=['probe'])
LENGTHS = ('short', 'medium', 'long')


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen metadata changed: {path}; use a new experiment directory')
    else:
        atomic_json(path, value)


def assign_folds(rows, seed=42):
    """Predeclared five folds, stratified by dataset/length or RULER task."""
    groups = sorted({(r['dataset'], r.get('length', r['subtask'])) for r in rows})
    folds = {}; offset = 0
    for dataset, stratum in groups:
        ordered = sorted((r for r in rows if (r['dataset'], r.get('length', r['subtask'])) == (dataset, stratum)),
                         key=lambda r: (fingerprint([seed, 'fold', dataset, r['id']]), r['id']))
        for j, row in enumerate(ordered):
            folds[f"{dataset}:{row['id']}"] = (offset+j) % 5
        offset = (offset+len(ordered)) % 5
    return folds


def roles(settings):
    if settings.get('naive_reuse'):
        return ('ruler',) if settings['dataset'] == 'ruler' else ('primary',)
    return ('ruler', 'features') if settings['dataset'] == 'ruler' else ('primary', 'extra', 'features')


def scheduled(settings, role):
    if settings.get('naive_reuse'):
        if role not in roles(settings):
            raise ValueError('Naive reuse has no extra or features stage')
        return ['naive-reuse']
    return [a['id'] for a in ACTIONS] if role == 'ruler' else SCHEDULE[role]


def actions_for(settings):
    if settings.get('naive_reuse'):
        from runner.naive_reuse import ACTION
        return [dict(ACTION)]
    return ACTIONS


def device_role(settings, role):
    return ('ruler' if settings['dataset'] == 'ruler' else 'primary') if role == 'features' else role


def protocol_for(settings, role, plan_sha, groups):
    protocol = dict(schema='tp2-data-stage-v2', dataset=settings['dataset'], tp=settings['tp'],
                    kind='features' if role == 'features' else 'fixed-controls', hardware_profile=settings.get('hardware_profile','server'),
                    model=settings['model'], prepared=settings['prepared'], cache_root=settings['cache_root'],
                    groups=groups, actions=actions_for(settings), scheduled_actions=scheduled(settings, role), plan_sha256=plan_sha,
                    answer_validation=PROBE_ANSWER_VALIDATION if role == 'features' else NATIVE_ANSWER_VALIDATION)
    if settings.get('execution_profile'):
        protocol['execution_profile'] = settings['execution_profile']
        protocol['evaluation_protocol'] = settings['evaluation_protocol']
    if 'hardware_preflight' in settings:
        protocol['hardware_preflight'] = settings['hardware_preflight']
    if settings.get('naive_reuse'):
        protocol['naive_reuse'] = True
    if role == 'features':
        protocol['feature_profile'] = DEFINITIONS
    return protocol


def publish_protocols(base):
    base = Path(base); settings = read(base/'settings.json')
    for role in roles(settings):
        path = base/device_role(settings, role)/'devices.json'
        if path.exists():
            freeze(base/role/'protocol.json', protocol_for(settings, role, file_hash(base/'plan.json'), read(path)['groups']))


def prepare(base):
    base = Path(base); settings = read(base/'settings.json'); dataset = settings['dataset']
    prepared = Path(settings['prepared'])
    if settings.get('naive_reuse'):
        from scripts.naive_reuse_inputs import validate_existing
        validate_existing(settings)
    elif dataset == 'longbench-v2':
        from scripts.longbench_v2 import prepare as prepare_inputs
        prepare_inputs(SimpleNamespace(model=Path(settings['model']), data=Path(settings['data']), output=prepared))
    else:
        from scripts.ruler import prepare as prepare_inputs
        prepare_inputs(SimpleNamespace(model=Path(settings['model']), ruler=Path(settings['data']), output=prepared,
                                       samples=settings.get('extension', {}).get('target_samples_per_task', settings['samples_per_task']), seed=settings['seed'],
                                       thinking=settings.get('execution_profile') == 'ruler-thinking'))
    receipt = read(prepared/'preparation.json')
    if file_hash(prepared/'manifest.jsonl') != receipt['files']['manifest.jsonl']:
        raise ValueError('Prepared manifest changed')
    manifest = [json.loads(line) for line in (prepared/'manifest.jsonl').read_text().splitlines() if line.strip()]
    if 'extension' in settings:
        from scripts.ruler_extension import verify_prepared_extension
        freeze(base/'extension.json', verify_prepared_extension(settings, receipt, manifest))
    rows = []; counts = {}; start = ruler_start(settings)
    for index, row in enumerate(manifest):
        task = row['subtask']
        ordinal = index if dataset == 'longbench-v2' else counts.get(task, 0)
        counts[task] = counts.get(task, 0)+1
        if ordinal < start:
            continue
        sample = read(prepared/row['prepared'])
        value = dict(row, ordinal=ordinal, sha256=receipt['files'][row['prepared']], dataset=dataset,
                     input_tokens=len(sample['token_ids']), evaluation_protocol=sample.get('evaluation_protocol'))
        if dataset == 'longbench-v2':
            length = sample['source_metadata']['length']
            if length not in LENGTHS:
                raise ValueError('Expected official LongBench short/medium/long labels')
            value.update(length=length, source_id=sample['source_id'])
        if settings.get('execution_profile') == 'ruler-thinking':
            from runner.thinking_budget import KEY, validate_sample_policy
            validate_sample_policy(sample)
            value[KEY] = sample[KEY]
        rows.append(value)
    validate_rows(settings, rows)
    freeze(base/'rows.json', rows)
    plan = dict(schema='tp2-data-plan-v2', settings_sha256=file_hash(base/'settings.json'),
                rows_sha256=file_hash(base/'rows.json'), preparation_sha256=file_hash(prepared/'preparation.json'),
                samples=len(rows), answers=len(rows)*len(actions_for(settings)),
                probes=0 if settings.get('naive_reuse') else len(rows), actions=actions_for(settings),
                schedule={role: scheduled(settings, role) for role in roles(settings)},
                feature_profile=None if settings.get('naive_reuse') else DEFINITIONS, evaluation='training', training_overlap=True,
                train_ids=[r['id'] for r in rows], heldout_ids=[], folds=assign_folds(rows), seed=42)
    if settings.get('naive_reuse'):
        plan.update(evaluation='fixed-control', training_overlap=False, train_ids=[], folds={})
    if 'extension' in settings:
        plan['extension_sha256'] = file_hash(base/'extension.json')
    freeze(base/'plan.json', plan)
    publish_protocols(base)
    return plan


def ruler_samples(settings):
    count = settings.get('samples_per_task', 100)
    allowed = (30, 200) if settings.get('execution_profile') == 'ruler-thinking' else (100, 200, 500)
    if 'extension' in settings:
        ruler_start(settings)
        ext = settings['extension']
        allowed = (ext['target_samples_per_task']-ext['start'],)
    if type(count) is not int or count not in allowed:
        raise ValueError('RULER collection requires 30/200 thinking, 100 or 200 or 500 non-thinking, or an explicit extension delta')
    return count


def ruler_start(settings):
    if 'extension' not in settings:
        return 0
    from scripts.ruler_extension import validate_extension
    validate_extension(settings)
    return settings['extension']['start']


def validate_rows(settings, rows):
    count = ruler_samples(settings) if settings['dataset'] == 'ruler' else None
    expected = 503 if count is None else 13*count
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError('Incomplete/duplicate dataset membership')
    if settings['dataset'] == 'longbench-v2':
        if len({r['source_id'] for r in rows}) != 503 or {r['ordinal'] for r in rows} != set(range(503)):
            raise ValueError('Expected 503 unique source rows')
    else:
        from scripts.ruler import TASKS
        start = ruler_start(settings)
        if settings.get('execution_profile') == 'ruler-thinking':
            from runner.thinking_budget import PROTOCOL, KEY, validate_policy
            if any(r.get('evaluation_protocol') != PROTOCOL for r in rows):
                raise ValueError('Thinking RULER evaluation identity mismatch')
            for row in rows:
                from scripts.ruler import CAPS
                policy = validate_policy(row[KEY])
                if policy['answer_cap'] != CAPS.get(row['subtask']) or row['input_tokens']+policy['output_reserve'] > 82304:
                    raise ValueError('Thinking row task cap or capacity mismatch')
        if (set(r['subtask'] for r in rows) != set(TASKS) or
                any(sorted(r['ordinal'] for r in rows if r['subtask'] == t) != list(range(start, start+count)) for t in TASKS)):
            raise ValueError(f'Expected 13 tasks x{count} source ordinals')


def load(base, check_code=False):
    base = Path(base); settings = read(base/'settings.json'); plan = read(base/'plan.json')
    if (file_hash(base/'settings.json') != plan['settings_sha256']
            or file_hash(base/'rows.json') != plan['rows_sha256']):
        raise ValueError('Frozen experiment metadata changed')
    rows = read(base/'rows.json'); validate_rows(settings, rows)
    if 'extension' in settings and plan.get('extension_sha256') != file_hash(base/'extension.json'):
        raise ValueError('Extension preparation receipt changed')
    if (plan['actions'] != actions_for(settings) or plan['schedule'] != {r: scheduled(settings, r) for r in roles(settings)}
            or plan['feature_profile'] != (None if settings.get('naive_reuse') else DEFINITIONS)):
        raise ValueError('Unexpected experiment scope')
    if settings.get('naive_reuse') and (plan['probes'] != 0 or plan['answers'] != len(rows)
            or file_hash(Path(settings['prepared'])/'preparation.json') != plan['preparation_sha256']):
        raise ValueError('Naive reuse scope or prepared receipt changed')
    if check_code:
        from scripts.router_control import code_hashes
        if settings['code'] != code_hashes():
            raise ValueError('Implementation changed; keep the original checkout for resume')
    return settings, plan, rows


def stage(base, role, check_code=False):
    settings, plan, rows = load(base, check_code)
    base = Path(base)
    protocol = read(base/role/'protocol.json')
    groups = read(base/device_role(settings, role)/'devices.json')['groups']
    if protocol != protocol_for(settings, role, file_hash(base/'plan.json'), groups):
        raise ValueError('Stage metadata changed')
    return protocol, rows
