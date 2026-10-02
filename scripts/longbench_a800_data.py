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
    return ('ruler', 'features') if settings['dataset'] == 'ruler' else ('primary', 'extra', 'features')


def scheduled(settings, role):
    return [a['id'] for a in ACTIONS] if role == 'ruler' else SCHEDULE[role]


def device_role(settings, role):
    return ('ruler' if settings['dataset'] == 'ruler' else 'primary') if role == 'features' else role


def protocol_for(settings, role, plan_sha, groups):
    protocol = dict(schema='tp2-data-stage-v2', dataset=settings['dataset'], tp=settings['tp'],
                    kind='features' if role == 'features' else 'fixed-controls', hardware_profile=settings.get('hardware_profile','server'),
                    model=settings['model'], prepared=settings['prepared'], cache_root=settings['cache_root'],
                    groups=groups, actions=ACTIONS, scheduled_actions=scheduled(settings, role), plan_sha256=plan_sha,
                    answer_validation=PROBE_ANSWER_VALIDATION if role == 'features' else NATIVE_ANSWER_VALIDATION)
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
    if dataset == 'longbench-v2':
        from scripts.longbench_v2 import prepare as prepare_inputs
        prepare_inputs(SimpleNamespace(model=Path(settings['model']), data=Path(settings['data']), output=prepared))
    else:
        from scripts.ruler import prepare as prepare_inputs
        prepare_inputs(SimpleNamespace(model=Path(settings['model']), ruler=Path(settings['data']), output=prepared,
                                       samples=settings['samples_per_task'], seed=settings['seed']))
    receipt = read(prepared/'preparation.json')
    if file_hash(prepared/'manifest.jsonl') != receipt['files']['manifest.jsonl']:
        raise ValueError('Prepared manifest changed')
    manifest = [json.loads(line) for line in (prepared/'manifest.jsonl').read_text().splitlines() if line.strip()]
    rows = []; counts = {}
    for index, row in enumerate(manifest):
        sample = read(prepared/row['prepared']); task = row['subtask']
        ordinal = index if dataset == 'longbench-v2' else counts.get(task, 0)
        counts[task] = counts.get(task, 0)+1
        value = dict(row, ordinal=ordinal, sha256=receipt['files'][row['prepared']], dataset=dataset,
                     input_tokens=len(sample['token_ids']), evaluation_protocol=sample.get('evaluation_protocol'))
        if dataset == 'longbench-v2':
            length = sample['source_metadata']['length']
            if length not in LENGTHS:
                raise ValueError('Expected official LongBench short/medium/long labels')
            value.update(length=length, source_id=sample['source_id'])
        rows.append(value)
    validate_rows(settings, rows)
    freeze(base/'rows.json', rows)
    plan = dict(schema='tp2-data-plan-v2', settings_sha256=file_hash(base/'settings.json'),
                rows_sha256=file_hash(base/'rows.json'), preparation_sha256=file_hash(prepared/'preparation.json'),
                samples=len(rows), answers=len(rows)*len(ACTIONS), probes=len(rows), actions=ACTIONS,
                schedule={role: scheduled(settings, role) for role in roles(settings)},
                feature_profile=DEFINITIONS, evaluation='training', training_overlap=True,
                train_ids=[r['id'] for r in rows], heldout_ids=[], folds=assign_folds(rows), seed=42)
    freeze(base/'plan.json', plan)
    publish_protocols(base)
    return plan


def validate_rows(settings, rows):
    expected = 503 if settings['dataset'] == 'longbench-v2' else 1300
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError('Incomplete/duplicate dataset membership')
    if settings['dataset'] == 'longbench-v2':
        if len({r['source_id'] for r in rows}) != 503 or {r['ordinal'] for r in rows} != set(range(503)):
            raise ValueError('Expected 503 unique source rows')
    else:
        from scripts.ruler import TASKS
        if (set(r['subtask'] for r in rows) != set(TASKS) or
                any(sorted(r['ordinal'] for r in rows if r['subtask'] == t) != list(range(100)) for t in TASKS)):
            raise ValueError('Expected 13 tasks x100 source ordinals')


def load(base, check_code=False):
    base = Path(base); settings = read(base/'settings.json'); plan = read(base/'plan.json')
    if (file_hash(base/'settings.json') != plan['settings_sha256']
            or file_hash(base/'rows.json') != plan['rows_sha256']):
        raise ValueError('Frozen experiment metadata changed')
    rows = read(base/'rows.json'); validate_rows(settings, rows)
    if (plan['actions'] != ACTIONS or plan['schedule'] != {r: scheduled(settings, r) for r in roles(settings)}
            or plan['feature_profile'] != DEFINITIONS):
        raise ValueError('Unexpected experiment scope')
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
