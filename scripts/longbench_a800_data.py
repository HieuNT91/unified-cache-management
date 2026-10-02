"""Frozen two-group LongBench v2 scope; CPU preparation and small metadata reads."""
import json
from pathlib import Path
from types import SimpleNamespace
from runner.setups import atomic_json, file_hash, fingerprint
from runner.longbench_features import DEFINITIONS
from runner.corpus_records import NATIVE_ANSWER_VALIDATION, PROBE_ANSWER_VALIDATION

RATIOS = (1, 5, 10, 20, 30, 40, 50, 60)
ACTIONS = [dict(id='nocache', method='baseline', ratio=None)] + [
    dict(id=f'prophetkv-{p}', method='prophetkv', ratio=p/100) for p in RATIOS]
SCHEDULE = dict(primary=['nocache', 'prophetkv-1', 'prophetkv-5', 'prophetkv-20'],
                extra=[f'prophetkv-{p}' for p in (10, 30, 40, 50, 60)], features=['probe'])
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


def protocol_for(settings, role, plan_sha):
    protocol = dict(schema='longbench-a800-stage-v1', dataset='longbench-v2',
                    kind='features' if role == 'features' else 'fixed-controls',
                    model=settings['model'], prepared=settings['prepared'], cache_root=settings['cache_root'],
                    groups=[settings['groups'][1 if role == 'extra' else 0]], actions=ACTIONS,
                    scheduled_actions=SCHEDULE[role], plan_sha256=plan_sha,
                    answer_validation=PROBE_ANSWER_VALIDATION if role == 'features' else NATIVE_ANSWER_VALIDATION)
    if role == 'features':
        protocol['feature_profile'] = DEFINITIONS
    return protocol


def prepare(base):
    from scripts.longbench_v2 import prepare as prepare_inputs
    base = Path(base); settings = read(base/'settings.json')
    prepared = Path(settings['prepared'])
    prepare_inputs(SimpleNamespace(model=Path(settings['model']), data=Path(settings['data']), output=prepared))
    receipt = read(prepared/'preparation.json')
    manifest = [json.loads(line) for line in (prepared/'manifest.jsonl').read_text().splitlines() if line.strip()]
    rows = []
    for ordinal, row in enumerate(manifest):
        sample = read(prepared/row['prepared'])
        length = sample['source_metadata']['length']
        if length not in LENGTHS:
            raise ValueError('Expected official LongBench short/medium/long labels')
        rows.append(dict(row, ordinal=ordinal, sha256=receipt['files'][row['prepared']],
                         dataset='longbench-v2', length=length, source_id=sample['source_id'],
                         input_tokens=len(sample['token_ids']), evaluation_protocol=sample.get('evaluation_protocol')))
    if len(rows) != 503 or len({r['id'] for r in rows}) != 503 or len({r['source_id'] for r in rows}) != 503:
        raise ValueError('Expected exactly 503 distinct original LongBench v2 rows')
    freeze(base/'rows.json', rows)
    plan = dict(schema='longbench-a800-two-group-v1', settings_sha256=file_hash(base/'settings.json'),
                rows_sha256=file_hash(base/'rows.json'), preparation_sha256=file_hash(prepared/'preparation.json'),
                samples=503, answers=503*9, probes=503, actions=ACTIONS, schedule=SCHEDULE,
                feature_profile=DEFINITIONS, evaluation='training', training_overlap=True,
                train_ids=[r['id'] for r in rows], heldout_ids=[], folds=assign_folds(rows), seed=42)
    freeze(base/'plan.json', plan)
    for role in SCHEDULE:
        freeze(base/role/'protocol.json', protocol_for(settings, role, file_hash(base/'plan.json')))
    return plan


def load(base, check_code=False):
    base = Path(base); settings = read(base/'settings.json'); plan = read(base/'plan.json')
    if (file_hash(base/'settings.json') != plan['settings_sha256']
            or file_hash(base/'rows.json') != plan['rows_sha256']):
        raise ValueError('Frozen experiment metadata changed')
    rows = read(base/'rows.json')
    if (len(rows) != 503 or len({r['id'] for r in rows}) != 503
            or plan['actions'] != ACTIONS or plan['schedule'] != SCHEDULE or plan['feature_profile'] != DEFINITIONS):
        raise ValueError('Unexpected experiment scope')
    if check_code:
        from scripts.router_control import code_hashes
        if settings['code'] != code_hashes():
            raise ValueError('Implementation changed; keep the original checkout for resume')
    return settings, plan, rows


def stage(base, role, check_code=False):
    settings, plan, rows = load(base, check_code)
    protocol = read(Path(base)/role/'protocol.json')
    if protocol != protocol_for(settings, role, file_hash(Path(base)/'plan.json')):
        raise ValueError('Stage metadata changed')
    return protocol, rows
