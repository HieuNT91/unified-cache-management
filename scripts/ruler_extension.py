"""Read-only parent linkage for L40 thinking 30 + 170 collection."""
import json
import re
from pathlib import Path

from runner.setups import file_hash, fingerprint


SCHEMA = 'ruler-thinking-extension-30-to-200-v1'
PARENT_FILES = ('settings.json', 'plan.json', 'rows.json', 'ruler-data.json',
                'ruler-data.json.sha256', 'ruler/complete.json',
                'ruler/controls-complete.json', 'features/complete.json')


def read(path):
    return json.loads(Path(path).read_text())


def validate_extension(settings):
    ext = settings.get('extension', {})
    if (settings.get('execution_profile') != 'ruler-thinking'
            or settings.get('hardware_profile') not in ('l40-tp2', 'l40-tp4')
            or settings.get('samples_per_task') != 170
            or ext.get('schema') != SCHEMA or ext.get('start') != 30
            or ext.get('target_samples_per_task') != 200
            or not Path(ext.get('parent_root', '')).is_absolute()
            or not Path(ext.get('parent_prepared', '')).is_absolute()
            or set(ext.get('parent_files', {})) != set(PARENT_FILES)
            or not re.fullmatch('[0-9a-f]{64}', ext.get('parent_preparation_sha256', ''))
            or any(not re.fullmatch('[0-9a-f]{64}', v) for v in ext.get('parent_files', {}).values())):
        raise ValueError('Invalid L40 thinking 30-to-200 extension provenance')


def check_parent(ext):
    root = Path(ext['parent_root'])
    for name, digest in ext['parent_files'].items():
        if file_hash(root/name) != digest:
            raise ValueError(f'Parent experiment changed: {name}')
    if file_hash(Path(ext['parent_prepared'])/'preparation.json') != ext['parent_preparation_sha256']:
        raise ValueError('Parent preparation receipt changed')


def configure_extension(args, settings):
    from scripts.longbench_a800_data import load, stage
    from scripts.router_dataset import load as load_dataset
    root = Path(args.extend_from).resolve()
    old, plan, rows = load(root)
    if (old['dataset'] != 'ruler' or old.get('execution_profile') != 'ruler-thinking'
            or old.get('samples_per_task') != 30 or 'extension' in old
            or old.get('seed') != 42):
        raise ValueError('Extension parent must be a completed 30/task thinking RULER run')
    for key in ('tp', 'hardware_profile', 'evaluation_protocol', 'thinking_cap', 'answer_caps', 'engine_window', 'sampling'):
        if old.get(key) != settings.get(key):
            raise ValueError(f'Extension must preserve parent {key}')
    # Do not let preparation, cache cleanup or result publication touch the parent.
    old_paths = [root, Path(old['prepared']).resolve(), Path(old['cache_root']).resolve()]
    new_paths = [Path(p).resolve() for p in (args.root, args.prepared, args.cache_root)]
    for new in new_paths:
        for prior in old_paths:
            if new == prior or new in prior.parents or prior in new.parents:
                raise ValueError('Extension requires fresh result/prepared/cache paths outside the parent paths')
    for i, new in enumerate(new_paths):
        if any(new == other or new in other.parents or other in new.parents for other in new_paths[i+1:]):
            raise ValueError('Extension result/prepared/cache paths must be separate')
    completed = read(root/'ruler/complete.json')
    if (completed.get('complete') is not True or completed.get('owned_engines_exited') is not True
            or completed.get('plan_sha256') != file_hash(root/'plan.json')):
        raise ValueError('Parent experiment is not complete')
    for role, name, count_key, expected in (
            ('ruler', 'controls-complete.json', 'answers', 390*12),
            ('features', 'complete.json', 'probes', 390)):
        stage(root, role)
        done = read(root/role/name)
        if (done.get('owned_engines_exited') is not True or done.get(count_key) != expected
                or done.get('protocol_sha256') != file_hash(root/role/'protocol.json')):
            raise ValueError(f'Parent {role} completion receipt is invalid')
    dataset = load_dataset(root/'ruler-data.json')
    if (dataset['provenance'].get('plan_sha256') != file_hash(root/'plan.json')
            or any(dataset['provenance'].get(k) != old.get(k) for k in
                   ('tp', 'seed', 'samples_per_task', 'hardware_profile', 'execution_profile', 'evaluation_protocol'))
            or {r['id']: r['input_sha256'] for r in dataset['rows']} != {r['id']: r['sha256'] for r in rows}):
        raise ValueError('Parent portable dataset does not match the completed plan')
    prepared = Path(old['prepared']).resolve()
    if plan.get('preparation_sha256') != file_hash(prepared/'preparation.json'):
        raise ValueError('Parent preparation changed since collection')
    return dict(schema=SCHEMA, start=30, target_samples_per_task=200,
                parent_root=str(root), parent_prepared=str(prepared),
                parent_files={n: file_hash(root/n) for n in PARENT_FILES},
                parent_preparation_sha256=file_hash(prepared/'preparation.json'))


def checked_json(root, name, receipt):
    if receipt['files'].get(name) != file_hash(root/name):
        raise ValueError(f'Prepared artifact changed: {root/name}')
    return read(root/name)


def verify_prepared_extension(settings, receipt, manifest):
    """One-time CPU prefix/content audit before publishing the delta plan."""
    from scripts.ruler import TASKS
    validate_extension(settings)
    ext = settings['extension']; check_parent(ext)
    prior = Path(ext['parent_prepared']); prepared = Path(settings['prepared'])
    old_receipt = read(prior/'preparation.json')
    before, after = old_receipt['spec'], receipt['spec']
    if (before.get('samples') != 30 or after.get('samples') != 200
            or {k: v for k, v in before.items() if k != 'samples'} !=
               {k: v for k, v in after.items() if k != 'samples'}):
        raise ValueError('Parent/new generator, tokenizer, assets, versions or policy differ; prefix reuse refused')
    if file_hash(prior/'manifest.jsonl') != old_receipt['files']['manifest.jsonl']:
        raise ValueError('Parent manifest changed')
    old_manifest = [json.loads(line) for line in (prior/'manifest.jsonl').read_text().splitlines() if line.strip()]
    old_by_id = {r['id']: r for r in old_manifest}
    parent_rows = read(Path(ext['parent_root'])/'rows.json')
    if (len(old_manifest) != 390 or len(old_by_id) != 390
            or {r['id']: old_receipt['files'][r['prepared']] for r in old_manifest}
               != {r['id']: r['sha256'] for r in parent_rows}):
        raise ValueError('Parent prepared membership does not match its collected rows')
    seen_ids = set(); task_counts = {}; selected = []
    # Raw comparison proves the source stream prefix, including evaluation fields.
    for task in TASKS:
        name = f'raw/{task}/validation.jsonl'
        for directory, rec in ((prior, old_receipt), (prepared, receipt)):
            if rec['files'].get(name) != file_hash(directory/name):
                raise ValueError(f'Raw source changed: {directory/name}')
        old_raw = [json.loads(s) for s in (prior/name).read_text().splitlines() if s.strip()]
        new_raw = [json.loads(s) for s in (prepared/name).read_text().splitlines() if s.strip()]
        if len(old_raw) != 30 or len(new_raw) != 200 or new_raw[:30] != old_raw:
            raise ValueError(f'200-row source prefix differs from original 30 rows: {task}; no GPU work scheduled')
    token_hashes = {task: set() for task in TASKS}
    for row in manifest:
        task = row['subtask']; ordinal = task_counts.get(task, 0)
        task_counts[task] = ordinal+1
        if task not in TASKS or row['id'] != f'{task}-{ordinal:03d}' or row['id'] in seen_ids:
            raise ValueError('Unexpected full-200 manifest membership/order')
        seen_ids.add(row['id'])
        sample = checked_json(prepared, row['prepared'], receipt)
        token_digest = fingerprint(sample['token_ids'])
        if token_digest in token_hashes[task]:
            raise ValueError(f'Duplicate prompt in 200-row source: {row["id"]}')
        token_hashes[task].add(token_digest)
        if ordinal < 30:
            old_row = old_by_id.get(row['id'])
            if old_row != row:
                raise ValueError(f'Parent manifest prefix differs: {row["id"]}')
            old_sample = checked_json(prior, old_row['prepared'], old_receipt)
            # Only the declared generator batch-size hash is expected to differ.
            canonical = lambda s: {k: v for k, v in s.items() if k != 'generator_config_sha256'}
            if canonical(sample) != canonical(old_sample):
                raise ValueError(f'Parent token/layout/policy prefix differs: {row["id"]}')
        else:
            selected.append(row['id'])
    if task_counts != {t: 200 for t in TASKS}:
        raise ValueError('Expected full 13x200 prepared batch')
    check_parent(ext)
    return dict(schema=SCHEMA, extension=ext, prepared_manifest_sha256=receipt['files']['manifest.jsonl'],
                prefix_prompts_checked=390, new_prompts=2210, selected_ids_sha256=fingerprint(selected),
                comparison='exact raw rows and prepared samples except generator batch-size hash')


def combine_exports(base, delta, delta_path):
    """Publish a portable union; each row keeps its original input/record hashes."""
    from scripts.router_dataset import load, save
    from scripts.longbench_a800_data import read as read_settings
    settings = read_settings(Path(base)/'settings.json'); ext = settings['extension']
    check_parent(ext)
    parent_path = Path(ext['parent_root'])/'ruler-data.json'
    parent = load(parent_path)
    if set(r['id'] for r in parent['rows']) & set(r['id'] for r in delta['rows']):
        raise ValueError('Parent and extension exports overlap')
    provenance = {k: v for k, v in delta['provenance'].items() if k not in ('extension', 'plan_sha256')}
    provenance.update(samples_per_task=200, collection='verified-30-plus-170',
        preparation_extension_sha256=file_hash(Path(base)/'extension.json'),
        components=[dict(path=str(path), sha256=file_hash(path), provenance=value['provenance'])
                    for path, value in ((parent_path, parent), (Path(delta_path), delta))])
    rows = sorted(parent['rows']+delta['rows'], key=lambda r: r['id'])
    return save(Path(base)/'ruler-data-200.json', 'ruler', delta['actions'], rows, provenance)
