"""Read-only parent linkage for L40 thinking 30+170 and L20 non-thinking 200+300."""
import json
import re
from pathlib import Path

from runner.setups import file_hash, fingerprint


SCHEMA = 'ruler-thinking-extension-30-to-200-v1'
L20_SCHEMA = 'ruler-nonthinking-extension-200-to-500-v1'
# d0483b1 -> 498e9dd: unchanged non-thinking formatter/generator/template/query
# functions; optional thinking preparation added, then check_model import moved.
LEGACY_NONTHINKING_ADAPTER = 'c0d46130b722ef1e704a42d66444898c50062fb84011ca572db5d8146f776262'
PARENT_FILES = ('settings.json', 'plan.json', 'rows.json', 'ruler-data.json',
                'ruler-data.json.sha256', 'ruler/complete.json',
                'ruler/controls-complete.json', 'features/complete.json')
CONTROL_FILES = ('settings.json', 'plan.json', 'rows.json', 'ruler/controls-complete.json',
                 'ruler/protocol.json', 'features/protocol.json', 'ruler/devices.json')
# cd5a3c0 -> 498e9dd changes only the check_model import in scripts/ruler.py
# from run to runner.preparation. The moved check_model function has identical
# AST; generator/formatting logic is byte-for-byte unchanged. This is a specific
# source mapping, not permission to ignore arbitrary adapter changes.
ADAPTER_IMPORT_MOVE = (
    '65d47398285cc6ab322dec3901cb938d03013eb08b79a665cc9c7711a14865a5',
    'd616631cfa085110bfed5cf41f559b5951a71701bbc2c347d3238ac0e402eebc',
)
# d0483b1 -> cd5a3c0 changes only sample_provenance: optional thinking
# metadata is retained when present. Every chunking/token/layout function is
# unchanged, and non-thinking samples have neither optional field.
NONTHINKING_CHUNKER_MOVE = (
    '54d1ad351fcd3aed2dc0be3dc58e8e2e0ecc9d581472cb1c0ce05d0d7cc51cf0',
    '7b6f3af413cf8ac4eb43384a0519be5bf461677ec1c5703d7d01ed2fb32fe940',
)


def read(path):
    return json.loads(Path(path).read_text())


def validate_extension(settings):
    ext = settings.get('extension', {})
    scope = ext.get('parent_scope', 'complete')
    expected_files = CONTROL_FILES if scope == 'controls-complete' else PARENT_FILES
    thinking = (settings.get('execution_profile') == 'ruler-thinking'
                and settings.get('hardware_profile') in ('l40-tp2', 'l40-tp4')
                and settings.get('samples_per_task') == 170 and ext.get('schema') == SCHEMA
                and ext.get('start') == 30 and ext.get('target_samples_per_task') == 200)
    nonthinking = (settings.get('execution_profile') in (None, 'ruler')
                   and settings.get('hardware_profile') == 'l20-tp2' and settings.get('tp') == 2
                   and settings.get('samples_per_task') == 300 and ext.get('schema') == L20_SCHEMA
                   and ext.get('start') == 200 and ext.get('target_samples_per_task') == 500)
    if (not (thinking or nonthinking)
            or not Path(ext.get('parent_root', '')).is_absolute()
            or not Path(ext.get('parent_prepared', '')).is_absolute()
            or scope not in ('complete', 'controls-complete')
            or set(ext.get('parent_files', {})) != set(expected_files)
            or not re.fullmatch('[0-9a-f]{64}', ext.get('parent_preparation_sha256', ''))
            or any(not re.fullmatch('[0-9a-f]{64}', v) for v in ext.get('parent_files', {}).values())):
        raise ValueError('Invalid RULER extension provenance')


def check_parent(ext):
    root = Path(ext['parent_root'])
    for name, digest in ext['parent_files'].items():
        if file_hash(root/name) != digest:
            raise ValueError(f'Parent experiment changed: {name}')
    if file_hash(Path(ext['parent_prepared'])/'preparation.json') != ext['parent_preparation_sha256']:
        raise ValueError('Parent preparation receipt changed')


def configure_extension(args, settings, groups=None):
    from scripts.longbench_a800_data import load, stage
    root = Path(args.extend_from).resolve()
    old, plan, rows = load(root)
    thinking = settings.get('execution_profile') == 'ruler-thinking'
    start, target = (30, 200) if thinking else (200, 500)
    schema = SCHEMA if thinking else L20_SCHEMA
    if (old['dataset'] != 'ruler' or old.get('execution_profile') != settings.get('execution_profile')
            or old.get('samples_per_task') != start or 'extension' in old
            or old.get('seed') != 42):
        raise ValueError(f'Extension parent must be a {start}/task RULER run with matching thinking mode and completed controls')
    for key in ('tp', 'hardware_profile', 'evaluation_protocol', 'thinking_cap', 'answer_caps', 'engine_window', 'sampling'):
        if old.get(key) != settings.get(key):
            raise ValueError(f'Extension must preserve parent {key}')
    if not thinking and groups != read(root/'ruler/devices.json')['groups']:
        raise ValueError('L20 extension must preserve the five parent GPU UUID pairs and ordering')
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
    # Completed controls are sufficient to select new inputs. Probes and the
    # parent's final export may still be pending; never invent their receipts.
    for role in ('ruler', 'features'):
        stage(root, role)
    marker = root/'ruler/controls-complete.json'
    if not marker.is_file():
        raise ValueError('Parent controls are not complete: missing ruler/controls-complete.json')
    done = read(marker)
    if (done.get('owned_engines_exited') is not True or done.get('answers') != 13*start*12
            or done.get('protocol_sha256') != file_hash(root/'ruler/protocol.json')):
        raise ValueError('Parent controls completion receipt is invalid')
    prepared = Path(old['prepared']).resolve()
    if plan.get('preparation_sha256') != file_hash(prepared/'preparation.json'):
        raise ValueError('Parent preparation changed since collection')
    return dict(schema=schema, start=start, target_samples_per_task=target,
                parent_root=str(root), parent_prepared=str(prepared),
                parent_scope='controls-complete',
                parent_files={n: file_hash(root/n) for n in CONTROL_FILES},
                parent_preparation_sha256=file_hash(prepared/'preparation.json'))


def completed_collection(root):
    from scripts.longbench_a800_data import load, stage
    from scripts.router_dataset import load as load_dataset
    root = Path(root)
    old, _, rows = load(root)
    completed = read(root/'ruler/complete.json')
    if (completed.get('complete') is not True or completed.get('owned_engines_exited') is not True
            or completed.get('plan_sha256') != file_hash(root/'plan.json')):
        raise ValueError('Parent experiment is not complete')
    for role, name, count_key, expected in (
            ('ruler', 'controls-complete.json', 'answers', len(rows)*12),
            ('features', 'complete.json', 'probes', len(rows))):
        stage(root, role)
        done = read(root/role/name)
        if (done.get('owned_engines_exited') is not True or done.get(count_key) != expected
                or done.get('protocol_sha256') != file_hash(root/role/'protocol.json')):
            raise ValueError(f'Parent {role} completion receipt is invalid')
    dataset = load_dataset(root/'ruler-data.json')
    if (dataset['provenance'].get('plan_sha256') != file_hash(root/'plan.json')
            or any(dataset['provenance'].get(k) != old.get(k) for k in
                   ('tp', 'seed', 'samples_per_task', 'execution_profile', 'evaluation_protocol'))
            or (('hardware_profile' in dataset['provenance'] or old.get('hardware_profile') in ('l40-tp2', 'l40-tp4'))
                and dataset['provenance'].get('hardware_profile') != old.get('hardware_profile'))
            or {r['id']: r['input_sha256'] for r in dataset['rows']} != {r['id']: r['sha256'] for r in rows}):
        raise ValueError('Parent portable dataset does not match the completed plan')
    return dataset


def checked_json(root, name, receipt):
    if receipt['files'].get(name) != file_hash(root/name):
        raise ValueError(f'Prepared artifact changed: {root/name}')
    return read(root/name)


def verify_prepared_extension(settings, receipt, manifest):
    """One-time CPU prefix/content audit before publishing the delta plan."""
    from scripts.ruler import TASKS
    validate_extension(settings)
    ext = settings['extension']; check_parent(ext)
    start, target = ext['start'], ext['target_samples_per_task']
    prior = Path(ext['parent_prepared']); prepared = Path(settings['prepared'])
    old_receipt = read(prior/'preparation.json')
    before, after = old_receipt['spec'], receipt['spec']
    import_move = (before.get('adapter'), after.get('adapter')) == ADAPTER_IMPORT_MOVE
    nonthinking_move = (ext['schema'] == L20_SCHEMA
                        and (before.get('adapter'), after.get('adapter')) ==
                        (LEGACY_NONTHINKING_ADAPTER, ADAPTER_IMPORT_MOVE[1]))
    chunker_move = (ext['schema'] == L20_SCHEMA
                    and (before.get('chunker'), after.get('chunker')) == NONTHINKING_CHUNKER_MOVE)
    excluded = {'samples', 'adapter'} if import_move or nonthinking_move else {'samples'}
    if chunker_move:
        excluded.add('chunker')
    differences = sorted(k for k in before.keys() | after.keys() if k not in excluded
                         and (k not in before or k not in after or before[k] != after[k]))
    if (before.get('samples') != start or after.get('samples') != target
            or differences):
        raise ValueError('Parent/new generator, tokenizer, assets, versions or policy differ; '
                         f'prefix reuse refused (fields: {differences}; '
                         f'samples: {before.get("samples")} -> {after.get("samples")})')
    if file_hash(prior/'manifest.jsonl') != old_receipt['files']['manifest.jsonl']:
        raise ValueError('Parent manifest changed')
    old_manifest = [json.loads(line) for line in (prior/'manifest.jsonl').read_text().splitlines() if line.strip()]
    old_by_id = {r['id']: r for r in old_manifest}
    parent_rows = read(Path(ext['parent_root'])/'rows.json')
    if (len(old_manifest) != 13*start or len(old_by_id) != 13*start
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
        if len(old_raw) != start or len(new_raw) != target or new_raw[:start] != old_raw:
            raise ValueError(f'{target}-row source prefix differs from original {start} rows: {task}; no GPU work scheduled')
    token_hashes = {task: set() for task in TASKS}
    for row in manifest:
        task = row['subtask']; ordinal = task_counts.get(task, 0)
        task_counts[task] = ordinal+1
        if task not in TASKS or row['id'] != f'{task}-{ordinal:03d}' or row['id'] in seen_ids:
            raise ValueError('Unexpected full-batch manifest membership/order')
        seen_ids.add(row['id'])
        sample = checked_json(prepared, row['prepared'], receipt)
        token_digest = fingerprint(sample['token_ids'])
        if token_digest in token_hashes[task]:
            raise ValueError(f'Duplicate prompt in source: {row["id"]}')
        token_hashes[task].add(token_digest)
        if ordinal < start:
            old_row = old_by_id.get(row['id'])
            if old_row != row:
                raise ValueError(f'Parent manifest prefix differs: {row["id"]}')
            old_sample = checked_json(prior, old_row['prepared'], old_receipt)
            # This hash includes batch size and the specifically mapped adapter.
            # Every actual sample field (including tokens/layout/policy) must match.
            canonical = lambda s: {k: v for k, v in s.items() if k != 'generator_config_sha256'}
            if canonical(sample) != canonical(old_sample):
                raise ValueError(f'Parent token/layout/policy prefix differs: {row["id"]}')
        else:
            selected.append(row['id'])
    if task_counts != {t: target for t in TASKS}:
        raise ValueError(f'Expected full 13x{target} prepared batch')
    check_parent(ext)
    result = dict(schema=ext['schema'], extension=ext, prepared_manifest_sha256=receipt['files']['manifest.jsonl'],
                prefix_prompts_checked=13*start, new_prompts=13*(target-start), selected_ids_sha256=fingerprint(selected),
                comparison='exact raw rows and prepared samples except generator batch-size hash')
    if import_move:
        result['comparison'] = 'exact raw rows and prepared samples except generator configuration hash; verified adapter import move'
        result['adapter_compatibility'] = dict(parent_sha256=before['adapter'], new_sha256=after['adapter'],
            reason='check_model import moved from run to runner.preparation; function AST unchanged',
            source_commits=['cd5a3c0060555b6849ad48aaf1fbf14ed0c5fe48', '498e9ddceed742cf0ff82801806606c9be1cd414'])
    if nonthinking_move:
        result['comparison'] = 'exact raw rows and prepared samples except generator configuration hash; verified unchanged non-thinking adapter path'
        result['adapter_compatibility'] = dict(parent_sha256=before['adapter'], new_sha256=after['adapter'],
            reason='non-thinking formatter/generator/template/query unchanged; optional thinking branch added and check_model import moved',
            source_commits=['d0483b1', '498e9ddceed742cf0ff82801806606c9be1cd414'])
    if chunker_move:
        result['chunker_compatibility'] = dict(parent_sha256=before['chunker'], new_sha256=after['chunker'],
            reason='only optional thinking fields added to sample_provenance; non-thinking provenance and all token/layout functions unchanged',
            source_commits=['d0483b1', 'cd5a3c0060555b6849ad48aaf1fbf14ed0c5fe48'])
    return result


def combine_exports(base, delta, delta_path, allow_pending=False):
    """Publish a portable union; each row keeps its original input/record hashes."""
    from scripts.router_dataset import save
    from scripts.longbench_a800_data import read as read_settings
    settings = read_settings(Path(base)/'settings.json'); ext = settings['extension']
    check_parent(ext)
    start, target = ext['start'], ext['target_samples_per_task']
    parent_path = Path(ext['parent_root'])/'ruler-data.json'
    required = ('ruler/complete.json', 'features/complete.json', 'ruler-data.json', 'ruler-data.json.sha256')
    missing = [name for name in required if not (parent_path.parent/name).is_file()]
    if missing:
        message = (f'Parent probes/final export pending; delta{target-start} retained, no complete{target} dataset yet. '
                   'Finish the original run probes, then run launcher merge. Missing: '+', '.join(missing))
        if allow_pending:
            print(message, flush=True)
            return None
        raise ValueError(message)
    parent = completed_collection(parent_path.parent)
    if set(r['id'] for r in parent['rows']) & set(r['id'] for r in delta['rows']):
        raise ValueError('Parent and extension exports overlap')
    provenance = {k: v for k, v in delta['provenance'].items() if k not in ('extension', 'plan_sha256')}
    provenance.update(samples_per_task=target, collection=f'verified-{start}-plus-{target-start}',
        preparation_extension_sha256=file_hash(Path(base)/'extension.json'),
        components=[dict(path=str(path), sha256=file_hash(path), provenance=value['provenance'])
                    for path, value in ((parent_path, parent), (Path(delta_path), delta))])
    rows = sorted(parent['rows']+delta['rows'], key=lambda r: r['id'])
    return save(Path(base)/f'ruler-data-{target}.json', 'ruler', delta['actions'], rows, provenance)


def merge_extension(base):
    """CPU-only union after both independent collections have finished."""
    from scripts.longbench_a800_data import load
    base = Path(base)
    settings, _, _ = load(base)
    if 'extension' not in settings:
        raise ValueError('merge requires an extension run')
    delta = completed_collection(base)
    result = combine_exports(base, delta, base/'ruler-data.json')
    output = base/f"ruler-data-{settings['extension']['target_samples_per_task']}.json"
    print(f"Merged {len(result['rows'])} prompts: {output}", flush=True)
    return result
