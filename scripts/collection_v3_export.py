"""CPU-only snapshot exports; answer seeds share one captured feature vector."""
import fcntl
import json
import os
import re
from pathlib import Path
import tempfile
import numpy as np
from runner.corpus_records import record_dir
from runner.setups import file_hash, fingerprint
from scripts.collection_v3 import ACTIONS, RATIOS, Snapshot, seeds, validate_declaration, answer_metrics
from scripts import export_features as ef


def complete_rows(snapshot):
    role = 'ruler' if snapshot.settings['dataset'] == 'ruler' else 'primary'
    for row in snapshot.rows:
        required = [('features', 'probe', row['id'], None)] + [
            (role, a['id'], row['id'], seed) for a in ACTIONS for seed in seeds(snapshot.settings)]
        if not all(k in snapshot.receipts for k in required):
            continue
        probe = snapshot.record('features', 'probe', row)
        outcomes = {seed: {a['id']: answer_metrics(snapshot.record(role, a['id'], row, seed)) for a in ACTIONS}
                    for seed in seeds(snapshot.settings)}
        for values in outcomes.values():
            for answer in values.values():
                if (answer['token_sha256'] != probe['token_sha256']
                        or answer.get('evaluation_protocol') != row.get('evaluation_protocol')):
                    raise ValueError('Answer/probe input or evaluation provenance differs')
        yield row, probe, outcomes


def protected_output(output):
    root = Path(__file__).resolve().parents[1]
    protected = [root/'.analysis'/name for name in ('ruler-features.npz', 'longbenchv2.npz')]
    if output.is_symlink() or output.resolve() in [p.resolve() for p in protected]:
        raise ValueError('Protected analysis NPZ cannot be overwritten')
    if output.exists() and any(p.exists() and os.path.samefile(output, p) for p in protected):
        raise ValueError('Protected analysis NPZ cannot be overwritten')


def publish(output, arrays, metadata):
    """One rename publishes a verified replacement; failed work retains old bytes."""
    protected_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output)+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            _, old = ef.verify_npz(output)
            if any(old.get(k) != metadata.get(k) for k in ('schema', 'feature_version', 'plan_sha256', 'dataset', 'actions')):
                raise ValueError('Existing NPZ belongs to another run/schema')
        with tempfile.TemporaryDirectory(prefix='.feature-export-', dir=output.parent) as tmp:
            temporary = Path(tmp)/output.name
            ef.publish_npz(temporary, arrays, metadata)
            os.replace(temporary, output)
    return output


def export_compact(root, output=None, quiet=False, workers=ef.DEFAULT_WORKERS):
    if type(workers) is not int or workers <= 0:
        raise ValueError('workers must be a positive integer')
    root = Path(root).resolve()
    # Independent from supervisor/publication locks; never hold up inference.
    with (root/'feature-export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = Snapshot(root)
        dataset = snapshot.settings['dataset']
        output = Path(output).absolute() if output else root/('longbenchv2.npz' if dataset == 'longbench-v2' else 'ruler-features.npz')
        protected_output(output)
        if not any(k[1] == 'probe' for k in snapshot.receipts):
            if not quiet:
                print('chưa có data feature probe', flush=True)
            return None
        selected = list(complete_rows(snapshot))
        if not selected:
            if not quiet:
                print('chưa đủ data', flush=True)
            return None
        jobs = [(i, str(record_dir(root/'features', 'probe', row['id'])), probe, row,
                 snapshot.settings['tp'], RATIOS) for i, (row, probe, _) in enumerate(selected)]
        vectors = {}
        for i, values, seconds in ef.extracted_prompts(jobs, min(workers, len(jobs))):
            if i in vectors or not 0 <= i < len(jobs):
                raise ValueError('Duplicate/unexpected extracted sequence')
            vectors[i] = (values, seconds)
        if len(vectors) != len(jobs):
            raise ValueError('Incomplete feature extraction')
        expanded = [(i, row, probe, seed, outcomes[seed]) for i, (row, probe, outcomes) in enumerate(selected)
                    for seed in seeds(snapshot.settings)]
        fields = ('accuracy', 'thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens',
                  'forced_tokens', 'thinking_cap_reached', 'answer_cap_reached', 'output_cap_reached')
        arrays = {('y_accuracy' if field == 'accuracy' else field): np.array([
            [values[a['id']][field] for a in ACTIONS] for _, _, _, _, values in expanded]) for field in fields}
        arrays.update(
            X=np.asarray([vectors[i][0] for i, *_ in expanded], dtype=np.float32),
            y_ttft=np.asarray([[values[a['id']]['timings']['ttft_seconds'] for a in ACTIONS]
                               for _, _, _, _, values in expanded], dtype=np.float32),
            feature_names=np.array(ef.FEATURE_NAMES), feature_groups=np.array([s[1] for s in ef.FEATURE_SPEC]),
            action_names=np.array([a['id'] for a in ACTIONS]),
            prompt_ids=np.array([f"{r['id']}--seed-{seed}" for _, r, _, seed, _ in expanded]),
            sequence_ids=np.array([r['id'] for _, r, _, _, _ in expanded]),
            source_ids=np.array([str(r.get('source_id', r['id'])) for _, r, _, _, _ in expanded]),
            seeds=np.array([seed for _, _, _, seed, _ in expanded], dtype=np.int64),
            group_ids=np.array([r['id'] for _, r, _, _, _ in expanded]),
            folds=np.array([snapshot.plan['folds'][f"{dataset}:{r['id']}"] for _, r, _, _, _ in expanded], dtype=np.int64),
            task=np.array([r['subtask'] for _, r, _, _, _ in expanded]),
            domain=np.array([r.get('domain', '') for _, r, _, _, _ in expanded]),
            sub_domain=np.array([r.get('sub_domain', '') for _, r, _, _, _ in expanded]),
            length_group=np.array([r.get('length', '') for _, r, _, _, _ in expanded]),
            input_hashes=np.array([r['sha256'] for _, r, _, _, _ in expanded]),
            token_hashes=np.array([probe['token_sha256'] for _, _, probe, _, _ in expanded]),
            ordinals=np.array([r['ordinal'] for _, r, _, _, _ in expanded], dtype=np.int64),
            dataset=np.array(dataset),
            evaluation_protocol=np.array([r.get('evaluation_protocol') or '' for _, r, _, _, _ in expanded]),
            probe_overhead_seconds=np.array([probe['timings']['routing_overhead_seconds'] for _, _, probe, _, _ in expanded]),
            control_gpu_uuids=np.array([[v[a['id']]['gpu_uuids'] for a in ACTIONS] for _, _, _, _, v in expanded]),
            probe_gpu_uuids=np.array([probe['gpu_uuids'] for _, _, probe, _, _ in expanded]),
            offline_extraction_seconds=np.array([vectors[i][1] for i, *_ in expanded]),
            outcomes_json=np.array([json.dumps(values, sort_keys=True) for _, _, _, _, values in expanded]))
        arrays['X_missing'] = np.isnan(arrays['X'])
        if (np.isinf(arrays['X']).any() or not np.isfinite(arrays['y_accuracy']).all()
                or not np.isfinite(arrays['y_ttft']).all() or (arrays['y_ttft'] <= 0).any()):
            raise ValueError('Invalid exported feature/outcome values')
        snapshot.unchanged()
        metadata = dict(schema='attention-router-compact-v2', feature_version=ef.VERSION,
                        dataset=dataset, actions=ACTIONS, samples=len(expanded), sequences=len(selected), feature_count=100,
                        feature_definitions=[dict(name=n, group=g, definition=d) for n, g, d in ef.FEATURE_SPEC],
                        conventions=ef.CONVENTIONS, plan_sha256=snapshot.pins['plan.json'],
                        source_settings=snapshot.settings, source_protocols=snapshot.protocols,
                        source_pins=snapshot.pins, source_record_receipts_sha256=fingerprint(
                            {k: v for k, v in snapshot.pins.items() if k.endswith('/validated.json')}),
                        feature_layers=list(ef.HEAD_LAYERS), total_q_heads=64,
                        split='Seeds of one sequence share group_ids and folds; no fitting.',
                        validation='Committed receipt snapshot, runtime retirement/identity, stored attention replay; no inference.',
                        exporter_sha256=file_hash(Path(__file__)))
        publish(output, arrays, metadata)
    if not quiet:
        print(f'Saved {output}: {len(selected)} sequences, {len(expanded)} rows, 100 features, 9 outcomes', flush=True)
    return output


def export_portable(base, output):
    from scripts.longbench_a800_control import engines_idle
    from scripts.longbench_a800_data import read
    from scripts.router_dataset import example, save
    snapshot = Snapshot(base)
    for role, protocol in snapshot.protocols.items():
        name = 'complete.json' if role == 'features' else 'controls-complete.json'
        receipt = read(snapshot.root/role/name)
        expected = len(snapshot.rows)*(1 if role == 'features' else 9*len(seeds(snapshot.settings)))
        if (receipt.get('owned_engines_exited') is not True
                or receipt.get('protocol_sha256') != file_hash(snapshot.root/role/'protocol.json')
                or receipt.get('probes' if role == 'features' else 'answers') != expected):
            raise ValueError('Finish all declared stages before portable export')
        engines_idle(snapshot.root/role)
    rows = []
    from collections import defaultdict
    pins_by_sequence = defaultdict(list)
    for (_, case, sequence, answer_seed), path in snapshot.receipts.items():
        pins_by_sequence[sequence].append((case, answer_seed, str(path.relative_to(snapshot.root))))
    for row, probe, outcomes in complete_rows(snapshot):
        for seed, values in outcomes.items():
            pins = {path: snapshot.pins[path] for case, answer_seed, path in pins_by_sequence[row['id']]
                    if case == 'probe' or answer_seed == seed}
            value = example(row, probe, values, pins, snapshot.settings['dataset'])
            value.update(id=f"{row['id']}--seed-{seed}", sequence_id=row['id'], source_id=row.get('source_id', row['id']),
                         seed=seed, group_id=row['id'], ordinal=row['ordinal'], input_tokens=row['input_tokens'],
                         fold=snapshot.plan['folds'][f"{snapshot.settings['dataset']}:{row['id']}"])
            value.update({k: row[k] for k in ('domain', 'sub_domain') if k in row})
            rows.append(value)
    if len(rows) != len(snapshot.rows)*len(seeds(snapshot.settings)):
        raise ValueError('Portable export requires every sequence/method/seed')
    snapshot.unchanged()
    provenance = dict(snapshot.settings, plan_sha256=snapshot.pins['plan.json'],
                      validation='committed-results-and-record-pins; no offline attention replay',
                      timing='baseline then feature probes then ProphetKV, with exited engine barriers')
    return save(output, snapshot.settings['dataset'], ACTIONS, rows, provenance)


def validate_portable(data):
    from runner.longbench_features import FEATURES, DEFINITIONS
    from runner.layout import PROMPT_PROTOCOL
    from runner.thinking_budget import KEY, validate_thinking_outcome
    from scripts.longbench_a800_data import validate_rows
    provenance = data['provenance']
    validate_declaration(provenance)
    if (data.get('schema') != 'attention-router-dataset-v3' or data.get('actions') != ACTIONS
            or data.get('dataset') != provenance['dataset'] or data.get('prompt_protocol') != PROMPT_PROTOCOL
            or data.get('feature_names') != list(FEATURES) or data.get('feature_definitions') != DEFINITIONS
            or data.get('payload_sha256') != fingerprint({k: v for k, v in data.items() if k != 'payload_sha256'})):
        raise ValueError('Invalid portable v3 schema/provenance')
    grouped = {}
    for row in data['rows']:
        sequence, seed = row['sequence_id'], row['seed']
        if (type(seed) is not int or seed not in seeds(provenance) or row['id'] != f'{sequence}--seed-{seed}'
                or row['group_id'] != sequence or type(row['fold']) is not int or not 0 <= row['fold'] < 5
                or row['dataset'] != data['dataset'] or set(row['outcomes']) != {a['id'] for a in ACTIONS}
                or set(row['features']) != set(FEATURES) or not row['source_pins']
                or not re.fullmatch('[0-9a-f]{64}', row.get('input_sha256', ''))
                or any(not re.fullmatch('[0-9a-f]{64}', pin) for pin in row['source_pins'].values())
                or not ef.number(row['probe_overhead_seconds'])):
            raise ValueError('Invalid portable answer identity/features')
        peers = grouped.setdefault(sequence, {})
        if seed in peers or (peers and any(row[k] != next(iter(peers.values()))[k] for k in
                ('source_id', 'features', 'fold', 'input_sha256', 'subtask', 'ordinal', 'evaluation_protocol'))):
            raise ValueError('Duplicate or inconsistent sequence seeds')
        peers[seed] = row
        for value in row['features'].values():
            if value is not None and (not ef.number(value) or value > 1+1e-6):
                raise ValueError('Invalid portable feature')
        groups = set()
        for value in row['outcomes'].values():
            if (not ef.number(value['accuracy'], upper=1) or not ef.number(value['ttft_seconds'], positive=True)
                    or type(value.get('seed')) is not int or value['seed'] != seed):
                raise ValueError('Invalid portable outcome')
            gpu = value['gpu_uuids']
            if len(gpu) != 2 or len(set(gpu)) != 2 or any(not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', g) for g in gpu):
                raise ValueError('Invalid portable GPU identity')
            if gpu != provenance['groups'][row['ordinal'] % len(provenance['groups'])]:
                raise ValueError('Portable outcome changed its frozen shard assignment')
            groups.add(tuple(gpu))
            if KEY in row:
                validate_thinking_outcome(value, row[KEY])
            else:
                from scripts.ruler import CAPS
                counts = ('thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens', 'forced_tokens')
                if (any(type(value.get(k)) is not int or value[k] < 0 for k in counts)
                        or value['thinking_tokens'] or value['forced_tokens']
                        or value['answer_tokens'] + value['control_tokens'] != value['output_tokens']
                        or value['output_tokens'] > CAPS[row['subtask']]
                        or type(value.get('output_cap_reached')) is not bool
                        or value.get('thinking_cap_reached') is not False
                        or value.get('answer_cap_reached') != value['output_cap_reached']):
                    raise ValueError('Invalid portable non-thinking token accounting')
        if len(groups) != 1:
            raise ValueError('Sequence methods changed GPU pair')
    if any(set(values) != set(seeds(provenance)) for values in grouped.values()):
        raise ValueError('Missing portable answer seeds')
    original = [dict(next(iter(values.values())), id=sequence) for sequence, values in grouped.items()]
    validate_rows(provenance, original)
    return data
