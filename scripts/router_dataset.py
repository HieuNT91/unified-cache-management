#!/usr/bin/env python3
"""Export small portable attention-feature/outcome datasets from committed data."""
import argparse
import json
import math
import re
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.longbench_features import FEATURES, DEFINITIONS
from runner.setups import atomic_json, file_hash, fingerprint
from runner.tree_policy import actions
from runner.corpus_records import accepted, protocol_identity
from runner.layout import PROMPT_PROTOCOL
from runner.thinking_budget import validate_thinking_outcome

SCHEMA = 'attention-router-dataset-v2'


def validate(data):
    if (data.get('schema') != SCHEMA or data.get('dataset') not in ('ruler', 'longbench-v2')
            or data.get('feature_names') != list(FEATURES) or data.get('feature_definitions') != DEFINITIONS
            or data.get('prompt_protocol') != PROMPT_PROTOCOL):
        raise ValueError('Incompatible dataset/feature version; all eight fixed layers and every TP head are required')
    if data.get('payload_sha256') != fingerprint({k: v for k, v in data.items() if k != 'payload_sha256'}):
        raise ValueError('Dataset checksum mismatch')
    from scripts.longbench_a800_data import ACTIONS
    if data.get('actions') != ACTIONS:
        raise ValueError('Portable training requires all 12 canonical measured actions')
    provenance = data.get('provenance', {})
    tp = provenance.get('tp')
    thinking = provenance.get('execution_profile') == 'ruler-thinking'
    l40 = provenance.get('hardware_profile') == 'l40-tp4'
    if (type(tp) is not int or provenance.get('seed') != 42
            or (l40 and (tp != 4 or not thinking or data['dataset'] != 'ruler'))
            or (not l40 and tp != 2)):
        raise ValueError('Portable collection requires TP2 or L40 thinking TP4, with seed42')
    from runner.thinking_budget import PROTOCOL, KEY, validate_policy
    if thinking and (data['dataset'] != 'ruler' or data['provenance'].get('evaluation_protocol') != PROTOCOL):
        raise ValueError('Invalid thinking dataset provenance')
    inventory = actions(data['actions']); rows = data['rows']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate dataset membership')
    if data['dataset'] == 'longbench-v2' and len(rows) != 503:
        raise ValueError('LongBench data must contain all 503 prompts')
    if data['dataset'] == 'ruler':
        from scripts.ruler import TASKS
        from collections import Counter
        from scripts.longbench_a800_data import ruler_samples
        count = ruler_samples(data['provenance'])
        if len(rows) != 13*count or Counter(r['subtask'] for r in rows) != Counter({t: count for t in TASKS}):
            raise ValueError(f'RULER data must contain all 13x{count} prompts')
    if data['dataset'] == 'longbench-v2' and len({r.get('source_id') for r in rows}) != 503:
        raise ValueError('LongBench source identities must be unique')
    for row in rows:
        if thinking:
            from scripts.ruler import CAPS
            policy = validate_policy(row[KEY])
            if row.get('evaluation_protocol') != PROTOCOL or policy['answer_cap'] != CAPS.get(row['subtask']):
                raise ValueError('Thinking row protocol/cap mismatch')
        elif row.get('evaluation_protocol') == PROTOCOL or KEY in row:
            raise ValueError('Thinking rows require thinking provenance')
        if (row['dataset'] != data['dataset'] or set(row['features']) != set(FEATURES)
                or set(row['outcomes']) != set(inventory)):
            raise ValueError('Missing features/actions or inconsistent dataset identity')
        for value in row['features'].values():
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not -1e-6 <= value <= 1+1e-6):
                raise ValueError('Feature must be null or finite attention coverage/agreement')
        if not math.isfinite(row['probe_overhead_seconds']) or row['probe_overhead_seconds'] < 0:
            raise ValueError('Invalid probe overhead')
        if (not row.get('source_pins') or not re.fullmatch('[0-9a-f]{64}', row.get('input_sha256', ''))
                or not row.get('subtask') or any(not re.fullmatch('[0-9a-f]{64}', v) for v in row['source_pins'].values())):
            raise ValueError('Missing input/measurement provenance')
        if row['dataset'] == 'longbench-v2' and row.get('length') not in ('short', 'medium', 'long'):
            raise ValueError('Missing official LongBench length label')
        devices = []
        for value in row['outcomes'].values():
            if thinking:
                validate_thinking_outcome(value, policy)
            gpu = value.get('gpu_uuids', [])
            if len(gpu) != tp or len(set(gpu)) != tp or any(not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', g) for g in gpu):
                raise ValueError(f'Each outcome must preserve its TP{tp} GPU identities')
            devices.append(tuple(gpu))
            if (not math.isfinite(value['accuracy']) or not 0 <= value['accuracy'] <= 1
                    or not math.isfinite(value['ttft_seconds']) or value['ttft_seconds'] <= 0):
                raise ValueError('Invalid measured outcome')
        if data['dataset'] == 'ruler' and len(set(devices)) != 1:
            raise ValueError('RULER actions of one prompt must share a TP group')
        if data['dataset'] == 'longbench-v2':
            from scripts.longbench_a800_data import SCHEDULE
            pairs = []
            for role in ('primary', 'extra'):
                current = {tuple(row['outcomes'][a]['gpu_uuids']) for a in SCHEDULE[role]}
                if len(current) != 1:
                    raise ValueError('A800 role actions of one prompt must share a TP2 pair')
                pairs.append(next(iter(current)))
            if set(pairs[0]) & set(pairs[1]):
                raise ValueError('A800 primary and extra pairs must be disjoint')
    return data



def load(path):
    path = Path(path); sidecar = Path(str(path)+'.sha256')
    words = sidecar.read_text().split() if sidecar.is_file() else []
    if not words or words[0] != file_hash(path):
        raise ValueError('Missing or changed portable dataset checksum')
    return validate(json.loads(path.read_text()))


def save(path, dataset, inventory, rows, provenance):
    value = dict(schema=SCHEMA, dataset=dataset, prompt_protocol=PROMPT_PROTOCOL,
                 feature_names=list(FEATURES), feature_definitions=DEFINITIONS,
                 actions=inventory, rows=rows, provenance=provenance)
    value['payload_sha256'] = fingerprint(value); validate(value)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path)+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sidecar = Path(str(path)+'.sha256')
        if path.exists():
            # Recover only a missing checksum after an interrupted atomic export.
            prior = load(path) if sidecar.exists() else validate(json.loads(path.read_text()))
            if prior != value:
                raise FileExistsError('Portable dataset is immutable; choose another output path')
            if sidecar.exists():
                return value
        else:
            atomic_json(path, value)
        temporary = Path(str(sidecar)+'.tmp')
        temporary.write_text(file_hash(path)+'  '+path.name+'\n'); temporary.replace(sidecar)
    return value


def example(row, probe, outcomes, source_pins, dataset):
    result = dict(id=row['id'], dataset=dataset, subtask=row['subtask'], input_sha256=row['sha256'],
                  evaluation_protocol=row.get('evaluation_protocol'), features=probe['features'],
                  probe_overhead_seconds=probe['timings']['routing_overhead_seconds'],
                  source_pins=source_pins, outcomes={})
    from runner.thinking_budget import KEY
    if KEY in row:
        result[KEY] = row[KEY]
    if dataset == 'longbench-v2':
        result.update(length=row['length'], source_id=row['source_id'])
    for case, record in outcomes.items():
        if record is None:
            raise ValueError(f"Missing measured {case} outcome for {row['id']}")
        result['outcomes'][case] = dict(accuracy=record['accuracy'], ttft_seconds=record['timings']['ttft_seconds'],
            gpu_uuids=record['gpu_uuids'], thinking_tokens=record['thinking_tokens'], answer_tokens=record['answer_tokens'],
            output_tokens=record['output_tokens'], control_tokens=record['control_tokens'],
            output_cap_reached=record['output_cap_reached'])
        if KEY in row:
            result['outcomes'][case].update({k:record[k] for k in ('generated_answer_tokens','generated_thinking_open_tokens','forced_tokens',
                'thinking_cap_reached','answer_cap_reached','thinking_closure')})
            result['outcomes'][case]['first_answer_content_seconds'] = record['timings']['first_answer_content_seconds']
    return result


def export_collection(base, output):
    from scripts.longbench_a800_data import load as load_plan, stage, read, roles, scheduled, ACTIONS
    base = Path(base); settings, plan, rows = load_plan(base)
    stages = roles(settings)
    for role in stages:
        filename = 'complete.json' if role == 'features' else 'controls-complete.json'
        path = base/role/filename
        if not path.is_file() or not read(path)['owned_engines_exited']:
            raise ValueError('Finish all controls and feature engines before export')
    from scripts.longbench_a800_control import engines_idle
    for role in stages:
        engines_idle(base/role)
    protocols = {role: stage(base, role)[0] for role in stages}
    values = []
    for row in rows:
        probe = accepted(base/'features', 'probe', row, protocols['features'])
        if probe is None:
            raise ValueError('Missing accepted feature capture')
        outcomes = {}; pins = {}
        for role in stages:
            for case in scheduled(settings, role):
                path = base/role/'records'/case/row['id']/'validated.json'
                pins[str(path.relative_to(base))] = file_hash(path)
                if case != 'probe':
                    outcomes[case] = accepted(base/role, case, row, protocols[role])
        values.append(example(row, probe, outcomes, pins, settings['dataset']))
    return save(output, settings['dataset'], ACTIONS, values, dict(plan_sha256=file_hash(base/'plan.json'), tp=settings['tp'],
                **({'hardware_profile': 'l40-tp4', 'hardware_preflight': settings['hardware_preflight']}
                   if settings.get('hardware_profile') == 'l40-tp4' else {}),
                **{k:settings[k] for k in ('execution_profile','evaluation_protocol') if k in settings},
                seed=42, samples_per_task=settings['samples_per_task'] if settings['dataset'] == 'ruler' else None,
                validation='committed-results-and-record-pins; per-request runtime validation, no offline attention replay',
                timing=f'TP{settings["tp"]} controls; independent feature probes collected after all control engines exited'))


def export_longbench(base, output):
    return export_collection(base, output)


def export_ruler(root, output):
    """Read a completed coverage-five corpus generated on any server."""
    from scripts.corpus_inputs import prepared_rows
    from scripts.corpus_control import idle
    root = Path(root)
    if (root/'settings.json').exists() and json.loads((root/'settings.json').read_text()).get('schema') == 'tp2-data-config-v2':
        return export_collection(root, output)
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol.get('dataset') != 'ruler' or protocol.get('kind') != 'collection' or protocol.get('feature_profile') != DEFINITIONS:
        raise ValueError('RULER requires a new collection configured with --feature-profile coverage-five; old layer means cannot recover per-head coverage')
    idle(root)
    prepared = Path(protocol['prepared'])
    limit = json.loads((root/'tranche.json').read_text())['limit_per_task']
    rows = [r for r in prepared_rows(prepared) if r['ordinal'] < limit]
    if len(rows) != 13*limit:
        raise ValueError('RULER preparation does not cover the configured tranche')
    from scripts.ruler import EVALUATION_PROTOCOL
    values = []
    for row in rows:
        row = dict(row, evaluation_protocol=EVALUATION_PROTOCOL)
        probe = accepted(root, 'probe', row, protocol)
        if probe is None:
            raise ValueError(f"Missing feature capture: {row['id']}")
        outcomes = {case: accepted(root, case, row, protocol) for case in actions(protocol['actions'])}
        pins = {str(Path('records')/case/row['id']/'validated.json'):
                file_hash(root/'records'/case/row['id']/'validated.json') for case in ['probe', *outcomes]}
        values.append(example(row, probe, outcomes, pins, 'ruler'))
    return save(output, 'ruler', protocol['actions'], values, dict(protocol_sha256=protocol_identity(protocol),
                validation='committed-results-and-record-pins; no offline attention replay',
                timing='Measured on the original RULER corpus hardware'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ruler', 'longbench-v2'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = (export_ruler if args.dataset == 'ruler' else export_longbench)(args.root.resolve(), args.output.resolve())
    print(f"Exported {len(value['rows'])} {args.dataset} rows, {len(value['actions'])} measured actions: {args.output}")


if __name__ == '__main__':
    main()
