#!/usr/bin/env python3
"""Debug AdaTreeV1 on one existing prepared Qwen3-32B input, without preparation."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runner.cache.setups import atomic_json, file_hash
from runner.corpus.dataset import relative
from runner.corpus.records import NATIVE_ANSWER_VALIDATION
from runner.inputs.preparation import check_model
from runner.inputs.layout import boundaries_for
from runner.router.adatree import METHOD, SCHEMA, validate_routing, answer
from runner.router.profiles import validate
from runner.tree.collection import SCHEMA as TREE_SCHEMA, actions
from runner.tree.prophetkv import PROTOCOL as TREE_PROTOCOL, budget


def prepare_run(args):
    """Read existing receipts and one selected row; never create prepared inputs."""
    devices = args.devices.split(',')
    if (len(devices) != 4 or len(set(devices)) != 4
            or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', device) for device in devices)):
        raise ValueError('Local debug requires four distinct explicit GPU UUIDs')
    paths = [args.output, args.cache_root, args.prepared, args.model]
    for i, left in enumerate(paths):
        for right in paths[i+1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError('Result, cache, prepared and model directories must be separate')
    if args.output.exists() or (args.cache_root.exists() and any(args.cache_root.iterdir())):
        raise ValueError('Debug requires fresh result and empty cache directories')
    check_model(args.model)
    receipt_path = args.prepared/'preparation.json'
    manifest_path = args.prepared/'manifest.jsonl'
    receipt = json.loads(receipt_path.read_text())
    if receipt['files'].get('manifest.jsonl') != file_hash(manifest_path):
        raise ValueError('Prepared manifest differs from its receipt')
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row['id'] == args.prompt_id]
    if len(selected) != 1:
        raise ValueError('Expected one exact manifest row ID (for example longbench-v2-461); '
                         'numeric source indices are not prompt IDs: ' + args.prompt_id)
    row = dict(selected[0])
    sample_path = relative(args.prepared, row['prepared'])
    sample_hash = file_hash(sample_path)
    if receipt['files'].get(row['prepared']) != sample_hash:
        raise ValueError('Prepared input differs from its receipt')
    sample = json.loads(sample_path.read_text())
    profile = 'ruler-thinking' if args.dataset == 'ruler' else 'longbench-v2'
    validate(sample, profile)
    if sample['model_config_sha256'] != file_hash(args.model/'config.json'):
        raise ValueError('Prepared input belongs to a different model')
    spec = receipt['spec']
    hashes = spec.get('tokenizer_hashes' if args.dataset == 'ruler' else 'tokenizer_files', {})
    if 'config.json' not in hashes or not any(name.startswith('tokenizer') for name in hashes):
        raise ValueError('Prepared model/tokenizer provenance is incomplete')
    for name, digest in hashes.items():
        if Path(name).name != name or file_hash(args.model/name) != digest:
            raise ValueError('Prepared model/tokenizer changed: ' + name)
    if sample['boundaries'] != boundaries_for(len(sample['token_ids']), min(sample['question_positions']), 4096):
        raise ValueError('AdaTreeV1 requires the original chunk4096 input layout')
    for percentage in (5, 20, 40):
        budget(sample['boundaries'], percentage/100)
    if args.dataset == 'longbench-v2' and args.seed not in spec.get('answer_seeds', []):
        raise ValueError('Seed is not part of the prepared LongBench protocol')
    if args.dataset == 'ruler' and args.seed != spec.get('sampling', {}).get('seed', 0):
        raise ValueError('Seed differs from the prepared RULER protocol')
    row.update(sha256=sample_hash, ordinal=sample['source_row'], dataset=args.dataset,
               input_tokens=len(sample['token_ids']), evaluation_protocol=sample['evaluation_protocol'])
    definitions = actions([5, 20, 40])
    protocol = dict(schema=TREE_SCHEMA, tree_protocol=TREE_PROTOCOL, dataset=args.dataset,
        execution_profile=profile, model=str(args.model), prepared=str(args.prepared),
        cache_root=str(args.cache_root), tp=4, groups=[devices],
        hardware_profile='rtx4500ada-tree-validation', percentages=[5, 20, 40],
        actions=definitions, scheduled_actions=[definition['id'] for definition in definitions],
        phase_order=['cached'], answer_validation=NATIVE_ANSWER_VALIDATION,
        debug_method=METHOD, debug_schema=SCHEMA, local_validation=True,
        validation_prompt_ids=[args.prompt_id])
    pins = {str(path): dict(sha256=file_hash(path), mtime_ns=path.stat().st_mtime_ns)
            for path in (receipt_path, manifest_path, sample_path)}
    return protocol, row, pins


def unchanged_inputs(pins):
    for name, expected in pins.items():
        path = Path(name)
        if file_hash(path) != expected['sha256'] or path.stat().st_mtime_ns != expected['mtime_ns']:
            raise RuntimeError('Prepared input or receipt changed during debug: ' + name)


def run(args):
    protocol, row, pins = prepare_run(args)
    if args.dry_run:
        print(json.dumps(dict(protocol=protocol, row=row, input_pins=pins), indent=2))
        return
    from runner.cache.setups import check_environment
    from scripts.router.corpus_control import check_hardware
    from runner.corpus.runtime import Engine
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(protocol['groups'][0]), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        ENABLE_SPARSE='TRUE', VLLM_USE_V1='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_ALLOW_INSECURE_SERIALIZATION='1', VLLM_ATTENTION_BACKEND='FLASH_ATTN',
        VLLM_USE_REROPE='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    check_environment(4)
    protocol['hardware'] = check_hardware(protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'protocol.json', protocol)
    atomic_json(args.output/'input-pins.json', pins)
    engine = None
    started = time.perf_counter()
    try:
        engine = Engine(args.output, protocol, 0, 'cached', uuid.uuid4().hex, [row])
        engine.begin(row)
        record, diagnostics = answer(engine, args.seed)
        record['cache_retirement'] = engine.end()
    except BaseException as error:
        if engine is not None:
            engine.record_failure(error, 'adatree-debug', METHOD)
        atomic_json(args.output/'failure.json', dict(error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        if engine is not None:
            engine.close()
    unchanged_inputs(pins)
    record.update(debug_schema=SCHEMA, prepared_inputs_unchanged=True,
                  total_debug_seconds=time.perf_counter()-started)
    atomic_json(args.output/'diagnostics.json', diagnostics)
    atomic_json(args.output/'result.json', record)
    atomic_json(args.output/'complete.json', dict(complete=True, owned_engine_shutdown=True,
        prepared_inputs_unchanged=True, answers=1, independent_probes=0,
        result_sha256=file_hash(args.output/'result.json')))
    print(json.dumps(dict(output=str(args.output), action=record['executed_action'],
        accuracy=record['accuracy'], routing=record['routing'], timings=record['timings']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'prepared', 'output', 'cache-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--dataset', choices=('ruler', 'longbench-v2'), required=True)
    parser.add_argument('--prompt-id', required=True, help='Exact manifest row ID, e.g. longbench-v2-461 or niah_single_1-000')
    parser.add_argument('--devices', required=True, help='Four comma-separated GPU UUIDs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='Read selected prepared metadata only; no GPU access')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    run(args)


if __name__ == '__main__':
    main()
