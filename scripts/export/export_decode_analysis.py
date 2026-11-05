#!/usr/bin/env python3
"""Standalone, streaming LongBench v2 decode export (Python >=3.9, stdlib only).

python export_decode_analysis.py EXPERIMENT_DIR --output decode.jsonl.gz
Only committed result.json files and small identity receipts are read. Never
opens diagnostics, attention archives, prepared inputs, a model or a launcher.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

SCHEMA = 'longbench-decode-analysis-v1'
POLICY = 'ruler_thinking_budget'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True).encode())


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.pins = {}

    def read(self, name, expected=None, pin=True):
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), f'Path escapes experiment: {name}')
        raw = path.read_bytes()
        sha = digest(raw)
        require(expected is None or sha == expected, f'Checksum mismatch: {path}')
        if pin:
            require(name not in self.pins or self.pins[name] == sha, f'Source changed: {path}')
            self.pins[name] = sha
        return json.loads(raw), sha

    def unchanged(self):
        # Results were hashed once against immutable commit receipts. Recheck
        # only metadata/receipts, never the large diagnostic or result corpus.
        for name, expected in list(self.pins.items()):
            self.read(name, expected)


def seed_schedule(protocol):
    if protocol.get('collection_protocol') != 'tp2-data-v3':
        return [None]
    seeds = protocol.get('answer_seeds')
    require(isinstance(seeds, list) and seeds
            and all(type(s) is int and s >= 0 for s in seeds)
            and seeds == sorted(set(seeds))
            and protocol.get('answers_per_question') == len(seeds), 'Invalid seed schedule')
    return seeds


def source(root):
    reader = Reader(root)
    settings, settings_sha = reader.read('settings.json')
    plan, plan_sha = reader.read('plan.json')
    require(settings.get('dataset') == 'longbench-v2', 'Expected a LongBench v2 experiment root')
    require(plan.get('settings_sha256') == settings_sha, 'settings.json differs from plan')
    require(valid_hash(plan.get('rows_sha256')), 'Missing rows hash in plan')
    rows, _ = reader.read('rows.json', plan['rows_sha256'])
    require(isinstance(rows, list) and rows, 'Empty experiment rows')
    by_id = {r['id']: r for r in rows}
    require(len(by_id) == len(rows) and len({r['source_id'] for r in rows}) == len(rows),
            'Duplicate prompt or public question IDs')
    for row in rows:
        require(isinstance(row['source_id'], str) and row['source_id']
                and re.fullmatch('[a-zA-Z0-9_-]+', row['id'])
                and valid_hash(row.get('sha256')), 'Invalid input identity')
    tree = settings.get('schema') == 'tree-prophetkv-thinking-collection-v1'
    roles = ['tree'] if tree else ['primary', 'extra']
    stages = []
    for role in roles:
        if not (reader.root / role / 'protocol.json').exists():
            require(role == 'extra', f'Missing {role}/protocol.json')
            continue
        protocol, protocol_file_sha = reader.read(f'{role}/protocol.json')
        require(protocol.get('dataset') == 'longbench-v2'
                and protocol.get('plan_sha256') in (plan_sha, fingerprint(plan)),
                f'{role}: protocol/plan mismatch')
        methods = protocol.get('scheduled_actions')
        require(isinstance(methods, list) and methods and len(set(methods)) == len(methods)
                and all(isinstance(m, str) and re.fullmatch('[a-zA-Z0-9_-]+', m) for m in methods),
                f'{role}: invalid method inventory')
        if not tree:
            require(plan.get('schedule', {}).get(role) == methods, f'{role}: schedule differs from plan')
        seeds = seed_schedule(protocol)
        require(seeds == seed_schedule(settings), f'{role}: seed schedule differs from settings')
        stages.append(dict(role=role, protocol=protocol, seeds=seeds,
                           methods=[m for m in methods if m != 'probe'],
                           protocol_sha256=fingerprint({k: v for k, v in protocol.items()
                                                       if k not in ('prepared', 'model', 'cache_root')}),
                           protocol_file_sha256=protocol_file_sha))
    return dict(reader=reader, settings=settings, rows=by_id, stages=stages, tree=tree)


def check_baseline(main, baseline, ids):
    require(any('nocache' in s['methods'] for s in baseline['stages']), 'Baseline has no nocache method')
    require(main['settings'].get('generation_protocol')
            == baseline['settings'].get('generation_protocol'), 'Baseline generation protocol differs')
    require(set(seed_schedule(main['settings'])) <= set(seed_schedule(baseline['settings'])),
            'Baseline seed schedule does not cover tree experiment')
    for pid in ids:
        row, other = main['rows'][pid], baseline['rows'].get(pid)
        require(other is not None, f'Baseline missing prompt: {pid}')
        for key in ('source_id', 'sha256', 'subtask', 'ordinal', 'evaluation_protocol', POLICY,
                    'references', 'scoring'):
            require(row.get(key) == other.get(key), f'Baseline input differs: {pid} ({key})')


def compact(record, row, stage, method, seed, include_token_ids):
    pid = row['id']
    require(record.get('prompt_id') == pid and record.get('method') == method
            and record.get('executed_action') == method
            and record.get('input_sha256') == row['sha256'], f'Result identity mismatch: {pid}/{method}')
    if seed is not None:
        require(type(record.get('seed')) is int and record['seed'] == seed
                and record.get('sequence_id') == pid, f'Result seed mismatch: {pid}/{method}')
    protocol = stage['protocol']
    groups = protocol['groups']
    group = row['ordinal'] % len(groups)
    require(record.get('group') == group and record.get('gpu_uuids') == groups[group]
            and record.get('cache_immutable') is True, f'Invalid worker/cache identity: {pid}')
    retired = record.get('retirement', [])
    require(sorted(r['rank'] for r in retired) == list(range(protocol['tp']))
            and all(r['quiescent'] and not r['transfers']['pending'] and not r['request_bookkeeping']
                    for r in retired), f'Missing retirement evidence: {pid}')
    if protocol.get('answer_validation') is not None:
        require(record.get('answer_validation') == protocol['answer_validation'], 'Answer validation mode differs')
    if POLICY in row:
        require(record.get(POLICY) == row[POLICY]
                and record.get('evaluation_protocol') == row.get('evaluation_protocol'), 'Thinking policy differs')
    if seed is not None:
        require(valid_hash(record.get('token_sha256')), f'Missing submitted token identity: {pid}')
    for key in ('thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens'):
        require(type(record.get(key)) is int and record[key] >= 0, f'Invalid {key}: {pid}')
    tokens = record.get('output_token_ids')
    require(isinstance(tokens, list) and len(tokens) == record['output_tokens']
            and all(type(t) is int and t >= 0 for t in tokens)
            and record['output_tokens'] == sum(record[k] for k in ('thinking_tokens', 'answer_tokens', 'control_tokens')),
            f'Invalid output token accounting: {pid}')
    require(type(record.get('accuracy')) in (int, float) and math.isfinite(record['accuracy'])
            and 0 <= record['accuracy'] <= 1, f'Invalid accuracy: {pid}')
    require(isinstance(record.get('prediction'), str) and isinstance(record.get('answer_text'), str),
            f'Missing decoded output/answer_text: {pid}')
    if 'scored_text' in record:
        require(isinstance(record['scored_text'], str), f'Invalid scored_text: {pid}')
    # Raw decode contains the entire thinking and answer, without redecoding,
    # heuristically splitting delimiters, normalizing whitespace or truncating.
    value = dict(type='answer', question_id=row['source_id'], prompt_id=pid, seed=seed,
                 method=method, output=record['prediction'], answer_text=record['answer_text'])
    for key in ('scored_text', 'accuracy', 'thinking_tokens', 'answer_tokens', 'control_tokens',
                'output_tokens', 'generated_answer_tokens', 'forced_tokens', 'thinking_closure',
                'thinking_cap_reached', 'answer_cap_reached', 'output_cap_reached', 'unfinished_thinking',
                'max_output_tokens', 'input_sha256', 'token_sha256', 'evaluation_protocol'):
        if key in record:
            value[key] = record[key]
    if include_token_ids:
        value['output_token_ids'] = tokens
    return value


class LimitedWriter:
    def __init__(self, raw, limit):
        self.raw, self.limit, self.count = raw, limit, 0

    def write(self, data):
        require(self.count + len(data) <= self.limit,
                'Compressed export exceeds size limit; select --methods, --seeds or --question-ids. '
                'No output published; no text was truncated.')
        written = self.raw.write(data)
        self.count += written
        return written

    def flush(self):
        self.raw.flush()


def export(root, output=None, *, methods=None, seeds=None, question_ids=None,
           include_token_ids=False, max_mb=100, quiet=False):
    require(math.isfinite(max_mb) and 0 < max_mb <= 100, '--max-mb must be in (0, 100] (decimal MB)')
    main = source(root)
    reader = main['reader']
    output = Path(output).absolute() if output else reader.root / 'decode-analysis.jsonl.gz'
    require(str(output).endswith('.jsonl.gz'), 'Output must end in .jsonl.gz')
    require(not output.exists() and not output.is_symlink(), f'Output exists: {output}')
    sources = [main]
    if main['tree'] and (reader.root / 'baseline-source').exists():
        sources.append(source(reader.root / 'baseline-source'))
    ids = set(main['rows'])
    if question_ids:
        lookup = {r['source_id']: pid for pid, r in main['rows'].items()}
        lookup.update({pid: pid for pid in ids})
        require(set(question_ids) <= set(lookup), 'Unknown --question-ids')
        ids = {lookup[q] for q in question_ids}
    selected_seeds = seed_schedule(main['settings'])
    if seeds is not None:
        require(set(seeds) <= set(selected_seeds) and seeds, 'Unknown/empty --seeds')
        selected_seeds = sorted(set(seeds))
    if len(sources) > 1:
        check_baseline(main, sources[1], ids)
    available = {m for s in main['stages'] for m in s['methods']}
    if len(sources) > 1:
        available.add('nocache')
    require(methods is None or (methods and set(methods) <= available), 'Unknown/empty --methods')
    selected_methods = set(methods) if methods else available
    # Avoid writing into immutable stage, copied-baseline or prepared directories.
    for src in sources:
        forbidden = [src['reader'].root / name for name in ('primary', 'extra', 'tree', 'features', 'baseline-source')]
        forbidden += [Path(src['settings'][k]).resolve() for k in ('prepared', 'model', 'cache_root')
                      if src['settings'].get(k)]
        require(not any(output.resolve().is_relative_to(p.resolve()) for p in forbidden),
                'Output must be outside stage, baseline, prepared, model and cache directories')
    pending, inventory = [], {}
    for index, src in enumerate(sources):
        rd = src['reader']
        for stage in src['stages']:
            for method in stage['methods']:
                if method not in selected_methods or (index and method != 'nocache'):
                    continue
                require(method not in inventory, f'Duplicate method across stages: {method}')
                inventory[method] = len(ids) * len(selected_seeds)
                folder = rd.root / stage['role'] / 'records' / method
                pattern = '*/validated.json' if stage['seeds'] == [None] else '*/seed-*/validated.json'
                # Freeze only receipt names; do not recurse into diagnostic trees.
                for receipt_path in sorted(folder.glob(pattern)):
                    if stage['seeds'] == [None]:
                        pid, seed = receipt_path.parent.name, None
                    else:
                        pid = receipt_path.parent.parent.name
                        seed = int(receipt_path.parent.name.removeprefix('seed-'))
                    require(pid in src['rows'] and seed in stage['seeds'], 'Unexpected committed identity')
                    if pid in ids and seed in selected_seeds:
                        pending.append((index, stage, method, pid, seed, str(receipt_path.relative_to(rd.root))))
    require(pending, 'No committed answers match the selection; no output created')
    manifest = dict(type='manifest', schema=SCHEMA, dataset='longbench-v2',
                    validation='committed-result-checksums-and-identity; no diagnostic replay',
                    max_compressed_bytes=int(max_mb * 1_000_000), include_token_ids=include_token_ids,
                    output_text='exact prediction: complete decoded thinking plus answer',
                    expected_per_method=inventory, selected_seeds=selected_seeds,
                    sources=[dict(name='experiment' if i == 0 else 'baseline-source',
                                  kv_chunk_size=src['settings'].get('kv_chunk_size'),
                                  metadata_sha256=dict(src['reader'].pins)) for i, src in enumerate(sources)])
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, seen, token_pins, initializations = Counter(), set(), {}, {}
    with tempfile.TemporaryDirectory(prefix='.decode-export-', dir=output.parent) as tmp:
        staged = Path(tmp) / 'export.jsonl.gz'
        with staged.open('wb') as raw:
            limited = LimitedWriter(raw, manifest['max_compressed_bytes'])
            with gzip.GzipFile(filename='', fileobj=limited, mode='wb', compresslevel=9, mtime=0) as compressed:
                def write(value):
                    compressed.write((json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                                                 allow_nan=False) + '\n').encode('utf-8'))
                write(manifest)
                for index, stage, method, pid, seed, receipt_name in pending:
                    src = sources[index]
                    rd = src['reader']
                    receipt, receipt_sha = rd.read(receipt_name)
                    require(receipt.get('complete') is True and receipt.get('protocol_sha256') == stage['protocol_sha256'],
                            f'Invalid commit receipt: {receipt_name}')
                    result_sha = receipt.get('files', {}).get('result.json')
                    require(valid_hash(result_sha), f'Missing result hash: {receipt_name}')
                    result_name = str(Path(receipt_name).parent / 'result.json')
                    record, _ = rd.read(result_name, result_sha, pin=False)
                    value = compact(record, src['rows'][pid], stage, method, seed, include_token_ids)
                    initial_name = str(Path(stage['role']) / record['initialization'])
                    initial_sha = record['initialization_sha256']
                    require(valid_hash(initial_sha), 'Missing initialization hash')
                    key = (index, initial_name)
                    if key not in initializations:
                        initial, sha = rd.read(initial_name, initial_sha)
                        initializations[key] = (initial, sha)
                    initial, sha = initializations[key]
                    require(sha == initial_sha and initial.get('validated') is True, 'Invalid initialization receipt')
                    if method == 'nocache':
                        require(record.get('num_cached_tokens') == 0
                                and not initial['engine_config'].get('kv_transfer_config'), 'Invalid no-cache baseline')
                    key = (pid, seed, method)
                    require(key not in seen, f'Duplicate answer: {key}')
                    seen.add(key)
                    token_sha = record.get('token_sha256')
                    if token_sha is not None:
                        require(token_pins.setdefault((pid, seed), token_sha) == token_sha,
                                f'Baseline/method submitted tokens differ: {pid}/seed-{seed}')
                    value.update(source='experiment' if index == 0 else 'baseline-source',
                                 record_path=result_name, result_sha256=result_sha, receipt_sha256=receipt_sha)
                    write(value)
                    counts[method] += 1
                    if not quiet and len(seen) % 500 == 0:
                        print(f'Exported {len(seen)}/{len(pending)} committed answers...', file=sys.stderr, flush=True)
                for src in sources:
                    src['reader'].unchanged()
                write(dict(type='summary', answers=len(seen), available_per_method=dict(counts),
                           missing_per_method={m: n-counts[m] for m, n in inventory.items()},
                           text_truncated=False))
            raw.flush()
            os.fsync(raw.fileno())
        # Atomic publication without clobbering another exporter or an old file.
        os.link(staged, output)
    if not quiet:
        print(f'Saved {output}: {len(seen)} answers, {output.stat().st_size / 1_000_000:.2f} MB compressed. '
              f'Per method: {dict(counts)}. Full text preserved.')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--methods', nargs='+', help='Exact method IDs, e.g. nocache tree-prophetkv-5')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--question-ids', nargs='+', help='Public benchmark _id or internal longbench-v2-NNN')
    parser.add_argument('--include-token-ids', action='store_true', help='Also retain the generated output token ID array')
    parser.add_argument('--max-mb', type=float, default=100, help='Compressed size limit, decimal MB (default/max: 100)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    try:
        export(args.experiment_dir, args.output, methods=args.methods, seeds=args.seeds,
               question_ids=args.question_ids, include_token_ids=args.include_token_ids,
               max_mb=args.max_mb, quiet=args.quiet)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Cannot export decodes: {error}\n')


if __name__ == '__main__':
    main()
