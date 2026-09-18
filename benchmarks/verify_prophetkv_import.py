#!/usr/bin/env python3
"""CPU-only integrity and scoring checks for the imported source snapshot."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'benchmarks/PROPHETKV_IMPORT_MANIFEST.json').read_text())
    failures = []
    parsed = 0
    for entry in manifest['files']:
        path = root / entry['destination']
        if not path.is_file():
            failures.append(f"Missing: {entry['destination']}")
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            failures.append(f"Hash mismatch: {entry['destination']}")
        if path.suffix == '.py':
            try:
                ast.parse(data, filename=str(path))
                parsed += 1
            except SyntaxError as error:
                failures.append(str(error))

    bench = load_module('imported_ruler_helpers', root / 'benchmarks/cacheblend_ruler.py')
    official = load_module('official_ruler_metrics', root / 'benchmarks/vendor/RULER/scripts/eval/synthetic/constants.py')
    for task in bench.TASK_MAX_TOKENS:
        prediction, references = 'Value is ABC and 123.', ['abc', '123', 'missing']
        metric = official.string_match_part if task.startswith('qa_') else official.string_match_all
        actual = round(100 * bench.score_prediction(task, prediction, references), 2)
        if actual != metric([prediction], [references]):
            failures.append(f'RULER score mismatch: {task}')
    if bench.score_prediction('qa_1', 'new\x00york', ['new\nyork']) != 1:
        failures.append('RULER control-character cleanup mismatch')

    print(json.dumps({
        'valid': not failures,
        'manifest_files': len(manifest['files']),
        'python_files_parsed': parsed,
        'ruler_tasks_checked': len(bench.TASK_MAX_TOKENS),
        'gpu_execution': False,
        'failures': failures,
    }, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
