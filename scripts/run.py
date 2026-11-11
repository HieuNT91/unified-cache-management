#!/usr/bin/env python3
"""Download pinned sources or prepare the original AdaTree benchmark inputs."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.cache.setups import file_hash

LONGBENCH_REVISION = '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
LONGBENCH_SHA256 = '15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2'
LONGBENCH_URL = ('https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/'
                 + LONGBENCH_REVISION + '/data.json')
RULER_VERSIONS = {'PyYAML': '6.0.3', 'nltk': '3.10.3', 'numpy': '2.2.6',
                  'scipy': '1.15.3', 'transformers': '4.53.2', 'wonderwords': '3.0.1'}
RULER_ASSETS = {
    'PaulGrahamEssays.json': '8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd',
    'english_words.json': 'affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca',
    'hotpotqa.json': 'e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8',
    'squad.json': '80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8',
}


def validate_ruler_assets(source):
    from scripts.data.ruler import source_check
    source_check(source)
    for name, expected in RULER_ASSETS.items():
        if file_hash(source/'scripts/data/synthetic/json'/name) != expected:
            raise ValueError('RULER asset differs from the original 6500-row corpus: '+name)


def validate_ruler_versions():
    for name, expected in RULER_VERSIONS.items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = 'not installed'
        if actual != expected:
            raise ValueError(f'Preparing RULER requires {name}=={expected}; found {actual}. Install requirements.txt.')


def validate_longbench(path):
    if file_hash(path) != LONGBENCH_SHA256:
        raise ValueError('LongBench source differs from the original 503-row corpus')


def download_longbench(path):
    """Never replace an existing source, including a source with the wrong hash."""
    if path.exists():
        validate_longbench(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download')
    owned = False
    try:
        with temporary.open('xb') as target:
            owned = True
            with urllib.request.urlopen(LONGBENCH_URL, timeout=60) as source:
                shutil.copyfileobj(source, target)
        validate_longbench(temporary)
        # Publish without replacing a concurrently installed source.
        os.link(temporary, path)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download', 'prepare'))
    parser.add_argument('dataset', choices=('ruler', 'longbench-v2'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples-per-task', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepared-answer-seeds', type=int, default=4)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        parser.error('Data download/preparation must be CPU-only')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args.source = args.source.resolve()
    if args.command == 'download':
        if args.dataset == 'ruler':
            from scripts.data.ruler import assets
            assets(SimpleNamespace(ruler=args.source))
            validate_ruler_assets(args.source)
        else:
            download_longbench(args.source)
            print(json.dumps(dict(source=str(args.source), sha256=LONGBENCH_SHA256)))
        return
    if args.model is None or args.output is None:
        parser.error('prepare requires --model and --output')
    if args.seed != 42 or args.samples_per_task != 500:
        parser.error('This run requires the original seed42 and complete 500/task RULER batch')
    if args.dataset == 'ruler':
        validate_ruler_versions()
        validate_ruler_assets(args.source)
        from scripts.data.ruler import prepare
        prepare(SimpleNamespace(ruler=args.source, model=args.model, output=args.output,
                                samples=500, seed=42, thinking=True, collection_v3=True))
    else:
        from scripts.data.longbench_v2 import prepare
        validate_longbench(args.source)
        prepare(SimpleNamespace(data=args.source, model=args.model, output=args.output,
                                collection_v3=True, answers_per_question=args.prepared_answer_seeds))


if __name__ == '__main__':
    main()
