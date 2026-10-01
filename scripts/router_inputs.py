#!/usr/bin/env python3
"""Original task-batch router preparation and separate legacy policy source readers."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runner.config import validate_sample
from runner.setups import atomic_json, file_hash, setup_lock
COHORT = ROOT/'scripts/router_cohort.json'


def load_spec():
    spec=json.loads(COHORT.read_text())
    if spec['schema'] != 'clean-router-cohort-v1' or len(spec['entries']) != 1300:
        raise ValueError('Wrong frozen cohort')
    return spec


def verify_sources(ruler, model):
    from importlib.metadata import version
    spec=load_spec()
    for name,expected in spec['source_files'].items():
        if file_hash(Path(ruler)/name) != expected:
            raise ValueError('Official RULER asset differs from frozen training source: '+name)
    for name,expected in spec['model_fingerprints'].items():
        if file_hash(Path(model)/name) != expected:
            raise ValueError('Frozen model/tokenizer mismatch: '+name)
    for name,expected in spec['versions'].items():
        if version(name) != expected:
            raise ValueError(f'Frozen generator requires {name}=={expected}')
    return spec


def format_frozen(tokenizer,row,entry):
    raise ValueError('Legacy marker/padding inputs cannot be relabeled; rebuild original RULER task batches')


_TOKENIZER=None
_WORK=None

def initialize(ruler,model,output):
    global _TOKENIZER,_WORK
    from transformers import AutoTokenizer
    _TOKENIZER=AutoTokenizer.from_pretrained(model,local_files_only=True)
    _WORK=(Path(ruler),Path(model),Path(output))


def regenerate(entry):
    raise ValueError('Legacy per-sample regeneration is incompatible; prepare original RULER task batches')


def verify_prepared(output):
    from scripts.corpus_inputs import original_rows
    return original_rows(Path(output))


def prepare(ruler,model,output,workers=8):
    from types import SimpleNamespace
    from scripts.ruler_64000 import prepare as prepare_original
    prepare_original(SimpleNamespace(ruler=Path(ruler),model=Path(model),output=Path(output),samples=500,seed=42))
    return verify_prepared(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ruler',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):
        raise ValueError('Preparation must be CPU only')
    prepare(a.ruler.resolve(),a.model.resolve(),a.output.resolve(),a.workers)
