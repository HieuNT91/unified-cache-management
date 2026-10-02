#!/usr/bin/env python3
"""Original task-batch RULER preparation for router experiments."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load_prepared(output):
    from scripts.corpus_inputs import original_rows
    return original_rows(Path(output))


def prepare(ruler,model,output,workers=8):
    from types import SimpleNamespace
    from scripts.ruler import prepare as prepare_original
    prepare_original(SimpleNamespace(ruler=Path(ruler),model=Path(model),output=Path(output),samples=500,seed=42))
    return load_prepared(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ruler',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):
        raise ValueError('Preparation must be CPU only')
    prepare(a.ruler.resolve(),a.model.resolve(),a.output.resolve(),a.workers)
