#!/usr/bin/env python3
"""CPU-only offline depth-1..5 search; preserves existing studies/runtime policies."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runner.deeper_router_study import DEFAULT,main
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','search','report','all','resume'])
    p.add_argument('--output',type=Path,default=DEFAULT)
    a=p.parse_args();main(a.stage,a.output)
