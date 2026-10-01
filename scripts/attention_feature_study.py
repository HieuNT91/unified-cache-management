#!/usr/bin/env python3
"""CPU-only entry point; set numeric thread controls before importing NumPy."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runner.attention_study import main, DEFAULT
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','extract','search','report','all','resume'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();main(args.stage,args.output)
