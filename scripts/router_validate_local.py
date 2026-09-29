#!/usr/bin/env python3
"""CPU-only data compatibility check; reads source artifacts, never imports training code."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.router_inputs import initialize, format_frozen, load_spec, regenerate, verify_sources
from runner.setups import atomic_json, file_hash
_SOURCE=None
_TOKENIZER=None

def init(source,model):
    global _SOURCE,_TOKENIZER
    from transformers import AutoTokenizer
    _SOURCE=Path(source)
    _TOKENIZER=AutoTokenizer.from_pretrained(model,local_files_only=True)

def check(entry):
    source=json.loads((_SOURCE/'inputs'/(entry['expected']['id']+'.json')).read_text())
    sample,row=format_frozen(_TOKENIZER,source['generator_source'],entry)
    return dict(id=row['id'],frozen=row['frozen'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-data',type=Path,required=True);p.add_argument('--ruler',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):raise ValueError('CPU-only validation')
    verify_sources(a.ruler,a.model)
    entries=load_spec()['entries']
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(a.training_data),str(a.model))) as pool:
        results=[]
        for result in pool.map(check,entries):
            results.append(result)
            if len(results)%100==0:print(f'Reconstructed {len(results)}/1300 frozen token/span inputs',flush=True)
    # Independent official-generator invocation samples all generator families;
    # remote prepare still regenerates and verifies all 1300 before inference.
    initialize(str(a.ruler),str(a.model),str(a.output/'independent'))
    generated=[]
    for task in ('niah_single_1','qa_1','qa_2','fwe','cwe','vt'):
        entry=next(e for e in entries if e['identity']['task']==task)
        generated.append(regenerate(entry))
    atomic_json(a.output/'validation.json',dict(complete=True,reconstructed_inputs=len(results),
        independent_generations=len(generated),all_frozen_input_and_metadata_hashes_match=True,
        gpu_inference=False,cohort_sha256=file_hash(ROOT/'scripts/router_cohort.json')))
    print('CPU compatibility validation complete.',flush=True)
