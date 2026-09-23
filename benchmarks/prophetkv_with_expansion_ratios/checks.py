"""CPU scheduling/configuration checks and optional CUDA selector parity checks."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from prophetkv_common import dump, sha
from prophetkv_gpu import ROOT, GPU_UUIDS, check
from settings import PARENT, CASES, PARAMETERS, METHOD

sys.path.insert(0,str(PARENT/'private_ucm/ucm/sparse'))
from prophetkv.selection import ExpansionConfig, select_expansion, select_expansion_reference


def configuration_checks():
    from dataclasses import asdict
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch
    import cacheblend_prophetkv as harness
    import worker
    p=json.loads((ROOT/'protocol.json').read_text())
    assert p['case_parameters']==PARAMETERS
    assert set(p['cases'])==set(CASES) and len(p['samples'])==200
    expected=set()
    for gpu,phases in p['phases'].items():
        assert sorted(phases)==sorted(CASES)
        for meta in p['samples']:
            if meta['physical_gpu']==int(gpu):
                for case in phases:
                    key=(meta['id'],case)
                    assert key not in expected
                    expected.add(key)
    assert len(expected)==600
    assert all(m['tokens']<=65536 for m in p['samples'])
    stub=ModuleType('vllm');stub.LLM=lambda **kwargs:kwargs
    config_stub=ModuleType('vllm.config');config_stub.KVTransferConfig=lambda **kwargs:SimpleNamespace(**kwargs)
    sample=json.loads(Path(p['samples'][0]['input_path']).read_text())
    with patch.dict(sys.modules,{'vllm':stub,'vllm.config':config_stub}):
        for case in CASES:
            config=ExpansionConfig(**PARAMETERS[case])
            assert asdict(config)==PARAMETERS[case]
            cfg=harness.build(SimpleNamespace(case=case,cache_dir=Path('/unused')),sample,p)
            sparse=cfg['kv_transfer_config'].kv_connector_extra_config['ucm_sparse_config']['ProphetKV']
            assert sparse['method']==METHOD and sparse['ratio']==config.total_ratio
            assert sparse['expansion']==PARAMETERS[case]
            assert cfg['tensor_parallel_size']==1 and not cfg['enable_prefix_caching']
            assert cfg['max_model_len']==65792 and cfg['enforce_eager'] and cfg['dtype']=='bfloat16'
    return dict(configurations=len(CASES),scheduled_new_measurements=len(expected),samples=200)


def cuda_checks():
    import os
    check(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUIDS[1] or torch.cuda.device_count()!=1:
        raise RuntimeError('Selector CUDA checks require physical GPU 1 exclusively by UUID')
    generator=torch.Generator().manual_seed(20260923)
    comparisons=0
    def compare(scores,case):
        nonlocal comparisons
        config=ExpansionConfig(**PARAMETERS[case])
        expected,ed=select_expansion_reference(scores,4096,config,diagnostics=True)
        actual,ad=select_expansion(scores.cuda(),4096,config,diagnostics=True)
        assert torch.equal(expected,actual.cpu()),case
        assert ed=={k:v.item() if torch.is_tensor(v) else v for k,v in ad.items()}
        comparisons+=1
    for case in CASES:
        for n in (0,1,11,100,1024,61440):
            for values in (torch.ones(n),torch.rand(n,generator=generator),
                           torch.randint(0,9,(n,),generator=generator).double()):
                compare(values,case)
        compare(torch.tensor([1.,1.+2**-50,0.,0.,1.,1.,0.,0.,0.,0.],dtype=torch.float64),case)
    saved=list(sorted((PARENT/'records').glob('*/prophetkv_with_expansion.diagnostics.json')))
    chosen=[]
    for task in ('niah_multikey_2','niah_multikey_3','cwe','qa_1'):
        chosen.append(next(path for path in saved if path.parent.name.startswith(task+'-')))
    for path in chosen:
        event=next(d for w in json.loads(path.read_text()) for d in w['diagnostics'] if d['kind']=='prophetkv_selection')
        for case in CASES:compare(torch.tensor(event['scores']),case)
    return dict(exact_mask_comparisons=comparisons,saved_probe_paths=[str(p) for p in chosen],
                gpu_uuid=GPU_UUIDS[1])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    cpu=configuration_checks();print(json.dumps(cpu),flush=True)
    if args.cuda:
        cuda=cuda_checks()
        from run import hashes
        dump(ROOT/'selector-validation.json',dict(complete=True,cpu=cpu,cuda=cuda,
             case_parameters=PARAMETERS,source_hashes=hashes(ROOT/'private_ucm/ucm/sparse/prophetkv'),at=time.time()))
        print(json.dumps(cuda),flush=True)
