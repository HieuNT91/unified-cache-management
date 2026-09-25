#!/usr/bin/env python3
"""One fresh, isolated engine for ProphetKV or corrected CacheBlend."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
if os.environ.get('SELECTOR_UCM_ROOT'):sys.path.insert(0,os.environ['SELECTOR_UCM_ROOT'])
from prophetkv_common import dump,load_sample,cache_snapshot,generate,score
from common import verify_gpu_visibility, TIMING, CASES, model_limit
from cacheblend_ruler import wait_for_cache

CONTROLS=('prophetkv-0','prophetkv-100')
TIMING='engine_step_first_token_monotonic'

def worker_probe(worker,drain=False):
    import torch
    import ucm
    from ucm.sparse.state import get_ucm_sparse,has_ucm_sparse
    root=Path(os.environ['SELECTOR_UCM_ROOT']).resolve()
    if not Path(ucm.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong UCM imported')
    from vllm.distributed import get_tensor_model_parallel_rank
    devices = verify_gpu_visibility()
    rank = get_tensor_model_parallel_rank()
    if torch.cuda.device_count() != len(devices):
        raise RuntimeError('Unexpected CUDA device count')
    actual = str(torch.cuda.get_device_properties(worker.device).uuid)
    expected = devices[rank]['uuid']
    if actual.removeprefix('GPU-').lower() != expected.removeprefix('GPU-').lower():
        raise RuntimeError(f'Physical GPU mismatch: {actual} != {expected}')
    r=dict(ucm_path=ucm.__file__,pid=os.getpid(),rank=rank,
           physical_gpu=devices[rank]['index'],visible_uuid=expected)
    if drain:
        entries=get_ucm_sparse().selection_diagnostics if has_ucm_sparse() else []
        r['diagnostics']=[{k:v.detach().cpu().tolist() if torch.is_tensor(v) else v for k,v in d.items()} for d in entries]
        entries.clear()
    return r

def setup(worker):
    from rope_window import ensure_rope_window
    audits = [ensure_rope_window(layer.self_attn.rotary_emb, 81920)
              for layer in worker.model_runner.model.model.layers]
    from ucm.sparse.prophetkv.runtime import setup as apply
    return dict(sparse=apply(worker), rope_windows=audits)

def set_request(worker,**kwargs):
    from ucm.sparse.prophetkv.runtime import set_request as apply
    return apply(worker,**kwargs)

def audit_capture(worker,enable,output_path=None,suffix_tokens=256):
    """Smoke-only model equivalence sampling after every complete decoder layer."""
    import torch
    if not enable:
        for h in worker.prophet_audit_hooks:h.remove()
        worker.prophet_audit_hooks=[]
        if output_path is None:raise ValueError('Missing audit artifact path')
        from vllm.distributed import get_tensor_model_parallel_rank
        output_path = output_path + f'.rank{get_tensor_model_parallel_rank()}.pt'
        torch.save([worker.prophet_model_audit],output_path)
        result=dict(path=output_path,layers=len(worker.prophet_model_audit))
        worker.prophet_model_audit={}
        return result
    model=worker.model_runner.model.model
    worker.prophet_model_audit={};worker.prophet_audit_hooks=[]
    def hook(index):
        def capture(module,args,result):
            if index in worker.prophet_model_audit:return
            hidden,residual=result
            total=hidden.float()+residual.float()
            # Common fresh suffix positions are identical in dense and sparse paths.
            worker.prophet_model_audit[index]=total[-suffix_tokens:].detach().cpu()
        return capture
    for i,layer in enumerate(model.layers):
        worker.prophet_audit_hooks.append(layer.register_forward_hook(hook(i)))
    return {'enabled':True}

def build(args,sample,p):
    from vllm import LLM
    from vllm.config import KVTransferConfig
    cfg=dict(model=p['model'],tokenizer=p['model'],trust_remote_code=True,enforce_eager=True,
        dtype='bfloat16',max_model_len=model_limit(p, sample),
        max_num_batched_tokens=model_limit(p, sample),
        num_gpu_blocks_override=model_limit(p, sample)//64,
        max_num_seqs=1,gpu_memory_utilization=p['gpu_memory_utilization'],block_size=64,enable_prefix_caching=False,
        distributed_executor_backend='mp',tensor_parallel_size=p['tensor_parallel_size'],disable_custom_all_reduce=True,
        generation_config='vllm',seed=0,enable_chunked_prefill=False,
        worker_cls='rope_worker.Worker')
    if sample['context_target'] == 65536:
        cfg['rope_scaling'] = p['rope_scaling_64k']
    if args.case!='baseline':
        parameters = p['case_parameters'][args.case]
        method = 'prophetkv'
        ratio = parameters['total_ratio']
        sparse=dict(chunk_end_token_id=sample['token_ids'][sample['boundaries'][1]-1],method=method,
            ratio=ratio,component_audit=False,compute_meta={'model.layers.1.self_attn.attn':
                dict(ratio=ratio,selection_metric='k',selection_diagnostics=True)})
        persistent = getattr(args, 'persistent', False)
        cfg['kv_transfer_config']=KVTransferConfig(
            kv_connector='PersistentBlendConnector' if persistent else 'UCMBlendConnector',
            kv_connector_module_path=('ucm.integration.vllm.persistent_connector' if persistent
                                      else 'ucm.integration.vllm.blend_connector'),kv_role='kv_both',
            kv_connector_extra_config=dict(ucm_connectors=[dict(ucm_connector_name='UcmNfsStore',
                ucm_connector_config=dict(storage_backends=str(args.cache_dir),use_direct=False,timeout_ms=120000))],
                ucm_sparse_config={'ProphetKV':sparse,'Blend':sparse}))
    return LLM(**cfg)
