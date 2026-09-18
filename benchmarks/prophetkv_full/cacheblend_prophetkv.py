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
from prophetkv_common import sha,dump,load_sample,cache_snapshot,generate,score
from prophetkv_gpu import GPU_UUIDS,check
from cacheblend_ruler import wait_for_cache

CASES=('baseline','prophetkv-10','prophetkv-20')
CONTROLS=('prophetkv-0','prophetkv-100')
TIMING='engine_step_first_token_monotonic'

def worker_probe(worker,drain=False):
    import torch
    import ucm
    from ucm.sparse.state import get_ucm_sparse,has_ucm_sparse
    root=Path(os.environ['SELECTOR_UCM_ROOT']).resolve()
    if not Path(ucm.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong UCM imported')
    gpu=int(os.environ['SELECTOR_PHYSICAL_GPU'])
    if os.environ['CUDA_VISIBLE_DEVICES']!=GPU_UUIDS[gpu] or torch.cuda.device_count()!=1:raise RuntimeError('Wrong GPU')
    r=dict(ucm_path=ucm.__file__,pid=os.getpid(),physical_gpu=gpu,visible_uuid=GPU_UUIDS[gpu])
    if drain:
        entries=get_ucm_sparse().selection_diagnostics if has_ucm_sparse() else []
        r['diagnostics']=[{k:v.detach().cpu().tolist() if torch.is_tensor(v) else v for k,v in d.items()} for d in entries]
        entries.clear()
    return r

def setup(worker):
    from ucm.sparse.prophetkv.runtime import setup as apply
    return apply(worker)

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
        torch.save([worker.prophet_model_audit],output_path)
        result=dict(path=output_path,sha256=sha(output_path),layers=len(worker.prophet_model_audit))
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
        dtype='bfloat16',max_model_len=((sample['tokens']+128+63)//64)*64,
        max_num_batched_tokens=((sample['tokens']+128+63)//64)*64,
        max_num_seqs=1,gpu_memory_utilization=.93,block_size=64,enable_prefix_caching=False,
        distributed_executor_backend='mp',tensor_parallel_size=1,disable_custom_all_reduce=True,
        generation_config='vllm',seed=0,enable_chunked_prefill=False)
    if args.case!='baseline':
        method='prophetkv' if args.case.startswith('prophetkv') else 'cacheblend'
        ratio=.1 if args.case=='populate' else int(args.case.split('-')[-1])/100
        sparse=dict(chunk_end_token_id=sample['token_ids'][sample['boundaries'][1]-1],method=method,
            ratio=ratio,component_audit=args.smoke,compute_meta={'model.layers.1.self_attn.attn':
                dict(ratio=ratio,selection_metric='k',selection_diagnostics=True)})
        cfg['kv_transfer_config']=KVTransferConfig(kv_connector='UCMBlendConnector',
            kv_connector_module_path='ucm.integration.vllm.blend_connector',kv_role='kv_both',
            kv_connector_extra_config=dict(ucm_connectors=[dict(ucm_connector_name='UcmNfsStore',
                ucm_connector_config=dict(storage_backends=str(args.cache_dir),use_direct=False,timeout_ms=120000))],
                ucm_sparse_config={'ProphetKV':sparse,'Blend':sparse}))
    return LLM(**cfg)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--case',choices=(*CASES,*CONTROLS,'populate'),required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    gpu=int(os.environ['SELECTOR_PHYSICAL_GPU']);check(gpu)
    if Path.cwd()!=Path('/tmp') or 'PYTHONPATH' in os.environ or os.environ['CUDA_VISIBLE_DEVICES']!=GPU_UUIDS[gpu]:
        raise RuntimeError('Unsafe launch environment')
    p=json.loads(args.protocol.read_text());sample=load_sample(args.sample)
    if sample['physical_gpu']!=gpu:raise RuntimeError('Sample GPU changed')
    if {n:version(n) for n in p['runtime_versions']}!=p['runtime_versions']:raise RuntimeError('Runtime changed')
    if p['sources']['benchmarks/prophetkv_full/cacheblend_prophetkv.py']!=sha(__file__):raise RuntimeError('Worker changed')
    for path,digest in p['runtime_file_hashes'].items():
        if sha(path)!=digest:raise RuntimeError('Patched runtime changed')
    ids,b=sample['token_ids'],sample['boundaries']
    chunks=[ids[a:z] for a,z in zip(b[:-2],b[1:-1])]
    cache_args=SimpleNamespace(model_path=Path(p['model']),tensor_parallel_size=1,
        cache_dir=args.cache_dir,cache_ready_timeout_seconds=600)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    verification=None
    if args.case not in ('baseline','populate'):verification=wait_for_cache(cache_args,chunks)
    t=time.perf_counter();llm=build(args,sample,p);load_seconds=time.perf_counter()-t
    llm.collective_rpc(setup)
    imports=llm.collective_rpc(worker_probe)
    print('verified_imports '+json.dumps(imports),flush=True)
    t=time.perf_counter();generate(llm.llm_engine,[100,200,300,400,500,600,700,800]*16,4,'warmup-128')
    warmup_seconds=time.perf_counter()-t
    provenance=dict(protocol_sha256=sha(args.protocol),input_sha256=sha(args.sample),
        sample_id=sample['id'],prompt_sha256=sample['prompt_sha256'],physical_gpu=gpu,gpu_uuid=GPU_UUIDS[gpu],
        worker_imports=imports,model_load_seconds=load_seconds,warmup_seconds=warmup_seconds)
    if args.case=='populate':
        t=time.perf_counter()
        for i,c in enumerate(chunks):
            generate(llm.llm_engine,c,1,f'populate-{i}')
            print(f'cache_population chunk={i+1}/{len(chunks)}',flush=True)
        verification=wait_for_cache(cache_args,chunks)
        dump(args.output,dict(**provenance,cache_build_seconds=time.perf_counter()-t,cache_verification=verification))
    else:
        def metadata(rid):
            if args.case!='baseline':llm.collective_rpc(set_request,kwargs=dict(request_id=rid,
                boundaries=b,question_positions=sample['query']['positions']))
        t=time.perf_counter();metadata('prime-actual-prompt')
        generate(llm.llm_engine,ids,1,'prime-actual-prompt')
        prime_seconds=time.perf_counter()-t
        llm.collective_rpc(worker_probe,kwargs={'drain':True})
        before=cache_snapshot(args.cache_dir)
        metadata('measured-1-0')
        if args.smoke and args.case in ('baseline','prophetkv-100'):
            llm.collective_rpc(audit_capture,kwargs={'enable':True,'suffix_tokens':b[-1]-b[-2]})
        print('MEASURE_BEGIN',flush=True)
        result,ttft,total=generate(llm.llm_engine,ids,16 if args.smoke else 128,'measured-1-0')
        print('MEASURE_END',flush=True)
        diagnostics=llm.collective_rpc(worker_probe,kwargs={'drain':True})
        if args.smoke and args.case in ('baseline','prophetkv-100'):
            audit_path=args.output.with_suffix('.model-audit.pt')
            receipt=llm.collective_rpc(audit_capture,kwargs={'enable':False,'output_path':str(audit_path)})
            if receipt[0]['layers']!=36 or receipt[0]['sha256']!=sha(audit_path):raise RuntimeError('Model audit transport failed')
        after=cache_snapshot(args.cache_dir)
        if before!=after:raise RuntimeError('Offline cache changed during measured inference')
        output=result.outputs[0];value,extracted=score(sample,output.text)
        dp=args.output.with_suffix('.diagnostics.json');dump(dp,diagnostics)
        record=dict(**provenance,schema_version=2,case=args.case,smoke=args.smoke,dataset=sample['dataset'],label=sample['label'],
            context_target=sample['context_target'],chunk_size=sample['chunk_size'],source_row=sample['source_row'],
            prompt_tokens=len(ids),max_output_tokens=16 if args.smoke else 128,
            timing_source=TIMING,ttft_seconds=ttft,generation_seconds=total,prime_seconds=prime_seconds,
            prediction=output.text,output_token_ids=list(output.token_ids),output_tokens=len(output.token_ids),
            finish_reason=output.finish_reason,score=value,extracted_answer=extracted,references=sample['source_metadata']['references'],
            num_cached_tokens=getattr(result,'num_cached_tokens',None),cache_verification=verification,
            warm_cache=True,cache_unchanged=True,online_mask_reused=False,diagnostics_sha256=sha(dp))
        dump(args.output,record)
        print(f'result case={args.case} score={value} ttft={ttft:.3f} total={total:.3f}',flush=True)
    llm.llm_engine.engine_core.shutdown()
    print('WORKER_COMPLETE',flush=True)

if __name__=='__main__':main()
