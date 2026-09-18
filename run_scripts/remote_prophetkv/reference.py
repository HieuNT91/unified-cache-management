"""Independent native dense-prefill reference after exact first-chunk reuse.

No UCM connector, ProphetKV probe, pruning, or custom attention is installed.
This is a numerical control only, never a measured no-cache baseline.
"""
import argparse,json,os,sys
from pathlib import Path
if os.environ.get('PROPHETKV_REFERENCE_HELPERS'):sys.path.insert(0,os.environ['PROPHETKV_REFERENCE_HELPERS'])
from prophetkv_common import load_sample,generate,dump
from common import verify_gpu_visibility
from worker import audit_capture,worker_probe,build
from types import SimpleNamespace

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=json.loads(args.protocol.read_text());s=load_sample(args.sample)
    verify_gpu_visibility()
    if Path.cwd()!=Path('/tmp') or 'PYTHONPATH' in os.environ:
        raise RuntimeError('Unsafe reference launch')
    # Independent native vLLM: enable only its exact-prefix cache.
    from vllm import LLM
    length=((s['tokens']+191)//64)*64
    cfg=dict(model=p['model'],tokenizer=p['model'],trust_remote_code=True,enforce_eager=True,
        dtype='bfloat16',max_model_len=length,max_num_batched_tokens=length,max_num_seqs=1,
        gpu_memory_utilization=p['gpu_memory_utilization'],block_size=64,enable_prefix_caching=True,
        distributed_executor_backend='mp',tensor_parallel_size=p['tensor_parallel_size'],disable_custom_all_reduce=True,
        generation_config='vllm',seed=0,enable_chunked_prefill=False)
    if s['context_target']==65536:cfg['rope_scaling']=p['rope_scaling_64k']
    llm=LLM(**cfg)
    imports=llm.collective_rpc(worker_probe)
    generate(llm.llm_engine,[100,200,300,400,500,600,700,800]*16,4,'warmup-128')
    prefix=s['boundaries'][1]
    generate(llm.llm_engine,s['token_ids'][:prefix],1,'exact-prefix-only')
    llm.collective_rpc(audit_capture,kwargs={'enable':True,'suffix_tokens':s['fresh_suffix_tokens']})
    result,ttft,total=generate(llm.llm_engine,s['token_ids'],16,'dense-native-prefix-reference')
    hits=getattr(result,'num_cached_tokens',None)
    if hits!=prefix:raise RuntimeError(f'Expected exactly {prefix} prefix tokens, got {hits}')
    artifact=args.output.with_suffix('.model-audit.pt')
    capture=llm.collective_rpc(audit_capture,kwargs={'enable':False,'output_path':str(artifact)})
    out=result.outputs[0]
    dump(args.output,dict(control='native_dense_exact_prefix',measured=False,
        sample_id=s['id'],gpu_devices=p['gpu_devices'],
        exact_prefix_tokens=prefix,num_cached_tokens=hits,computed_prompt_tokens=s['tokens']-prefix,
        output_token_ids=list(out.token_ids),prediction=out.text,worker_imports=imports,
        capture=capture,ttft_seconds=ttft,generation_seconds=total))
    llm.llm_engine.engine_core.shutdown()
    print('REFERENCE_COMPLETE',flush=True)
if __name__=='__main__':main()
