"""Independent native dense-prefill reference after exact first-chunk reuse.

No UCM connector, ProphetKV probe, pruning, or custom attention is installed.
This is a numerical control only, never a measured no-cache baseline.
"""
import argparse,json,os,sys
from pathlib import Path
if os.environ.get('PROPHETKV_REFERENCE_HELPERS'):sys.path.insert(0,os.environ['PROPHETKV_REFERENCE_HELPERS'])
from prophetkv_common import load_sample,generate,dump,sha
from prophetkv_gpu import check,GPU_UUIDS
from cacheblend_prophetkv import audit_capture,worker_probe

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=json.loads(args.protocol.read_text());s=load_sample(args.sample)
    gpu=s['physical_gpu'];check(gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUIDS[gpu] or Path.cwd()!=Path('/tmp') or 'PYTHONPATH' in os.environ:
        raise RuntimeError('Unsafe reference launch')
    from vllm import LLM
    length=((s['tokens']+191)//64)*64
    llm=LLM(model=p['model'],tokenizer=p['model'],trust_remote_code=True,enforce_eager=True,
        dtype='bfloat16',max_model_len=length,max_num_batched_tokens=length,max_num_seqs=1,
        gpu_memory_utilization=.93,block_size=64,enable_prefix_caching=True,
        distributed_executor_backend='mp',tensor_parallel_size=1,disable_custom_all_reduce=True,
        generation_config='vllm',seed=0,enable_chunked_prefill=False)
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
    dump(args.output,dict(control='native_dense_exact_prefix',measured=False,protocol_sha256=sha(args.protocol),
        input_sha256=sha(args.sample),sample_id=s['id'],physical_gpu=gpu,gpu_uuid=GPU_UUIDS[gpu],
        exact_prefix_tokens=prefix,num_cached_tokens=hits,computed_prompt_tokens=s['tokens']-prefix,
        output_token_ids=list(out.token_ids),prediction=out.text,worker_imports=imports,
        audit_sha256=sha(artifact),capture=capture,ttft_seconds=ttft,generation_seconds=total,
        source_sha256=sha(__file__)))
    llm.llm_engine.engine_core.shutdown()
    print('REFERENCE_COMPLETE',flush=True)
if __name__=='__main__':main()
