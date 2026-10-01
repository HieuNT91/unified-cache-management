"""Worker RPCs for corpus capture, separate from the frozen router action schema."""
import os
import time
from pathlib import Path


def mode(worker,ratio,capture=False):
    from runner.worker import configure
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tp_group
    if not 0 < ratio <= 1 or (capture and ratio!=.01):raise ValueError('Invalid corpus action')
    result=configure(worker,'prophetkv',ratio)
    sparse=get_ucm_sparse()
    if sparse.router_arrays is not None or sparse.router_capture:raise RuntimeError('Capture leaked across requests')
    sparse.router_capture=capture
    get_tp_group().barrier()
    return dict(**result,capture=capture)


def capture_state(worker):
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank
    sparse=get_ucm_sparse()
    return dict(rank=get_tensor_model_parallel_rank(),capture=bool(sparse.router_capture),
                arrays=sparse.router_arrays is not None)


def alignment_state(worker):
    """Post-generation evidence that repeated setup preserved unit delta scale."""
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank
    sparse=get_ucm_sparse();connector=sparse.connector
    delta=connector.cos_sin_cache
    native=sparse.model.layers[0].self_attn.rotary_emb.cos_sin_cache
    return dict(rank=get_tensor_model_parallel_rank(),
                normalized=bool(getattr(connector,'prophet_delta_normalized',False)),
                delta_amplitude=float(delta[0,:delta.shape[-1]//2].float().mean().item()),
                native_amplitude=float(native[0,:native.shape[-1]//2].float().mean().item()),
                aliases_native_table=delta.data_ptr()==native.data_ptr())


def export(worker,output,sample,inventory=None):
    import numpy as np
    import torch
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank
    from runner.corpus import selection
    from runner.tree_policy import actions
    inventory=actions(inventory)
    from runner.setups import file_hash
    sparse=get_ucm_sparse();rank=get_tensor_model_parallel_rank()
    path=Path(output)/f'attention.rank{rank}.npz';temporary=path.with_suffix('.npz.tmp')
    try:
        if not sparse.router_capture or sparse.router_arrays is None:raise RuntimeError('Missing independent probe capture')
        start=time.perf_counter();torch.cuda.synchronize()
        layers,scores,local_mean=[v.cpu().numpy() for v in sparse.router_arrays]
        export_seconds=time.perf_counter()-start
        arrays=dict(layers=layers,scores=scores,local_mean=local_mean,
            context_positions=np.arange(sample['boundaries'][-2],dtype=np.int64),
            boundaries=np.asarray(sample['boundaries'],dtype=np.int64),
            question_positions=np.asarray(sample['question_positions'],dtype=np.int64),
            original_to_formatted=np.asarray(sample.get('original_to_formatted',list(range(len(sample['token_ids'])))),dtype=np.int64))
        arrays.update({a:selection(scores,sample['boundaries'][1],d['ratio']) for a,d in inventory.items() if a!='nocache'})
        start=time.perf_counter()
        with temporary.open('wb') as handle:
            np.savez_compressed(handle,**arrays);handle.flush();os.fsync(handle.fileno())
        temporary.replace(path)
        serialization_seconds=time.perf_counter()-start
        return dict(rank=rank,path=path.name,sha256=file_hash(path),export_seconds=export_seconds,
                    serialization_seconds=serialization_seconds)
    finally:
        sparse.router_arrays=None;sparse.router_capture=False
        temporary.unlink(missing_ok=True)


def tree_mode(worker,request_id,definition,capture=False):
    """Generic actions using the same request-scoped dense guard as frozen routers."""
    from runner.worker import configure
    from runner.router_guard import is_dense_request
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
    dense=definition['method']=='baseline'
    if is_dense_request(request_id)!=dense or (capture and definition['ratio']!=.01):raise ValueError('Invalid tree request tag/probe')
    configure(worker,'prophetkv',definition['ratio'] or .01)
    sparse=get_ucm_sparse()
    if sparse.router_arrays is not None or sparse.router_capture:raise RuntimeError('Capture leaked')
    sparse.router_capture=capture;sparse.router_native_layers=set()
    sparse.connector.router_dense_id=request_id if dense else None
    sparse.connector.store.router_dense=dense
    sparse.router_store_before=dict(sparse.connector.store.operations)
    from ucm.sparse.prophetkv.attention import install
    for layer in sparse.model.layers:install(layer.self_attn.attn,sparse)
    get_tp_group().barrier()
    return dict(rank=get_tensor_model_parallel_rank(),request_id=request_id,definition=definition,capture=capture)
