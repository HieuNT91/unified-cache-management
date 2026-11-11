"""Worker audits for capture isolation and native YaRN alignment."""


def capture_state(worker):
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank
    sparse=get_ucm_sparse()
    return dict(rank=get_tensor_model_parallel_rank(),capture=bool(sparse.router_capture),
                arrays=sparse.router_arrays is not None or getattr(sparse,'router_head_arrays',None) is not None
                or bool(getattr(sparse,'router_head_layers',())))


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
