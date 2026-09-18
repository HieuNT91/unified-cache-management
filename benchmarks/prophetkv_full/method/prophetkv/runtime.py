"""All-layer probing using the loaded Qwen3 model and once-aligned paged KV."""
import time
import torch
from .selection import context_importance, select, LayerAlignment


def project(layer,hidden,residual,positions):
    if residual is None:
        residual=hidden
        hidden=layer.input_layernorm(hidden)
    else:
        hidden,residual=layer.input_layernorm(hidden,residual)
    a=layer.self_attn
    qkv,_=a.qkv_proj(hidden)
    q,k,v=qkv.split([a.q_size,a.kv_size,a.kv_size],dim=-1)
    q=a.q_norm(q.reshape(-1,a.num_heads,a.head_dim)).reshape(-1,a.q_size)
    k=a.k_norm(k.reshape(-1,a.num_kv_heads,a.head_dim)).reshape(-1,a.kv_size)
    q,k=a.rotary_emb(positions,q,k)
    return q.reshape(-1,a.num_heads,a.head_dim),k.reshape(-1,a.num_kv_heads,a.head_dim),v.reshape(-1,a.num_kv_heads,a.head_dim),residual


def setup(worker):
    from ucm.sparse.state import get_ucm_sparse,has_ucm_sparse
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    if not has_ucm_sparse():return {'baseline':True}
    sparse=get_ucm_sparse()
    model=worker.model_runner.model.model
    if model.__class__.__name__!='Qwen3Model' or len(model.layers)!=36:
        raise RuntimeError('Supported model: 36-layer Qwen3 only')
    sparse.model=model
    group=get_kv_transfer_group()
    connector=getattr(group,"connector",group)
    sparse.connector=connector
    if hasattr(connector,'prophet_aligned'):return {'installed':True}
    guard=LayerAlignment()
    connector.prophet_aligned=guard.layers
    bind=connector.bind_connector_metadata
    wait=connector.wait_for_layer_load
    def bind_once(meta):
        guard.reset()
        return bind(meta)
    def wait_once(name):
        guard.apply(name,wait)
    connector.bind_connector_metadata=bind_once
    connector.wait_for_layer_load=wait_once
    return {'installed':True,'layers':36}


def set_request(worker,request_id,boundaries,question_positions):
    from ucm.sparse.state import get_ucm_sparse
    from .selection import RequestMetadata
    sparse=get_ucm_sparse()
    sparse.request_state.arm(RequestMetadata(request_id,tuple(boundaries),tuple(question_positions)))


@torch.inference_mode()
def probe(sparse,positions,embeddings):
    from vllm.forward_context import get_forward_context
    from torch.nn.attention.bias import causal_lower_right
    context=get_forward_context()
    meta=sparse.request
    b=meta.boundaries
    suffix_tokens=b[-1]-b[-2]
    device=positions.device
    qp=torch.tensor(meta.question_positions,device=device)-b[-2]
    suffix_positions=positions[-suffix_tokens:]
    if not torch.equal(suffix_positions,torch.arange(b[-2],b[-1],device=device)):
        raise RuntimeError('Fresh suffix positions changed')
    hidden=embeddings[-suffix_tokens:].clone();residual=None
    scores=torch.zeros(b[-2],device=device,dtype=torch.float32)
    blocks=sparse.attn_metadata.block_table[0].long()
    cp=torch.arange(b[-2],device=device)
    slots=blocks[cp//64]*64+cp%64
    started=time.perf_counter()
    for i,layer in enumerate(sparse.model.layers):
        name=f'model.layers.{i}.self_attn.attn'
        sparse.connector.wait_for_layer_load(name)
        cache=context.no_compile_layers[name].kv_cache[context.virtual_engine]
        ck=cache[0,slots//64,slots%64];cv=cache[1,slots//64,slots%64]
        q,k,v,residual=project(layer,hidden,residual,suffix_positions)
        scores+=context_importance(q[qp],ck)
        full_k=torch.cat((ck,k));full_v=torch.cat((cv,v))
        output=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],
            full_k.transpose(0,1)[None],full_v.transpose(0,1)[None],
            attn_mask=causal_lower_right(suffix_tokens,len(full_k)),enable_gqa=True,dropout_p=0.)
        hidden,_=layer.self_attn.o_proj(output[0].transpose(0,1).reshape(suffix_tokens,-1))
        hidden,residual=layer.post_attention_layernorm(hidden,residual)
        hidden=layer.mlp(hidden)
        del ck,cv,full_k,full_v,q,k,v,output
    torch.cuda.synchronize()
    eligible=scores[b[1]:]
    selected=select(eligible,b[1],sparse.ratio)
    sparse.selection_diagnostics.append(dict(kind='prophetkv_selection',request_id=meta.request_id,
        scores=eligible,selected_positions=selected,eligible_count=len(eligible),selected_count=len(selected),
        question_positions=list(meta.question_positions),prefix_tokens=b[1],suffix_tokens=suffix_tokens,
        probe_layers=36,probe_tokens_per_layer=suffix_tokens,probe_seconds=time.perf_counter()-started,
        normalization='all_context_keys',fusion='sum_layers_fp32',ratio=sparse.ratio,
        selection_stage='before_layer_0_qkv',alignment_count=len(sparse.connector.prophet_aligned)))
    return selected
