"""Tree-only construction, sink probing and TP selection integration."""
import math
import time
import torch

from runner.tree_prophetkv import METHOD, PROTOCOL, budget, read_f3, write_f3


def begin_population(sparse, layout, request_id, tokens, start, count):
    config = getattr(sparse, 'tree_f3_config', None)
    if config is None:
        return
    from runner.layout import validate_request_metadata
    validate_request_metadata(layout, request_id, list(tokens))
    if layout['phase'] != 'populate':
        return
    if config['readonly'] or start != 0 or count != len(tokens):
        raise RuntimeError('F3 requires writable single-step chunk construction')
    if getattr(sparse, 'f3rec', None) is not None:
        raise RuntimeError('Previous F3 construction was not retired')
    from .tree_selection import F3Recorder
    hf = sparse.model_config_hf
    if hf.num_hidden_layers != 64:
        raise RuntimeError('This tree-prophetkv integration requires Qwen3-32B')
    sparse.f3_tokens = tuple(tokens)
    sparse.f3_request_id = request_id
    sparse.f3rec = F3Recorder(sparse.f3_tokens,
        getattr(hf, 'head_dim', hf.hidden_size // hf.num_attention_heads), hf.num_hidden_layers)


def record_layer(sparse, layer_index, query, key):
    recorder = getattr(sparse, 'f3rec', None)
    if recorder is None:
        return False
    if recorder.add_layer(layer_index, query, key):
        from vllm.distributed import get_tensor_model_parallel_rank
        values = recorder.result()  # all ranks enter both reductions
        config = sparse.tree_f3_config
        if get_tensor_model_parallel_rank() == 0:
            write_f3(config['cache_dir'], config['namespace'], sparse.f3_tokens, values)
        sparse.f3rec = None
        sparse.f3_tokens = None
    return True


def prepare_sinks(sparse, bounds, device):
    _, forced = budget(bounds, sparse.ratio)
    positions = torch.tensor(forced, device=device, dtype=torch.long)
    tokens = torch.tensor([sparse.prompt_token_ids[p] for p in forced], device=device)
    return positions, sparse.model.get_input_embeddings(tokens), None


def probe_sinks(sparse, layer, hidden, residual, positions, ck, cv):
    """Repair S4 in gathered copies only, preserving the source probe algorithm."""
    from .runtime import project
    from .attention import positional_kernel
    q, k, v, residual = project(layer, hidden, residual, positions)
    ck[positions] = k
    cv[positions] = v
    output = torch.empty_like(q)
    zero = torch.zeros(1, device=q.device, dtype=torch.int32)
    q_len = torch.tensor([len(q)], device=q.device, dtype=torch.int32)
    k_len = torch.tensor([len(ck)], device=q.device, dtype=torch.int32)
    positional_kernel(sparse)(q, ck, cv, output, zero, q_len, zero, k_len, positions,
                             len(q), is_causal=True, sm_scale=1 / math.sqrt(layer.self_attn.head_dim))
    h, _ = layer.self_attn.o_proj(output.reshape(len(q), -1))
    h, residual = layer.post_attention_layernorm(h, residual)
    return layer.mlp(h), residual


def tree_selection(sparse, scores, bounds):
    from vllm.distributed import tensor_model_parallel_all_reduce, get_tp_group
    from .tree_selection import select_tree
    started = time.perf_counter()
    group = get_tp_group()
    scores = tensor_model_parallel_all_reduce(scores) / group.world_size
    eligible = scores[bounds[1]:]
    config = sparse.tree_f3_config
    links = []
    for a, z in zip(bounds[1:-2], bounds[2:-1]):
        links.extend(read_f3(config['cache_dir'], config['namespace'], sparse.prompt_token_ids[a:z]))
    chosen, _ = select_tree(links, eligible.double().cpu().numpy(), bounds, sparse.ratio)
    selected = torch.tensor(chosen, device=scores.device, dtype=torch.long)
    if group.world_size > 1:
        selected = group.broadcast(selected, src=0)
    count, forced = budget(bounds, sparse.ratio)
    return eligible, selected, dict(selection_method=METHOD, tree_protocol=PROTOCOL,
        unit_budget=count, forced=forced, f3_links=links, f3_layers=list(range(16)),
        probe_question_positions=list(range(min(sparse.request.question_positions), bounds[-1])),
        recompute_mask='prefix-fresh-earlier-own-chunk-causal', suffix_mask='prefix-fresh-causal-suffix',
        decode_mask='native-unmasked', tree_seconds=time.perf_counter()-started)
