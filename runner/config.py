"""One configuration shared by all three methods; importable without CUDA."""
from ucm.sparse.prophetkv.layers import resolve_layers

METHODS = ('baseline', 'prophetkv', 'selective_prophetkv')
WINDOW = 131072
PREFILL_BUDGET = 16384
ROPE = dict(rope_type='yarn', factor=4.0, original_max_position_embeddings=32768)
VERSIONS = {'vllm': '0.9.2', 'torch': '2.7.0', 'transformers': '4.53.2', 'uc-manager': '0.3.0'}


def engine_config(model, method, ratio=.2, layers=None, count=None, tp=4,
                  memory=.9, cache_dir=None, end_token=None, persistent=None):
    if method not in METHODS:
        raise ValueError('Unknown method')
    if not 0 <= ratio <= 1 or tp not in (1, 2, 4, 8) or not 0 < memory < 1:
        raise ValueError('Invalid ratio, tensor parallel size or memory utilization')
    if method == 'baseline' and (layers is not None or count is not None):
        raise ValueError('Baseline has no scoring layers')
    cfg = dict(model=str(model), tokenizer=str(model), dtype='bfloat16',
               max_model_len=WINDOW, max_num_batched_tokens=PREFILL_BUDGET,
               enable_chunked_prefill=True, enable_prefix_caching=False,
               enforce_eager=True, max_num_seqs=1, block_size=64,
               tensor_parallel_size=tp, distributed_executor_backend='mp',
               disable_custom_all_reduce=True, gpu_memory_utilization=memory,
               generation_config='vllm', seed=0, quantization=None,
               cpu_offload_gb=0, swap_space=0, rope_scaling=dict(ROPE),
               worker_cls='runner.worker.Worker')
    if method != 'baseline':
        selected = resolve_layers(method, layers, count)
        if cache_dir is None or type(end_token) is not int:
            raise ValueError('Cached methods require a cache directory and delimiter')
        sparse = dict(method=method, ratio=ratio, chunk_end_token_id=end_token,
                      component_audit=False, compute_meta={})
        if method == 'selective_prophetkv':
            sparse['scoring_layers'] = list(selected)
        cfg['kv_transfer_config'] = dict(kv_connector='PersistentBlendConnector',
            kv_connector_module_path='ucm.integration.vllm.persistent_connector',
            kv_role='kv_both', kv_connector_extra_config=dict(
                ucm_connectors=[dict(ucm_connector_name='UcmNfsStore',
                    ucm_connector_config=dict(storage_backends=str(cache_dir),
                        use_direct=False, timeout_ms=120000))],
                ucm_sparse_config={'ProphetKV': sparse, 'Blend': sparse}))
    if persistent is not None and method != 'baseline':
        cfg['kv_transfer_config']['kv_connector_extra_config']['persistent_setup'] = persistent
    return cfg


def validate_preparation(kv_chunk_size=4096, context_length=WINDOW):
    if type(kv_chunk_size) is not int or kv_chunk_size % 64 or not 64 <= kv_chunk_size <= PREFILL_BUDGET:
        raise ValueError('kv-chunk-size must be a multiple of 64 from 64 to 16384')
    if type(context_length) is not int or not 1 <= context_length <= WINDOW:
        raise ValueError('context-length must be from 1 to 131072')


def validate_sample(sample, kv_chunk_size=4096, context_length=WINDOW):
    validate_preparation(kv_chunk_size, context_length)
    ids, bounds, question = sample['token_ids'], sample['boundaries'], sample['question_positions']
    if not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError('Invalid token IDs')
    if len(bounds) < 4 or bounds[0] != 0 or bounds[-1] != len(ids):
        raise ValueError('Need at least two context chunks and a fresh suffix')
    if any(type(b) is not int for b in bounds) or any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Invalid boundaries')
    if any(b % 64 for b in bounds[:-1]) or bounds[-1] - bounds[-2] < 256:
        raise ValueError('Context must be block aligned with at least 256 fresh tokens')
    if any(b - a > kv_chunk_size for a, b in zip(bounds[:-2], bounds[1:-1])):
        raise ValueError('Offline context chunks exceed kv-chunk-size')
    if not question or question != sorted(set(question)) or any(
            type(q) is not int or not bounds[-2] <= q < bounds[-1] for q in question):
        raise ValueError('Question must be entirely in the fresh suffix')
    marker = ids[bounds[1] - 1]
    # Consecutive padding blocks are merged by the connector.
    for start, end in zip(bounds[:-2], bounds[1:-1]):
        if ids[end - 1] != marker:
            raise ValueError('Context delimiters differ')
        seen = False
        for pos in range(start + 63, end, 64):
            if ids[pos] == marker:
                seen = True
            elif seen:
                raise ValueError('Internal delimiter splits a context chunk')
    if any(ids[p] == marker for p in range(bounds[-2] + 63, len(ids), 64)):
        raise ValueError('Delimiter in fresh suffix')
    if len(ids) > context_length:
        raise ValueError('Formatted prompt exceeds context-length; truncation is forbidden')
    budget = sample['max_output_tokens']
    if type(budget) is not int or not 1 <= budget <= 16384 or len(ids) + budget > WINDOW:
        raise ValueError('Prompt plus output exceeds the 131072-token window')
    if type(sample['thinking']) is not bool:
        raise ValueError('thinking must be boolean')
    return sample
