"""Shared clean method configuration; importable without CUDA."""
from ucm.sparse.prophetkv.layers import resolve_layers

METHODS = ('baseline', 'prophetkv', 'selective_prophetkv', 'router')
WINDOW = 131072
PREFILL_BUDGET = 16384
ROPE = dict(rope_type='yarn', factor=4.0, original_max_position_embeddings=32768)
VERSIONS = {'vllm': '0.9.2', 'torch': '2.7.0', 'transformers': '4.53.2', 'uc-manager': '0.3.0'}


def engine_config(model, method, ratio=None, layers=None, count=None, tp=4,
                  memory=.9, cache_dir=None, end_token=None, persistent=None, router_policy=None, router_id="router1", router_profile=False):
    if method not in METHODS:
        raise ValueError('Unknown method')
    if method == 'router':
        from runner.router_policy import load_policy
        if ratio is not None or layers is not None or count is not None:
            raise ValueError('Router requests reject manual ratio/layer overrides')
        if router_policy is None or router_id not in ('router1','router2','router3'):
            raise ValueError('Router requires --router-policy and a valid --router-id')
        policy = load_policy(router_policy)
        from runner.layout import PROMPT_PROTOCOL
        if policy.get('prompt_protocol') != PROMPT_PROTOCOL:
            raise ValueError('Policy belongs to an incompatible prompt protocol; no automatic refit')
        router_profile = True
        ratio = .01
    elif router_policy is not None:
        raise ValueError('--router-policy requires --method router')
    ratio = .2 if ratio is None else ratio
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
    if router_profile:
        if tp != 4:
            raise ValueError('Frozen router profile requires TP4')
        cfg.update(max_model_len=65920, num_gpu_blocks_override=1031,
                   hf_overrides={'max_position_embeddings':32768})
    if method != 'baseline':
        selected = resolve_layers(method, layers, count)
        if cache_dir is None:
            raise ValueError('Cached methods require a cache directory')
        sparse = dict(method='prophetkv' if method == 'router' else method, ratio=ratio,
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
    if type(kv_chunk_size) is not int or kv_chunk_size % 64 or not 64 <= kv_chunk_size <= 4096:
        raise ValueError('kv-chunk-size must be a multiple of 64 from 64 to 4096')
    if type(context_length) is not int or not 1 <= context_length <= WINDOW:
        raise ValueError('context-length must be from 1 to 131072')


def validate_sample(sample, kv_chunk_size=4096, context_length=WINDOW):
    validate_preparation(kv_chunk_size, context_length)
    from runner.layout import PROMPT_PROTOCOL, token_hash, validate_layout
    ids, bounds, question = sample['token_ids'], sample['boundaries'], sample['question_positions']
    if sample.get('prompt_protocol') != PROMPT_PROTOCOL or sample.get('token_sha256') != token_hash(ids):
        raise ValueError('Incompatible/corrupt prompt protocol; rebuild from source or verified mapping')
    validate_layout(ids, bounds, question)
    if any(b-a > kv_chunk_size for a,b in zip(bounds[:-2], bounds[1:-1])):
        raise ValueError('Offline context chunks exceed kv-chunk-size')
    if len(ids) > context_length:
        raise ValueError('Formatted prompt exceeds context-length; truncation is forbidden')
    budget = sample['max_output_tokens']
    if type(budget) is not int or not 1 <= budget <= 16384 or len(ids) + budget > WINDOW:
        raise ValueError('Prompt plus output exceeds the 131072-token window')
    if sample.get('evaluation_protocol'):
        from scripts.ruler import CAPS, EVALUATION_PROTOCOL
        if sample['evaluation_protocol'] != EVALUATION_PROTOCOL or sample['max_output_tokens'] != CAPS.get(sample.get('task')) or len(ids) + budget > 65536 or sample['thinking']:
            raise ValueError('Invalid original RULER budget/decoding protocol')
    if type(sample['thinking']) is not bool:
        raise ValueError('thinking must be boolean')
    return sample
