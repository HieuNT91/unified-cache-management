"""Shared clean method configuration; importable without CUDA."""
METHODS = ('baseline', 'prophetkv', 'tree-prophetkv', 'adatree-v1')
WINDOW = 131072
PREFILL_BUDGET = 16384
ROPE = dict(rope_type='yarn', factor=4.0, original_max_position_embeddings=32768)
VERSIONS = {'vllm': '0.9.2', 'torch': '2.7.0', 'transformers': '4.53.2', 'uc-manager': '0.3.0'}


def engine_config(model, method, ratio=None, tp=4,
                  memory=.9, cache_dir=None,
                  adatree_chunk_size=4096):
    if method not in METHODS:
        raise ValueError('Unknown method')
    adaptive = method == 'adatree-v1'
    if adaptive:
        from ucm.sparse.prophetkv.adatree import feature_names
        feature_names(adatree_chunk_size)
        if ratio is not None:
            raise ValueError('AdaTreeV1 rejects manual ratio overrides')
        ratio = .05  # Minimum leaf budget; routing chooses the measured answer's ratio.
    ratio = .2 if ratio is None else ratio
    if not 0 <= ratio <= 1 or tp not in (1, 2, 4, 8) or not 0 < memory < 1:
        raise ValueError('Invalid ratio, tensor parallel size or memory utilization')
    cfg = dict(model=str(model), tokenizer=str(model), dtype='bfloat16',
               max_model_len=WINDOW, max_num_batched_tokens=PREFILL_BUDGET,
               enable_chunked_prefill=True, enable_prefix_caching=False,
               enforce_eager=True, max_num_seqs=1, block_size=64,
               tensor_parallel_size=tp, distributed_executor_backend='mp',
               disable_custom_all_reduce=True, gpu_memory_utilization=memory,
               generation_config='vllm', seed=0, quantization=None,
               cpu_offload_gb=0, swap_space=0, rope_scaling=dict(ROPE),
               worker_cls='runner.runtime.worker.Worker')
    if method != 'baseline':
        if cache_dir is None:
            raise ValueError('Cached methods require a cache directory')
        native_method = 'tree-prophetkv' if adaptive else method
        sparse = dict(method=native_method, ratio=ratio,
                      component_audit=False, compute_meta={})
        cfg['kv_transfer_config'] = dict(kv_connector='PersistentBlendConnector',
            kv_connector_module_path='ucm.integration.vllm.persistent_connector',
            kv_role='kv_both', kv_connector_extra_config=dict(
                ucm_connectors=[dict(ucm_connector_name='UcmNfsStore',
                    ucm_connector_config=dict(storage_backends=str(cache_dir),
                        use_direct=False, timeout_ms=120000))],
                ucm_sparse_config={'ProphetKV': sparse, 'Blend': sparse}))
    if method in ('tree-prophetkv', 'adatree-v1'):
        from pathlib import Path
        from runner.tree.prophetkv import PROTOCOL
        cfg['kv_transfer_config']['kv_connector_extra_config']['tree_f3'] = dict(
            protocol=PROTOCOL, cache_dir=str(Path(cache_dir).resolve()),
            namespace=str(Path(cache_dir).resolve()), readonly=False)
        if adaptive:
            cfg['kv_transfer_config']['kv_connector_extra_config']['adatree_v1'] = True
            cfg['kv_transfer_config']['kv_connector_extra_config']['adatree_chunk_size'] = adatree_chunk_size
    return cfg


def validate_preparation(kv_chunk_size=4096, context_length=WINDOW):
    if type(kv_chunk_size) is not int or kv_chunk_size % 64 or not 64 <= kv_chunk_size <= 4096:
        raise ValueError('kv-chunk-size must be a multiple of 64 from 64 to 4096')
    if type(context_length) is not int or not 1 <= context_length <= WINDOW:
        raise ValueError('context-length must be from 1 to 131072')


def validate_sample(sample, kv_chunk_size=4096, context_length=WINDOW):
    validate_preparation(kv_chunk_size, context_length)
    from runner.inputs.layout import PROMPT_PROTOCOL, token_hash, validate_layout
    ids, bounds, question = sample['token_ids'], sample['boundaries'], sample['question_positions']
    if sample.get('prompt_protocol') != PROMPT_PROTOCOL or sample.get('token_sha256') != token_hash(ids):
        raise ValueError('Incompatible/corrupt prompt protocol; rebuild from source or verified mapping')
    validate_layout(ids, bounds, question)
    if any(b-a > kv_chunk_size for a,b in zip(bounds[:-2], bounds[1:-1])):
        raise ValueError('Offline context chunks exceed kv-chunk-size')
    if len(ids) > context_length:
        raise ValueError('Formatted prompt exceeds context-length; truncation is forbidden')
    budget = sample['max_output_tokens']
    from runner.inputs.prepared_policy import validate_sample_policy
    from runner.inputs.prepared_policy import RULER_PROTOCOL, LONGBENCH_PROTOCOL
    thinking_ruler = sample.get('evaluation_protocol') in (RULER_PROTOCOL, LONGBENCH_PROTOCOL)
    if thinking_ruler:
        validate_sample_policy(sample)
    if type(budget) is not int or not 1 <= budget <= (WINDOW if thinking_ruler else 16384) or len(ids) + budget > WINDOW:
        raise ValueError('Prompt plus output exceeds the 131072-token window')
    if sample.get('evaluation_protocol') and not thinking_ruler:
        from scripts.data.ruler import CAPS, EVALUATION_PROTOCOL
        if sample['evaluation_protocol'] != EVALUATION_PROTOCOL or sample['max_output_tokens'] != CAPS.get(sample.get('task')) or len(ids) + budget > 65536 or sample['thinking']:
            raise ValueError('Invalid original RULER budget/decoding protocol')
    if type(sample['thinking']) is not bool:
        raise ValueError('thinking must be boolean')
    return sample
