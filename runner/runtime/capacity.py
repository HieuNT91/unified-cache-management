"""Original prepared-input validation and hardware-independent runtime budgets."""
from runner.inputs.config import engine_config,validate_sample
INPUT_LIMITS = {'ruler-thinking': 65536, 'longbench-v2': 114688}


def validate(sample,dataset):
    if dataset not in INPUT_LIMITS:raise ValueError('Unsupported dataset')
    validate_sample(sample,4096,INPUT_LIMITS[dataset])
    if sample['thinking'] is not True:
        raise ValueError('Dataset execution profile mismatch')
    # References remain reporting metadata; generate submits only tokens/layout.
    return sample


def allocation(dataset):
    """Runtime capacity; prepared samples keep their original generation metadata."""
    from runner.runtime.output_budget import OUTPUT_CAP
    p = dict(input_tokens=INPUT_LIMITS[dataset], output_tokens=OUTPUT_CAP, thinking=True)
    p['kv_tokens'] = p['input_tokens'] + p['output_tokens']
    p['kv_blocks'] = (p['kv_tokens'] + 63) // 64
    return p


def config(model,dataset,cached,cache,*,tp=4,memory=.95):
    from runner.runtime.tensor_parallel import validate_tp
    validate_tp(tp)
    if not 0 < memory < 1:
        raise ValueError('GPU memory utilization must be between 0 and 1')
    p=allocation(dataset)
    result=engine_config(model,'prophetkv' if cached else 'baseline',ratio=.01,tp=tp,
                        cache_dir=cache,memory=memory)
    # Let vLLM measure activation/workspace and available KV blocks. Do not
    # override its measured block capacity or offload model weights to the CPU.
    result.update(max_model_len=p['kv_tokens'],
                  hf_overrides=dict(max_position_embeddings=32768,ucm_activation_tile=4096))
    return result


def validate_initialization(receipts,dataset,*,tp=4):
    from runner.runtime.tensor_parallel import validate_tp
    validate_tp(tp)
    p=allocation(dataset)
    if (len(receipts)!=tp or
            sorted(r.get('rank', -1) for r in receipts)!=list(range(tp))):
        raise ValueError('Missing initialization rank')
    blocks=[r.get('kv_blocks') for r in receipts]
    if (any(type(n) is not int or n<p['kv_blocks'] for n in blocks)
            or len(set(blocks))!=1):
        raise ValueError('KV cache capacity audit failed: insufficient or inconsistent blocks')
    for r in receipts:
        if (r['kv_tokens']!=p['kv_tokens'] or r['block_size']!=64 or
                len(r['rope'])!=64 or any(not a['formula_bitwise_equal'] or a['table_positions']!=131072 for a in r['rope'])):
            raise ValueError('KV or native YaRN4 initialization audit failed')
