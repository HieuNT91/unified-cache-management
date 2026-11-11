"""Original prepared-input validation and hardware-independent runtime budgets."""
from runner.inputs.config import engine_config,validate_sample
PROFILES={
    'ruler-thinking':dict(input_tokens=65536,output_tokens=None,thinking=True),
    'ruler':dict(input_tokens=65536,output_tokens=None,thinking=False,kv_tokens=65920,kv_blocks=1031),
    'longbench-v2':dict(input_tokens=114688,output_tokens=16384,thinking=True),
}


def validate(sample,dataset):
    if dataset not in PROFILES:raise ValueError('Unsupported dataset')
    p=PROFILES[dataset];validate_sample(sample,4096,p['input_tokens'])
    if sample['thinking']!=p['thinking'] or (p['output_tokens'] is not None and sample['max_output_tokens']!=p['output_tokens']):
        raise ValueError('Dataset execution profile mismatch')
    if dataset=='ruler-thinking':
        from runner.runtime.thinking_budget import validate_sample_policy
        validate_sample_policy(sample)
    if dataset=='ruler':
        from scripts.data.ruler import EVALUATION_PROTOCOL
        if sample.get('evaluation_protocol')!=EVALUATION_PROTOCOL:
            raise ValueError('RULER requires original task-batch protocol; rebuild from source')
    # References remain reporting metadata; generate submits only tokens/layout.
    return sample


def allocation(dataset):
    """Runtime capacity; prepared samples keep their original generation metadata."""
    from runner.router.adatree import answer_generation
    p = dict(PROFILES[dataset])
    if p['thinking']:
        p['output_tokens'] = answer_generation()['output_reserve']
        p['kv_tokens'] = p['input_tokens'] + p['output_tokens']
        p['kv_blocks'] = (p['kv_tokens'] + 63) // 64
    return p


def config(model,dataset,cached,cache,delimiter=None,*,tp=4,memory=.95):
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
