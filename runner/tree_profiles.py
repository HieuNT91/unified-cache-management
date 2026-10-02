"""Execution constraints are independent of a learned tree's actions/features."""
from runner.config import engine_config,validate_sample
PROFILES={
    'ruler':dict(input_tokens=65536,output_tokens=None,thinking=False,kv_tokens=65920,kv_blocks=1031),
    'longbench-v2':dict(input_tokens=114688,output_tokens=16384,thinking=True,kv_tokens=131072,kv_blocks=2048),
}


def hardware_profile(name='server',dataset='ruler',tp=4):
    if name=='server':
        return dict(names=('A800','L20'),memory=.95 if tp==2 else .9,workspace_gib=8.,
                    kv_cache_mode='auto' if tp==2 else 'fixed')
    if name=='l20-tp2' and dataset=='ruler':
        return dict(names=('L20',),memory=.95,workspace_gib=3.,
                    kv_cache_mode='auto' if tp==2 else 'fixed')
    if name=='rtx4500ada' and dataset in PROFILES:
        # Local 24 GiB TP4 profile: validated clean YaRN4/16K-prefill runs,
        # with 4096-token activation tiles and full original-position KV.
        return dict(names=('RTX 4500 Ada Generation',),memory=.95,workspace_gib=2.5,kv_cache_mode='fixed')
    raise ValueError('Unsupported hardware/dataset profile')


def validate(sample,dataset):
    if dataset not in PROFILES:raise ValueError('Unsupported dataset')
    p=PROFILES[dataset];validate_sample(sample,4096,p['input_tokens'])
    if sample['thinking']!=p['thinking'] or (p['output_tokens'] is not None and sample['max_output_tokens']!=p['output_tokens']):
        raise ValueError('Dataset execution profile mismatch')
    if dataset=='ruler':
        from scripts.ruler import EVALUATION_PROTOCOL
        if sample.get('evaluation_protocol')!=EVALUATION_PROTOCOL:
            raise ValueError('RULER requires original task-batch protocol; rebuild from source')
    # References remain reporting metadata; generate submits only tokens/layout.
    return sample


def allocation(dataset,hardware='server'):
    p=dict(PROFILES[dataset])
    if hardware=='rtx4500ada':p.update(kv_tokens=65920,kv_blocks=1031)
    return p


def config(model,dataset,cached,cache,delimiter=None,hardware='server',tp=4):
    from runner.tensor_parallel import validate_tp
    validate_tp(tp)
    p=allocation(dataset,hardware)
    profile=hardware_profile(hardware,dataset,tp)
    result=engine_config(model,'prophetkv' if cached else 'baseline',ratio=.01,tp=tp,cache_dir=cache,
                        memory=profile['memory'])
    # In TP2 server runs vLLM profiles runtime memory and spends the remaining
    # budget on KV. Context/output limits remain independent of cache capacity.
    result.update(max_model_len=p['kv_tokens'],
                  hf_overrides=dict(max_position_embeddings=32768,ucm_activation_tile=4096))
    if profile['kv_cache_mode']=='fixed':result['num_gpu_blocks_override']=p['kv_blocks']
    return result


def validate_initialization(receipts,dataset,hardware='server',tp=4):
    from runner.tensor_parallel import validate_tp
    validate_tp(tp)
    p=allocation(dataset,hardware)
    automatic=hardware_profile(hardware,dataset,tp)['kv_cache_mode']=='auto'
    ranks = [r.get('rank') for r in receipts]
    # Legacy TP4 initialization receipts did not include rank IDs.
    if len(receipts)!=tp or ((tp==2 or any(r is not None for r in ranks)) and
                            sorted(-1 if r is None else r for r in ranks)!=list(range(tp))):
        raise ValueError('Missing initialization rank')
    blocks=[r.get('kv_blocks') for r in receipts]
    if (any(type(n) is not int or n<p['kv_blocks'] for n in blocks) or
            len(set(blocks))!=1 or (not automatic and blocks[0]!=p['kv_blocks'])):
        raise ValueError('KV cache capacity audit failed: insufficient, inconsistent or unexpected blocks')
    for r in receipts:
        if (r['kv_tokens']!=p['kv_tokens'] or r['block_size']!=64 or
                len(r['rope'])!=64 or any(not a['formula_bitwise_equal'] or a['table_positions']!=131072 for a in r['rope'])):
            raise ValueError('KV or native YaRN4 initialization audit failed')
