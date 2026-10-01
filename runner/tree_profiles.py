"""Execution constraints are independent of a learned tree's actions/features."""
from runner.config import engine_config,validate_sample
PROFILES={
    'ruler':dict(input_tokens=65536,output_tokens=None,thinking=False,kv_tokens=65920,kv_blocks=1031),
    'longbench-v2':dict(input_tokens=114688,output_tokens=16384,thinking=True,kv_tokens=131072,kv_blocks=2048),
}


def hardware_profile(name='server',dataset='ruler'):
    if name=='server':
        return dict(names=('A800','L20'),memory=.9,workspace_gib=8.)
    if name=='rtx4500ada' and dataset in PROFILES:
        # Local 24 GiB TP4 profile: validated clean YaRN4/16K-prefill runs,
        # with 4096-token activation tiles and full original-position KV.
        return dict(names=('RTX 4500 Ada Generation',),memory=.95,workspace_gib=2.5)
    raise ValueError('Unsupported hardware/dataset profile')


def validate(sample,dataset):
    if dataset not in PROFILES:raise ValueError('Unsupported dataset')
    p=PROFILES[dataset];validate_sample(sample,4096,p['input_tokens'])
    if sample['thinking']!=p['thinking'] or (p['output_tokens'] is not None and sample['max_output_tokens']!=p['output_tokens']):
        raise ValueError('Dataset execution profile mismatch')
    if dataset=='ruler':
        from scripts.ruler_64000 import EVALUATION_PROTOCOL
        if sample.get('evaluation_protocol')!=EVALUATION_PROTOCOL:
            raise ValueError('RULER requires original task-batch protocol; rebuild from source')
    # References remain reporting metadata; generate submits only tokens/layout.
    return sample


def allocation(dataset,hardware='server'):
    p=dict(PROFILES[dataset])
    if hardware=='rtx4500ada':p.update(kv_tokens=65920,kv_blocks=1031)
    return p


def config(model,dataset,cached,cache,delimiter=None,hardware='server'):
    p=allocation(dataset,hardware)
    result=engine_config(model,'prophetkv' if cached else 'baseline',ratio=.01,tp=4,cache_dir=cache,
                        memory=hardware_profile(hardware,dataset)['memory'])
    result.update(max_model_len=p['kv_tokens'],num_gpu_blocks_override=p['kv_blocks'],
                  hf_overrides=dict(max_position_embeddings=32768,ucm_activation_tile=4096))
    return result


def validate_initialization(receipts,dataset,hardware='server'):
    p=allocation(dataset,hardware)
    if len(receipts)!=4:raise ValueError('Missing initialization rank')
    for r in receipts:
        if (r['kv_tokens']!=p['kv_tokens'] or r['kv_blocks']!=p['kv_blocks'] or r['block_size']!=64 or
                len(r['rope'])!=64 or any(not a['formula_bitwise_equal'] or a['table_positions']!=131072 for a in r['rope'])):
            raise ValueError('KV or native YaRN4 initialization audit failed')
