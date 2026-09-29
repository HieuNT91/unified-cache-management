"""Execution constraints are independent of a learned tree's actions/features."""
from runner.config import engine_config,validate_sample
PROFILES={
    'ruler':dict(input_tokens=64000,output_tokens=256,thinking=False,kv_tokens=65920,kv_blocks=1031),
    'longbench-v2':dict(input_tokens=114688,output_tokens=16384,thinking=True,kv_tokens=131072,kv_blocks=2048),
}


def validate(sample,dataset):
    if dataset not in PROFILES:raise ValueError('Unsupported dataset')
    p=PROFILES[dataset];validate_sample(sample,4096,p['input_tokens'])
    if sample['thinking']!=p['thinking'] or sample['max_output_tokens']!=p['output_tokens']:
        raise ValueError('Dataset execution profile mismatch')
    if dataset=='ruler' and len(sample['token_ids'])!=64000:raise ValueError('RULER requires exactly 64000 formatted tokens')
    if any(k in sample for k in ('references','outputs','gold','answer')):raise ValueError('References leaked into inference input')
    return sample


def config(model,dataset,cached,cache,delimiter):
    p=PROFILES[dataset]
    result=engine_config(model,'prophetkv' if cached else 'baseline',ratio=.01,tp=4,cache_dir=cache,end_token=delimiter)
    result.update(max_model_len=p['kv_tokens'],num_gpu_blocks_override=p['kv_blocks'],
                  hf_overrides=dict(max_position_embeddings=32768,ucm_activation_tile=4096))
    return result


def validate_initialization(receipts,dataset):
    p=PROFILES[dataset]
    if len(receipts)!=4:raise ValueError('Missing initialization rank')
    for r in receipts:
        if (r['kv_tokens']!=p['kv_tokens'] or r['kv_blocks']!=p['kv_blocks'] or r['block_size']!=64 or
                len(r['rope'])!=64 or any(not a['formula_bitwise_equal'] or a['table_positions']!=131072 for a in r['rope'])):
            raise ValueError('KV or native YaRN4 initialization audit failed')
