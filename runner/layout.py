"""Versioned token-preserving layouts and request metadata (no CUDA imports)."""
import hashlib
import json

PROMPT_PROTOCOL = 'rpkv-original-tokens-v1'
CACHE_PROTOCOL = 'rpkv-local-chunk-kv-v1'
METADATA_KEY = 'rpkv_layout'


def token_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def boundaries_for(length, question_start, chunk_size=4096):
    if type(length) is not int or length < 1 or not 0 <= question_start < length:
        raise ValueError('Invalid prompt length/question start')
    if type(chunk_size) is not int or chunk_size % 64 or not 64 <= chunk_size <= 4096:
        raise ValueError('Context chunks must be block-aligned and at most 4096 tokens')
    suffix = max(0, min(question_start, length - 256) // 64 * 64)
    return list(range(0, suffix, chunk_size)) + ([suffix] if suffix else [0]) + [length]


def validate_layout(tokens, boundaries, question_positions, sparse=False):
    if not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('Invalid token IDs')
    b, q = boundaries, question_positions
    if (not isinstance(b, (list, tuple)) or len(b) < 2 or b[0] != 0 or b[-1] != len(tokens)
            or any(type(v) is not int for v in b) or any(a >= z for a,z in zip(b,b[1:]))):
        raise ValueError('Invalid boundaries: require contiguous increasing prompt coverage')
    if (not q or list(q) != sorted(set(q)) or any(type(p) is not int or not b[-2] <= p < len(tokens) for p in q)):
        raise ValueError('Question must be entirely in the fresh suffix')
    suffix = max(0, min(q[0], len(tokens)-256)//64*64)
    if b[-2] != suffix or any(v % 64 for v in b[:-1]):
        raise ValueError('Invalid block-aligned suffix start')
    if any(z-a > 4096 for a,z in zip(b[:-2],b[1:-1])):
        raise ValueError('Context chunks exceed 4096 tokens')
    if sparse and len(b) < 4:
        raise ValueError('Sparse inference requires at least two context chunks; use baseline for this input')


def stamp_sample(sample, template=None):
    sample['prompt_protocol'] = PROMPT_PROTOCOL
    sample['token_sha256'] = token_hash(sample['token_ids'])
    if template is not None:
        sample['template_sha256'] = hashlib.sha256(template.encode()).hexdigest()
    return sample


def request_metadata(request_id, tokens, phase, sample=None):
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('Request identity required')
    if phase == 'read':
        if sample is None or list(tokens) != sample['token_ids']:
            raise ValueError('Read metadata requires the unchanged prepared input')
        if sample.get('prompt_protocol') != PROMPT_PROTOCOL or sample.get('token_sha256') != token_hash(tokens):
            raise ValueError('Incompatible or corrupt prompt identity; rebuild from source')
        bounds, question = sample['boundaries'], sample['question_positions']
        validate_layout(tokens, bounds, question)
    elif phase == 'populate':
        if not tokens or len(tokens) % 64 or len(tokens) > 4096:
            raise ValueError('Populate requires one complete local block-aligned context chunk')
        bounds, question = [0, len(tokens)], []
    else:
        raise ValueError('Request phase must be populate/read')
    return dict(protocol=PROMPT_PROTOCOL, cache_protocol=CACHE_PROTOCOL, request_id=request_id,
                phase=phase, token_sha256=token_hash(tokens), boundaries=list(bounds),
                question_positions=list(question))


def validate_request_metadata(metadata, request_id, tokens, sparse=True):
    if not isinstance(metadata, dict):
        raise ValueError('Missing explicit rpkv request metadata')
    if (metadata.get('protocol') != PROMPT_PROTOCOL or metadata.get('cache_protocol') != CACHE_PROTOCOL
            or metadata.get('request_id') != request_id or metadata.get('token_sha256') != token_hash(tokens)):
        raise ValueError('Request metadata identity/token hash mismatch')
    phase = metadata.get('phase')
    if phase == 'read':
        validate_layout(tokens, metadata.get('boundaries'), metadata.get('question_positions'), sparse=sparse)
    elif phase == 'populate':
        expected = request_metadata(request_id, tokens, phase)
        if metadata != expected:
            raise ValueError('Invalid populate layout')
    else:
        raise ValueError('Missing/invalid request phase')
    return metadata


def sample_provenance(sample):
    return {key:sample.get(key) for key in ('prompt_protocol','token_sha256','template_sha256',
        'evaluation_protocol','generator_revision','generator_config_sha256')}


def validate_policy_protocol(policy, sample):
    if policy.get('prompt_protocol') != PROMPT_PROTOCOL or policy.get('evaluation_protocol') != sample.get('evaluation_protocol'):
        raise ValueError('Policy is incompatible with prompt/evaluation protocol; no automatic refit')
