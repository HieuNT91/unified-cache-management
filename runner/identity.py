"""Stable, CPU-only cache hashing shared by inventory and both connector ranks."""
import hashlib
import pickle

CACHE_FORMAT = 'ucm-bf16-block64-v2'
HASH_PROTOCOL = 4


class BlockHasher:
    def __init__(self, model_identity, tp, rank, persistent=False):
        self.meta = f'{model_identity}:{tp}:torch.bfloat16:{rank}'.encode()
        self.protocol = HASH_PROTOCOL if persistent else pickle.HIGHEST_PROTOCOL

    def __call__(self, value):
        raw = value if isinstance(value, bytes) else pickle.dumps(value, protocol=self.protocol)
        return hashlib.md5(self.meta + raw).digest()


def setup_seed(fingerprint):
    return ('UCM_SETUP', CACHE_FORMAT, fingerprint)


def block_keys(hasher, tokens, seed, block_size=64):
    return chain_keys(hasher, tokens, hasher(seed), block_size)


def chain_keys(hasher, tokens, parent, block_size=64):
    result = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = hasher((parent, tuple(tokens[start:start + block_size])))
        result.append(parent)
    return result


def shard_name(key, model_identity, tp, rank, persistent=False):
    return (key if rank == 0 else BlockHasher(model_identity, tp, rank, persistent)(key)).hex()
