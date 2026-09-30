"""Supported checkpoints; importable without CUDA.

Qwen3-32B with YaRN 4x is the official profile and keeps every original
constant. Qwen3-4B-Instruct-2507 is an explicit addition: it has a native
262144-position RoPE table, so it runs without rope scaling, and its window
is sized for one 24 GB card (prompt plus output must still fit).
"""
import json
from pathlib import Path

PROFILES = {
    'qwen3-32b': dict(num_hidden_layers=64, hidden_size=5120, window=131072,
                      rope=dict(rope_type='yarn', factor=4.0, original_max_position_embeddings=32768)),
    'qwen3-4b-instruct-2507': dict(num_hidden_layers=36, hidden_size=2560, window=73728,
                                   max_position_embeddings=262144, rope=None),
}


def profile_of(config):
    """Match an HF config dict (or object) to exactly one supported profile."""
    get = config.get if isinstance(config, dict) else (lambda k, d=None: getattr(config, k, d))
    if get('model_type') != 'qwen3' or get('quantization_config'):
        raise ValueError('Use an unquantized Qwen3 checkpoint')
    for name, profile in PROFILES.items():
        if (get('num_hidden_layers'), get('hidden_size')) != (profile['num_hidden_layers'], profile['hidden_size']):
            continue
        native = profile.get('max_position_embeddings')
        if native is not None and get('max_position_embeddings') != native:
            continue
        return name, profile
    raise ValueError('Unsupported checkpoint; supported profiles: ' + ', '.join(PROFILES))


def model_profile(model):
    return profile_of(json.loads((Path(model) / 'config.json').read_text()))
