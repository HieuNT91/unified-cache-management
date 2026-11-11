"""Model identity check shared by the benchmark preparation scripts."""
import json


def check_model(path):
    cfg = json.loads((path / 'config.json').read_text())
    if (cfg.get('model_type'), cfg.get('num_hidden_layers'), cfg.get('hidden_size')) != ('qwen3', 64, 5120):
        raise ValueError('Use a local Qwen3-32B checkpoint')
    if cfg.get('quantization_config'):
        raise ValueError('Use the unquantized BF16 checkpoint')
