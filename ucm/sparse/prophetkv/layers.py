"""CPU-safe validation of the Qwen3 scoring-layer policy (64 layers for Qwen3-32B)."""


def resolve_layers(method, layers=None, count=None, num_layers=64):
    if method not in ('prophetkv', 'selective_prophetkv', 'router'):
        raise ValueError('Unsupported ProphetKV method')
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError('Invalid decoder layer count')
    if method in ('prophetkv', 'router'):
        if layers is not None or count is not None:
            raise ValueError('Use selective_prophetkv to configure scoring layers')
        return tuple(range(num_layers))
    if (layers is None) == (count is None):
        raise ValueError('Specify exactly one of scoring layers or layer count')
    if count is not None:
        if type(count) is not int or not 1 <= count <= num_layers:
            raise ValueError(f'Layer count must be an integer in [1, {num_layers}]')
        # A count always means the last N decoder layers, in ascending order.
        return tuple(range(num_layers - count, num_layers))
    layers = tuple(layers)
    if not layers or any(type(i) is not int or not 0 <= i < num_layers for i in layers):
        raise ValueError(f'Layer IDs must be integers in [0, {num_layers - 1}]')
    if len(set(layers)) != len(layers):
        raise ValueError('Duplicate layer IDs')
    return tuple(sorted(layers))
