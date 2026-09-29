"""CPU-safe validation of the 64-layer Qwen3 scoring policy."""


def resolve_layers(method, layers=None, count=None):
    if method not in ('prophetkv', 'selective_prophetkv', 'router'):
        raise ValueError('Unsupported ProphetKV method')
    if method in ('prophetkv', 'router'):
        if layers is not None or count is not None:
            raise ValueError('Use selective_prophetkv to configure scoring layers')
        return tuple(range(64))
    if (layers is None) == (count is None):
        raise ValueError('Specify exactly one of scoring layers or layer count')
    if count is not None:
        if type(count) is not int or not 1 <= count <= 64:
            raise ValueError('Layer count must be an integer in [1, 64]')
        # A count always means the last N decoder layers, in ascending order.
        return tuple(range(64 - count, 64))
    layers = tuple(layers)
    if not layers or any(type(i) is not int or not 0 <= i < 64 for i in layers):
        raise ValueError('Layer IDs must be integers in [0, 63]')
    if len(set(layers)) != len(layers):
        raise ValueError('Duplicate layer IDs')
    return tuple(sorted(layers))
