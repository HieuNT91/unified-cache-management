"""Policies for a bounded, prompt-at-a-time cache sweep (CPU importable)."""
from ucm.sparse.prophetkv.lifecycle import TrackedStore
from ucm.sparse.prophetkv.layers import resolve_layers


def request_phase(request_id, layouts):
    parts = request_id.split(':')
    if len(parts) != 4 or parts[1] not in layouts or parts[2] not in ('populate', 'read'):
        raise RuntimeError('Unknown temporary-cache request')
    return parts[2], layouts[parts[1]]


class TemporaryStore(TrackedStore):
    """Only explicitly tagged construction requests may write to the backend."""
    writable = False

    def dump_data(self, *args, **kwargs):
        if not self.writable:
            raise RuntimeError('Temporary KV is read-only during inference')
        return super().dump_data(*args, **kwargs)


def configure_sparse(sparse, method, ratio, layers=None):
    selected = resolve_layers(method, layers)
    if not 0 <= ratio <= 1:
        raise ValueError('Invalid ratio')
    if (sparse.request_state.pending is not None or getattr(sparse, 'request', None) is not None
            or getattr(sparse, 'prefill_state', None) is not None
            or sparse.active or sparse.connector.worker_request_id is not None
            or sparse.connector.store.pending):
        raise RuntimeError('Retire the request before changing the scoring policy')
    sparse.method, sparse.ratio, sparse.scoring_layers = method, ratio, selected
    return dict(method=method, ratio=ratio, scoring_layers=list(selected))
