"""Policies for a bounded, prompt-at-a-time cache sweep (CPU importable)."""
from ucm.sparse.prophetkv.lifecycle import TrackedStore, RepeatedLoad
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


class TreeTemporaryStore(TemporaryStore):
    """Submit each prompt's KV shard once, including still-queued file writes.

    F3 population computes every chunk in full, even shared prefixes. Native
    dump completion only retires D2H; checking file existence cannot prevent a
    second queued write from replacing a shard after the readiness snapshot.
    """
    def __init__(self, store, max_load_blocks=None):
        super().__init__(store, max_load_blocks=max_load_blocks)
        self.namespace = None
        self.submitted_shards = set()
        self.dump_failure = None

    def begin_namespace(self, namespace):
        if self.dump_failure is not None:
            raise RuntimeError('Tree KV dump failed; restart the worker') from self.dump_failure
        if namespace != self.namespace:
            if self.pending:
                raise RuntimeError('Retire transfers before changing tree KV namespace')
            # The driver verifies publication and deletes the previous prompt
            # only after retirement. New namespace hashes cannot alias old ones.
            self.submitted_shards.clear()
            self.namespace = namespace

    def dump_data(self, keys, shards, addresses):
        if not self.writable:
            raise RuntimeError('Temporary KV is read-only during inference')
        if self.router_dense:
            raise RuntimeError('UCM transfer attempted during dense fallback')
        if self.dump_failure is not None:
            raise RuntimeError('Tree KV dump failed; restart the worker') from self.dump_failure
        if self.namespace is None:
            raise RuntimeError('Missing tree KV namespace')
        if len(keys) != len(shards) or len(keys) != len(addresses):
            raise ValueError('Cache dump hashes, shards and source rows differ')
        keep, new = [], set()
        for i, (key, shard) in enumerate(zip(keys, shards)):
            identity = (key, shard)
            if identity not in self.submitted_shards and identity not in new:
                keep.append(i)
                new.add(identity)
        if not keep:
            return RepeatedLoad(())  # Existing wait/drain API: no backend task.
        # Reserve before submission. A backend exception can leave partial work
        # queued, so never retry those keys in the same worker after an error.
        self.submitted_shards.update(new)
        try:
            return super().dump_data([keys[i] for i in keep], [shards[i] for i in keep],
                                     [addresses[i] for i in keep])
        except Exception as error:
            self.dump_failure = error
            raise

    def drain(self):
        result = super().drain()
        # The upstream save callback logs submission failures rather than
        # propagating them. Retirement must still reject this construction.
        if self.dump_failure is not None:
            raise RuntimeError('Tree KV dump failed; restart the worker') from self.dump_failure
        return result


def configure_sparse(sparse, method, ratio, layers=None):
    selected = resolve_layers(method, layers)
    if not 0 <= ratio <= 1:
        raise ValueError('Invalid ratio')
    if (sparse.request_state.pending is not None or getattr(sparse, 'request', None) is not None
            or getattr(sparse, 'prefill_state', None) is not None
            or sparse.active or sparse.connector.worker_request_id is not None
            or sparse.connector.store.pending):
        raise RuntimeError('Retire the request before changing the scoring policy')
    if method == 'tree-prophetkv' and getattr(sparse, 'tree_f3_config', None) is None:
        raise ValueError('tree-prophetkv needs a tree-enabled construction engine')
    sparse.method, sparse.ratio, sparse.scoring_layers = method, ratio, selected
    return dict(method=method, ratio=ratio, scoring_layers=list(selected))
