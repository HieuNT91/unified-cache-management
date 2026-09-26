"""Persistent store policies; no CUDA imports. Readers cannot write or delete KV."""
from pathlib import Path

from runner.identity import shard_name
from ucm.sparse.prophetkv.lifecycle import TrackedStore


class SetupStore(TrackedStore):
    def __init__(self, store, config, rank=0):
        super().__init__(store)
        self.config = config
        self.rank = rank
        self.cache = Path(config['cache_dir'])

    def _path(self, name):
        return self.cache / 'kv' / name[:8] / name

    def _valid(self, name):
        path = self._path(name)
        return (not path.is_symlink() and not path.parent.is_symlink() and path.is_file()
                and path.stat().st_size == self.config['bytes_per_shard'])

    def lookup(self, keys):
        # Scheduler rank-0 keys are reusable only when EVERY TP shard is committed.
        return [all(self._valid(shard_name(key, self.config['fingerprint'],
                    self.config['tp'], rank, True)) for rank in range(self.config['tp']))
                for key in keys]

    def load_data(self, keys, *args, **kwargs):
        if any(not self._valid(key.hex()) for key in keys):
            raise RuntimeError('Persistent KV is missing or incomplete; rerun setup')
        return super().load_data(keys, *args, **kwargs)

    def dump_data(self, keys, shards, addresses):
        if self.config['readonly']:
            raise RuntimeError('Persistent setup is read-only; rerun setup to construct KV')
        # Complete shards survive a resume, even if another rank is incomplete.
        keep = [i for i, key in enumerate(keys) if not self._valid(key.hex())]
        if not keep:
            return None
        for i in keep:
            path = self._path(keys[i].hex())
            if path.is_symlink() or path.parent.is_symlink():
                raise RuntimeError('Refusing to replace a symlinked KV shard')
            path.unlink(missing_ok=True)
        return super().dump_data([keys[i] for i in keep], [shards[i] for i in keep],
                                 [addresses[i] for i in keep])

    def wait(self, task):
        if task is None:
            return None
        return super().wait(task)

    def __getattr__(self, name):
        # Deny alternate mutation interfaces as well as the connector dump path.
        if name in ('dump', 'delete', 'remove', 'clear', 'cc_store'):
            raise RuntimeError('Persistent setup store does not expose mutation interfaces')
        return super().__getattr__(name)
