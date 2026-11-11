"""Temporary independent chunk KV cache shared by AdaTree answer methods."""
from pathlib import Path
import time
from types import SimpleNamespace

class PromptCache:
    """One owned namespace; verify committed TP shards before reusing or deleting."""
    def __init__(self, cache, model, tp, namespace, sample):
        from runner.inputs.identity import BlockHasher, block_keys, shard_name
        from ucm.sparse.prophetkv.lifecycle import seed_value
        from runner.inputs.layout import validate_layout
        validate_layout(sample['token_ids'], sample['boundaries'], sample['question_positions'], sparse=True)
        self.cache, self.tp = Path(cache), tp
        self.chunks = list(dict.fromkeys(tuple(sample['token_ids'][a:b]) for a,b in
                           zip(sample['boundaries'][:-2], sample['boundaries'][1:-1])))
        self.args = SimpleNamespace(model_path=model, tensor_parallel_size=tp, cache_dir=self.cache,
            cache_ready_timeout_seconds=600, hash_seed=seed_value(namespace))
        hasher = BlockHasher(str(model), tp, 0)
        keys = {key for chunk in self.chunks for key in block_keys(hasher, chunk, self.args.hash_seed)}
        names = [shard_name(key, str(model), tp, rank) for key in keys for rank in range(tp)]
        self.files = {f'kv/{name[:8]}/{name}' for name in names}
        self.before = None

    def ready(self):
        from runner.cache.lifecycle import wait_for_cache
        started = time.monotonic()
        self.before = None
        result = wait_for_cache(self.args, self.chunks)
        # PcStore 0.3.0 completes dump tasks after device-to-host transfer;
        # the file queue may still be writing/renaming repeated prefix blocks.
        # Wait only for staging files belonging to this prompt, never ignore
        # or remove them, and take the immutable snapshot after publication.
        staging = {f'kv/.temp/{Path(name).name}' for name in self.files}
        while True:
            paths = list((self.cache/'kv').rglob('*'))
            if any(p.is_symlink() for p in paths):
                raise RuntimeError('Symlink in temporary cache')
            actual = {str(p.relative_to(self.cache)) for p in paths if p.is_file()}
            extra, missing = sorted(actual-self.files), sorted(self.files-actual)
            if not extra and not missing:
                # Recheck sizes after the last rename, including replacements.
                from runner.cache.lifecycle import verify_cache
                result.update(verify_cache(self.args, self.chunks))
                self.before = self.snapshot()
                result['readiness_wait_seconds'] = time.monotonic() - started
                return result
            detail = (
                f'cache={self.cache}, expected={len(self.files)}, actual={len(actual)}, '
                f'extra={len(extra)}, missing={len(missing)}, '
                f'extra_examples={extra[:10]}, missing_examples={missing[:10]}')
            if missing or not set(extra).issubset(staging):
                raise RuntimeError('Unexpected files in temporary prompt cache: ' + detail)
            remaining = self.args.cache_ready_timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError('Cache staging did not drain before readiness timeout: ' + detail)
            time.sleep(min(0.25, remaining))

    def snapshot(self):
        result = {}
        for name in self.files:
            path = self.cache/name
            if path.is_symlink() or path.parent.is_symlink():
                raise RuntimeError('Symlink in temporary cache')
            stat = path.stat()
            result[name] = [stat.st_size, stat.st_mtime_ns]
        return result

    def unchanged(self):
        if self.before is None:
            raise RuntimeError('Original KV changed during inference: readiness snapshot missing')
        after = self.snapshot()
        if self.before != after:
            changed = sorted(name for name in self.before if self.before[name] != after[name])
            examples = {name: dict(before=self.before[name], after=after[name]) for name in changed[:10]}
            raise RuntimeError('Original KV changed during inference: '
                               f'cache={self.cache}, changed_shards={len(changed)}, '
                               f'size_mtime_ns={examples}')

    def delete(self):
        from ucm.sparse.prophetkv.lifecycle import delete_retired_files
        self.unchanged()
        delete_retired_files(self.cache, self.files)
        return dict(deleted_shards=len(self.files), deleted_bytes=sum(s[0] for s in self.before.values()),
                    remaining_shards=0)
