"""Wait for warmup disk commits before snapshotting or deleting block files."""
from pathlib import Path
from types import SimpleNamespace
from cacheblend_ruler import wait_for_cache
from lifecycle import seed_value

WARMUP_TOKENS = [100, 200, 300, 400, 500, 600, 700, 800] * 16


def wait_for_warmup_cache(model, cache_dir, namespace):
    # Backend Wait() acknowledges the device-to-host copy, not the later
    # FileWorker write/rename. Require both committed full-size blocks first.
    return wait_for_cache(SimpleNamespace(
        model_path=Path(model), tensor_parallel_size=1, cache_dir=Path(cache_dir),
        cache_ready_timeout_seconds=600, hash_seed=seed_value(namespace)),
        [WARMUP_TOKENS])
