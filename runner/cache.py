"""UCM block readiness verification shared by setup and temporary caches."""
import json
import time


def failure_inventory(cache_dir, expected_files):
    """Read file metadata for a failure receipt, without loading or changing KV.

    This is a snapshot during failure handling, not a quiescence or integrity
    audit. Keep it outside the cache so later owned-cache cleanup preserves it.
    """
    from pathlib import Path
    cache_dir = Path(cache_dir)
    files, errors = {}, []
    try:
        for path in (cache_dir/'kv').rglob('*'):
            try:
                if not path.is_file() and not path.is_symlink():
                    continue
                stat = path.lstat()
                files[str(path.relative_to(cache_dir))] = dict(
                    bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    symlink=path.is_symlink())
            except OSError as error:
                errors.append(dict(path=str(path), error=str(error)))
    except OSError as error:
        errors.append(dict(path=str(cache_dir/'kv'), error=str(error)))
    expected, actual = set(expected_files), set(files)
    return dict(cache=str(cache_dir), cache_exists=cache_dir.exists(),
                captured_at_ns=time.time_ns(), quiescence_verified=False,
                expected_files=sorted(expected), actual_files=files,
                extra=sorted(actual-expected), missing=sorted(expected-actual),
                inventory_errors=errors)


def verify_cache(args, token_groups):
    """Check every expected block and TP shard in this UCM 0.3.0 NFS store.

    UCM's scheduler looks up rank 0 only. File sizes are also checked to catch
    incomplete writes. This verifies availability, not tensor correctness.
    """
    config = json.loads((args.model_path / "config.json").read_text())
    kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    expected_bytes = 64 * max(1, kv_heads // args.tensor_parallel_size) * head_dim * 2 * 2 * config["num_hidden_layers"]
    from runner.identity import BlockHasher, block_keys, shard_name
    identity = getattr(args, 'setup_fingerprint', str(args.model_path))
    persistent = hasattr(args, 'setup_fingerprint')
    tp = args.tensor_parallel_size
    hasher = BlockHasher(identity, tp, 0, persistent)
    keys = {key for tokens in token_groups
            for key in block_keys(hasher, tokens, getattr(args, 'hash_seed', 'UCM_HASH_SEED'))}
    missing = []
    for key in keys:
        for rank in range(tp):
            name = shard_name(key, identity, tp, rank, persistent)
            path = args.cache_dir / "kv" / name[:8] / name
            if not path.is_file() or path.stat().st_size != expected_bytes:
                missing.append((rank, name))
    if missing:
        raise RuntimeError(f"Cache incomplete: {len(missing)}/{len(keys) * tp} shards missing or wrong size; examples={missing[:3]}")
    return {"expected_unique_blocks": len(keys), "verified_shards": len(keys) * tp,
            "bytes_per_shard": expected_bytes, "complete": True}



def wait_for_cache(args, token_groups, clock=time.monotonic, sleep=time.sleep):
    """Wait for asynchronous dumps to finish, outside the measured TTFT."""
    started = clock()
    while True:
        try:
            result = verify_cache(args, token_groups)
            result['readiness_wait_seconds'] = clock() - started
            return result
        except RuntimeError as error:
            if not str(error).startswith('Cache incomplete:'):
                raise
            remaining = args.cache_ready_timeout_seconds - (clock() - started)
            if remaining <= 0:
                raise RuntimeError(f'Cache readiness timed out: {error}') from error
            sleep(min(0.25, remaining))
