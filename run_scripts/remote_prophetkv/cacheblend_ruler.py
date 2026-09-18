"""RULER prompt/scoring and UCM 0.3.0 block verification helpers.

Extracted unchanged from benchmarks/prophetkv_full/cacheblend_ruler.py.
No inference entry point or historical scheduler is included.
"""
import hashlib
import json
import pickle
import re
import struct
import time


def cacheblend_prompt(token_ids, tokenizer, end_token_id, payload_size, suffix_size):
    if len(token_ids) <= suffix_size + payload_size:
        raise ValueError("RULER prompt is too short to split into CacheBlend chunks")
    body = token_ids[:-suffix_size]
    suffix = token_ids[-suffix_size:]
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(body):
        # RULER contains long periodic haystacks, so unrelated chunks can have
        # identical token blocks. Numbered text keeps the independent chunk
        # hashes unique without placing an end-of-text token before its content.
        marker = tokenizer.encode(
            f"\n[Context chunk {chunk_index + 1}]\n", add_special_tokens=False
        )
        current_payload_size = payload_size - len(marker)
        if current_payload_size <= 0:
            raise ValueError("chunk payload is too small for its chunk marker")
        payload = body[start : start + current_payload_size]
        chunk = marker + payload
        padded_length = ((len(chunk) + 1 + 63) // 64) * 64
        chunks.append(chunk + [end_token_id] * (padded_length - len(chunk)))
        start += len(payload)
        chunk_index += 1
    blended = [token for chunk in chunks for token in chunk] + suffix
    return chunks, blended



def score_prediction(task, prediction, references):
    prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip().lower()
    if task in {"qa_1", "qa_2"}:
        return max(reference.lower() in prediction for reference in references)
    return sum(reference.lower() in prediction for reference in references) / len(references)



def prompt_digest(tokens):
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()



def verify_cache(args, token_groups):
    """Check every expected block and TP shard in this UCM 0.3.0 NFS store.

    UCM's scheduler looks up rank 0 only. File sizes are also checked to catch
    incomplete writes. This verifies availability, not tensor correctness.
    """
    config = json.loads((args.model_path / "config.json").read_text())
    kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    expected_bytes = 64 * max(1, kv_heads // args.tensor_parallel_size) * head_dim * 2 * 2 * config["num_hidden_layers"]
    metas = [f"{args.model_path}:{args.tensor_parallel_size}:torch.bfloat16:{rank}".encode()
             for rank in range(args.tensor_parallel_size)]
    def hashed(meta, value):
        raw = value if isinstance(value, bytes) else pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(meta + raw).digest()
    keys = set()
    for tokens in token_groups:
        parent = hashed(metas[0], "UCM_HASH_SEED")
        for start in range(0, len(tokens) - 63, 64):
            parent = hashed(metas[0], (parent, tuple(tokens[start:start + 64])))
            keys.add(parent)
    missing = []
    for key in keys:
        for rank, meta in enumerate(metas):
            name = (key if rank == 0 else hashed(meta, key)).hex()
            path = args.cache_dir / "kv" / name[:8] / name
            if not path.is_file() or path.stat().st_size != expected_bytes:
                missing.append((rank, name))
    if missing:
        raise RuntimeError(f"Cache incomplete: {len(missing)}/{len(keys) * len(metas)} shards missing or wrong size; examples={missing[:3]}")
    return {"expected_unique_blocks": len(keys), "verified_shards": len(keys) * len(metas),
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

