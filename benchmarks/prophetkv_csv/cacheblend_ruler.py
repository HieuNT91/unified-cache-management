#!/usr/bin/env python3
"""Run one RULER dataset through vLLM with or without UCM CacheBlend."""

import argparse
import hashlib
import pickle
import re
import struct
import shutil
from importlib.metadata import version
import json
import os
import time
from collections import defaultdict
from pathlib import Path

# Long 64K prefills leave large, differently sized temporary allocations.
# Expandable segments prevent false OOMs when consecutive RULER tasks differ
# slightly in prompt length.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TIMING_SOURCE = "engine_step_first_token_monotonic"
SCHEMA_VERSION = 2


TASK_MAX_TOKENS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("baseline", "full_cache", "cacheblend"), required=True
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--max-model-len", type=int, default=66048)
    parser.add_argument("--rope-factor", type=float)
    parser.add_argument("--chunk-payload-tokens", type=int, default=4095)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--cache-build-batch-size", type=int, default=1)
    parser.add_argument("--cache-settle-seconds", type=float, default=1.0)
    parser.add_argument("--cache-ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--final-cache-settle-seconds", type=float, default=0.0)
    parser.add_argument("--cache-timeout-ms", type=int, default=120000)
    parser.add_argument("--blend-recompute-ratio", type=float, default=0.2)
    parser.add_argument("--populate-only", action="store_true")
    parser.add_argument("--reuse-existing-cache", action="store_true")
    parser.add_argument(
        "--cacheblend-prompt-control",
        action="store_true",
        help=(
            "Apply CacheBlend chunk markers and padding in baseline mode without "
            "enabling the cache connector. This isolates prompt-layout effects."
        ),
    )
    parser.add_argument("--prompt-layout", choices=("chunked", "original"), default="chunked")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tasks", nargs="*", default=list(TASK_MAX_TOKENS))
    return parser.parse_args()


def load_rows(data_root, tasks, limit):
    rows = []
    for task in tasks:
        path = data_root / task / "validation.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        task_rows = [json.loads(line) for line in path.read_text().splitlines()]
        if limit is not None:
            task_rows = task_rows[:limit]
        rows.extend((task, row) for row in task_rows)
    return rows


def build_llm(args):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    kwargs = dict(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=32,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=64,
        enable_prefix_caching=False,
        distributed_executor_backend="mp",
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=True,
        generation_config="vllm",
    )
    if args.rope_factor is not None:
        kwargs["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": args.rope_factor,
            "original_max_position_embeddings": 32768,
        }
    if args.mode != "baseline":
        if args.cache_dir is None:
            raise ValueError("--cache-dir is required for CacheBlend")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="UCMBlendConnector",
            kv_connector_module_path="ucm.integration.vllm.blend_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "ucm_connectors": [
                    {
                        "ucm_connector_name": "UcmNfsStore",
                        "ucm_connector_config": {
                            "storage_backends": str(args.cache_dir),
                            "use_direct": False,
                            "timeout_ms": args.cache_timeout_ms,
                        },
                    }
                ],
                "ucm_sparse_config": {
                    "Blend": {
                        "chunk_end_token_id": (
                            args.chunk_end_token_id if args.mode == "cacheblend" else -1
                        ),
                        "compute_meta": {
                            "model.layers.1.self_attn.attn": {
                                "ratio": args.blend_recompute_ratio
                            }
                        },
                    }
                },
            },
        )
    return LLM(**kwargs)


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


def generate_timed(engine, prompt, sampling, request_id, clock=time.perf_counter):
    """Client-observed TTFT, including submission, scheduling and output delivery.

    The sampling output kind must be CUMULATIVE, never FINAL_ONLY. An empty
    decoded text still counts if the engine delivered a token (e.g. EOS).
    """
    if engine.has_unfinished_requests():
        raise RuntimeError("Timing requires an idle engine")
    started = clock()
    engine.add_request(request_id, prompt, sampling)
    first = None
    result = None
    while engine.has_unfinished_requests():
        outputs = engine.step()
        observed = clock()
        for output in outputs:
            if output.request_id != request_id:
                raise RuntimeError("Unexpected request during timing")
            if output.outputs and output.outputs[0].token_ids and first is None:
                first = observed - started
            if output.finished:
                result = output
    elapsed = clock() - started
    if first is None or result is None:
        raise RuntimeError("Engine did not deliver a first token and final output")
    return result, first, elapsed


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


def main():
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.inputs import TokensPrompt

    args = parse_args()
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be positive")
    if not 0 < args.blend_recompute_ratio <= 1:
        raise ValueError("--blend-recompute-ratio must be in (0, 1]")
    if (args.populate_only or args.reuse_existing_cache) and args.mode == "baseline":
        raise ValueError("cache control flags cannot be used with --mode baseline")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.cacheblend_prompt_control:
        args.prompt_layout = "chunked"
    if args.mode == "cacheblend" and args.prompt_layout != "chunked":
        raise ValueError("CacheBlend requires --prompt-layout chunked")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data_root, args.tasks, args.limit)
    if not rows:
        raise ValueError("No samples selected")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    args.chunk_end_token_id = tokenizer.pad_token_id
    if args.chunk_end_token_id is None:
        args.chunk_end_token_id = tokenizer.unk_token_id
    if args.chunk_end_token_id is None:
        raise ValueError("Tokenizer has neither a pad token nor an unknown token")

    llm = build_llm(args)
    warmup_ids = tokenizer.encode("Benchmark warmup: count to three. " * 32, add_special_tokens=False)[:128]
    llm.generate(TokensPrompt(prompt_token_ids=warmup_ids), SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)
    prompt_manifest = []
    records = []
    scores = defaultdict(list)
    timings = defaultdict(list)
    ttfts = defaultdict(list)
    cache_builds = defaultdict(list)
    started = time.perf_counter()
    with args.output.open("w", buffering=1) as output_file:
        for ordinal, (task, row) in enumerate(rows, start=1):
            prompt = row["input"] + row.get("answer_prefix", "")
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            sampling = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=TASK_MAX_TOKENS[task],
            )
            record = {
                "model": args.model_name,
                "mode": args.mode,
                "task": task,
                "sample_index": row["index"],
                "input_tokens": len(token_ids),
                "references": row["outputs"],
            }
            if args.mode == "cacheblend":
                record["blend_recompute_ratio"] = args.blend_recompute_ratio

            chunks = []
            request_ids = token_ids
            if args.prompt_layout == "chunked":
                chunks, request_ids = cacheblend_prompt(
                    token_ids, tokenizer, args.chunk_end_token_id,
                    args.chunk_payload_tokens, args.suffix_tokens,
                )
            if len(request_ids) + TASK_MAX_TOKENS[task] > args.max_model_len:
                raise ValueError("Prompt plus output budget exceeds --max-model-len")
            record.update(
                schema_version=SCHEMA_VERSION, prompt_layout=args.prompt_layout,
                prompt_sha256=prompt_digest(request_ids), request_tokens=len(request_ids),
                cache_chunks=len(chunks), max_output_tokens=TASK_MAX_TOKENS[task],
                timing_source=TIMING_SOURCE, cache_build_seconds=0.0,
                prompt_format="raw_completion", thinking_budget="task_output_limit",
            )
            prompt_manifest.append((task, row["index"], record["prompt_sha256"], TASK_MAX_TOKENS[task]))
            cache_groups = chunks if args.mode == "cacheblend" else [request_ids]
            if args.mode != "baseline" and not args.reuse_existing_cache:
                config = json.loads((args.model_path / "config.json").read_text())
                heads = config.get("num_key_value_heads", config["num_attention_heads"])
                dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
                estimate = sum(len(group) for group in cache_groups) * heads * dim * 4 * config["num_hidden_layers"]
                if shutil.disk_usage(args.cache_dir).free < estimate + 2 * 1024**3:
                    raise RuntimeError("Insufficient free space for this sample's KV cache")
                cache_started = time.perf_counter()
                for start in range(0, len(cache_groups), args.cache_build_batch_size):
                    batch = cache_groups[start:start + args.cache_build_batch_size]
                    llm.generate(
                        [TokensPrompt(prompt_token_ids=chunk) for chunk in batch],
                        SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False,
                    )
                record["cache_build_seconds"] = time.perf_counter() - cache_started

            if args.mode != "baseline" and args.cache_settle_seconds:
                time.sleep(args.cache_settle_seconds)

            if args.mode != "baseline":
                record["cache_verification"] = wait_for_cache(args, cache_groups)

            if args.populate_only:
                record["populated"] = True
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                cache_builds[task].append(record["cache_build_seconds"])
                print(
                    f"[{ordinal}/{len(rows)}] {args.model_name} {args.mode} population "
                    f"{task}: time={record['cache_build_seconds']:.3f}s",
                    flush=True,
                )
                continue

            for repeat in range(args.repeats):
                sampling.output_kind = RequestOutputKind.CUMULATIVE
                result, ttft, elapsed = generate_timed(
                    llm.llm_engine, TokensPrompt(prompt_token_ids=request_ids), sampling,
                    f"measured-{ordinal}-{repeat}",
                )
                measured = dict(record)
                measured.update(
                    repeat=repeat, generation_seconds=elapsed, ttft_seconds=ttft,
                    prediction=result.outputs[0].text,
                    output_tokens=len(result.outputs[0].token_ids),
                    finish_reason=result.outputs[0].finish_reason,
                    output_token_ids=list(result.outputs[0].token_ids),
                    num_cached_tokens=getattr(result, "num_cached_tokens", None),
                )
                measured["score"] = score_prediction(task, measured["prediction"], row["outputs"])
                records.append(measured)
                output_file.write(json.dumps(measured, ensure_ascii=False) + "\n")
                scores[task].append(measured["score"])
                timings[task].append(elapsed)
                ttfts[task].append(ttft)
                cache_builds[task].append(record["cache_build_seconds"])
                print(
                    f"[{ordinal}/{len(rows)} repeat={repeat}] {args.model_name} {args.mode} {task}: "
                    f"score={measured['score']:.3f}, ttft={ttft:.3f}s, total={elapsed:.3f}s, "
                    f"tokens={measured['output_tokens']}, finish={measured['finish_reason']}", flush=True,
                )

    if args.populate_only:
        if args.final_cache_settle_seconds:
            print(
                f"waiting {args.final_cache_settle_seconds:.0f}s for asynchronous cache writes",
                flush=True,
            )
            time.sleep(args.final_cache_settle_seconds)
        summary = {
            "model": args.model_name,
            "mode": f"{args.mode}_population",
            "samples": len(rows),
            "sequence_length": 65536,
            "mean_cache_build_seconds": sum(map(sum, cache_builds.values())) / len(rows),
            "wall_seconds": time.perf_counter() - started,
            "task_mean_cache_build_seconds": {
                task: sum(values) / len(values) for task, values in cache_builds.items()
            },
        }
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return

    summary = {
        "model": args.model_name,
        "mode": args.mode,
        "schema_version": SCHEMA_VERSION,
        "timing_source": TIMING_SOURCE,
        "runtime": {
            "model_path": str(args.model_path), "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "vllm": version("vllm"), "torch": version("torch"),
            "max_model_len": args.max_model_len, "rope_factor": args.rope_factor,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "prompt_layout": args.prompt_layout,
        "prompt_manifest": prompt_manifest,
        "repeats": args.repeats,
        "measurements": len(records),
        "warmup": {"input_tokens": len(warmup_ids), "max_output_tokens": 4},
        "truncated_measurements": sum(r["finish_reason"] == "length" for r in records),
        "samples": len(rows),
        "sequence_length": 65536,
        "score": 100 * sum(map(sum, scores.values())) / len(records),
        "ttft_seconds": sum(map(sum, ttfts.values())),
        "mean_ttft_seconds": sum(map(sum, ttfts.values())) / len(records),
        "generation_seconds": sum(map(sum, timings.values())),
        "wall_seconds": time.perf_counter() - started,
        "mean_cache_build_seconds": (
            sum(map(sum, cache_builds.values())) / len(records) if cache_builds else 0.0
        ),
        "task_scores": {
            task: 100 * sum(values) / len(values) for task, values in scores.items()
        },
        "task_generation_seconds": {
            task: sum(values) for task, values in timings.items()
        },
        "task_mean_ttft_seconds": {
            task: sum(values) / len(values) for task, values in ttfts.items()
        },
        "task_mean_cache_build_seconds": {
            task: sum(values) / len(values) for task, values in cache_builds.items()
        },
    }
    if args.mode == "cacheblend":
        summary["blend_recompute_ratio"] = args.blend_recompute_ratio
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
