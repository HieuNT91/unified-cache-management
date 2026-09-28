#!/usr/bin/env bash
# Remote L20 commands; all device selection is pinned to user-supplied UUIDs.
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE_ROOT/scripts/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
export MODEL_PATH="${MODEL_PATH:-/data/jh/ckpts/Qwen3-32B}"
export PYTHON_BIN="${PYTHON_BIN:-/data/jh/envs/ucm/bin/python}"
export RULER_SOURCE="${RULER_SOURCE:-$CODE_ROOT/.cache/vendor/RULER}"
export PREPARED_DIR="${PREPARED_DIR:-$CODE_ROOT/.cache/ruler-64000-thinking-prepared}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$CODE_ROOT/outputs/ruler-64000-thinking16k-l20}"
export CACHE_ROOT="${CACHE_ROOT:-$CODE_ROOT/.cache/ruler-temporary}"
GPU_A="${GPU_A:-GPU-9af1e932-db0c-d64a-e1d4-e86854f6168b,GPU-9487c377-cb87-c9cf-bdfd-7e9e0c5e93cf,GPU-568bf4a9-91fe-67c9-c454-c3f39345b17b,GPU-63808146-028b-5fb1-2e0c-0f4361156677}"
GPU_B="${GPU_B:-GPU-21292522-6b1d-d6fd-517d-4f5ad92c228f,GPU-c7cf4603-1994-5a84-cba8-a9005bc62dc9,GPU-1ef3727d-cbd0-0c94-eb49-7d58c12344d9,GPU-34e0b17f-40d7-6140-6f41-b92ab321b9f3}"
percentages=(1 5 10 15 20 30 40 60 80)
prepare() {
    CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/ruler_64000.py" prepare \
        --ruler "$RULER_SOURCE" --model "$MODEL_PATH" --output "$PREPARED_DIR" --samples 100
}
worker() {
    local shard="$1" devices="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$devices" bash "$CODE_ROOT/run.sh" sweep \
        --model "$MODEL_PATH" --manifest "$PREPARED_DIR/manifest.jsonl" \
        --output "$EXPERIMENT_DIR" --cache-root "$CACHE_ROOT" --tp 4 \
        --shard "$shard" --shards 2 --percentages "${percentages[@]}" \
        --layers 11 12 13 14 15 --context-length 64000 --exact-input-tokens 64000 "$@"
}
report() {
    CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/sweep_report.py" \
        --output "$EXPERIMENT_DIR" --manifest "$PREPARED_DIR/manifest.jsonl" \
        --percentages "${percentages[@]}" --shards 2 "$@"
}
case "${1:-}" in
    download)
        CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/ruler_64000.py" download --ruler "$RULER_SOURCE"
        ;;
    prepare) prepare ;;
    dry-run)
        worker 0 '' --dry-run
        worker 1 '' --dry-run
        ;;
    run|resume)
        mkdir -p "$EXPERIMENT_DIR"
        exec 9>"$EXPERIMENT_DIR/sweep.lock"
        flock -n 9 || { echo 'A coordinator already owns this sweep.' >&2; exit 1; }
        export UCM_SWEEP_COORDINATOR_PID=$$
        cd "$CODE_ROOT"
        resume_flags=()
        if [[ "$1" == resume ]]; then
            CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m runner.resume \
                --model "$MODEL_PATH" --manifest "$PREPARED_DIR/manifest.jsonl" \
                --output "$EXPERIMENT_DIR" --cache-root "$CACHE_ROOT" --tp 4 --shards 2 --percentages "${percentages[@]}" --context-length 64000 --exact-input-tokens 64000
            resume_flags=(--resume --skip-selective)
        else
            for name in group-0 group-1 baseline prophetkv-{1,5,10,15,20,30,40,60,80} selective-{1,5,10,15,20,30,40,60,80}; do
                [[ ! -e "$EXPERIMENT_DIR/$name" ]] || { echo "Existing result: $name; use a new EXPERIMENT_DIR." >&2; exit 1; }
            done
            prepare
        fi
        CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/longbench_v2.py" disk \
            --prepared "$PREPARED_DIR" --cache-root "$CACHE_ROOT" --groups 2
        log_suffix=""
        [[ "$1" != resume ]] || log_suffix="-resume-$(date +%Y%m%d-%H%M%S)"
        worker 0 "$GPU_A" "${resume_flags[@]}" >"$EXPERIMENT_DIR/group-a${log_suffix}.log" 2>&1 & pid_a=$!
        worker 1 "$GPU_B" "${resume_flags[@]}" >"$EXPERIMENT_DIR/group-b${log_suffix}.log" 2>&1 & pid_b=$!
        echo "Group A PID=$pid_a; Group B PID=$pid_b"
        status_a=0; status_b=0
        wait "$pid_a" || status_a=$?
        wait "$pid_b" || status_b=$?
        if [[ "$status_a" != 0 || "$status_b" != 0 ]]; then
            report
            echo "Sweep failed: A=$status_a B=$status_b; validated records retained." >&2
            exit 1
        fi
        report --final
        ;;
    status) report ;;
    aggregate) report --final ;;
    stop) "$PYTHON_BIN" "$CODE_ROOT/scripts/sweep_control.py" stop --output "$EXPERIMENT_DIR" ;;
    *) echo "Usage: bash scripts/l20_ruler.sh {download|prepare|dry-run|run|stop|resume|status|aggregate}" >&2; exit 2 ;;
esac
