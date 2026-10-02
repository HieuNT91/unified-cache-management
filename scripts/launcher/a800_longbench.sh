#!/usr/bin/env bash
# Remote-only commands. This script never selects local GPUs by numeric index.
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
export MODEL_PATH="${MODEL_PATH:-/mnt/sde/jh/ckpts/Qwen3-32B}"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/sde/jh/envs/ucm/bin/python}"
export LONGBENCH_DATA="${LONGBENCH_DATA:-/mnt/sde/jh/projects/unified-cache-management/.data/LongBench-v2/data.json}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$CODE_ROOT/outputs/longbench-v2-503-yarn4-thinking16k}"
export PREPARED_DIR="${PREPARED_DIR:-$EXPERIMENT_DIR/prepared}"
export CACHE_ROOT="${CACHE_ROOT:-$CODE_ROOT/.cache/longbench-temporary}"
GPU_A="${GPU_A:-GPU-6f2a33e5-aa6e-6681-3d49-1de956c38b3e,GPU-87070a52-3745-1eae-7b16-69594603f277,GPU-73ecc591-94c9-c261-412c-5e0cf51fa103,GPU-5224ff6c-63bb-3a1e-249c-15de01751b5d}"
GPU_B="${GPU_B:-GPU-1d441d83-c33c-4797-23be-b7355d92e3b6,GPU-83ea256b-cf28-6476-ce48-f34e04955379,GPU-b6ece736-772f-8ff9-e3a6-f4a0b369e7b4,GPU-9eed921f-1ef7-c923-f4ec-acf7181237a9}"
prepare() {
    CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/longbench_v2.py" prepare \
        --model "$MODEL_PATH" --data "$LONGBENCH_DATA" --output "$PREPARED_DIR"
}
worker() {
    local shard="$1" devices="$2"; shift 2
    cd "$CODE_ROOT"
    CUDA_VISIBLE_DEVICES="$devices" bash "$CODE_ROOT/run.sh" sweep \
        --model "$MODEL_PATH" --manifest "$PREPARED_DIR/manifest.jsonl" \
        --output "$EXPERIMENT_DIR" --cache-root "$CACHE_ROOT" --tp 4 \
        --shard "$shard" --shards 2 "$@"
}
report() {
    CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/sweep_report.py" \
        --output "$EXPERIMENT_DIR" --manifest "$PREPARED_DIR/manifest.jsonl" \
        --percentages 1 5 10 15 20 30 --shards 2 "$@"
}
case "${1:-}" in
    prepare) prepare ;;
    run|resume)
        mkdir -p "$EXPERIMENT_DIR"
        exec 9>"$EXPERIMENT_DIR/sweep.lock"
        flock -n 9 || { echo 'This sweep already has a live coordinator.' >&2; exit 1; }
        export UCM_SWEEP_COORDINATOR_PID=$$
        # Refuse partial/completed sweeps before acquiring GPUs.
        cd "$CODE_ROOT"
        resume_flags=()
        resume_device_group=""
        if [[ "$1" == resume ]]; then
            prepare_flags=()
            if [[ -n "${RESUME_EXCLUDE_PERCENTAGES:-}" ]]; then
                read -r -a excluded_percentages <<< "$RESUME_EXCLUDE_PERCENTAGES"
                prepare_flags+=(--exclude-percentages "${excluded_percentages[@]}")
            fi
            case "${RESUME_SINGLE_GROUP:-0}" in
                1) prepare_flags+=(--single-group-gpus "$GPU_A") ;;
                0) ;;
                *) echo 'RESUME_SINGLE_GROUP must be 0 or 1.' >&2; exit 2 ;;
            esac
            CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m runner.resume \
                --model "$MODEL_PATH" --manifest "$PREPARED_DIR/manifest.jsonl" \
                --output "$EXPERIMENT_DIR" --cache-root "$CACHE_ROOT" --tp 4 --shards 2 \
                --validation "${RESUME_VALIDATION:-fast}" "${prepare_flags[@]}"
            resume_flags=(--resume --skip-selective)
            resume_device_group="$(CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -c \
                'import json,sys; s=json.load(open(sys.argv[1])); print(",".join(s["gpu_uuids"][0]) if s.get("execution_mode")=="sequential" else "")' \
                "$EXPERIMENT_DIR/continuation.json")"
        else
            [[ "${RESUME_SINGLE_GROUP:-0}" == 0 && -z "${RESUME_EXCLUDE_PERCENTAGES:-}" ]] || {
                echo 'Scope/device changes apply to resume, not a new run.' >&2; exit 2;
            }
            for name in group-0 group-1 baseline prophetkv-{1,5,10,15,20,30} selective-{1,5,10,15,20,30}; do
                [[ ! -e "$EXPERIMENT_DIR/$name" ]] || { echo "Existing result: $name; use a new EXPERIMENT_DIR." >&2; exit 1; }
            done
            prepare
        fi
        physical_groups=2
        [[ -z "$resume_device_group" ]] || physical_groups=1
        CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/longbench_v2.py" disk \
            --prepared "$PREPARED_DIR" --cache-root "$CACHE_ROOT" --groups "$physical_groups"
        log_suffix=""
        [[ "$1" != resume ]] || log_suffix="-resume-$(date +%Y%m%d-%H%M%S)"
        if [[ -n "$resume_device_group" ]]; then
            for shard in 0 1; do
                worker "$shard" "$resume_device_group" "${resume_flags[@]}" \
                    >"$EXPERIMENT_DIR/group-${shard}${log_suffix}.log" 2>&1 & pid=$!
                echo "Sequential group $shard PID=$pid; devices=$resume_device_group"
                wait "$pid" || { report; echo "Group $shard failed; validated records retained." >&2; exit 1; }
            done
        else
            worker 0 "$GPU_A" "${resume_flags[@]}" >"$EXPERIMENT_DIR/group-a${log_suffix}.log" 2>&1 & pid_a=$!
            worker 1 "$GPU_B" "${resume_flags[@]}" >"$EXPERIMENT_DIR/group-b${log_suffix}.log" 2>&1 & pid_b=$!
            echo "Group A PID=$pid_a; Group B PID=$pid_b"
            status_a=0; status_b=0
            wait "$pid_a" || status_a=$?
            wait "$pid_b" || status_b=$?
            [[ "$status_a" == 0 && "$status_b" == 0 ]] || { echo "Sweep failed: A=$status_a B=$status_b" >&2; exit 1; }
        fi
        report --final
        ;;
    counts)
        CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/sweep_counts.py" \
            --output "$EXPERIMENT_DIR" --manifest "$PREPARED_DIR/manifest.jsonl" \
            --percentages 1 5 10 15 20 30 --shards 2
        ;;
    status) report ;;
    status_same_count) report --same-count ;;
    aggregate) report --final ;;
    stop) "$PYTHON_BIN" "$CODE_ROOT/scripts/sweep_control.py" stop --output "$EXPERIMENT_DIR" ;;
    *) echo "Usage: bash scripts/launcher/a800_longbench.sh {prepare|run|stop|resume|counts|status|status_same_count|aggregate}" >&2; exit 2 ;;
esac
