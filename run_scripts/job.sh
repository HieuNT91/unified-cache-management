#!/usr/bin/env bash
# Scope, devices and output directories for one of four concurrent TP=2 jobs.
set -euo pipefail
if [[ $# -lt 1 || ! $1 =~ ^[0-3]$ ]]; then
    echo 'Usage: bash run_scripts/job.sh {0|1|2|3} {plan|preflight|prepare|smoke|run|detach|status|logs|stop|report} [options]' >&2
    exit 2
fi
job_id="$1"
shift
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
export RESULT_ROOT="$RESULT_ROOT/job-$job_id"
export CACHE_ROOT="$CACHE_ROOT/job-$job_id"
export REMOTE_JOB_ID="$job_id"
export GPU_INDICES="$((2 * job_id)),$((2 * job_id + 1))"
if [[ $# -eq 0 ]]; then set -- plan; fi
exec bash "$RUN_SCRIPTS_DIR/_entry.sh" "$@"
