#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for job_id in 0 1 2 3; do
    bash "$script_dir/job.sh" "$job_id" detach "$@"
done
