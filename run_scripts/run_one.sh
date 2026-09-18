#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo 'Usage: bash run_scripts/run_one.sh {8192|16384|32768|65536} {baseline|prophetkv-5|prophetkv-10|prophetkv-20|prophetkv-30|prophetkv-40|prophetkv-50}' >&2
    exit 2
fi
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/_entry.sh" run --length "$1" --method "$2"
