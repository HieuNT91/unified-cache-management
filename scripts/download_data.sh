#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
dataset="${1:?Usage: scripts/download_data.sh ruler|longbench-v2}"
[[ $# == 1 ]] || { echo 'Expected a dataset only.' >&2; exit 2; }
case "$dataset" in
  ruler) source_path="${RULER_PATH:?Set RULER_PATH in config/.env}" ;;
  longbench-v2) source_path="${LONGBENCH_DATA:?Set LONGBENCH_DATA in config/.env}" ;;
  *) echo 'Dataset must be ruler or longbench-v2.' >&2; exit 2 ;;
esac
python_command=("${PYTHON_BIN:?Set PYTHON_BIN in config/.env}")
[[ -z "${PYTHON_WRAPPER:-}" ]] || python_command=("$PYTHON_WRAPPER" "${python_command[@]}")
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Each command below identifies its source and local destination for manual use.
# Existing files are retained; the final Python command checks original hashes.

# Publish only complete downloads. Never overwrite an existing destination.
download_file() {
  local url="$1" destination="$2" temporary
  [[ ! -e "$destination" ]] || return 0
  mkdir -p -- "$(dirname -- "$destination")"
  temporary="$(mktemp "${destination}.download.XXXXXX")"
  if curl --fail --location "$url" --output "$temporary" &&
    "${python_command[@]}" - "$temporary" "$destination" "$dataset" <<'CHECK'
import sys
from pathlib import Path
from runner.cache.setups import file_hash
from scripts.run import LONGBENCH_SHA256, RULER_ASSETS
path, destination = map(Path, sys.argv[1:3])
expected = LONGBENCH_SHA256 if sys.argv[3] == 'longbench-v2' else RULER_ASSETS[destination.name]
if file_hash(path) != expected:
    raise SystemExit(f'Download differs from the original source: {destination.name}')
CHECK
  then
    ln -- "$temporary" "$destination" || { rm -f -- "$temporary"; return 1; }
    rm -f -- "$temporary"
  else
    rm -f -- "$temporary"
    return 1
  fi
}

case "$dataset" in
  longbench-v2)
    download_file \
      'https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json' \
      "$source_path"
    ;;
  ruler)
    # RULER source. LFS word list is resolved and hash-checked by the final command.
    if [[ ! -d "$source_path" ]]; then
      GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/RULER.git "$source_path"
      git -C "$source_path" checkout --detach c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
    fi
    asset_dir="$source_path/scripts/data/synthetic/json"

    # english_words.json (Git LFS):
    # https://media.githubusercontent.com/media/NVIDIA/RULER/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/json/english_words.json
    # The original downloader handles its LFS pointer safely at the end.

    download_file \
      'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json' \
      "$asset_dir/squad.json"

    download_file \
      'https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json' \
      "$asset_dir/hotpotqa.json" || \
    download_file \
      'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json' \
      "$asset_dir/hotpotqa.json" || \
    download_file \
      'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json' \
      "$asset_dir/hotpotqa.json"

    # Paul Graham essays: URLs are listed in this upstream script.
    # For restricted networks, copy the original PaulGrahamEssays.json here.
    if [[ ! -f "$asset_dir/PaulGrahamEssays.json" ]]; then
      (cd "$asset_dir" && "${python_command[@]}" download_paulgraham_essay.py > download-essays.log 2>&1)
    fi

    # NLTK punkt / punkt_tab: the final command fetches only missing resources.
    # Manual installation: PYTHON_BIN -m nltk.downloader punkt punkt_tab
    # Offline: unpack tokenizers/punkt and tokenizers/punkt_tab into ~/nltk_data.
    ;;
esac

# Preserve original validation, LFS resolution, NLTK setup and asset receipt.
exec "${python_command[@]}" "$CODE_ROOT/scripts/run.py" download "$dataset" --source "$source_path"
