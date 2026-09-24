# Interim A800 results

Run from the repository directory on the remote server:

```bash
export RESULT_ROOT=/path/to/results/qwen3-32b-expansion-nonthinking
python3 run_scripts/a800_live_results.py
```

This standard-library utility works while the jobs run. It reads accepted record
JSON and checks its receipt hash and protocol/input identity. It acquires no job
locks, writes no experiment files, starts no processes, and imports no GPU/runtime
packages. Each invocation takes one snapshot and exits.

The script lives outside `run_scripts/a800_expansion/`, so adding it does not
change the runner's frozen source hashes. Keep the benchmark source directory
unchanged while its jobs run. If a branch contains unrelated runner changes,
fetch and restore only this utility instead of pulling those changes mid-run.

```bash
# Explicit result root, or just one job:
python3 run_scripts/a800_live_results.py --root "$RESULT_ROOT"
python3 run_scripts/a800_live_results.py --root "$RESULT_ROOT" --job 0

# Save a snapshot outside the experiment directory:
python3 run_scripts/a800_live_results.py --format csv > /tmp/a800-live.csv
python3 run_scripts/a800_live_results.py --format json > /tmp/a800-live.json
```

The table shows per-task/method sample counts, accuracy, mean/median TTFT,
paired sample counts, paired mean TTFT speedup, and length-limited outputs. It
also reports aggregate RULER metrics. Progress notices go to stderr so CSV/JSON
stdout can be redirected directly.

Only pairs with both baseline and method results contribute to speedup. Accuracy
and latency columns use every accepted result for that task/method, so unfinished
methods may cover different samples. These are provisional comparisons. Snapshot
reads are not globally simultaneous; newly accepted records can appear next time.
Full diagnostics/log validation remains the final aggregation command's job.

The displayed state comes from `progress.json`; it does not prove the process
is alive. With your original launch environment configured, use these existing
commands for process identity checks, current logs, and final aggregation:

```bash
bash run_scripts/a800_expansion/run.sh status
bash run_scripts/a800_expansion/run.sh logs 0
# After both jobs exit:
bash run_scripts/a800_expansion/run.sh aggregate
```

Tests (CPU-only, no third-party dependencies):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s run_scripts -p test_a800_live_results.py -v
```
