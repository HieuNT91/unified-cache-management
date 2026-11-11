# AdaTreeV1.1

Run **no-cache + AdaTree** with Qwen3-32B BF16.

## 1. Create the environment

Follow [docs/create_env.md](docs/create_env.md).

## 2. Download and prepare data

Follow [docs/download_and_prepare_data.md](docs/download_and_prepare_data.md).

## 3. Run

Set paths and GPU UUIDs in `config/.env`. Edit `TP`, `NUMBER_GPUS`, `CHUNK_SIZE`
and `GPU_MEMORY_UTILIZATION` at the top of `run.sh`. TP supports 1/2/4; the UUID
count must equal `NUMBER_GPUS`. For H20 TP1, use `TP=1`, `NUMBER_GPUS=1` and one
GPU UUID. No CPU offload is used; vLLM checks available KV capacity.

### LongBenchV2

```bash
bash run.sh start longbench-v2
```

Defaults: 503 inputs and answer seeds `(0 1)`. Set `LONGBENCH_ANSWER_SEEDS` in
`run.sh` to `(0)` or any list of distinct nonnegative integers. Runtime reserves
114688 input + 10000 output tokens. Results are saved under
`results/adatree-v1.1/longbench-v2/chunk4096-seeds-0-1/` with the selected chunk
size and seeds in the folder name.

### RULER64K

```bash
bash run.sh start ruler
```

Defaults: 13 tasks × 500 inputs, preparation seed 42 and answer seed 0.
`RULER_SAMPLES_PER_TASK` and `PREPARATION_SEED` in `run.sh` must match preparation.
RULER follows the original 65536 generation bound; prompts are not forced to
exactly 64000 tokens. Results are saved under
`results/adatree-v1.1/ruler/chunk4096/` with the selected chunk size.

Wait for a run to finish before reusing its GPUs. Chunk 4096 uses the AdaTreeV1
router; set `CHUNK_SIZE=2048` and run again to use the supplied D60/T5_1/D20 tree.
Use `bash run.sh resume DATASET` for missing answers or `bash run.sh stop DATASET`
to stop a run. Keep the same configuration when resuming; change `RUN_NAME` to
start a separate run.

```bash
python status.py                                      # all runs, full results
python status.py results/adatree-v1.1/ruler/chunk4096
python status.py results/adatree-v1.1/ruler/chunk4096 --mode same-input
```

Reports include accuracy, TTFT, thinking/answer tokens and per-task counts.
The default `full` mode uses all committed answers; `same-input` compares the
intersection of prompt/seed/token identities. TTFT includes online routing and
excludes independent KV construction.
