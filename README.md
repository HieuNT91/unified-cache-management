# AdaTreeV1.1

Run **no-cache + AdaTree**, Qwen3-32B BF16, LongBench v2 (503 inputs, seeds 0/1)
and RULER64K thinking (13×500 inputs, seed 42; answer seed 0). Chunk 4096 uses the
AdaTreeV1 router; chunk 2048 uses the supplied D60/T5_1/D20 tree. Both methods keep
8192 thinking / 10000 total output tokens and loop breaker 32..128 repeated 4 times.

Use Python 3.10, a compatible CUDA build of `uc-manager==0.3.0`, and
`pip install -r requirements.txt`. Model/tokenizer files must be the original
Qwen3-32B snapshot (`9216db5781bf21249d130ec9da846c4624c16137`). Copy
`config/.env.example` to `config/.env` and edit paths and GPU UUIDs. Edit `TP`, `NUMBER_GPUS`, `MODEL_NAME`, `CHUNK_SIZE` and hardware profile
at the top of `run.sh` (TP2/TP4; UUID count must equal NUMBER_GPUS).

```bash
cp config/.env.example config/.env
# Edit config/.env and run.sh first.
bash run.sh download ruler
bash run.sh download longbench-v2
bash run.sh prepare ruler
bash run.sh prepare longbench-v2
bash run.sh start ruler
# Wait for completion before reusing the same GPUs:
bash run.sh start longbench-v2
```

For existing data, point `RULER_PREPARED`/`LONGBENCH_PREPARED` to the original
prepared folders and skip download/prepare. Inputs stay read-only; both chunk
sizes preserve identical token IDs. Source folders do not need Git metadata;
fresh preparation checks asset hashes and package versions. RULER64K follows
the original 65536 generation bound, not a forced exact 64000-token prompt.

Set `CHUNK_SIZE=2048` and repeat `start` for the second configuration. Each run
is isolated in `results/adatree-v1.1/DATASET/chunkSIZE/`. `resume` runs missing
answers; `stop` stops only that run's verified owned processes. Reusing an
existing run requires the same configuration; set a new RUN_NAME for a new run.

```bash
python status.py                                      # all runs, full results
python status.py results/adatree-v1.1/ruler/chunk4096
python status.py results/adatree-v1.1/ruler/chunk4096 --mode same-input
# Optional TP4 RTX 4500 Ada smoke, exact existing prompt IDs:
bash run.sh smoke longbench-v2 longbench-v2-461
bash run.sh smoke ruler niah_single_1-000
bash run.sh smoke-status ruler
bash run.sh smoke-stop ruler
```

Reports show accuracy, TTFT (seconds), thinking/answer tokens and per-task counts.
`full` uses all committed answers per method; `same-input` uses the intersection
of prompt/seed/token identities. TTFT includes online routing and excludes
independent KV construction; the local smoke profile offloads 4 GiB/rank, so its
latency is not a server benchmark. Full-dataset results require the full runs.
