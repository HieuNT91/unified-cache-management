# Single-probe feature study and live comparison

This experiment uses all 260 completed RULER prompts for fitting, selection and
scoring. Six measured actions, the original 2,592 settings, depths 1–3 and at most
2 percentage points overall accuracy loss are fixed. No held-out claim or
remote deployment is included. Existing experiments and frozen runtimes stay intact.

The new package is `outputs/ruler13-gpu-feature-study-20260930/`. Its controller
uses a private, hash-pinned `frozen-code/`. Control processes have empty CUDA
visibility; only GPU workers receive the original physical 1–4 UUID group.

```bash
outputs/ruler13-gpu-feature-study-20260930/control.sh status
# After an interrupted attempt and process-identity inspection:
outputs/ruler13-gpu-feature-study-20260930/control.sh resume
```

The controller performs source verification and exact CPU feature replay, 260
all64/1% diagnostic one-token probes, engine shutdown, CPU search and policy
locking, then 520 routed answers with 520 fresh independent probes. It compares
the current six-action fast router and the selected winner for each prompt,
alternating their order. Each phase builds one temporary KV per prompt, primes,
retires every request, and deletes KV before publication. Initialization verifies
native YaRN4, eight committed TP4 warmup shards and the original 64,000-token /
65,920-KV / scheduled-prefill16,384 / greedy non-thinking256 profile.

The 76 additions are the original 40 saved-attention candidates plus 12 head
coverage, 12 query coverage, six head/query entropy and six first-token confidence
statistics. Eight layers are fixed at 7,15,23,31,39,47,55,63. Query indices are
`floor(i*(Q-1)/(min(16,Q)-1))`, including endpoints; all heads on every rank are
included. Native scoring executes unchanged. Separate bounded FP32 kernels use
full-context softmax before eligible-region normalization. They retain neither
logit token identities nor decoded probe text as policy inputs. Scratch exceeding
256 MiB/rank halts. The first scheduled probe checks bounded statistics against
an independent CPU FP32 reference, across all eight layers and four ranks.

The original five features and current policy are reproduced before search.
All 76 singles receive the full grid. The eight strongest singles form 28 pairs;
the best pair receives six triples. The current policy remains a candidate and
the winner receives drop-one refits. Selection uses measured feature costs and
accuracy, with stable ties by fewer additions, smaller tree and setting order.
Registered versioned study policies do not change existing five-feature formats.

Comparable simulation is selected answer TTFT plus traversal, with saved probe
TTFT charged only for dense. The cost-aware estimate adds measured resident CPU
reductions, transfers, synchronization and shared GPU work once per dependency.
Joint GPU head/query collection costs are conservatively charged in full when
any of those features is required; live execution omits unused branches. CPU
subset reductions use two fixed prompts/task, three repetitions, without archive
I/O. Confidence, attention kernels and reductions are timed separately. The
historical 3.65-second oracle is a reference, not a live prediction.

Each live answer consumes an immutable, single-use score handoff bound to input
hash, cache namespace, runtime hash, GPU UUID group and source probe request.
Sparse answers reload the original KV and align all64 layers without another
scoring probe. Dense answers verify zero reused KV, native all64 attention and
zero UCM lookup/load/store operations. Every answer's scores, masks and complete
output token IDs must match the saved fixed action on the same UUID group.
Mismatches halt acceptance and are never retried as operational failures.

Live TTFT begins before arming the fresh probe and ends at the first final token,
including features, transfer, retirement, decision, synchronization and reload.
Construction and ordinary priming are separate. Archive serialization happens
only after measurement. Neither live results nor their timing can trigger refit.

One detached supervisor uses identity checks, launch/run/publication locks, a
900-second GPU progress watchdog and one operational retry. Resume verifies
accepted records and collects only missing units. Busy assigned GPUs prevent a
new GPU phase; unrelated processes are never stopped. Startup receipts are written
once per GPU phase; user-facing status is on request.

Outputs: protocol/source pins, exact CPU replays, native-score archives and compact
GPU statistics, per-subset searches, ranked features, extraction costs,
interpretable policies, simulation decisions/per-task metrics, locked policy
receipt, paired live records, `report.txt`/`.json` and per-task/paired CSVs. Only a
winner meeting the accuracy limit and beating current measured live TTFT is called
a confirmed improvement.

CPU checks: `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest discover -s tests -v` in the pinned environment. Scheduled GPU
gates run in the authorized requests; no separate inference qualification is run.
