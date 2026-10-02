# Archived agent and experiment notes

Snapshot of the previous root `AGENTS.md`, taken on 2026-10-02 before the
documentation reorganization. The material below is historical evidence, not
instructions governing this worktree. Dates, PIDs, “current”, “pending” and
“active” refer to the time each entry was written. Verify receipts and process
identity before any separately authorized intervention.

Current rules are in [AGENTS.md](../../AGENTS.md), with
[experiment protection](../agents/experiment-rules.md) and current
[RULER](../protocols/ruler.md) / [LongBench v2](../protocols/longbench-v2.md)
protocols. Old delimiter, padded-length and chunk-size instructions do not apply
to new `rpkv` inputs. No historical scheduler should be restarted from these notes.

---

# Agent instructions

## Fixed controls without independent probes (2026-10-01)

- User requested removing independent diagnostic probes from the current
  implementation ONLY. Explicitly leave the active expanded experiment running
  with its existing frozen code, protocols and540-probe scope. Do not stop,
  restart, migrate or edit that package to apply this source change.
- New `scripts/rpkv_gpu_validation.py` runs require protocol field
  `answer_validation: native-answer-diagnostics-v1`. They run only the fixed
  answers and finalize with zero independent probes. Use fresh result paths
  and expected_probes=0 in new experiment manifests; do not relabel old results.
- Retain native per-answer score/mask replay, all-layer/rank checks, readiness,
  immutable cache and retirement. Cache construction, warmup and priming remain.
  Router probes needed for online decisions and historical collection readers
  keep their original behavior. Old protocol readers retain their probe gates.
  CPU regression passed240 tests, including six no-probe execution/acceptance/
  reporting tests. This source change has not been run on GPU; the active frozen
  experiment remains unchanged (179 frozen file hashes and process IDs verified).

## Expanded rpkv controls (2026-10-01)

- User authorized 40 samples per RULER task and20 LongBench v2 inputs with
  the same no-cache / ProphetKV1% /5% methods and physical GPUs1–4 by UUID.
  New package: `outputs/rpkv-ruler40-longbench20-20261001/`. Preserve the
  completed five-sample controls and their frozen code/results unchanged.
- Scope:520 RULER +20 LongBench inputs,1620 fresh answers and540 independent
  one-token validation probes. Seed42; one40-row original batch per RULER task.
  LongBench uses the previously approved full-text fitting pool and native16K
  output reserve. No truncation, router fitting or historical scheduler restart.
- Package code and cohort are frozen before launch. `sequence.json` tracks
  RULER -> LongBench -> independent final audit. Never duplicate supervisors.
  Verify startup once, then leave the detached run running; status on request.
  New final reports include TTFT, accuracy, thinking/answer token lengths and
  per-task metrics. Completion requires all1620 answers,540 probes and exits.
  Launched Oct1 at18:41 HKT; supervisor PID3077453, sequence PID3077458.
  Verify live identities before any intervention. Both are PPID1, independent
  sessions, SIGHUP-ignored, /dev/null stdin and CPU-only. First scheduled input
  niah_single_1-000 passed independent all-rank probe/1%/5% replay, eight warmup
  shards, immutable cache and retirement; both answers scored1 with thinking0.
  See results/ruler/startup-verification.json. Startup verification is complete.
  Read-only status: `python outputs/rpkv-ruler40-longbench20-20261001/status.py`.

## rpkv current scope (2026-10-01; supersedes historical protocols below)

- Current worktree: `.worktrees/rpkv`, branch `rpkv`. See [current overview](../../README.md) and [control studies](rpkv-controls.md).
- Original tokens only; no chunk markers/EOT padding. Metadata phase/layout and
  request/token identity are mandatory; never recover boundaries from delimiters.
- RULER uses the pinned original generators: 13 task batches, 500/task, seed42,
  total window65536 including task caps128/30/120/50/32. Native Qwen3 non-thinking
  is the documented adaptation. Score only newly generated text with upstream rules.
- Preserve old worktrees, artifacts, jobs and policies. No historical restart,
  real-policy refit, branch push or legacy artifact mutation is authorized.
- 2026-10-01 user separately authorized GPU validation on physical GPUs1–4,
  resolved/pinned by UUID: original RULER13 x5 and five random LongBench v2
  inputs, each no-cache / ProphetKV1% /5%, 210 answers. LongBench selection
  is seed42 from full-text inputs fitting KV65920 with the native16K output
  reserve, explicitly approved after the initial random cohort exceeded memory.
  RULER has native task caps and non-thinking; LongBench keeps thinking.
  Independent diagnostic probes verify sparse controls; no router fitting.
  Package: outputs/rpkv-gpu-validation-20261001/. Never duplicate its supervisor.
  Only accepted measurements constitute GPU evidence; CPU tests alone do not.
  Completed: all210 answers and70 probes passed independent final validation;
  owned GPU engines exited. See final/validation.json and final/report.md.
  The full CPU suite passed232 tests. Do not restart this completed scope.
  User authorized committing the implementation locally and providing push
  commands; no push is authorized.


- 2026-09-30 completed CPU-only depth1–5 router search on the existing260
  prompts/six measured actions: `outputs/ruler13-deeper-router-study-20260930/`.
  Final deliverables are in `finalized/`, superseding the primary-stage report
  for overall ranking. Guide: [DEEPER_ROUTER_STUDY.md](DEEPER_ROUTER_STUDY.md). No GPU inference,
  runtime policy integration or deployment. Preserve previous studies unchanged.
  Evaluated117 subsets x4320 settings=505440; reused116640 exact original depth1–3
  settings, completed388800 new fits. Includes five/seven/all45 controls,40 singles,
  28 pairs,6 triples and40 one-round drop-one refits of the primary all45 winner.
  Final ranking includes ablations; no recursive feature elimination. An initial
  partial pass is preserved separately, with completed identical subsets reused.
  Overall best: depth5/16 leaves/minleaf5, all45 minus gap10, positive loss,
  minimize-time, weight3, penalty.005. Accuracy90.0064% versus dense91.5769%
  (1.5705pp overall loss; floor89.5769231%); primary TTFT5.859440s versus
  incumbent8.605654s (-31.91%). CPU-adjusted5.904380s; incremental resident
  features44.940ms. Remaining gap to3.650979s hindsight:2.208461s. Actions
  dense/1/5/10/20/50=9/120/17/62/39/13. All260 fit/select/score; training-only,
  mixed sessions and assumed integrated probe reuse, not live measured latency.
  Best per maximum depth1–5:15.986091/11.191585/8.605654/6.326677/5.859440s.
  210 CPU tests passed; after orchestration correction,12 focused tests passed.
  Independent audits checked1560 answers,260 probes,all505440 candidate metrics
  and final rankings, exported decisions/boundaries/fallbacks and tree bounds.
  Full resume preserved523 completed files' hashes/mtimes and the prior study;
  finalization also proved immutable. See independent-validation.json and
  delivery-validation.json. Existing runtime policies remain unchanged.


- 2026-09-30 user authorized the 76-addition GPU feature study and live all260
  comparison. Package: `outputs/ruler13-gpu-feature-study-20260930/`;
  `control.sh status` / `resume`. Guide: [GPU_FEATURE_STUDY.md](GPU_FEATURE_STUDY.md).
  **Launched and startup independently verified Sep30 ~17:00 HKT.** Current
  supervisor PID2796747, worker PID2797080; verify identities before action.
  Supervisor PPID1, separate session, ignored SIGHUP, /dev/null stdin and empty
  CUDA visibility verified. Only original UUID-pinned physical GPUs1–4 are used.
  Leave this detached study running; no ongoing assistant polling or duplicate
  supervisor. Existing completed experiments and their frozen runtimes unchanged.
  Current runtime is the package's read-only, hash-pinned `frozen-code/`.
  All212 CPU tests passed, including actual vLLM RPC codec, bounded attention,
  policy boundaries, handoff isolation, cost sharing and synthetic final reports.
  All1560 source answers and260 original decisions independently checked; all260
  saved attention archives replayed exactly. Original six-action current fast
  policy:90.4487% accuracy,9.820171s comparable simulation.
  Initial startup halted at0 accepted records on untyped RPC NumPy transfer.
  Its one failed diagnostic probe, logs, runtime and exited process receipts are
  preserved in `startup-history/rpc-array-transport/`. Explicit FP32 byte packets
  fixed the transfer; no accepted record was replaced. An earlier CPU-only report
  JSON-type correction is preserved in `preflight-history/report-json-bool/`.
  First scheduled cwe-000 probe passed exact native scores/masks on all ranks,
  all64 selection, native YaRN4, eight committed warmup shards, immutable KV,
  retirement and deletion. All81 features finite.64 independent CPU FP32 reference
  checks (8 layers x4 ranks x head/query) passed; max error1.1921e-7. Analytic
  extra scratch bound113.15625MiB/rank, below256MiB. See
  `results/independent-startup-verification.json`. This verifies diagnostic
  startup; live handoff/output-equivalence GPU gates are scheduled later.
  Pipeline:260 accepted diagnostic all64/1% one-token probes -> owned engine
  exit -> CPU76 singles /28 pairs /6 triples, each2592 settings, depths1–3,
  <=2pp overall training loss -> locked winner ->520 routed answers with520
  fresh probes. Current/winner share one-use native-score handoff and alternate
  prompt order. No refit after live results. BF16 TP4, YaRN4, exact64000 inputs,
  KV65920, prefill16384 and greedy non-thinking256 remain unchanged.
  Cost-aware selection counts measured shared feature dependencies once; GPU
  attention uses the conservative measured joint head/query pass, while live
  computes required branches. CPU resident reductions exclude archive I/O.
  Every live sparse mask and full output must match its saved same-group fixed
  action; dense verifies native attention/zero reused KV/zero UCM operations.
  Live TTFT starts before probe and ends at final first token. Serialization is
  afterward; construction/priming separate.900s watchdog, one operational retry,
  missing-only resume; next GPU phase halts if assigned GPUs become occupied.
  Reports/ranked features/rules/per-task/paired CSVs publish automatically. Claim
  improvement only if winner meets <=2pp loss AND beats current measured live
  TTFT. Training-cohort findings only; no held-out claim or deployment.

- 2026-09-30 user authorized adding ProphetKV5/10% to the existing all260
  router pool and launching520 new answers. Package:
  `outputs/ruler13-router-add5-10-20260930/`; `control.sh status` / `resume`.
  **Launched and startup independently verified Sep30 12:45 HKT.** Supervisor
  PID2722396, worker PID2722558; verify live identities before intervention.
  Supervisor PPID1, separate session, ignored SIGHUP, /dev/null stdin and empty
  CUDA visibility verified. Worker uses original UUID-pinned physical GPUs1–4.
  Never duplicate this supervisor or restart the completed base collection.
  Runtime is package `frozen-code/`, read-only; preserve it unchanged.
  Entry point source: `scripts/corpus_extend.py`; guide:
  [RULER_ACTION_EXTENSION.md](RULER_ACTION_EXTENSION.md). Prior dirty worktree edits preserved.
  Base260 inputs/probes/1040 answers and source runtime are pinned in
  `results/sources.json`. All260 original archives/features replay exactly;
  BLAS/OMP must be1 for original float64 feature arithmetic. Initial CPU
  preflight failure under different BLAS settings is preserved separately in
  `preflight-history/blas-thread-setting/`; no GPU inference ran in that attempt.
  Derived5/10% masks use native FP32 scores, exact floors and stable ties;
  original archives/receipts are never rewritten. Combined reader validates
  each source record under its own original protocol.
  One persistent TP4 engine: original prompt order, ordinary priming, missing
  5% then10% answers, one temporary KV build/prompt then retirement/deletion.
  No new baseline or independent probe. BF16 YaRN4, input64000, chunk4096,
  scheduled prefill16384, non-thinking greedy256 remain unchanged. First cwe-000
  pair passed all-rank score/mask/all64-selection replay, YaRN normalization,
  eight committed warmup shards, immutability and retirement/deletion. See
  package `startup-verification.json`; startup done, status only on request.
  All171 CPU tests and9 extension tests passed. Watchdog900s, one operational
  retry, validation mismatch halt, resume only missing new records. After all
  520 answers validate and owned engines exit, supervisor pins260x6 matrix and
  automatically runs2592 settings with five features/depth1–3. <=2pp TRAINING
  loss; fastest primary plus maximum1%-usage objective. Previous11.31183s
  router retained as a candidate to prevent reported best feasible regression.
  Corrected cost: selected answer TTFT + traversal + saved probe TTFT ONLY for
  dense. Training-only simulation assuming integrated probe reuse, mixed timing
  sessions and unmeasured extra feature/sync/switch costs. No held-out claim,
  deeper trees or live integrated router. Reports/exports appear in
  `results/training/` after collection; they are not available at startup.


- 2026-09-30 user requested more1% routing and corrected-cost retuning. CPU-only
  all260 training study: `outputs/ruler13-router-more1pct-20260930/`. Searched
  2592 settings: two objectives (maximize1% count / minimize probe-reuse TTFT),
  signed/positive loss, depth1–3, minleaf5/10/20, three leaf penalties and24 loss
  weights.1638 candidates met actual training macro loss<=2pp. Same five numeric
  attention features; no task/query/reference features. All260 used for fitting,
  tuning and scoring; no held-out performance claim or GPU launch.
  Max-one best found:135/260 at1% (51.923%, up from95/260), accuracy89.7692%
  vs dense91.5769% (loss1.8077pp), corrected TTFT13.8952s/2.0609x. Actions
  dense/1/20/50=98/135/19/8. More dense choices make it slower than the previous
  router. Minimum-TTFT objective recovered the previous exact choices40/95/88/37,
  accuracy90.8718%, TTFT11.3118s/2.5316x. Max-one depth3/seven leaves/minleaf5;
  min-TTFT depth3/six leaves/minleaf20. Exports pass value/boundary/missing-feature
  checks. All2592 candidate accuracy/bounds and780 final decisions independently
  checked; see report.txt/.json, per-task.csv, rules and independent-validation.json.
  Finite greedy-tree search, not proven global optimality. Latency remains the
  hypothetical integrated-probe estimate with unmeasured extra feature/sync/switch
  costs excluded. Preserve earlier studies and their immutable artifacts.

- 2026-09-30 user requested corrected probe-reuse latency accounting. Saved in
  `outputs/ruler13-router-probe-reuse-accounting-20260930/`: selected answer TTFT
  plus tree traversal, plus per-prompt one-token probe TTFT ONLY for dense
  decisions. ProphetKV is assumed to reuse its normal probe. Same trees/actions;
  no refit, reranking or GPU run. All260 training fast-loss2pp:90.8718%,11.3118s,
  2.5316x; fast-loss5pp:86.5833%,8.6355s,3.3162x. Prior heldout104 fast-loss5pp:
  91.0897%,13.6059s,2.1047x. These are integrated-reuse ESTIMATES, not measured
  live router latency. Dense probe proxy (~2.06s) includes the collected one-token
  first-token work; incremental feature/transfer/TP-sync/switch costs remain
  unmeasured and excluded. Preserve earlier recorded-pipeline reports. See
  report.txt/.json and validation.json (1768 decisions replayed).

- 2026-09-30 user explicitly requested fitting, tuning and scoring on ALL260
  collected prompts. Completed CPU-only resubstitution study:
  `outputs/ruler13-router-all260-training-20260930/`. All216 clean tree settings
  fitted on260; selected accuracy-first and fastest within0/2/5pp TRAINING loss.
  This scope has NO held-out set; preserve earlier156/104 reports separately.
  Dense91.5769%/28.6370s. Accuracy-first92.6346%/36.9749s; fast-no-loss
  91.6474%/25.6530s; fast-loss2pp90.8718%/22.4236s/1.2771x; fast-loss5pp
  86.5833%/19.9845s/1.4330x. Times include original mean probe overhead11.4283s.
  TTFT audit: feature mean5.0273s but median.0630s/max47.0431s; replay mean2.8597s
  but median.5574s. Total probe median4.1834s. Three original37–47s feature cases
  reran CPU-only in~.06s with identical features; decision~.000214s including
  validation. Slow original intervals lack CPU/page-fault/scheduling telemetry;
  exact cause is unverified. Offline timings do not replace recorded costs or
  establish live-router latency. See report.txt, latency-explanation.txt,
  latency-audit.json, per-task.csv and delivery-validation.json. No GPU launch.

- 2026-09-30 CPU router training/simulation completed at user request. Artifacts:
  `outputs/ruler13-router-simulation-20260930/`. Frozen completed clean corpus:
  `outputs/ruler13-corpus20-clean-yarn4-20260929/`; all1040 answers/260 probes
  completed Sep30 01:56 HKT and owned processes exited. Do not restart collection.
  Training split seed42:156 rows (12/task), held-out104 (8/task). Unchanged clean
  trainer searched216 settings with five training-only folds and fitted five
  ranked policies. Two additional speed comparisons were selected from OOF only
  (fastest within2pp/5pp loss) before inspecting held-out performance. Seven fits,
  five distinct trees; router4 equals3 and router5 equals1. Router1 remains primary.
  All260 attention archives were independently replayed; five features matched
  recorded values exactly. Main estimates include recorded probe overhead even
  for dense fallback; no GPU inference, test-driven refit or timing recalibration.
  Held-out dense91.3782%/28.6364s; router1 91.1378%/36.9500s/.7750x;
  fast-loss5pp91.0897%/25.4381s/1.1257x, choosing dense/1/20/50 counts31/27/46/0.
  Held-out mean probe overhead12.4476s. No setting met the joint OOF<=2pp loss
  and>=4x target. See comparison.json/.txt/.html, per-task.csv, tree exports and
  validation.json. Paired20k within-task bootstrap intervals are descriptive;
  this small held-out cohort is not a general accuracy guarantee.

- 2026-09-29 local clean RULER13 data collection is launched on physical GPUs1–4
  by UUID. Package: `outputs/ruler13-corpus20-clean-yarn4-20260929/`; control:
  `control.sh status` / `status_same_count`. Exactly20 new prompts/task,
  260 prompts,260 independent all64/1% one-token probes,1040 answers from true
  connector-free dense and original all64 ProphetKV1/20/50%. Inputs are exactly
  64000 tokens; full source/question reconstruction passed for all260. BF16 TP4,
  native YaRN4 table131072, KV65920/1031 blocks, chunk/tile4096, prefill16384,
  greedy non-thinking256, local RTX4500Ada memory.95. Execution uses the package's
  read-only `frozen-code/`; never edit that copy or duplicate its supervisor.
  Supervisor initial PID2538953, CPU reporter2539705; verify live identities,
  detachment receipts, progress and logs before intervention. Baselines run first;
  then one cached engine collects each prompt's probe and three sparse answers.
  All-rank exact scores/masks must match the independent probe. Cached records
  explicitly audit unit-magnitude YaRN delta normalization after generation.
  NPZ archives retain all64 FP32 layers on all4 ranks plus native scores, local
  means, exact masks and position mappings. Features, outputs/token IDs, scores,
  timings and immutable-cache/retirement evidence are retained. CPU reporter
  writes live summaries, first-phase replay receipts, then a training snapshot
  and `collection-complete.json` after1040 answers/260 probes and owned exit.
  The corpus plan reserves2600 identities; only this260-prompt tranche runs.
  No tree fitting, later tranche or held-out inference is scheduled. Final
  supervisor receipt is `results/partial-260-validation.json` relative to the
  larger reserved plan. Startup verification only; status later on request.

- This worktree (`.worktrees/prophetkv-clean`) is the current codebase for new
  implementation, inference, launchers, and deployment. Use code under
  `.worktrees/` for current work.
- The parent repository's `benchmarks/` contains old/outdated code, preserved
  unchanged as reference for old experiments only. Do not import or invoke it
  for new work or restart its historical schedulers. Leave existing authorized
  training/export processes untouched; finalized artifacts may be read as data.

- Scope: Qwen3-32B BF16, YaRN 4× (32768 → 131072), TP configurable (default 4).
- Four public modes: `baseline`, `prophetkv`, `selective_prophetkv`, `router`.
- All modes use chunked prefill with at most 16384 scheduled tokens per step.
  Prefix caching stays disabled. Baseline has no UCM connector.
- Cached requests reserve full original-position prompt KV, load/align once,
  and retain one global selection across steps. Never use sparse-slot remapping.
- Empty repair ranges advance without forward/sampling; sample only after the
  full prompt. Reject preemption. Clear request state on finish/cancel/retirement.
- Probe the saved full suffix in at most 16384-query/projection batches, with
  all-context normalization and query-weighted means. Keep repaired KV intact.
- ProphetKV averages all 64 layers. Selective averages explicit zero-based IDs,
  or the last N layers; sum in ascending FP32 order, then divide by layer count.
- Preserve TP averaging, all-context-key normalization, stable position ties,
  exact floor budgets, original-position causal fusion and all-layer alignment.
- Context chunks default to 4096; setup accepts multiples of 64 through 16384.
  Keep the whole question in a fresh
  suffix of at least 256 tokens; never silently truncate inputs.
- Preserve cache readiness, all-rank selection replay and request retirement.
- `setup` freezes a JSONL collection and builds independent KV once. `run --setup`
  inherits model/TP/preparation and runs all prompts, or `--prompt-ids`, sequentially.
- Keep setup hashes separate from execution UUIDs. Algorithm/ratio/layers/output
  cap do not change KV identity; model/tokenizer/tokens/layout/runtime changes do.
- Exclusive build lock, shared reader locks, every-rank committed shards and
  retirement are mandatory. Resume only incomplete chunks; preserve complete shards.
- Cached collection readers never populate, write warmup KV, delete KV or silently
  rebuild. Baseline creates no connector and never looks up KV. Legacy `--input`
  retains its temporary cache. Setup completion is published atomically.
- Persistent setup GPU validation is pending assigned devices. The opt-in matrix
  is `tests/gpu_persistent_setup.py`; CPU tests use fake engines and local fixtures.
- Collection runs publish live aggregation after validated/retired prompts and
  final aggregation only after every selected prompt succeeds and engine shutdown.
  Report per-subtask and prompt-weighted overall TTFT, accuracy, thinking/answer
  content-token counts; keep control tokens and scored/unscored denominators explicit.
- Freeze per-prompt subtask/references/scoring as evaluation metadata; references
  never enter inference. Score only decoded answer content, excluding thinking.
  Missing references mean N/A; unfinished thinking has no answer content.
- Run CPU tests with `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v`.
- GPU execution needs a user request and explicit UUIDs. Never start, stop or
  resume historical experiments from other branches/worktrees; never use dummy jobs.
- Keep data, weights, caches, logs and results out of Git. See README for commands.
- vLLM 0.9.2 UUID compatibility is installed by `apply_all_patches` before
  attention backend imports, in drivers and spawned workers. Keep UUID visibility
  unchanged; NVML resolves UUIDs to physical indices only for metadata queries.
  Do not rely on manually modified site-packages or replace UUIDs with ordinals.

- 2026-09-26 focused GPU check is complete: two exact 64000-token NIAH single-3
  prompts × three modes, six validated measurements, all UUIDs correct. Both
  ProphetKV modes used 20%; selective layers [45,48,50,56,58]. User approved a
  test-only 64256-token/1004-block allocation on GPUs1–4; YaRN4 table131072 and
  scheduled budget16384 unchanged. Greedy non-thinking256. All engines exited
  and caches retired. See `outputs/niah-single3-64000-2samples-20260926/` for
  comparison and final-validation.json. Do not resume the completed driver.
  The default full-window configuration remains unchanged and GPU-unvalidated.

- Remote all-503 LongBench v2 commands: [A800_LONGBENCH.md](../deployment/A800_LONGBENCH.md) and
  `scripts/a800_longbench.sh`. No local GPU launch. Two UUID-pinned TP4 groups split
  even/odd prompt rows, each running baseline then all 12 cached configurations.
  Use `run.sh sweep`: build ONE prompt's KV, wait for all TP shards, reuse across
  vanilla/selective 1/5/10/15/20/30%, retire and delete before the next prompt.
  No collection KV setup in this launcher. One baseline engine plus one cached
  engine per group; policy changes require quiescence. Read requests cannot dump.
  Thinking/output16K, prefill16K, chunk4K, selective zero-based layers11–15 and
  preserved middle truncation to formatted input<=114688 remain unchanged.
  At most55.88GiB total retained KV, plus write/scratch/results overhead.
  Live reports serialize both writers; final requires503 validations/method and
  both engines exited. CPU tests cover bounded deletion and dynamic policies;
  `tests/gpu_temporary_sweep.py` is opt-in and has not been run on remote GPUs.
  Existing persistent setups and historical experiments must remain untouched.

- Remote deployment uses Git fetch and a separate server worktree, with existing
  data at `/mnt/sde/jh/projects/unified-cache-management/.data/LongBench-v2/data.json`.
  User explicitly wants no archive transfers. Prepare tokens offline with the
  local Qwen3 model/tokenizer; see the A800 guide for exact paths and commands.

- L20 RULER commands: [L20_RULER_64000.md](../deployment/L20_RULER_64000.md), `scripts/l20_ruler.sh`.
  User already installed the environment and cloned the clean branch at
  `/data/jh/unified-cache-management/ucm`; model `/data/jh/ckpts/Qwen3-32B`,
  Python `/data/jh/envs/ucm/bin/python`. Git deployment and official downloads only.
  Eight tasks x100: cwe/fwe/vt/qa_1/qa_2/niah_multivalue/niah_multikey_2/3.
  Exactly64000 formatted input tokens (not65536), native thinking/output16384,
  BF16 YaRN4, chunk4096, scheduled prefill16384. Baseline plus vanilla/selective
  1/5/10/15/20/30/40/60/80%, selected zero-based layers11–15:15200 measurements.
  Pinned NVIDIA generators retain source tokens; documented newline padding before
  user text attains exact length. No forced assistant prefix or truncation.
  Two UUID-pinned TP4 groups on remote0–3/4–7;8–9 stay free. Build each prompt once,
  reuse across18 cached policies, then retire/delete. No local GPU launch.
  Per-method live/final and combined status/aggregate report TTFT, answer-only RULER
  accuracy, thinking/answer content-token lengths per task and overall.

- A800/L20 mid-run scope change: `resume` on the existing launchers stops scheduling
  selective and runs only missing vanilla ratios; baselines must be complete.
  See [RESUME_PROPHETKV.md](../deployment/RESUME_PROPHETKV.md). Never delete completed records or restart
  baseline. Stop only the exact output-owned process tree before updating code.
  Resume validates original fingerprints, all-rank diagnostics, retirement and
  retained hashes; keep original reports and raw artifacts unchanged. New report
  state/session receipts live in `continuation/attempt-*`, selected by
  `continuation.json`. Final scope is3521 A800/8000 L20, selective discontinued.
  CPU-only recovery/locking/stop tests passed; remote GPU continuation is pending.

- A800 LongBench can explicitly resume on one surviving TP4 group with
  `RESUME_SINGLE_GROUP=1`, full UUIDs in `GPU_A`, and
  `RESUME_EXCLUDE_PERCENTAGES="15"`. See [A800_SINGLE_GROUP_RESUME.md](../deployment/A800_SINGLE_GROUP_RESUME.md).
  Keep both original logical shards; execute them sequentially on GPU_A and never
  use GPU_B in this mode. Preserve old results and their original device provenance.
  Active scope is baseline plus vanilla1/5/10/20/30%, 3018 total; retained15% and
  selective results are excluded from both ordinary and matched status. Resume
  inherits saved exclusions/device assignments. No local GPU or remote launch.

- Server-local launcher settings live in Git-ignored `.env`; `.env.example` is
  tracked. `scripts/server_env.sh` loads one trusted Bash assignment per line,
  preserving existing environment values before expanding later assignments.
  Both launchers load it before defaults. Optional `UCM_ENV_FILE` selects another
  file. Keep existing run/prepared/cache paths and UUID groups for a continuation.
  See [SERVER_ENV.md](../deployment/SERVER_ENV.md) for fetch-before-stop/update/resume commands.

- Resume first prints saved-file counts per method/shard and complete vanilla
  prompts; `counts` is read-only and does not load token files or diagnostics.
  `RESUME_VALIDATION=fast` skips saved diagnostic replay/hashing, retains result
  metadata/hash/input/config/retirement checks and requires diagnostic presence.
  Record the mode in receipts/reports; existing diagnostic pins are preserved
  but not verified in fast mode. Every NEW measurement remains fully validated.
  Full remains the absent-variable default; .env.example explicitly selects fast.
  Accept the pinned 16c1f75/2ae7f30 continuation runtime on upgrade, preserving
  accepted cohort fingerprints for repeated resumes. Unknown runtime drift fails.
- Historical remote fingerprints may include untracked `ucm/vendor/RULER` and
  `ucm/.cache/vendor/RULER` directories (both can coexist). Preserve both in the
  checkout. Resume can reconstruct a pinned release plus current vendor hashes,
  accepting only an exact original fingerprint match. Never remove vendor hashes
  or rewrite original receipts to force compatibility. Keep those files unchanged.

- Native router workflow: [A800_ROUTER.md](../deployment/A800_ROUTER.md), `scripts/a800_router.sh`.
  It supersedes the benchmark deployment adapter for clean inference. Keep active
  training and its existing exporter untouched; never import benchmark code.
  `runner.router_export` is a one-shot CPU consumer of finalized training data.
  Preserve three policies and router1 primary, with no A800 refit/recalibration.
  Router profile: BF16 TP4, YaRN4 table131072, separate KV65920/1031 blocks,
  chunk/tile4096, scheduled prefill<=16384, full question suffix>=256, greedy
  non-thinking256. Each answer has its own all64/1% single-token probe, exported
  native attention features, retirement, decision and TP barrier. Missing features
  fall back dense; corrupt data halts. Dense bypass is request-scoped; baseline
  remains connector-free. Total TTFT includes the full routing interval.
  Configure/prepare/verify/detach/status and missing-record resume use new paths,
  two disjoint A800 UUID groups, frozen1300 inputs,9100 answers/3900 probes.
  CPU suite and full frozen-input reconstruction are required; scheduled remote
  GPU gates remain unverified until an explicitly requested A800 run.
