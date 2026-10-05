# Thinking RULER on five L20 TP2 pairs

This is a separate collection in `tp2-router-data`: 13 tasks × 30 samples, seed42,
390 prompts, twelve actions, 4,680 answers and 390 one-token feature probes.
Each of five TP2 pairs owns 78 prompts (six per task); every action for a prompt
uses the same pair. Controls finish and their engines exit before feature
collection and portable export. There is no automatic fitting.

Implementation and CPU verification do not authorize an experiment launch or
real-data training. Keep existing configured checkouts, jobs and artifacts intact.
Use a new checkout and fresh preparation/result/cache directories when launching
this protocol after separate authorization. GPU behavior remains unverified.

## Configuration and commands

Copy [the example](../../.env.l20.thinking.example) to `.env.l20.thinking` at the
checkout root and set the model, pinned RULER source, interpreter and fresh paths.
The launcher uses the shared trusted Bash loader: existing environment variables
win; `UCM_ENV_FILE` can select another file. The usual `PYTHON_BIN`, `MODEL_PATH`,
`RULER_PATH`, `EXPERIMENT_DIR`, `PREPARED_DIR`, `CACHE_ROOT`, `TP`, `GPU_DEVICES`,
`SAMPLES_PER_TASK` and `SEED` names are unchanged. TP2,30/task and seed42 are fixed
for this profile. Configure resolves the ten devices to immutable UUID pairs.

```bash
cp .env.l20.thinking.example .env.l20.thinking
# Edit the copied configuration before use.
bash scripts/launcher/l20_ruler_thinking_data.sh configure
bash scripts/launcher/l20_ruler_thinking_data.sh prepare
# Only after GPU launch is separately authorized:
bash scripts/launcher/l20_ruler_thinking_data.sh detach
bash scripts/launcher/l20_ruler_thinking_data.sh status
bash scripts/launcher/l20_ruler_thinking_data.sh status_same_count
bash scripts/launcher/l20_ruler_thinking_data.sh report
```

`resume` and `stop` use the existing ownership/identity checks. Resume skips
committed records. There is no separate verify stage. The default paths contain
`ruler-l20-tp2-thinking-30-v1`; never point them at a prior non-thinking collection.
The original `l20_ruler_data.sh` retains its non-thinking100/200-row protocol.
See the [shared TP2 guide](A800_LONGBENCH_DATA.md) for lifecycle and storage details.

## Frozen input and generation protocol

Preparation calls the unchanged upstream generator once per task, with30 rows,
seed42, its original task parameters and65536-token source budget including the
original task answer reserve. Raw batches are retained unchanged. The adapter
validates the non-thinking template boundaries, extracts the complete user content
and renders the tokenizer's native thinking chat template. It does not truncate,
pad or move the original question. Token hashes and question positions are rebuilt.
The original answer prefix is deferred until the answer phase and emitted once.

Evaluation identity: `nvidia-ruler-c3f5e3b-qwen3-thinking-v1`.
Execution profile: `ruler-thinking`. Prepared samples, preparation metadata and
run rows freeze each tokenized prefix/transition, output reserve and phase policy.
The native Qwen3 template leaves the initial `<think>` for generation. It is
allowed only at the beginning; templates already opening thinking are supported
without another opener. Thinking delimiters are control tokens, even when the
tokenizer does not list them as special tokens.

The generated-content thinking cap is16,384. Natural `</think>` closes reasoning
earlier; otherwise the next token at the cap is forced to `</think>`. The request
then forces two newlines and the original answer prefix, followed by generated
answer content capped at128 for NIAH,30 for VT,120 for CWE,50 for FWE and32 for QA.
Whitespace counts as generated content in its phase. Forced tokens consume neither
phase budget; the optional initial opener and closure are controls. EOS is blocked
in thinking; the answer ends at EOS or its own cap even after early closure.
Further thinking blocks and other special controls are blocked.

Both phases use temperature0.6, top-p0.95, top-k20, min-p0 and seed0. One request
retains its KV through both phases. The process-local vLLM worker masks/forces
logits before sampling, and the scheduler stops at the answer budget. The policy
travels in `SamplingParams.extra_args`; state belongs to each request and is
released with its request bookkeeping. Construction, priming and one-token probes
carry no phase policy. Baseline and sparse answers share the same policy.

The engine window is82,304, with at least 1,286 KV blocks of 64 tokens and equal capacity
across both ranks. TP2 retains automatic sizing at95% memory. YaRN4 retains its
131,072-position table. Admission includes the complete prompt, optional opener,
all reasoning, closure, transition, prefix and answer reserve. Overflow halts and
retains raw data. No capacity check permits shortening or precision changes.
Existing readiness, native diagnostics, immutable-cache and retirement gates apply.

## Results and portable training

Only newly generated answer content reaches the existing RULER scorer and null
prediction counter. Reasoning, delimiters, transition text and the forced prefix
cannot earn credit. Records retain raw output text/IDs and full-stream
thinking/answer/control accounting; full-stream answer content includes the forced
newlines/prefix. `generated_answer_tokens` and `scored_text` describe the scored
portion. `forced_tokens`, `generated_thinking_open_tokens`, separate
`thinking_cap_reached`/`answer_cap_reached` flags and `thinking_closure` preserve
the phase details. `output_cap_reached` means either content cap was reached.

TTFT remains submission-to-first-generated-token, with the existing synchronization
cost semantics. `first_answer_content_seconds` adds latency to the first newly
generated answer-content token, excluding the forced prefix; it is null if there
is no such token. It includes the same synchronization/routing overhead as reported
TTFT. JSON/CSV/Markdown report both latency measures and separate cap counts.
Trainer costs continue to use TTFT plus the later feature-probe overhead, normalized
by the same prompt's baseline; they do not switch to answer-content latency.

The export remains `ruler-data.json` with its checksum. The importer requires the
complete390-row thinking cohort, twelve actions, TP2 provenance, frozen per-task
policies and consistent phase accounting. Existing100/200-row non-thinking RULER
and503-row LongBench bundles remain supported. An explicitly invoked CPU trainer
can use390 thinking RULER rows alone or893 rows combined with LongBench; it retains
the thinking evaluation identity and provenance. Training/evaluation overlap and
all existing limits on interpretation still apply.

## CPU verification

All289 CPU regression tests passed, including16 focused thinking tests. Coverage
includes phase boundaries, native template adaptation, real vLLM metadata
serialization and scheduler requests, worker logits, cleanup, capacity, relocation,
reports, and a synthetic390-row fit/export/import. A separate CPU check used the
cached Qwen3-32B tokenizer; no source cohort was generated. All12 shell launchers
passed `bash -n`; changed Python files, controller help and `git diff --check`
also passed. These checks do not establish GPU behavior or performance.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest discover -s tests -v
```
