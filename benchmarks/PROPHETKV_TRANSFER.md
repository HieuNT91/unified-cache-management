# ProphetKV implementation snapshot

This branch imports the current ProphetKV UCM/vLLM port from
`/home/thnguyen/unified-cache-management` for independent version tracking.
`PROPHETKV_IMPORT_MANIFEST.json` records the source and destination base commits,
upstream revisions, source paths, and SHA-256 hashes. Imported files retain their
original bytes. The seven files in `ucm/sparse/prophetkv/` also match the private
implementation pinned by the original 64K experiment.

The later local sync adds the full RULER/LongBench v2 study and persistent-engine
runner. `PROPHETKV_SYNC_MANIFEST.json` records the 40 added source files and their
hashes without changing the original import manifest. Both snapshots retain
their source bytes. The model remains Qwen3-4B-Instruct-2507, with 36 layers,
TP=1 and one prefill request at a time. Four independent GPU workers are supported;
this does not add tensor-parallel or Qwen3-32B validation.

## Code map

| Purpose | Location | Status |
| --- | --- | --- |
| ProphetKV registration and implementation | `ucm/sparse/factory.py`, `ucm/sparse/prophetkv/` | Imported, including numerical references and positional causal attention |
| Required connector/base hooks | `ucm/sparse/blend/{blend,selection}.py`, `ucm/integration/vllm/patch/0.9.2/` | Required by the current ProphetKV subclass and pinned runtime |
| RULER pilot and reporting | `benchmarks/run_prophetkv.py`, `prophetkv_*.py`, `cacheblend_prophetkv.py` | Original pilot; retains its original comparison configurations |
| Historical 64K ProphetKV sweep | `benchmarks/prophetkv64k/run_with_multivalue.py` | Six tasks, ten samples/task, four chunk layouts; superseded |
| Full RULER and LongBench v2 study | `benchmarks/prophetkv_full/` | Frozen 700 RULER and 184 LongBench v2 prompts, variable query suffixes and native dense-prefix controls |
| Persistent ProphetKV sweep | `benchmarks/prophetkv_persistent/run.py` | Baseline/10%/20%, 2,652 requests; normally twelve engine starts across GPUs 1–4 |
| Missing CSV measurements | `benchmarks/prophetkv_csv/` | Historical supplemental runner; depends on earlier experiment inputs |
| CSV reporting | `benchmarks/fill_kv_repair_{report,chunks}.py` | Reporting code only; result CSVs are not imported |
| RULER generation and official scoring | `benchmarks/vendor/RULER/` | Pinned upstream source and local word-list asset |
| LongBench v1 and v2 general evaluation | `test/common/uc_eval/`, `test/suites/E2E/test_evaluator.py` | Already identical in the destination; verified and included in the manifest |
| LongBench v2 official prompt and scoring reference | `benchmarks/vendor/LongBench-v2/` | Saved zero-shot template and upstream prediction script; reference only |

Standalone CacheBlend, QCFuse, selector, norm, random-skip and component-retention
experiments were excluded at the user's request. `cacheblend_ruler.py` is retained
as a direct dependency of ProphetKV for prompt formatting, RULER scoring, prompt
hashes and complete-cache verification. Its historical filename and contents are
preserved rather than refactored during the transfer. The registered ProphetKV
class currently inherits `Blend`; removing that dependency requires a separate
implementation change.

The positional attention and score kernels derive from the pinned QCFuse/SGLang
reference, but implement ProphetKV's selection procedure here. Their applicable
license and notice are preserved under `benchmarks/vendor/prophetkv-kernels/`.
No standalone QCFuse algorithm or experiment runner is imported.

## Dataset support and interpretation

The original top-level, `prophetkv64k`, and `prophetkv_csv` drivers prepare
**RULER only**; their non-RULER scoring branch is historical and incomplete.
The new `prophetkv_full` study supplies LongBench v2 native non-thinking prompts,
official answer extraction, question/choice positions, and fresh suffixes of
max(256, complete query-tail length). Its selection is based on actual context
token count below 65,536, without truncation or a length-category filter.
The frozen source study has 184 qualifying LongBench v2 prompts and 100 samples
for each of seven RULER tasks. Every configuration uses a 128-token output cap.

The persistent runner consumes those frozen inputs and assignments. It uses
one engine per GPU/configuration, a fixed 65,792-token capacity, request-specific
hash namespaces, tracked transfer retirement, per-request validation receipts,
and bounded sample caches. Prefix caching remains disabled; ProphetKV explicitly
loads independently populated chunks. TTFT includes online probing and selection.
Load, warmup, cache construction and session costs are recorded separately.

The source-machine qualification passed 20-request endurance per configuration,
A–B–A isolation, matched-capacity isolated-engine comparisons, failure/resume,
and all 36-layer native-reference controls, including an 871-token query suffix.
That qualification is not a completed 2,652-request accuracy result or validation
of the fork on another machine. Results and certificates are not copied here.
The 100% control matches independent native dense computation after exact
first-chunk reuse; it is not required to match cache-free BF16 computation.
This supersedes the older 3% no-cache comparison described in the preserved
`prophetkv_full/PROPHETKV.md`; its later `run_verified.py` implements the correction.

The LongBench v2 reference `pred.py` came from the saved upstream
`THUDM/LongBench` prediction script, not from a local QCFuse implementation. It
expects other upstream configuration files and is included for provenance and
answer-extraction reference, not as a standalone runnable CLI.

Current RULER formatting includes numbered boundaries and real padding tokens.
Chunk sizes therefore change the model's input tokens; use separate matching
full-prefill baselines. Rates apply to eligible cached tokens, excluding the
exact prefix and fresh suffix. Primary TTFT includes the online probe, selection,
transfer and recomputation. See `PROPHETKV.md` and the two experiment subdirectories
for the preserved protocols. These are a UCM/vLLM port, not an unmodified benchmark
or a claim of multi-GPU validation.

## Runtime and external inputs

The source experiment uses UCM 0.3.0, patched vLLM 0.9.2, PyTorch 2.7.0,
Transformers 4.53.2, Python 3.10 and CUDA 12.6. The newer repository checkout is
not a substitute for that installed runtime. The supervisors construct a private
UCM copy from the compatible installation and overlay the imported implementation.
The vLLM patch files are preserved; no package was installed or patched by this
transfer.

Weights, Python/CUDA environments, evaluation datasets, downloaded QA corpora,
live results, process receipts and KV caches are not included. In particular:

- RULER runners expect `.downloads/RULER`, `.data/RULER/...` and, for the CSV
  supplement, earlier frozen prompt/result artifacts. The generator source is
  versioned under `benchmarks/vendor/RULER`; QA/essay corpora remain external.
- LongBench datasets remain external. The generic evaluator accepts configured
  dataset paths; see `test/common/uc_eval/README.md`.
- Existing launchers retain source-machine GPU UUIDs, checkpoint paths, historical
  result names and absolute `/tmp` cache roots. Configure and pin new paths and
  GPU identities before a new experiment. Do not run historical launchers against
  caches or result directories owned by the original checkout's live jobs.
- `prophetkv_full.generate_sources()` checks for pre-generated 100-row datasets;
  it does not generate them. `prophetkv_persistent.prepare()` additionally requires
  the earlier full study's protocol, frozen inputs, private runtime and cleanup
  receipt. Copying these sources alone does not make preparation runnable.
- Dummy GPU jobs, idle placeholders, reservation and keep-alive workloads are
  permanently prohibited. Historical helper functions are retained only as part
  of the byte-identical source snapshots; never invoke them. The persistent
  scheduler does not call them. Do not launch historical schedulers that do.
- Physical GPU 0 remains forbidden. Before every GPU launch, inspect
  `nvidia-smi -L` and restrict visibility to the assigned allowed UUIDs. CPU-only
  preparation and validation expose no GPUs.

No experiment is launched or resumed as part of this source transfer.

## CPU verification

With a Python 3.10+ interpreter:

```bash
CUDA_VISIBLE_DEVICES='' python benchmarks/verify_prophetkv_import.py
```

With PyTorch installed, run each preserved suite in a separate process so that
the experiment-local modules cannot shadow another suite's modules:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s benchmarks -p test_prophetkv.py -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s benchmarks/prophetkv64k -p test_prophetkv.py -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s benchmarks/prophetkv_csv -p test_prophetkv.py -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s benchmarks/prophetkv_full -p test_prophetkv.py -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s benchmarks/prophetkv_persistent -p 'test_*.py' -v
```

The import verifier checks recorded file hashes, Python syntax and RULER scoring
against the vendored official reference without importing an inference engine.
It is a snapshot check: subsequent intentional implementation edits should be
tracked as new commits rather than rewriting the original source hashes.
