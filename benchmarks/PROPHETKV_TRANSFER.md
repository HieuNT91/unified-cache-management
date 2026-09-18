# ProphetKV implementation snapshot

This branch imports the current ProphetKV UCM/vLLM port from
`/home/thnguyen/unified-cache-management` for independent version tracking.
`PROPHETKV_IMPORT_MANIFEST.json` records the source and destination base commits,
upstream revisions, source paths, and SHA-256 hashes. Imported files retain their
original bytes. The seven files in `ucm/sparse/prophetkv/` also match the private
implementation pinned by the original 64K experiment.

The transfer does not add multi-GPU or Qwen3-32B support. The current model is
Qwen3-4B-Instruct-2507, with 36 layers, TP=1, one prefill request at a time,
and a fresh 256-token suffix containing the complete question.

## Code map

| Purpose | Location | Status |
| --- | --- | --- |
| ProphetKV registration and implementation | `ucm/sparse/factory.py`, `ucm/sparse/prophetkv/` | Imported, including numerical references and positional causal attention |
| Required connector/base hooks | `ucm/sparse/blend/{blend,selection}.py`, `ucm/integration/vllm/patch/0.9.2/` | Required by the current ProphetKV subclass and pinned runtime |
| RULER pilot and reporting | `benchmarks/run_prophetkv.py`, `prophetkv_*.py`, `cacheblend_prophetkv.py` | Original pilot; retains its original comparison configurations |
| Current 64K ProphetKV sweep | `benchmarks/prophetkv64k/run_with_multivalue.py` | Six tasks, ten samples/task, four chunk layouts; uses the sibling base runner |
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

The existing ProphetKV experiment drivers prepare **RULER only**. The generic
LongBench/LongBench v2 evaluator code is included, but a ProphetKV-specific
LongBench adapter and validated sweep have not yet been implemented. In
particular, the non-RULER branch of `prophetkv_common.score` refers to an undefined
`extract_answer`; it must not be treated as working LongBench v2 support.
Long questions/options exceeding 256 tokens also require a fresh-suffix adaptation.

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
```

The import verifier checks recorded file hashes, Python syntax and RULER scoring
against the vendored official reference without importing an inference engine.
It is a snapshot check: subsequent intentional implementation edits should be
tracked as new commits rather than rewriting the original source hashes.
