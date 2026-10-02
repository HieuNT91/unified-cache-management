# rpkv

Qwen3-32B inference with original-token RULER 64K inputs and metadata-only
ProphetKV chunk boundaries. This branch inherits `prophetkv-clean` at `c07faff`
and its then-uncommitted changes; other worktrees and experiment packages remain
separate.

Baseline, ProphetKV, selective ProphetKV and router answers use the same prepared
`token_ids`. Chunking adds no text or special tokens. RULER uses the pinned
upstream generator/scorer with a native Qwen3 non-thinking adapter; LongBench v2
keeps its own thinking, output budget and scoring protocol.

For the new twelve-action **TP2 A800/L20 data collection and CPU router training**,
follow [the deployment guide](docs/deployment/A800_LONGBENCH_DATA.md). A800 primary
can start independently; configure extra later. The workflow automatically reads
`.env.a800` / `.env.l20` and exports portable checksummed datasets. GPU execution
and real-data training have not been performed for this implementation.

## Code and documentation

| Path | Purpose |
|---|---|
| [AGENTS.md](AGENTS.md) | Short mandatory rules for work in this branch |
| [run.py](run.py), [run.sh](run.sh) | Public preparation/inference entry points |
| [runner/](runner/) | Layouts, orchestration, cache lifecycle, reporting and routers |
| [ucm/](ucm/) | Connector, storage, sparse attention and runtime integration |
| [scripts/](scripts/) | Dataset adapters, launchers, baseline/ProphetKV controls and router workflows |
| [scripts/launcher/](scripts/launcher/) | Shell launchers for A800/L20 and router/corpus workflows, plus their shared environment loader |
| [tests/](tests/) | CPU regressions and opt-in GPU validation |
| [docs/agents/](docs/agents/architecture.md) | Architecture, invariants, testing and experiment rules |
| [docs/protocols/](docs/protocols/ruler.md) | RULER and LongBench v2 input/decoding/scoring contracts |
| [docs/deployment/](docs/deployment/README.md) | A800/L20, `.env`, reporting and resume guides |
| [docs/studies/](docs/studies/README.md) | Historical designs, results and archived agent notes |

Local `inputs/`, `outputs/`, `.cache/` and `.exports/` hold generated material and
are not source to commit. Existing frozen runtimes and result packages must be
preserved; see [experiment rules](docs/agents/experiment-rules.md).

The maintained experiment scope is no-cache baseline, ProphetKV and router.
Attention-feature, GPU-feature, deeper-router and linked action-extension study
implementations have been removed from `runner/` and `scripts/`. Historical
designs remain in `docs/studies/`; original frozen experiment code stays with its
artifacts. Router collection, training, policy export and validation remain, along
with shared input preparation, scoring, setup/sweep and resume infrastructure.

## Prepare and check on CPU

Run commands from this worktree root. Use the compatible Python environment;
see [testing](docs/agents/testing.md) for dependencies and the validation matrix.

```bash
export PYTHON_BIN=/path/to/environment/bin/python
export MODEL_PATH=/path/to/Qwen3-32B
export RULER_SOURCE="$PWD/.cache/RULER-rpkv"
export CUDA_VISIBLE_DEVICES=''

"$PYTHON_BIN" scripts/ruler.py download --ruler "$RULER_SOURCE"
"$PYTHON_BIN" scripts/ruler.py prepare --ruler "$RULER_SOURCE" \
  --model "$MODEL_PATH" --output inputs/rpkv-ruler64k

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  -m unittest discover -s tests -v
```

Default RULER preparation is 13 tasks × 500 samples, seed 42, generated as one
batch per task before sharding. `max_seq_length=65536` includes the complete
prompt and task output reserve; it does not require exactly 64,000 input tokens.
For prefix placement, task caps, decoding and scoring, read the
[RULER protocol](docs/protocols/ruler.md). Use the
[LongBench v2 protocol](docs/protocols/longbench-v2.md) for that dataset.

## Inference and deployment

The public modes are `baseline`, `prophetkv`, `selective_prophetkv` and `router`.
Baseline disables the connector and prefix reuse. Sparse methods share the
metadata layout and keep the complete question in the fresh suffix. Persistent
setup builds reusable KV for a collection; temporary sweeps build one prompt's
KV and reuse it across configurations. See [architecture](docs/agents/architecture.md)
and [invariants](docs/agents/invariants.md).

Use the [deployment index](docs/deployment/README.md) for server settings and
resume. Inherited guides are marked where their original protocol differs from
`rpkv`; do not use padded historical inputs or automatically reuse old policies.
GPU execution, process intervention and publishing require applicable user scope.

New fixed-control runs use `answer_validation: native-answer-diagnostics-v1`
and `expected_probes: 0`. They retain native answer scoring/mask validation,
construction and priming. Router decision probes remain supported. Existing
frozen experiments retain their original independent-probe protocol.

There is no separate `verify` launcher command. Launchers read manifests/receipts,
and each input is checked when used. Runtime metadata, cache readiness, native
answer diagnostics and retirement checks remain. Resume defaults to `fast`;
reports summarize committed results without reloading models or replaying saved
diagnostics/scoring. New fixed-control final reports write `completion.json`
with `report_validation: committed-results-only-v1`, not an independent audit.

The [GPU control study record](docs/studies/rpkv-controls.md) records completed
5-sample controls and the separately launched 40-per-task/20-LongBench scope.
Their evidence does not validate every GPU edge case or the later no-probe source;
see [testing and evidence limits](docs/agents/testing.md).
