# traj4 / traj4f: copy-gated trajectory selection for ProphetKV

Replaces ProphetKV's top-B recompute selection. The last prompt token and 4 greedy draft
tokens, decoded over the stale cache, form a reader trajectory. If consecutive readers read
consecutive context positions (copy test `z >= 2`), the recompute set is solved as a Potts
chain cut along that trajectory (long contiguous runs); otherwise it is the probe's top-B.
Same budget as ProphetKV. Code: `ucm/sparse/prophetkv/trajectory.py`, `runtime.py`.

| variant | probe (non-copy branch) | chain price | copy-test null |
|---|---|---|---|
| `last:traj4` | last prompt token | `16384 * edge` | raw offset masses |
| `last:traj4f` | last prompt token | `UCM_TRAJ_LAMREL * sum(v)/sum(edge) * edge` (default 1.0) | length-unbiased |
| `question:traj4f` | question tokens | as `last:traj4f` | length-unbiased |

## Run

```bash
export UCM_TRAJ_LAMREL=1.0
PYTHON_BIN=/path/to/python MODEL=/path/to/qwen3-4b-instruct-2507 \
MANIFEST=prepared_ruler13/manifest.jsonl OUT=runs/traj CACHE_ROOT=cache/traj \
GPUS="0 1 2 3" PERCENTAGES="10 20" \
EXTRA_ARGS="--variants question last:traj4 last:traj4f" \
bash scripts/ruler64k_launch.sh
```

`question` is the ProphetKV baseline. Requires all-layer ProphetKV (`--no-selective`, set
by the launcher) and token-level selection. Each request logs `copy_z`, `copying`,
`draft_tokens` and `selected_positions` in its diagnostics.

## Results: RULER 64K, 13 tasks × 100 prompts (1300)

Qwen3-4B-Instruct-2507, 4096-token chunks, TP1, vLLM 0.9.2. `question:traj4f` and
`last:traj4f` were run together; the other columns come from separate runs on the same
1300 prompts.

**10% budget**

| task | dense | ProphetKV | `last:traj4` | `question:traj4f` | `last:traj4f` | `last:traj4f`, LAMREL=12 |
|---|---|---|---|---|---|---|
| niah_multikey_3 | 1.000 | 0.310 | **0.880** | 0.690 | 0.690 | 0.780 |
| niah_multikey_2 | 1.000 | 0.470 | **0.800** | 0.680 | 0.690 | 0.740 |
| niah_multikey_1 | 1.000 | **0.870** | 0.820 | 0.810 | 0.810 | 0.800 |
| cwe | 0.647 | 0.475 | **0.605** | 0.484 | 0.603 | 0.603 |
| niah_multivalue | 1.000 | **0.980** | 0.833 | 0.970 | 0.970 | 0.907 |
| niah_multiquery | 0.998 | **0.985** | 0.958 | 0.975 | 0.975 | 0.978 |
| qa_1 | 0.740 | **0.750** | 0.670 | 0.740 | 0.740 | 0.690 |
| qa_2 | 0.580 | 0.440 | 0.390 | **0.450** | **0.450** | **0.450** |
| vt | 0.976 | 0.926 | **0.934** | 0.924 | **0.934** | **0.934** |
| fwe | 0.923 | **0.993** | 0.990 | **0.993** | 0.990 | 0.990 |
| niah_single_1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| niah_single_2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| niah_single_3 | 1.000 | 0.990 | 1.000 | 0.990 | 1.000 | 1.000 |
| **mean** | 0.913 | 0.784 | **0.837** | 0.824 | 0.835 | 0.836 |

**20% budget**

| task | dense | ProphetKV | `last:traj4` | `question:traj4f` | `last:traj4f` | `last:traj4f`, LAMREL=12 |
|---|---|---|---|---|---|---|
| niah_multikey_3 | 1.000 | 0.400 | **0.920** | 0.650 | 0.650 | 0.810 |
| niah_multikey_2 | 1.000 | 0.610 | **0.850** | 0.830 | **0.850** | 0.810 |
| niah_multikey_1 | 1.000 | **0.880** | 0.870 | **0.880** | **0.880** | 0.870 |
| cwe | 0.647 | 0.587 | **0.664** | 0.591 | 0.656 | 0.656 |
| niah_multivalue | 1.000 | **0.993** | 0.948 | 0.990 | 0.990 | 0.975 |
| niah_multiquery | 0.998 | 0.988 | 0.973 | **0.990** | **0.990** | 0.988 |
| qa_1 | 0.740 | **0.790** | 0.670 | 0.740 | 0.740 | 0.670 |
| qa_2 | 0.580 | **0.500** | 0.410 | 0.490 | 0.490 | 0.460 |
| vt | 0.976 | **0.974** | 0.962 | **0.974** | 0.962 | 0.962 |
| fwe | 0.923 | 0.993 | 0.990 | **0.997** | 0.987 | 0.987 |
| niah_single_1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| niah_single_2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| niah_single_3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **mean** | 0.913 | 0.824 | **0.866** | 0.856 | 0.861 | 0.861 |

**Paired wins/losses** (prompts where the two scores differ):

| comparison | 10% | 20% |
|---|---|---|
| `last:traj4` vs ProphetKV | 194 / 115 | 153 / 81 |
| `last:traj4f` vs ProphetKV | 170 / 59 | 132 / 51 |

**TTFT** (`question:traj4f` / `last:traj4f`, same run): 9.96 / 9.91 s at 10%, 15.56 /
15.44 s at 20% on one machine (889 prompts); 3.86 / 3.83 s and 5.80 / 5.73 s on the other
(411 prompts, faster GPUs). The trajectory costs +0.3 to +0.6 s over ProphetKV.

## Ablations (same 1300 prompts)

| change | 10% | 20% | where it moves |
|---|---|---|---|
| probe: last token vs question (`last:traj4f` vs `question:traj4f`) | 75 / 21 | 61 / 28 | almost all `cwe` (copy gate closed there) |
| price + null: `last:traj4f` vs `last:traj4` | 91 / 63 | 59 / 53 | loses `niah_multikey_3`, wins `niah_multivalue`, QA |
| price: LAMREL 12 vs 1 (`last:traj4f`) | 39 / 52 | 28 / 33 | +0.09/+0.16 `niah_multikey_3`, −0.05/−0.07 `qa_1`, −0.06/−0.02 `niah_multivalue` |

The chain price trades `niah_multikey_3` (wants long runs) against `niah_multivalue` and QA
(want several shorter runs); no single value improves the mean.
