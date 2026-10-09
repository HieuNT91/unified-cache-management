# traj4f: copy-gated trajectory selection for ProphetKV

`traj4f` is a drop-in replacement for ProphetKV's recompute-set selection. It keeps
ProphetKV's probe and budget, decodes **4 greedy draft tokens** over the stale cache, and
uses them to decide whether the answer is about to **copy a contiguous span** from the
context. If it is, the recompute set is chosen as long contiguous runs along that span;
if it is not, the selection is exactly ProphetKV's top-B.

On Qwen3-4B-Instruct-2507, RULER 64K, at a 10% recompute budget:

- `question:traj4f` improves the mean over six non-saturated tasks from **0.578 to 0.683**
  (paired 37 wins / 14 losses), for **+0.33 s TTFT**.
- `last:traj4`, the earlier form with the last-token probe, reaches **92% of dense** over
  all 13 tasks (0.837 against ProphetKV's 0.784; 194/115 paired on 1300 prompts).

The two have the same mean on the tasks where both were run, but win on different tasks:
see [Question probe or last-token probe](#question-probe-or-last-token-probe).

## How it works

Everything below happens once per request, inside the ProphetKV probe, before the
selected tokens are recomputed.

1. **Probe (unchanged).** The question tokens read the stale context cache in every layer;
   the mean attention per context token is ProphetKV's score `u_q`.
2. **Readers.** One extra query row is taken for the **last prompt token** (`u_last`), and
   **4 greedy draft tokens** are decoded from it over the stale cache with paged
   FlashAttention. Nothing is written to the KV cache. Each draft gives one attention row
   over the context. The 5 rows (last token + 4 drafts) form a trajectory.
3. **Copy test.** If the model is copying a string, consecutive readers read consecutive
   positions: reader `t` reads `j` and reader `t+1` reads `j+1`. The trajectory mass at
   offset +1 is compared with the same mass at offsets −16..16 (excluding 0 and +1):

   ```
   edge_j = sum_t  w_t * h(a_t(j), a_{t+1}(j+1)),   h(p,q) = pq/(p+q),   w_t = sqrt(kappa_t kappa_{t+1})
   z      = (M(+1) - mean_k M(k)) / sd_k M(k)        every M(k) divided by its N-|k| overlapping terms
   ```

   `kappa_t` is the concentration of reader `t` (`sum a^2 / (sum a)^2`), so diffuse readers
   count less.
4. **Gate.**
   - `z < 2` (no copy evidence): select top-B of the question probe `u_q`. **Identical to
     ProphetKV.**
   - `z >= 2` (copying): solve the Potts chain cut

     ```
     max_x  sum_j u_j x_j  -  sum_j c_j [x_j != x_{j+1}]     subject to  sum_j x_j = B
     u = sum of the 5 reader rows,   c = edge * UCM_TRAJ_LAMREL * sum(u) / sum(edge)
     ```

     exactly by dynamic programming along the sequence, with bisection on the budget
     multiplier and a top-up by the best single tokens. Cutting is expensive where the
     readers move along consecutive tokens, so the solution is a set of long runs over
     the copy trajectory.

The budget `B = floor(ratio * N)`, where `N` is the context length after the exact prefix
chunk, is the same as ProphetKV's.

The `f` suffix means four corrections over the original `traj4`: the true last-token row
heads the trajectory (not the question probe), the null is unbiased in offset length, the
chain price is relative to the unary mass rather than the bare constant 16384, and the
non-copy branch ranks the question probe. `traj4f` beats `traj4` by 0.05 at 10%.

## How to run

The selection is a sweep variant: `<probe>:traj<D>[f]`. The probe (`question` or `last`)
ranks the non-copy branch and is the score exported to the diagnostics:

```bash
export UCM_TRAJ_LAMREL=1.0            # relative chain price; 1.0 is what was measured
PYTHON_BIN=/path/to/python MODEL=/path/to/qwen3-4b-instruct-2507 \
MANIFEST=prepared_ruler13/manifest.jsonl OUT=runs/traj4f CACHE_ROOT=cache/traj4f \
GPUS="0 1 2 3" PERCENTAGES="10 20" \
EXTRA_ARGS="--variants question question:traj4f last:traj4" \
bash scripts/ruler64k_launch.sh
```

- `question` is the ProphetKV baseline; all variants in one `--variants` list share the
  engine and the chunk cache, so the comparison is paired on the same prompts.
- `UCM_TRAJ_LAMREL` only affects the `f` variants; `last:traj4` uses the original
  absolute chain price (16384).
- Add `--skip-baseline` to reuse a dense baseline measured by an earlier run.
- `question:traj<D>f` sets a different number of drafts `D`. Only `D = 4` was measured.

Requirements, all enforced at runtime:

- all-layer ProphetKV (`--no-selective`, which the launcher passes); the drafts reuse
  every layer's suffix K/V;
- token-level selection (not `--units sent`).

Each request logs a `prophetkv_selection` event with `selection_method:
copy_gated_trajectory`, `copy_z`, `copying`, `draft_tokens`, `trajectory_lam_rel`,
`traj_true_last` and `traj_unbiased`. `run.py` validates the budget and range of the chain
cut, and replays the top-B exactly when `copying` is false.

## Results

`question:traj4f`. Qwen3-4B-Instruct-2507, RULER 64K (`niah_multikey_2/3`,
`niah_multivalue`, `cwe`, `qa_1`, `qa_2`; 40 prompts each, 240 in total), 4096-token
chunks, TP1, vLLM 0.9.2. This branch reproduces that run's recompute sets exactly (checked
on 6 prompts for both arms; `copy_z` agrees to six digits). Accuracy:

| task | ProphetKV 10% | traj4f 10% | ProphetKV 20% | traj4f 20% |
|---|---|---|---|---|
| niah_multikey_2 | 0.450 | **0.750** | 0.650 | **0.825** |
| niah_multikey_3 | 0.350 | **0.700** | 0.400 | **0.725** |
| niah_multivalue | 0.988 | 0.969 | 1.000 | 0.988 |
| cwe | 0.480 | 0.480 | 0.578 | 0.578 |
| qa_1 | 0.725 | 0.725 | 0.725 | 0.700 |
| qa_2 | 0.475 | 0.475 | 0.475 | 0.450 |
| **mean** | 0.578 | **0.683** | 0.638 | **0.711** |
| paired wins/losses vs ProphetKV | | **37/14** (p = 9e-4) | | **25/9** (p = 5e-3) |
| mean TTFT | 8.37 s | 8.70 s (**+0.33 s**) | 13.17 s | 13.55 s (**+0.38 s**) |

Against full recomputation (dense prefill, no reuse), on the 188 of these prompts that a
dense baseline run on the same machine also covers:

| | dense | ProphetKV | traj4f |
|---|---|---|---|
| mean accuracy, 10% budget | 0.828 | 0.577 | 0.680 |
| mean accuracy, 20% budget | 0.828 | 0.624 | 0.703 |
| mean TTFT, 10% budget | 43.80 s | 8.32 s | 8.63 s |

traj4f closes about 40% of the gap between ProphetKV and dense at either budget, at about
a fifth of dense TTFT. It still loses to dense 10/58 at 10%. The largest remaining gaps
are `niah_multikey_3` (1.00 against 0.66), `niah_multikey_2` (1.00 against 0.74) and
`cwe` (0.72 against 0.46). On `qa_1` both sparse methods score above dense (0.75 against
0.66).

The gain is concentrated on the two multi-key needle tasks, where the answer is a copied
uuid or number. How often the gate opened (`copy_z`, unbiased, same run):

| task | `copy_z` 10th pct | median | 90th pct | gate open |
|---|---|---|---|---|
| niah_multikey_2 | 2.24 | 2.99 | 3.29 | 95% |
| qa_2 | 2.45 | 2.56 | 2.87 | 100% |
| qa_1 | 2.24 | 2.50 | 3.10 | 100% |
| niah_multikey_3 | 2.22 | 2.25 | 2.28 | 100% |
| niah_multivalue | 2.10 | 2.15 | 2.62 | 100% |
| cwe | 1.26 | 1.33 | 1.38 | 0% |

On `cwe` the gate stays closed on all 40 prompts, so the selection is exactly ProphetKV's
and the scores are equal. On `qa_1`, `qa_2` and `niah_multivalue` the gate opens but the
chain cut neither helps nor hurts at 10%. The other seven RULER tasks were not run with
traj4f.

### Question probe or last-token probe

The probe also chooses which readers rank the non-copy branch. `last:traj4`, the original
selection with the last prompt token as the probe and none of the `f` corrections, was run
on all 13 RULER tasks in an earlier sweep. On the 134 prompts both runs share (same
machine, 10% budget):

| task | dense | ProphetKV | `question:traj4f` | `last:traj4` |
|---|---|---|---|---|
| niah_multikey_3 | 1.000 | 0.391 | 0.652 | **0.826** |
| niah_multikey_2 | 1.000 | 0.364 | 0.682 | **0.727** |
| cwe | 0.695 | 0.486 | 0.486 | **0.610** |
| niah_multivalue | 1.000 | 0.979 | **0.969** | 0.906 |
| qa_1 | 0.636 | 0.773 | **0.773** | 0.545 |
| qa_2 | 0.545 | 0.409 | **0.455** | 0.364 |
| mean | 0.818 | 0.572 | 0.675 | 0.668 |

The means are tied (18/23 paired), but the two probes split the tasks: the last-token
probe is better where the answer is copied or counted (`niah_multikey_3` reaches 0.957 at
20%), and the question probe is better on QA and multi-value retrieval. The two columns
come from separate runs, and `last:traj4f` has not been run.

`last:traj4` is the only variant measured on the full RULER set, 13 tasks × 100 prompts
(1300; 722 on one machine and 578 on another):

| task | dense | ProphetKV 10% | `last:traj4` 10% | ProphetKV 20% | `last:traj4` 20% |
|---|---|---|---|---|---|
| niah_multikey_3 | 1.000 | 0.310 | **0.880** | 0.400 | **0.920** |
| niah_multikey_2 | 1.000 | 0.470 | **0.800** | 0.610 | **0.850** |
| cwe | 0.648 | 0.475 | **0.605** | 0.587 | **0.664** |
| niah_multivalue | 1.000 | **0.980** | 0.833 | **0.993** | 0.948 |
| qa_1 | 0.740 | **0.750** | 0.670 | **0.790** | 0.670 |
| qa_2 | 0.580 | **0.440** | 0.390 | **0.500** | 0.410 |
| niah_multikey_1 | 1.000 | **0.870** | 0.820 | **0.880** | 0.870 |
| niah_multiquery | 0.998 | **0.985** | 0.958 | **0.988** | 0.973 |
| vt | 0.976 | 0.926 | 0.934 | 0.974 | 0.962 |
| fwe | 0.923 | 0.993 | 0.990 | 0.993 | 0.990 |
| niah_single_1/2/3 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 |
| **mean** | 0.913 | 0.784 | **0.837** | 0.824 | **0.866** |
| share of dense | | 86% | **92%** | 90% | **95%** |
| paired wins/losses vs ProphetKV | | | **194/115** (p = 4e-6) | | **153/81** (p = 2e-6) |

Its TTFT is 0.3–0.6 s above ProphetKV's. Absolute TTFT is not comparable across the two
machines, which have different GPUs.

## Things to know before relying on it

- **The gate is what protects `cwe`.** The threshold of 2.0 falls in a gap in the
  measured `copy_z`: `cwe` never exceeds 1.38, and every other task's 10th percentile is
  at least 2.10. In a reference simulator, applying the chain cut on every prompt
  regardless of `z` lost to the gated version 11/43 on a held-out set. That simulator used
  the absolute chain price, not `UCM_TRAJ_LAMREL`.
- **The TTFT cost** is one extra query row per layer for the last prompt token plus 4
  decode steps over the full stale context, measured at 64K only.
- **Small regressions** at 20% on `qa_1` (−0.025), `qa_2` (−0.025) and
  `niah_multivalue` (−0.012), each one or two prompts out of 40.
- **Measured on one model, at 64K, at TP1.** Other models, lengths and tensor-parallel
  degrees have not been run. The readers are all-reduced across TP ranks, so TP > 1 is
  supported, but it has not been checked against TP1.
