# traj4 / traj4f: copy-gated trajectory selection for ProphetKV

`traj4` and `traj4f` are drop-in replacements for ProphetKV's recompute-set selection. They
keep ProphetKV's probe and budget, decode **4 greedy draft tokens** over the stale cache,
and use them to decide whether the answer is about to **copy a contiguous span** from the
context. If it is, the recompute set is chosen as long contiguous runs along that span;
if it is not, the selection is ProphetKV's top-B.

Qwen3-4B-Instruct-2507, RULER 64K, all 13 tasks × 100 prompts (1300):

| | dense | ProphetKV | `last:traj4` | `question:traj4f` | **`last:traj4f`** |
|---|---|---|---|---|---|
| mean accuracy, 10% budget | 0.913 | 0.784 | **0.837** | 0.824 | 0.835 |
| mean accuracy, 20% budget | 0.913 | 0.824 | **0.866** | 0.856 | 0.861 |

- Both `last:traj4` and `last:traj4f` beat ProphetKV clearly (`last:traj4f`: 170/59 paired
  at 10%, 132/51 at 20%), for +0.3 to +0.6 s TTFT.
- Their means are tied, but they trade tasks: `last:traj4` is much better on
  `niah_multikey_3`, `last:traj4f` on `niah_multivalue` and QA. See
  [Full results](#full-results-13-tasks--100-prompts).
- `question:traj4f` differs from `last:traj4f` only when the copy gate is closed (in
  practice `cwe`) and loses there, so the last-token probe is the one to use.

## How it works

Everything below happens once per request, inside the ProphetKV probe, before the
selected tokens are recomputed.

1. **Probe (unchanged).** The probe tokens (the question for `question:`, the last prompt
   token for `last:`) read the stale context cache in every layer; the mean attention per
   context token is ProphetKV's score `u`.
2. **Readers.** One query row is taken for the **last prompt token**, and **4 greedy draft
   tokens** are decoded from it over the stale cache with paged FlashAttention. Nothing is
   written to the KV cache. Each draft gives one attention row over the context. The 5
   rows (last token + 4 drafts) form a trajectory.
3. **Copy test.** If the model is copying a string, consecutive readers read consecutive
   positions: reader `t` reads `j` and reader `t+1` reads `j+1`. The trajectory mass at
   offset +1 is compared with the same mass at offsets −16..16 (excluding 0 and +1):

   ```
   edge_j = sum_t  w_t * h(a_t(j), a_{t+1}(j+1)),   h(p,q) = pq/(p+q),   w_t = sqrt(kappa_t kappa_{t+1})
   z      = (M(+1) - mean_k M(k)) / sd_k M(k)
   ```

   `kappa_t` is the concentration of reader `t` (`sum a^2 / (sum a)^2`), so diffuse readers
   count less.
4. **Gate.**
   - `z < 2` (no copy evidence): select top-B of the probe `u`. **Identical to ProphetKV.**
   - `z >= 2` (copying): solve the Potts chain cut

     ```
     max_x  sum_j v_j x_j  -  sum_j c_j [x_j != x_{j+1}]     subject to  sum_j x_j = B
     v = sum of the 5 reader rows,   c = price * edge
     ```

     exactly by dynamic programming along the sequence, with bisection on the budget
     multiplier and a top-up by the best single tokens. Cutting is expensive where the
     readers move along consecutive tokens, so the solution is a set of long runs over
     the copy trajectory.

The budget `B = floor(ratio * N)`, where `N` is the context length after the exact prefix
chunk, is the same as ProphetKV's.

### traj4 and traj4f

`traj4f` (`f` suffix) changes four things in `traj4`:

| | `traj4` | `traj4f` |
|---|---|---|
| first trajectory row | the probe row | the true last-prompt-token row |
| null of the copy test | raw masses (offset `k` sums over `N - |k|` terms) | each `M(k)` divided by its `N - |k|` terms |
| chain price | `c = 16384 * edge` | `c = UCM_TRAJ_LAMREL * sum(v) / sum(edge) * edge` (default 1.0) |
| non-copy branch ranks | the first trajectory row | the probe `u` |

With the `last:` probe the first and last rows coincide, so `last:traj4` and `last:traj4f`
differ only in the null and in the chain price. The absolute price 16384 has no relation
to the scale of `v` and makes the chain cut keep very long runs; the relative price makes
shorter ones.

## How to run

The selection is a sweep variant: `<probe>:traj<D>[f]`, with probe `question` or `last`:

```bash
export UCM_TRAJ_LAMREL=1.0            # relative chain price of the f variants
PYTHON_BIN=/path/to/python MODEL=/path/to/qwen3-4b-instruct-2507 \
MANIFEST=prepared_ruler13/manifest.jsonl OUT=runs/traj CACHE_ROOT=cache/traj \
GPUS="0 1 2 3" PERCENTAGES="10 20" \
EXTRA_ARGS="--variants question last:traj4 last:traj4f" \
bash scripts/ruler64k_launch.sh
```

- `question` is the ProphetKV baseline. All variants in one `--variants` list share the
  engine and the chunk cache, so the comparison is paired on the same prompts.
- `UCM_TRAJ_LAMREL` only affects the `f` variants; `traj4` always uses 16384.
- Add `--skip-baseline` to reuse a dense baseline measured by an earlier run.
- `traj<D>` sets a different number of drafts `D`. Only `D = 4` was measured.
- To split one manifest across machines, give each machine `SHARD_OFFSET` and
  `SHARDS_TOTAL` (see the header of `ruler64k_launch.sh`).

Requirements, all enforced at runtime:

- all-layer ProphetKV (`--no-selective`, which the launcher passes); the drafts reuse
  every layer's suffix K/V;
- token-level selection (not `--units sent`).

Each request logs a `prophetkv_selection` event with `selection_method:
copy_gated_trajectory`, `copy_z`, `copying`, `draft_tokens`, `selected_positions`,
`trajectory_lam_rel`, `traj_true_last` and `traj_unbiased`. `run.py` validates the budget
and range of the chain cut, and replays the top-B exactly when `copying` is false.

## Full results: 13 tasks × 100 prompts

Qwen3-4B-Instruct-2507, RULER 64K, 4096-token chunks, TP1, vLLM 0.9.2, 1300 prompts.
`question:traj4f` and `last:traj4f` were run together (paired); dense, ProphetKV and
`last:traj4` come from an earlier run on the same 1300 prompts.

| task | dense | ProphetKV 10% | `last:traj4` 10% | `question:traj4f` 10% | `last:traj4f` 10% | ProphetKV 20% | `last:traj4` 20% | `question:traj4f` 20% | `last:traj4f` 20% |
|---|---|---|---|---|---|---|---|---|---|
| niah_multikey_3 | 1.000 | 0.310 | **0.880** | 0.690 | 0.690 | 0.400 | **0.920** | 0.650 | 0.650 |
| niah_multikey_2 | 1.000 | 0.470 | **0.800** | 0.680 | 0.690 | 0.610 | **0.850** | 0.830 | **0.850** |
| niah_multikey_1 | 1.000 | **0.870** | 0.820 | 0.810 | 0.810 | **0.880** | 0.870 | **0.880** | **0.880** |
| cwe | 0.648 | 0.475 | **0.605** | 0.484 | 0.603 | 0.587 | **0.664** | 0.591 | 0.656 |
| niah_multivalue | 1.000 | **0.980** | 0.833 | 0.970 | 0.970 | **0.993** | 0.948 | 0.990 | 0.990 |
| niah_multiquery | 0.998 | **0.985** | 0.958 | 0.975 | 0.975 | 0.988 | 0.973 | **0.990** | **0.990** |
| qa_1 | 0.740 | **0.750** | 0.670 | 0.740 | 0.740 | **0.790** | 0.670 | 0.740 | 0.740 |
| qa_2 | 0.580 | 0.440 | 0.390 | **0.450** | **0.450** | **0.500** | 0.410 | 0.490 | 0.490 |
| vt | 0.976 | 0.926 | **0.934** | 0.924 | **0.934** | **0.974** | 0.962 | **0.974** | 0.962 |
| fwe | 0.923 | **0.993** | 0.990 | **0.993** | 0.990 | 0.993 | 0.990 | **0.997** | 0.987 |
| niah_single_1/2/3 | 1.000 | 0.997 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **mean** | 0.913 | 0.784 | **0.837** | 0.824 | 0.835 | 0.824 | **0.866** | 0.856 | 0.861 |
| share of dense | | 86% | 92% | 90% | 91% | 90% | 95% | 94% | 94% |

Paired wins/losses (prompts where the two scores differ):

| | 10% | 20% |
|---|---|---|
| `last:traj4f` vs ProphetKV | 170 / 59 | 132 / 51 |
| `last:traj4f` vs `question:traj4f` | 75 / 21 | 61 / 28 |
| `last:traj4f` vs `last:traj4` | 91 / 63 | 59 / 53 |

`last:traj4f` against `last:traj4`, by task, 10% / 20%:

- loses on `niah_multikey_3` (2/21, 1/28) and on `niah_multikey_2` at 10% (1/12);
- wins on `niah_multivalue` (47/4, 17/2), `niah_multiquery` (13/6, 9/2), `qa_1` (12/5,
  9/2) and `qa_2` (8/2, 9/1).

`last:traj4f` wins more prompts than `last:traj4` at 10%, but its losses on
`niah_multikey_3` are whole answers lost (score 1 → 0), while many of its wins on
`niah_multivalue` are partial credit, so the means come out tied.

Mean TTFT, `question:traj4f` / `last:traj4f`: 9.96 / 9.91 s at 10% and 15.56 / 15.44 s at
20% on one machine (889 prompts), 3.86 / 3.83 s and 5.80 / 5.73 s on the other (411
prompts). Absolute TTFT is not comparable across the two machines, which have different
GPUs.

## Things to know before relying on it

- **Neither variant dominates.** Pick `last:traj4` if the workload is dominated by
  copying one long string (uuid-like keys), `last:traj4f` if it has several answers
  spread over the context or QA.
- **The gate is what protects `cwe`.** On `cwe` the copy gate stays closed, so the
  selection is the probe's top-B; the gain over ProphetKV there comes from using the
  last-token probe instead of the question.
- **The TTFT cost** is one extra query row per layer for the last prompt token plus 4
  decode steps over the full stale context, measured at 64K only.
- **Measured on one model, at 64K, at TP1.** Other models, lengths and tensor-parallel
  degrees have not been run. The readers are all-reduced across TP ranks, so TP > 1 is
  supported, but it has not been checked against TP1.
