# Qwen3-32B ProphetKV accuracy diagnosis

All 120 YaRN-2x measurements are complete. This note distinguishes observations from hypotheses; no additional GPU experiments were run for this analysis.

## Findings

The main accuracy loss is CWE. At 20% recomputation, VT scores 98% versus 100% without reuse; CWE scores 42% versus 74%. At 50%, VT recovers 100%, but CWE reaches only 57%. YaRN 2x improves several scores relative to 4x without materially changing TTFT, but does not eliminate the CWE gap.

CWE asks for the ten most common words throughout a numbered list. In the local generator, the target words occur 30 times and distractors 3 times. VT follows one four-hop variable chain. These require different kinds of evidence aggregation.

The implementation first builds independent chunk KV caches. It probes the fresh suffix through all 64 layers, ranks cached tokens using attention from the question tokens (mean over heads, sum over layers, mean across TP ranks), and repairs one fixed top-k token set through the model. Unselected K/V remains present and available to attention, but retains representations computed without preceding chunks. RoPE realignment restores positions; it cannot reconstruct those missing contextual computations.

**Working hypothesis:** the query-attention repair budget does not sufficiently recover global frequency information and the contextual dependencies needed for CWE. A token receiving high question attention is not necessarily the token whose recomputation would most improve the final answer. Probing itself uses the approximate cached context. These mechanisms are plausible explanations, not experimentally isolated causes.

## Comparison with the 4B checkpoint on the same ten source rows

References match for every paired row. Model, native context configuration, chat/tokenizer details, TP and memory adaptation still differ; this does not isolate a parameter-count effect. The 4B checkpoint is Instruct-2507, configured for 262144 positions with RoPE theta 5000000. The original 32B checkpoint config is 40960 positions with theta 1000000, extended here using YaRN with original position count 32768.

| CWE method | 4B accuracy | 32B YaRN 2x accuracy |
|---|---:|---:|
| baseline | 35% | 74% |
| prophetkv-5 | 29% | 35% |
| prophetkv-20 | 38% | 42% |
| prophetkv-30 | 39% | 47% |
| prophetkv-40 | 48% | 44% |
| prophetkv-50 | not measured | 57% |

The 32B baseline advantage is large (74% versus 35%), while the sparse results are much closer. At 20%, 32B is still slightly higher in absolute accuracy (42% versus 38%); at 40%, it is lower (44% versus 48%). It is inaccurate to infer a universal model-size effect from these ten-row comparisons.

## Checks and direct failure examples

- No measured 2x output hit the 128-token limit (0/120). Failures include incorrect words and repeated entries in completed answers.
- CWE row 0: baseline gives nine correct distinct reference words (90%). At 20%, the completed output lists “passive” ten times and scores 10%. At 30%, the same prompt scores 70%. This is a content/degeneration error, not a token-limit failure.
- All 80 adjacent-ratio comparisons have identical saved probe scores and nested masks. The 30% to 40% accuracy decline occurs while adding repaired tokens, rather than changing the score computation.
- Both task qualification cases passed the native-prefix 100% recomputation control with bitwise-equal audited fresh-suffix activations across all 64 layers and four ranks. All cache-shard availability/size, hit, tensor-write/preservation and engine-isolation gates passed. These checks reduce the likelihood of gross cache, TP or attention errors; they do not certify every sparse semantic path or every measured prompt at 100% recomputation.

## Post-hoc mask coverage

Reference-word token spans were located by decoding frozen prompt IDs, re-encoding with offsets, and requiring exact token-ID roundtrip equality. The denominator is reference-word tokens in the eligible cached region, excluding the exact first chunk and fresh suffix. Ground-truth references were used only for this CPU analysis, never for inference or selection.

| Recompute ratio | Reference-word token positions repaired, mean across 10 CWE prompts |
|---|---:|
| prophetkv-5 | 6.8% |
| prophetkv-20 | 39.3% |
| prophetkv-30 | 58.9% |
| prophetkv-40 | 75.4% |
| prophetkv-50 | 87.8% |

The selector does prioritize reference words, but even repairing about 88% of their token positions at 50% leaves a 17-point CWE gap. Literal answer-word coverage is therefore not a sufficient explanation; other contextual dependencies and nonlinear interactions may matter. Coverage alone cannot establish causality.

## Paper scope and remaining uncertainty

The ProphetKV paper evaluates Llama-3.1-8B, Qwen2.5-14B and Qwen3-14B, with its main RULER setting at 8K and 512-token chunks. Qwen3-14B CWE/FWE were excluded because generation repeated the input and exceeded the limit. Its headline accuracy does not validate this Qwen3-32B, 64K, 4096-chunk UCM/vLLM port. Source: https://arxiv.org/html/2602.02579v3#S5

A decisive next diagnostic would run a full 128-token 100%-repair comparison on all ten CWE prompts, then compare selection strategies at a fixed budget. The current 100% qualification covers one smoke prompt per task, not the full measured cohort. Such further inference has not been launched.

All results use the same memory tiling. The 2x table has 256 extra positions computed from unchanged factor-2 frequencies to retain the identical 65792-token engine allocation; its original 65536 entries are bitwise preserved. The 2x measurements ran later than 4x. This is ten prompts/task with one timing each.

Artifacts: `cpu-analysis.json`, `mask-consistency.json`, `analyze.py`, `../final/`, and `../yarn-comparison/`.
