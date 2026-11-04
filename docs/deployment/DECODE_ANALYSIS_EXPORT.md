# Export decoded answers for error analysis (under 100 MB)

Copy only `scripts/export_decode_analysis.py` to the experiment server. It runs
with Python 3.9+ and the standard library; no repository imports, launcher, GPU,
NumPy, tokenizer or model are needed.

```bash
python3 /path/to/export_decode_analysis.py /path/to/EXPERIMENT_DIR \
  --output /path/to/longbench-decodes.jsonl.gz
```

Pass the experiment root containing `settings.json`, `plan.json` and `rows.json`,
not `records/`. Supports A800 LongBench v2 collection v3 (`primary/`), legacy
primary/extra controls, and tree experiments (`tree/`). For tree experiments,
an existing `baseline-source/` is included automatically, with only `nocache`
and matching inputs/seeds. A tree experiment without that folder exports just
its tree methods. No remote action is performed by the script.

The output is **one gzip-compressed JSONL file, at most 100,000,000 bytes**.
All decoded text is preserved, including whitespace and Unicode. Compression
ratio depends on the outputs: a sufficiently large collection may exceed the
limit. In that case export fails and removes its temporary file; it never
silently truncates text, drops answers or publishes a partial file. Select a
smaller scope explicitly, for example:

```bash
python3 /path/to/export_decode_analysis.py /path/to/EXPERIMENT_DIR \
  --output /path/to/agent-decodes.jsonl.gz \
  --question-ids longbench-v2-171 longbench-v2-199 longbench-v2-211 \
                 longbench-v2-305 longbench-v2-387 longbench-v2-456
```

`--question-ids` accepts either internal prompt IDs or the public benchmark's
`_id`. `--methods nocache tree-prophetkv-5 tree-prophetkv-20` and `--seeds 0 1`
can further restrict the export. Filters are explicit; baseline is included
only when `nocache` is selected (or no method filter is supplied). Unknown
filters fail. `--max-mb` can lower the limit. Existing files are never overwritten;
use a new output path for each snapshot.

## File contents

Each answer line has `type: "answer"` and:

| Field | Meaning |
|---|---|
| `question_id` | Public LongBench v2 `_id`, from the experiment's pinned rows |
| `prompt_id`, `seed` | Internal sequence ID and answer seed; seed is null for legacy records |
| `method` | Exact recorded method, including repair rate |
| `output` | Exact full `prediction`: decoded thinking **and** answer |
| `answer_text` | Exact answer-channel text stored by the output analyzer |
| `scored_text` | Exact text used for scoring, when recorded; excludes forced transition/prefix |
| `thinking_tokens`, `answer_tokens` | Recorded content token counts, not character/word estimates |
| `generated_answer_tokens` | Answer-generation count when available; can differ from `answer_tokens` because of forced transition tokens |
| `control_tokens`, `output_tokens` | Recorded control and total output counts |
| `accuracy` | Recorded score; export does not rescore |
| Closure/cap fields | `thinking_closure`, `thinking_cap_reached`, `answer_cap_reached`, `output_cap_reached`, etc., when present |
| Identity/provenance | Input/token hashes, evaluation protocol, source role/path and result/receipt hashes |

No context, input tokens, question/options, references, attention, caches or
diagnostic payloads are exported. The full decode retains the thinking text
even if a decoder omitted special delimiters; the exporter does not guess a
thinking/answer split. `answer_text` and `scored_text` are taken directly from
the record, never inferred by stripping a suffix. No generated token ID arrays
are included by default. Add `--include-token-ids` to retain `output_token_ids`
for tokenizer-based analysis; the same size limit still applies.

To join the earlier outcomes NPZ, use `prompt_id + "--seed-" + str(seed)` for
seeded records, or just `prompt_id` for legacy records, together with `method`.
Use the hashes to distinguish incompatible inputs across experiments.

The first line (`type: "manifest"`) records schema, source metadata hashes,
chunk size when declared, selection and expected counts. The last line
(`type: "summary"`) reports exported and missing answers per method. Missing
answers are not filled; live/incomplete experiments can be exported. This is
a snapshot of committed receipt paths, not a claim that the run is complete.

```python
import gzip
import json

with gzip.open("longbench-decodes.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row["type"] == "answer":
            print(row["question_id"], row["method"], row["seed"])
            print(row["thinking_tokens"], row["answer_tokens"])
            print(row["output"])
```

## Read cost and validation

Only small metadata/initialization receipts, `validated.json` and the associated
`result.json` are read. The 97 GB of diagnostic/attention artifacts are not
traversed or opened. One result is loaded and compressed at a time. Some full
results contain additional JSON fields, so the bytes read are the sizes of
the selected result files, not just their exported fields.

Checks cover metadata/plan pins, receipt protocol identity, result checksum,
prompt/method/seed/input/policy and worker identity, initialization, recorded
retirement, token accounting and cross-method submitted-token identity. Small
metadata and receipts are checked again before atomic publication. Diagnostic
archives and prepared input content are not independently replayed or audited.
Source records are read-only; no experiment jobs are started or stopped.
