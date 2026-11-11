# Download and prepare data

Set up the environment using [create_env.md](create_env.md), then copy
`config/.env.example` to `config/.env` and edit the paths. Both scripts read this
file automatically. The paths below use the example defaults; `$CODE_ROOT` is
the repository root.

## Download

Run from the repository root:

```bash
bash scripts/download_data.sh ruler
bash scripts/download_data.sh longbench-v2
```

Downloads require Git and `curl`. URLs and destinations are listed in
`scripts/download_data.sh`. After successful downloads, the main files are:

```text
$CODE_ROOT/data/
├── LongBench-v2/
│   └── data.json                    # Original dataset: 503 questions
└── RULER/                           # NVIDIA/RULER source checkout
    ├── .git/
    ├── asset-downloads.json          # Asset sources and hashes
    ├── scripts/
    │   ├── synthetic.yaml
    │   └── data/
    │       └── synthetic/
    │           ├── constants.py
    │           ├── ...
    │           └── json/
    │               ├── english_words.json
    │               ├── squad.json
    │               ├── hotpotqa.json
    │               ├── PaulGrahamEssays.json
    │               ├── download_paulgraham_essay.py
    │               └── download-essays.log  # If the essay downloader ran
    └── ...
```

NLTK resources are stored separately, usually under:

```text
~/nltk_data/tokenizers/
├── punkt/
└── punkt_tab/
```

The actual NLTK location depends on the environment and existing resources.
Downloading does not create prepared inputs or independent KV caches.

For manual or offline setup, place the original files at the same destinations
and install the NLTK resources before preparing. RULER needs its source scripts
as well as the four JSON assets. Preserve the original asset bytes, especially
`PaulGrahamEssays.json`; fetching the essays again may produce different content.
Preparation checks source asset hashes.

## Prepare

Set `MODEL_PATH` in `config/.env` to the original Qwen3-32B model/tokenizer
snapshot (`9216db5781bf21249d130ec9da846c4624c16137`).
Edit the preparation hyperparameters at the top of `prepare_data.sh`:
`RULER_SAMPLES_PER_TASK=500` and `PREPARATION_SEED=42` by default. Keep the
corresponding values in `run.sh` aligned.

```bash
bash prepare_data.sh ruler
bash prepare_data.sh longbench-v2
```

Preparation runs on CPU and writes tokenized inputs, manifests and metadata to:

```text
$CODE_ROOT/data/prepared/
├── ruler64k/                        # RULER_PREPARED
└── longbench-v2/                    # LONGBENCH_PREPARED
```

Override source and output paths in `config/.env`. Independent KV caches are
created during benchmark execution.

## Existing data

If source data is already available, set `RULER_PATH` and `LONGBENCH_DATA` and
skip downloading. RULER source folders do not need Git metadata. Fresh
preparation checks source asset hashes and required package versions.

If prepared datasets already exist, point `RULER_PREPARED` and
`LONGBENCH_PREPARED` to them and skip both download and preparation. Keep
`RULER_SAMPLES_PER_TASK` and `PREPARATION_SEED` in `run.sh` consistent with the
existing RULER metadata. Runs use the supplied manifest without a fixed cohort
size.

Prepared inputs stay read-only. Both chunk sizes preserve identical token IDs;
changing LongBench answer seeds reuses the same inputs. Prepared files retain
their original budget metadata, separately from runtime generation settings.
