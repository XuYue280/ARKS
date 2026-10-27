# ARKS one-command pipeline

`arks_pipeline.sh` runs the complete ARKS workflow for one model with one
command: download, weight export, rank profiling, first-order rank
allocation, per-experiment compression, codebook calibration, WikiText-2
perplexity, adaptive coefficient sparsity, and a summary. It calls the
existing Python stages, checkpoints every stage, and resumes automatically.

```bash
cd transformer-sparsification
source .venv/bin/activate
chmod +x final/arks_pipeline.sh

final/arks_pipeline.sh --model facebook/opt-125m --experiments all --seed 0 \
    --output-dir arks_runs --cache-dir arks_cache
```

Other models only need the model id and the extra rank budget:

```bash
final/arks_pipeline.sh --model facebook/opt-350m  --extra-budget-mb 5.0
final/arks_pipeline.sh --model facebook/opt-1.3b  --extra-budget-mb 20.0
final/arks_pipeline.sh --model facebook/opt-2.7b  --extra-budget-mb 40.0
final/arks_pipeline.sh --model meta-llama/Llama-3.2-1B            # 64.06 MB default
final/arks_pipeline.sh --model meta-llama/Llama-3.2-3B --extra-budget-mb 190.0
```

Gated Llama checkpoints use the token stored by `huggingface-cli login`
(or `HF_TOKEN`). Layer counts, matrix shapes, the number of matrices and the
vocabulary size are read from the model configuration; nothing is
hard-coded per model.

## Experiments

| Name            | Directory                | Coefficients          | Calibration |
|-----------------|--------------------------|-----------------------|-------------|
| `fp16`          | `adaptive_baseline_fp16` | FP16 scalar           | no          |
| `int8`          | `adaptive_baseline_int8` | INT8 scalar           | no          |
| `vq_nocalib`    | `adaptive_vq_nocalib`    | VQ codebook           | no          |
| `vq_calibrated` | `adaptive_vq_calibrated` | VQ codebook           | yes         |
| `sparsity`      | `adaptive_sparsity`      | FP16 or INT8 scalar   | no          |

`--experiments all` runs all five. To evaluate only the scalar baselines
without any coefficient quantization codebooks (no VQ, no calibration, no
sparsity), select the two baselines:

```bash
nohup setsid final/arks_pipeline.sh --model facebook/opt-350m --extra-budget-mb 5.0 \
    --experiments fp16,int8 --output-dir arks_runs --cache-dir arks_cache \
    > arks_runs/opt-350m.out 2>&1 < /dev/null &
```

`nohup setsid ... &` keeps the run alive after the terminal is closed;
follow it with `tail -f arks_runs/opt-350m/seed_0/logs/run_*.log` and check
whether it is alive with `pgrep -af "arks_pipeline.sh --model"`. Running the
same command later with `--experiments all` adds the remaining experiments
without redoing the finished stages.

The rank profile and rank allocation are
computed once per run and shared by every experiment (they do not depend
on the coefficient representation), and the calibrated VQ experiment starts
from a copy of the uncalibrated VQ artifacts instead of compressing again.
Both are bit-for-bit identical to the previous per-experiment recomputation.

## Output layout

```
<output-dir>/<model-tag>/seed_<seed>[_<run-tag>]/
├── logs/run_<timestamp>.log        full log of every invocation
├── hardware.json  model_info.json  plan.json  plan.env  config.env
├── shared/rank_profile/            rank-6 K-subspaces profile
├── shared/rank_allocation/         first-order scores and selected ranks
├── adaptive_baseline_fp16/{artifacts,results,rank_allocation}
├── adaptive_baseline_int8/...
├── adaptive_vq_nocalib/...
├── adaptive_vq_calibrated/{artifacts,calibration,results,rank_allocation}
├── adaptive_sparsity/results/
├── stale/                          contents of stages that had to be redone
└── summary.csv  summary.json
<cache-dir>/models/<model-tag>/{fetch,weights,dense_ppl.json}
<cache-dir>/tokens/                 tokenized dataset splits
```

Result files keep their previous names and columns
(`inference_<timestamp>.csv/json`, `block_codebook_calibration.csv`,
`adaptive_sparsity.csv`, `first_order_rank_scores.csv`, ...).

## Checkpointing and resume

Every stage directory carries `.arks_stage.json` with the hash of its
configuration, the relevant script sources, and the stamps of the stages it
depends on. Running the same command again skips complete stages whose hash
still matches and reruns the rest. A stage whose configuration, code, or
upstream inputs changed is moved to `stale/` and recomputed, together with
everything downstream of it. `--force compress`, `--force calibrate`,
`--force all`, ... redo stages on demand; `--dry-run` shows what would run.

Runs for different models, seeds, and `--run-tag` values live in separate
directories and never overwrite each other.

## Hardware detection and execution settings

At start the script detects usable CPU cores, RAM, the GPU and its free VRAM,
plus the model's layer count, hidden size, vocabulary, and matrix shapes, and
chooses:

| Setting                  | Meaning                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `DEVICE`                 | `cuda` when a GPU is present, else `cpu`                                 |
| `WORKERS`                | compression processes: `min(matrices, (cores-2)/threads, RAM bound, 48)` |
| `THREADS_PER_WORKER`     | `1` (results independent of core count) or `legacy`                      |
| `LOSS_CHUNK_TOKENS`      | `0` when full-vocabulary logits fit in VRAM, else `512`                   |
| `MODEL_STAGING`          | `cuda` when the dense model plus one block's working set fits, else `cpu`|
| `CALIBRATION_STREAMING`  | `on` (linear-in-depth block calibration)                                 |

Each value, the reason for it, memory estimates, and warnings are printed and
saved to `plan.json`. Every value can be overridden
(`--workers`, `--threads-per-worker`, `--loss-chunk-tokens`, `--model-staging`,
`--calibration-streaming`, `--device`).

Settings that define the experiment (ranks, K, codebook size, sample counts,
sequence lengths, epochs, ...) are never changed automatically. If they do not
fit the detected hardware, a warning names the setting to change.

## Numerical behaviour

* Rank profiling, allocation, compression, calibration, and inference use the
  same code paths, arguments, and defaults as `run_final.sh`.
* The K-subspaces fit and the coefficient k-means depend on the BLAS/OpenMP
  thread count (a property of scikit-learn and BLAS, unchanged by this
  pipeline). They do not depend on the number of worker processes. Therefore:
  * `--threads-per-worker legacy` (one process, library default threads)
    reproduces results of the original scripts bit-for-bit on the same
    machine;
  * the default `--threads-per-worker auto` (one thread per worker, many
    workers) gives results that are identical across machines and worker
    counts, but differ from legacy runs at floating-point reduction level.
* `LOSS_CHUNK_TOKENS > 0` evaluates the LM head in slices; perplexities agree
  with the single-pass evaluation to about `1e-4` relative. It is enabled
  automatically only when the full logits would not fit.
* Gradient offloading in the allocator, dense-target caching and weight-graph
  reuse in calibration, the token and dense-perplexity caches, and
  streaming block inputs are exact (bit-for-bit) optimizations.

Use `arks_compare_runs.py --run-dir A --reference-dir B` to compare two runs
(or a run against an old `run_final.sh` output tree).
