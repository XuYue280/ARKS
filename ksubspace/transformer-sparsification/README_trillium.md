# Running ARKS on Trillium

## The one thing that trips everyone up

Trillium is **two separate Slurm clusters**:

| login node | Slurm cluster | hardware |
|---|---|---|
| `tri-login0{1..6}` | `trillium` | CPU only — `GresTypes=(null)`, **no GPUs at all** |
| `trig-login01` | `grillium` | 60 nodes x 4x H100-80GB |

ARKS needs a GPU, so **everything must be submitted from `trig-login01`**.
Jobs submitted from a `tri-login*` node can never get a GPU. `run.sh submit`
checks `ClusterName` and refuses to run on the wrong one.

`/home` and `/scratch` are shared between both clusters, so there is nothing to
copy — just `ssh trig-login01` and work in the same directory.

## Quick start

```bash
ssh trig-login01
cd /scratch/xuy280/ARKS/ksubspace/transformer-sparsification

./run.sh setup                      # once: build .venv from the /cvmfs wheelhouse
./run.sh download facebook/opt-6.7b  # login node only (needs network)
./run.sh verify   facebook/opt-6.7b  # prove the offline read path works
./run.sh submit   facebook/opt-6.7b  # two-phase job array
./run.sh status                      # squeue + newest log tails
./run.sh collect                     # -> results_summary.csv
```

`./run.sh submit` with no models submits the default set
(opt-6.7b, opt-13b, opt-30b, Llama-3.1-8B).

## Why the stages are split this way

Compute nodes have **no outbound network** and a **read-only `$HOME`/`$PROJECT`**.
Nothing in `final1/` passes `local_files_only=`, so every Hub and `datasets`
access must be served from cache.

* `download` (login node) fetches the model snapshot **and** pre-tokenises every
  corpus into `arks_cache/tokens/`. With the token cache warm, compute nodes
  never call `load_dataset()` at all — which is what makes C4's pinned shard and
  PTB's raw `githubusercontent` URL work offline.
* `verify` re-runs that read path with `HF_HUB_OFFLINE=1` and hard-fails if
  anything is missing. Always run it before submitting.

## The two-phase job array

`fp16` and `int8` share `shared/rank_profile` and `shared/rank_allocation`, and
`arks_pipeline.sh`'s `run_stage()` **moves** a stage directory into `stale/`
whenever its stamp is not complete-and-matching. Two concurrent jobs on the same
model would therefore pull in-progress directories out from under each other.

So `submit` sends two arrays:

* **phase A** — `--experiments fp16`: fetch, export, profile, allocate, fp16
* **phase B** — `--experiments int8` with `--dependency=aftercorr:<A>`: reuses
  the finished shared stages, runs int8 only

`aftercorr` matches task-to-task, so each model's int8 starts as soon as *its own*
fp16 finishes, rather than waiting for the slowest model in the array.

Set `PHASES=fp16` (or `int8`) to submit one phase at a time — required on the
`debug` partition, whose QoS caps a user at a single submitted job.

## Perplexity corpora

`--eval-dataset` takes **one** corpus per invocation, and it feeds only the
*inference* stage's hash. So each worker runs the pipeline once per corpus and
everything upstream reports `[skip]`. Results are copied into `ppl/<corpus>/`
after each pass, because the next pass moves `results/` into `stale/` and the
result rows carry no dataset field.

Recipes reproduce the GPTQ/SparseGPT reference numbers (opt-125m, dense fp16,
seqlen 2048) — see `arks_common.EVAL_RECIPES`:

| corpus | split | assembly | target | measured |
|---|---|---|---|---|
| wikitext2 | test | `"\n\n".join`, full | 27.65 | 27.6509 |
| ptb | test | **stripped** lines, `" ".join`, full | 38.99 | 38.9823 |
| c4 | validation | `[:1100]`, `" ".join`, first 256x2048 | 26.56 | 26.5589 |

PTB's `.strip()` is load-bearing: unstripped lines give 30.22, not 38.98.

## Resource choices, and why

* `--gpus-per-node=1` — Trillium allows **exactly 1 GPU, or a multiple of 4**.
* **Never pass `--mem`.** Trillium rejects it outright; memory is implicit
  (186 GiB per GPU, 745 GiB for a whole node).
* `--cpus-per-task=24` — one quarter of the node's 96 cores, matching one GPU.
* `--workers` is **pinned per model** in `MODEL_TABLE`. `arks_plan.py` sizes the
  compression pool from `/proc/meminfo`, which is not cgroup-aware and reports
  the node's 755 GiB rather than the job's 186 GiB share. On auto it would start
  far too many workers and get OOM-killed.
* `OMP/OPENBLAS/MKL/FLEXIBLAS_NUM_THREADS` are set to `$SLURM_CPUS_PER_TASK`;
  otherwise BLAS sizes itself from the node core count and oversubscribes every
  SVD in the K-subspaces fit loop.
* `TRITON_CACHE_DIR` points at `$SCRATCH`; Llama's rotary kernels JIT through
  Triton, whose default cache is `$HOME/.triton` — read-only on compute nodes.

## Models that do NOT fit

`arks_plan.py:143` hard-errors (`exit 3`) when
`model_bytes_fp16 * 1.15 > vram_total`, before doing any work:

| model | fp16 weights | needs | one H100 has |
|---|---|---|---|
| `facebook/opt-66b` | 122.9 GiB | 141.4 GiB | 79.6 GiB |
| `meta-llama/Llama-3.1-70B` | 131.4 GiB | 151.1 GiB | 79.6 GiB |

`final1/` has no `device_map`, `max_memory`, or `accelerate` dispatch anywhere —
models are loaded with a plain `.to(device)`. So a 4-GPU (320 GiB) allocation
would not help without real model-parallel support. These two are deliberately
absent from `MODEL_TABLE`.

### What running them would actually cost

Extrapolated from the measured opt-30b run (profile 2971s, allocate 11431s,
compress 1976s, three inference passes 8720+8143+6258s = 11.0h total), assuming
a 4-GPU whole node (320 GiB VRAM, 96 cores, 745 GiB RAM, workers capped at 48):

| model | profile | allocate | compress | 3x inference | phase A | phase B |
|---|---|---|---|---|---|---|
| opt-66b | 0.8h | **7.0h** | 0.5h | **14.2h** | 22.4h | 14.7h |
| Llama-3.1-70B | 0.8h | **7.3h** | 0.5h | **14.9h** | 23.5h | 15.4h |

Two blockers, not one:

1. **VRAM** — needs `device_map`/`max_memory` at the three model-load sites plus
   relaxing the single-GPU gate at `arks_plan.py:143`.
2. **Walltime** — phase A lands at 22–23.5h against Trillium's 24h ceiling. That
   margin is not real: node contention was measured at 1.47x on 2026-09-18 when
   three other jobs shared trig0021.

**`allocate` and the inference weight-reconstruction are both single-threaded**
and together account for **87%** of phase A, so more GPUs and more cores do
nothing for them. Parallelising those two loops the way `compress` already does
(`ProcessPoolExecutor` in `bench_compress_minibatch_vq.py:576`) is the change
that would matter — it would cut opt-30b from 11h to roughly 4h as well.

## Gated models (Llama)

`meta-llama/*` is `gated=manual`: it needs both an approved licence request and a
valid token. An `hf_oauth_*` token expires; once expired it makes
`huggingface_hub` send `Authorization:` and get **401 even on public repos**.

```bash
printf %s "hf_YOURTOKEN" > .hf_token && chmod 600 .hf_token
```

`run.sh` reads `.hf_token` (or `$HF_TOKEN`) and ignores anything that looks like
an expired oauth token.

## Fixes applied to final1/

All additive; the `default` recipe preserves the previous behaviour.

| file | fix |
|---|---|
| `arks_common.py` | `EVAL_RECIPES` / `build_eval_text` for per-corpus assembly |
| `arks_common.py` | `Path.with_suffix()` truncated token-cache names at the first `.` — `facebook/opt-6.7b` -> `facebook_opt-6.pt`, so every dotted model id (all Llama-3.x) could cache only one corpus |
| `arks_common.py` | `verification_mode="no_checks"` — a pinned C4 shard otherwise raises `ExpectedMoreSplitsError` |
| `arks_pipeline.sh` | `resolve_dataset` emitted the literal string `"None"` as PTB's config |
| `arks_pipeline.sh` | field-shift: PTB's empty config field vanished under `read` (IFS whitespace collapses runs), sliding the data-files URL into `--dataset-config`. Now newline-delimited |
| `arks_fetch.py` | offline branch — `list_repo_files()` is a live Hub call that cannot work on a compute node |
