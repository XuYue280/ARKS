#!/bin/bash
# =============================================================================
# run.sh -- Trillium (SciNet / Alliance Canada) batch runner for final1/arks_pipeline.sh
#
#   Docs: https://docs.alliancecan.ca/wiki/Trillium_Quickstart
#
# WHERE THIS RUNS
#   Trillium is two Slurm clusters, not one:
#     tri-login0{1..6}  -> cluster "trillium", CPU-only   (GresTypes=(null))
#     trig-login01      -> cluster "grillium", 60 nodes x 4x H100-80GB
#   ARKS needs a GPU, so everything here must be submitted from trig-login01.
#   `submit` refuses to run anywhere else rather than silently queueing a
#   CPU-only job that can never get a GPU.
#
# WHY THE STAGES ARE SPLIT THE WAY THEY ARE
#   Compute nodes have NO outbound network and a read-only $HOME/$PROJECT.
#   Nothing in final1/ passes local_files_only=, so every Hub/datasets access
#   has to be satisfied from cache. `download` (login node, has network) fills
#   both the HF cache and -- more importantly -- the *token* cache, so compute
#   nodes never call load_dataset() at all. `verify` re-checks that offline
#   read path before anything is queued.
#
# THE TWO-PHASE JOB ARRAY
#   fp16 and int8 share shared/rank_profile and shared/rank_allocation inside
#   one run directory, and arks_pipeline.sh's run_stage() *moves* a stage
#   directory to stale/ when its stamp is not complete+matching. Two concurrent
#   jobs on one model would therefore yank in-progress directories out from
#   under each other. So:
#     phase A (array, one task per model): --experiments fp16
#         runs fetch -> export -> profile -> allocate -> fp16 compress+eval
#     phase B (array, one task per model): --experiments int8
#         --dependency=aftercorr:<A>, reuses the completed shared stages
#   That is still "one fp16 job and one int8 job per model", with the expensive
#   shared stages computed exactly once and no race.
#
# PERPLEXITY CORPORA
#   --eval-dataset takes ONE corpus per invocation, and it only feeds the
#   inference stage's hash -- so each worker runs the pipeline once per corpus
#   and only inference re-runs. Results are harvested into ppl/<corpus>/ after
#   each pass, because run_stage() moves the previous results/ into stale/.
#
#   Corpus recipes reproduce the GPTQ/SparseGPT reference numbers
#   (verified on facebook/opt-125m, dense fp16, seqlen 2048):
#       wikitext2  test        "\n\n".join, full          -> 27.6509  (target 27.65)
#       ptb        test        stripped, " ".join, full   -> 38.9823  (target 38.99)
#       c4         validation  [:1100], " ".join, 256x2048 -> 26.5589 (target 26.56)
#
# USAGE
#   ./run.sh setup                 # login node: create .venv
#   ./run.sh download [models...]  # login node: models + tokenised corpora (network)
#   ./run.sh verify   [models...]  # login node: prove the offline read path works
#   ./run.sh smoke                 # submit opt-125m only (correctness gate)
#   ./run.sh submit   [models...]  # submit the two-phase array for MODELS
#   ./run.sh submit --dry-run      # print the manifest + sbatch lines, submit nothing
#   ./run.sh collect               # gather every PPL into results_summary.csv
#   ./run.sh status                # squeue + newest log tails
#   ./run.sh worker M P            # internal: one array task (manifest, phase)
#
# ENVIRONMENT OVERRIDES
#   ACCOUNT PARTITION GPUS_PER_NODE CPUS_PER_TASK MEM ARRAY_THROTTLE
#   OUTPUT_DIR CACHE_DIR SEED HF_HUB_CACHE_DIR HF_DATASETS_CACHE_DIR
# =============================================================================

set -euo pipefail

# Slurm copies the batch script into /var/spool/slurm/slurmd/job<N>/, so
# dirname($0) is the spool directory on a compute node, not the repo.
# SLURM_SUBMIT_DIR survives the `--export=NONE` sbatch wrapper.
if [[ -z "${ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT="$SLURM_SUBMIT_DIR"
  else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
fi
cd "$ROOT"

VENV="$ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
PIPELINE="$ROOT/final1/arks_pipeline.sh"
LOG_DIR="$ROOT/logs"
JOB_DIR="$ROOT/logs/jobs"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/arks_runs}"
CACHE_DIR="${CACHE_DIR:-$ROOT/arks_cache}"
SEED="${SEED:-0}"

# Shared HF caches. Default to the 292 GB of OPT checkpoints already on scratch
# so nothing is re-downloaded. Deliberately NOT $HOME: read-only on compute nodes.
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-/scratch/xuy280/server/transformer-sparsification/.cache/models}"
HF_DATASETS_CACHE_DIR="${HF_DATASETS_CACHE_DIR:-/scratch/xuy280/server/transformer-sparsification/.cache/datasets}"
# A HF_HOME with no stale token file in it: an expired hf_oauth_* token makes
# huggingface_hub send Authorization: and get 401 even on public repos.
HF_HOME_DIR="${HF_HOME_DIR:-$ROOT/.hf_home}"

ACCOUNT="${ACCOUNT:-def-kazem}"
PARTITION="${PARTITION:-compute}"
GPUS_PER_NODE_OVERRIDE="${GPUS_PER_NODE:-}"  # set to force 1 or 4 for every row
GPUS_PER_NODE=1                              # replaced per-manifest in do_submit
CPUS_PER_TASK_OVERRIDE="${CPUS_PER_TASK:-}"  # else 24 per requested GPU
CPUS_PER_TASK=24                             # replaced per-manifest in do_submit
# NOTE: Trillium REJECTS --mem/--mem-per-cpu outright. Memory is implicit:
# 186 GiB per GPU, or the node's 745 GiB for a whole-node job. MEM_GIB is only
# used here to size the compression worker pool, never passed to sbatch.
MEM_GIB="${MEM_GIB:-186}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-}"
# Which phases to submit. The debug QoS caps a user at ONE submitted job
# (MaxSubmitPU=1), so iterate there with PHASES=fp16, then PHASES=int8.
PHASES="${PHASES:-fp16 int8}"
# afterany-chained jobs per phase. A model whose phase A exceeds Trillium's 24h
# ceiling (opt-66b ~22.6h, Llama-3.1-70B ~23.7h before contention) needs more
# than one job; every link but the first runs with RESUME=1, so it continues
# from the stage stamps and skips corpora its predecessors already harvested.
CHAIN_LINKS="${CHAIN_LINKS:-1}"   # raise for models whose phase A exceeds 24h
# MODEL_PARALLEL is deliberately NOT defaulted here: each MODEL_TABLE row
# carries its own model_parallel value, and an unset variable is what lets the
# row win. Export MODEL_PARALLEL=layerwise only to override every row at once.

MODULE_LIST="${MODULE_LIST:-StdEnv/2023 gcc/12.3 arrow python/3.11.5 cuda/12.6}"

# Evaluation corpora: name:split:max_ppl_samples  (0 = whole split)
EVAL_SPECS=("wikitext2:test:0" "ptb:test:0" "c4:validation:256")

# ---------------------------------------------------------------------------
# Model table: model | extra_budget_mb | workers | time_fp16 | time_int8
#
# extra_budget_mb: 100 for OPT, 200 for Llama, as specified. opt-125m keeps the
#   published 1.88 MB -- it is the correctness smoke test, and 1.88 is the
#   setting its reference numbers were produced with.
#
# workers: arks_plan.py sizes the compression pool from /proc/meminfo, which on
#   a shared GPU node reports the NODE's 768 GB rather than this job's cgroup
#   share. Left on auto it would start far too many workers and get OOM-killed,
#   so every row pins it. Budget is per_worker = 32*largest_matrix_elems + 512MB
#   against ~80% of $MEM, capped by (cpus-2).
#
# Deliberately absent -- these cannot run on one H100 and arks_plan.py exits 3
# (model_bytes_fp16 * 1.15 > vram_total) before doing any work:
#   facebook/opt-66b          122.9 GiB fp16 -> needs 141.4 GiB, have 79.6
#   meta-llama/Llama-3.1-70B  131.4 GiB fp16 -> needs 151.1 GiB, have 79.6
# Running those needs real model-parallel support (device_map/max_memory),
# which final1/ does not have today.
# ---------------------------------------------------------------------------
# Walltimes are calibrated from measured runs, not guessed, and deliberately
# kept at or under 12h: Trillium warns that a >12h request schedules later, and
# lowering opt-30b's 20h to 12h moved it from PENDING to RUNNING within seconds.
# Measured phase A: opt-125m 5min, opt-6.7b 2h01. The dominant term for large
# models is `allocate`, whose per-matrix scoring loop is single-threaded and so
# gains nothing from more workers. Every stage is checkpointed, so a timeout
# resumes rather than restarts -- margin here is ~1.5-2x, not 4x.
# columns: model | extra_budget_mb | workers | time_fp16 | time_int8 | model_parallel | gpus
#
# model_parallel=layerwise streams one decoder layer to the GPU at a time for
# both the perplexity evaluation and the rank allocator's backward. Verified
# bit-for-bit identical to the all-on-GPU path on opt-125m -- all three corpora
# for inference, and all 72 importance scores for the allocator.
#
# The layerwise rows still ask for 4 GPUs, but for HOST RAM, not VRAM: the
# model sits on the host (123/131 GiB) and --offload-gradients parks every
# gradient there too (122/128 GiB), so the job needs ~245/260 GiB. A 1-GPU
# share is 186 GiB; the whole node is 745 GiB.
# $MODEL_PARALLEL overrides every row.
MODEL_TABLE=(
  "facebook/opt-125m|1.88|16|01:00:00|01:00:00|none|1"
  # Added for the baseline comparison table. budget follows the documented
  # rule (100 OPT / 200 Llama), so these land at a HIGHER rho than the
  # 0.60 the baselines are run at -- a fixed 100 MB is 15.8% of opt-350m
  # but 0.79% of opt-6.7b. Same effect already visible on Llama-3.2-1B
  # (8.46% -> rho 0.832). Flag it wherever these points are plotted.
  # workers=10 sized for CPUS_PER_TASK=12 (the pool is capped at cpus-2);
  # walltimes are the size interpolation x1.5 for that smaller pool.
  "facebook/opt-350m|100.0|10|02:00:00|02:00:00|none|1"
  "facebook/opt-1.3b|100.0|10|03:00:00|03:00:00|none|1"
  "facebook/opt-2.7b|100.0|10|04:00:00|03:00:00|none|1"
  "meta-llama/Llama-3.2-3B|200.0|10|04:00:00|03:00:00|none|1"
  # Small enough that the whole pipeline fits the debug partition's 2 h cap
  # (measured scaling off Llama-3.1-8B: ~20 min fp16, ~13 min int8). Submit it
  # with PARTITION=debug -- that queue is near-empty while `compute` is days
  # deep -- one phase at a time, since the debug QoS is MaxSubmitPU=1.
  "meta-llama/Llama-3.2-1B|200.0|22|02:00:00|02:00:00|none|1"
  "facebook/opt-6.7b|100.0|22|05:00:00|03:00:00|none|1"
  "facebook/opt-13b|100.0|22|07:00:00|04:00:00|none|1"
  "facebook/opt-30b|100.0|20|12:00:00|08:00:00|none|1"
  "meta-llama/Llama-3.1-8B|200.0|22|05:00:00|03:00:00|none|1"
  "facebook/opt-66b|100.0|48|20:00:00|14:00:00|layerwise|4"
  "meta-llama/Llama-3.1-70B|200.0|48|20:00:00|14:00:00|layerwise|4"
)
DEFAULT_MODELS=("facebook/opt-6.7b" "facebook/opt-13b" "facebook/opt-30b" "meta-llama/Llama-3.1-8B")

# ---------------------------------------------------------------------------

log() { echo "[run.sh] $*"; }
die() { echo "[run.sh] ERROR: $*" >&2; exit 1; }

table_row() {
  local model="$1" row
  for row in "${MODEL_TABLE[@]}"; do
    [[ "${row%%|*}" == "$model" ]] && { echo "$row"; return 0; }
  done
  return 1
}

load_cluster_modules() {
  if type module &>/dev/null; then
    # gcc before arrow: arrow is compiler-dependent and will not resolve
    # without a toolchain loaded first.
    # shellcheck disable=SC2086
    module load $MODULE_LIST
  fi
}

# Environment shared by login-node prep and compute-node workers.
common_env() {
  mkdir -p "$HF_HOME_DIR" "$CACHE_DIR/tokens" "$OUTPUT_DIR" "$LOG_DIR" "$JOB_DIR"
  export HF_HOME="$HF_HOME_DIR"
  export HF_HUB_CACHE="$HF_HUB_CACHE_DIR"
  export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR"
  export HF_DATASETS_CACHE="$HF_DATASETS_CACHE_DIR"
  export ARKS_TOKEN_CACHE="$CACHE_DIR/tokens"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
  # Llama routes rotary embeddings through a Triton JIT kernel whose default
  # cache is $HOME/.triton -- read-only on compute nodes.
  export TRITON_CACHE_DIR="$ROOT/.triton_cache"
  mkdir -p "$TRITON_CACHE_DIR"
  # Gated repos (meta-llama/*) need a token. Prefer an explicit HF_TOKEN; else
  # fall back to a token file, but only a real one -- an expired hf_oauth_*
  # makes huggingface_hub send Authorization: and get 401 even on public repos.
  if [[ -z "${HF_TOKEN:-}" && -r "$ROOT/.hf_token" ]]; then
    HF_TOKEN=$(tr -d " \t\n\r" < "$ROOT/.hf_token")
    [[ "$HF_TOKEN" == hf_* && "$HF_TOKEN" != hf_oauth_* ]] && export HF_TOKEN || unset HF_TOKEN
  fi
}

ensure_python() {
  [[ -x "$PYTHON_BIN" ]] || die "no interpreter at $PYTHON_BIN -- run: $0 setup"
}

models_from_args() {
  if (( $# )); then printf '%s\n' "$@"; else printf '%s\n' "${DEFAULT_MODELS[@]}"; fi
}

# ---- setup ----------------------------------------------------------------

do_setup() {
  load_cluster_modules
  log "creating venv at $VENV"
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  # --no-index: install from the Alliance wheelhouse on /cvmfs, not PyPI.
  # transformers is pinned to the 4.x line on purpose: final1/ calls
  # from_pretrained(torch_dtype=...) and walks model.model.decoder.layers,
  # both of which changed in transformers 5.
  "$VENV/bin/pip" install --no-index --upgrade pip setuptools wheel
  "$VENV/bin/pip" install --no-index \
    "torch==2.14.0" "numpy<3" scipy scikit_learn safetensors sentencepiece zstandard \
    "transformers==4.57.6" "datasets==5.0.0" "accelerate==1.14.0" pandas joblib
  log "venv ready"
  "$VENV/bin/python" -c "import torch,transformers,datasets,sklearn;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('transformers',transformers.__version__)"
}

# ---- download (login node, needs network) ---------------------------------

do_download() {
  load_cluster_modules; ensure_python; common_env
  local models; mapfile -t models < <(models_from_args "$@")
  local ds; ds=$(printf '%s\n' "${EVAL_SPECS[@]}" | cut -d: -f1 | paste -sd,)
  log "downloading ${#models[@]} model(s); corpora: $ds"
  local m rc=0
  for m in "${models[@]}"; do
    log "--- $m ---"
    "$PYTHON_BIN" "$ROOT/tools/prewarm_caches.py" --model "$m" --datasets "$ds" || rc=1
  done
  (( rc == 0 )) || die "one or more downloads failed (gated Llama needs a valid HF_TOKEN)"
  log "download complete"
}

# ---- verify (login node: prove the offline path works) --------------------

do_verify() {
  load_cluster_modules; ensure_python; common_env
  local models; mapfile -t models < <(models_from_args "$@")
  log "offline read-path check for ${#models[@]} model(s)"
  # Exactly the env a compute node gets.
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  ARKS_ROOT="$ROOT" \
  MODELS="$(printf '%s\n' "${models[@]}")" \
  EVAL_SPECS_ENV="${EVAL_SPECS[*]}" \
  "$PYTHON_BIN" - <<'PY'
import argparse, os, sys
sys.path.insert(0, os.path.join(os.environ["ARKS_ROOT"], "final1"))
from transformers import AutoConfig, AutoTokenizer
from bench_inference_minibatch_vq import test_loader, resolve_dataset
from arks_common import cached_token_ids, dataset_text_column, DATASET_PRESETS

models = [m for m in os.environ["MODELS"].splitlines() if m.strip()]
specs = [s.split(":") for s in os.environ["EVAL_SPECS_ENV"].split()]
RECIPE = {"wikitext2": "wikitext2", "ptb": "ptb", "c4": "c4"}
ok = True
for m in models:
    try:
        AutoConfig.from_pretrained(m)
        tok = AutoTokenizer.from_pretrained(m, use_fast=False)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        print(f"[verify] OK   model={m}")
    except Exception as e:
        ok = False; print(f"[verify] FAIL model={m}: {e}"); continue
    for name, split, _cap in specs:
        ns = argparse.Namespace(dataset=name, dataset_config=None, dataset_split=split,
                                dataset_text_column="auto", dataset_data_files=None,
                                dataset_recipe=RECIPE.get(name, "default"))
        try:
            test_loader(tok, 2048, 1, 1, m, **resolve_dataset(ns))
            print(f"[verify] OK   eval  {m} {name}/{split}")
        except Exception as e:
            ok = False; print(f"[verify] FAIL eval  {m} {name}/{split}: {e}")
    p = DATASET_PRESETS["wikitext2"]
    try:
        cached_token_ids(tok, model_id=m, dataset_name=p["name"], dataset_config=p["config"],
                         split="train", variant="join_all",
                         build_text=lambda ds: "\n\n".join(ds[dataset_text_column(ds, "auto")]),
                         data_files=None)
        print(f"[verify] OK   calib {m} wikitext2/train")
    except Exception as e:
        ok = False; print(f"[verify] FAIL calib {m}: {e}")
if not ok:
    print("[verify] FAILED -- run ./run.sh download before submitting"); sys.exit(1)
print("[verify] all offline checks passed")
PY
}

# ---- worker (compute node) ------------------------------------------------

do_worker() {
  local manifest="$1" phase="$2"
  # Third argument, not an env var: Trillium wraps sbatch with --export=NONE,
  # so only the script's own arguments survive to the compute node.
  [[ "${3:-}" == "resume" ]] && export RESUME=1
  [[ -f "$manifest" ]] || die "manifest not readable: $manifest"
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "SLURM_ARRAY_TASK_ID unset (submit via $0 submit)"

  load_cluster_modules; ensure_python; common_env

  # Compute nodes have no network. With the token caches warm (see download),
  # nothing below should need either of these -- they are here so a cache miss
  # fails loudly and immediately instead of hanging on a connect timeout.
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1

  # flexiblas/OpenMP size themselves from the node's core count, not this job's
  # cgroup share, and oversubscribe every SVD in the K-subspaces fit loop.
  local threads="${SLURM_CPUS_PER_TASK:-1}"
  export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads" FLEXIBLAS_NUM_THREADS="$threads"

  local line; line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")
  [[ -n "$line" ]] || die "no manifest line at index $SLURM_ARRAY_TASK_ID"
  local model budget workers _ta _tb row_parallel _gpus
  IFS='|' read -r model budget workers _ta _tb row_parallel _gpus <<< "$line"
  # $MODEL_PARALLEL (if the submitter exported one) wins over the table row.
  local parallel="${MODEL_PARALLEL:-${row_parallel:-none}}"

  local experiment expdir
  case "$phase" in
    fp16) experiment=fp16; expdir=adaptive_baseline_fp16 ;;
    int8) experiment=int8; expdir=adaptive_baseline_int8 ;;
    *) die "unknown phase: $phase" ;;
  esac

  local tag="${model##*/}"
  local run_dir="$OUTPUT_DIR/$tag/seed_${SEED}"

  echo "=========================================================="
  echo "[worker] task=$SLURM_ARRAY_TASK_ID host=$(hostname) phase=$phase"
  echo "[worker] model=$model budget=${budget}MB workers=$workers threads=$threads model-parallel=$parallel"
  echo "[worker] run_dir=$run_dir"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo "=========================================================="

  local spec name split cap rc=0
  for spec in "${EVAL_SPECS[@]}"; do
    IFS=: read -r name split cap <<< "$spec"
    echo
    if [[ "${RESUME:-0}" == "1" ]] && harvested "$run_dir" "$expdir" "$name"; then
      echo "### [$phase] $model -- eval corpus: $name ALREADY HARVESTED, skipping"
      continue
    fi
    echo "### [$phase] $model -- eval corpus: $name (split=$split, max_ppl_samples=$cap)"
    # Only the inference stage hashes the eval corpus, so corpora 2 and 3 reuse
    # the profile/allocate/compress artifacts and just re-run inference.
    if "$PIPELINE" \
        --model "$model" \
        --experiments "$experiment" \
        --extra-budget-mb "$budget" \
        --calibration-dataset wikitext2 \
        --eval-dataset "$name" --eval-split "$split" --max-ppl-samples "$cap" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" --cache-dir "$CACHE_DIR" \
        --device cuda --workers "$workers" \
        --model-parallel "$parallel" \
        --python "$PYTHON_BIN"; then
      harvest_ppl "$run_dir" "$expdir" "$name"
    else
      rc=1
      echo "[worker] FAILED on corpus $name (continuing to the next corpus)"
    fi
  done
  return $rc
}

# True when this corpus already has a usable harvested result. A model whose
# phase A does not fit one job's walltime is submitted as a chain of jobs on
# afterany, and each link skips what its predecessors already produced -- the
# pipeline's own stage stamps cover everything upstream of inference, but not
# which corpus an inference was for.
harvested() {
  local dir="$1/ppl/$3/$2"
  [[ -d "$dir" ]] || return 1
  "$PYTHON_BIN" - "$dir" <<'PY'
import glob, json, sys
for path in glob.glob(sys.argv[1] + "/inference_*.json"):
    try:
        data = json.load(open(path))
    except Exception:
        continue
    for row in (data if isinstance(data, list) else [data]):
        if isinstance(row, dict) and row.get("ppl") is not None:
            raise SystemExit(0)
raise SystemExit(1)
PY
}

# run_stage() moves the previous results/ into stale/ when the next corpus
# runs, and the result rows carry no dataset field -- so snapshot each corpus's
# output under ppl/<corpus>/ while it is still the live one.
harvest_ppl() {
  local run_dir="$1" expdir="$2" corpus="$3"
  local src="$run_dir/$expdir/results" dst="$run_dir/ppl/$corpus/$expdir"
  [[ -d "$src" ]] || { echo "[harvest] no results dir at $src"; return 0; }
  # Replace, do not accumulate: inference_<timestamp>.json is a new filename on
  # every run, so copying into a dirty destination leaves results from earlier
  # (possibly failed) attempts behind and collect() counts them all.
  rm -rf "$dst"
  mkdir -p "$dst"
  cp -a "$src"/. "$dst"/ 2>/dev/null || true
  echo "[harvest] $corpus -> $dst"
}

# ---- submit ---------------------------------------------------------------

do_submit() {
  local dry=0
  [[ "${1:-}" == "--dry-run" ]] && { dry=1; shift; }

  local cluster; cluster=$(scontrol show config 2>/dev/null | awk -F= '/^ClusterName/{gsub(/ /,"",$2);print $2}')
  if [[ "$cluster" != "grillium" ]]; then
    die "this is cluster '${cluster:-unknown}', which has no GPUs.
       ARKS needs an H100. Submit from the GPU login node instead:
         ssh trig-login01
         cd $ROOT && ./run.sh submit ${*:-}"
  fi

  ensure_python; common_env
  [[ "$ROOT" == /scratch/* ]] || log "WARNING: $ROOT is not under /scratch (compute nodes cannot write \$HOME)"

  local models; mapfile -t models < <(models_from_args "$@")
  local manifest="$JOB_DIR/manifest_$(date +%Y%m%d_%H%M%S)_$$.tsv"
  : > "$manifest"
  local m row
  for m in "${models[@]}"; do
    row=$(table_row "$m") || die "model not in MODEL_TABLE: $m"
    echo "$row" >> "$manifest"
  done
  local n; n=$(wc -l < "$manifest")
  (( n > 0 )) || die "nothing to submit"

  log "manifest ($n task(s)) -> $manifest"
  sed 's/^/[run.sh]   /' "$manifest"

  # Per-array walltime is one number, so take the longest row in each phase.
  local time_a time_b
  time_a=$(cut -d'|' -f4 "$manifest" | sort -r | head -1)
  time_b=$(cut -d'|' -f5 "$manifest" | sort -r | head -1)
  # One --gpus-per-node per array, so a mixed manifest takes the larger ask.
  # Trillium allows exactly 1 or a multiple of 4, so this is 1 or 4.
  local row_gpus
  row_gpus=$(cut -d'|' -f7 "$manifest" | sort -rn | head -1)
  GPUS_PER_NODE="${GPUS_PER_NODE_OVERRIDE:-${row_gpus:-1}}"
  # 24 cores per GPU share; a 4-GPU ask is the whole 96-core node.
  CPUS_PER_TASK="${CPUS_PER_TASK_OVERRIDE:-$((24 * GPUS_PER_NODE))}"

  if [[ "$GPUS_PER_NODE" != "1" && $((GPUS_PER_NODE % 4)) -ne 0 ]]; then
    die "GPUS_PER_NODE=$GPUS_PER_NODE: Trillium allows exactly 1 or a multiple of 4"
  fi

  local spec="0-$((n - 1))"
  [[ -n "$ARRAY_THROTTLE" ]] && spec="${spec}%${ARRAY_THROTTLE}"

  # --export is deliberately not used for configuration: Trillium wraps sbatch
  # with `--export=NONE --get-user-env`, so the manifest path and phase are
  # passed as script arguments, which always survive.
  build_cmd() {
    local phase="$1" tl="$2"; shift 2
    # Trillium rejects an explicit --partition on a whole-node (4-GPU) job:
    # "Requested partition compute does not support whole node jobs". Omitting
    # it lets the scheduler route to compute_full_node itself.
    local -a part=()
    (( GPUS_PER_NODE < 4 )) && part=(--partition="$PARTITION")
    printf '%s\n' sbatch --parsable \
      --job-name="arks-$phase" --account="$ACCOUNT" "${part[@]}" \
      --array="$spec" --nodes=1 --gpus-per-node="$GPUS_PER_NODE" \
      --cpus-per-task="$CPUS_PER_TASK" --time="$tl" \
      --output="$LOG_DIR/arks-$phase-%A_%a.out" --error="$LOG_DIR/arks-$phase-%A_%a.err" \
      "$@" "$ROOT/run.sh" worker "$manifest" "$phase" ${RESUME_ARG:+"$RESUME_ARG"}
  }

  local want_a=0 want_b=0 ph
  for ph in $PHASES; do
    case "$ph" in fp16) want_a=1 ;; int8) want_b=1 ;; *) die "bad PHASES entry: $ph" ;; esac
  done

  # submit_chain <phase> <timelimit> [dependency...] -> echoes the last job id.
  # Link 1 runs fresh; links 2..CHAIN_LINKS wait on afterany (not afterok --
  # the whole point is to pick up after a walltime kill) and run with resume.
  submit_chain() {
    local phase="$1" tl="$2"; shift 2
    local -a dep=("$@")
    local last="" link cmd
    for (( link = 1; link <= CHAIN_LINKS; link++ )); do
      RESUME_ARG=""
      (( link > 1 )) && RESUME_ARG="resume"
      local -a this_dep=("${dep[@]}")
      (( link > 1 )) && this_dep=(--dependency="afterany:$last")
      mapfile -t cmd < <(build_cmd "$phase" "$tl" "${this_dep[@]}")
      log "phase $phase link $link/$CHAIN_LINKS: ${cmd[*]}" >&2
      if (( dry )); then last="<${phase}${link}>"; else
        last=$("${cmd[@]}")
        log "  submitted $last${this_dep[*]:+ (${this_dep[*]})}" >&2
      fi
    done
    RESUME_ARG=""
    echo "$last"
  }

  local jid_a="" jid_b=""
  if (( want_a )); then
    jid_a=$(submit_chain fp16 "$time_a" | tail -1)
  fi

  if (( want_b )); then
    # aftercorr: int8 task i starts as soon as fp16 task i finishes, so models
    # pipeline independently instead of waiting for the slowest fp16 task.
    # aftercorr requires the fp16 task to *succeed*. That is right for a
    # single job, but wrong for a chain, where the last link timing out is the
    # expected way to run out of walltime -- int8 would be cancelled for a
    # reason that is not a failure. int8 needs nothing from fp16 anyway (only
    # the shared profile/allocate stages), so a chain waits on afterany.
    local kind="aftercorr"
    (( CHAIN_LINKS > 1 )) && kind="afterany"
    local dep=()
    [[ -n "$jid_a" ]] && dep=(--dependency="$kind:$jid_a")
    (( dry )) && [[ -z "$jid_a" ]] && dep=(--dependency="$kind:<A>")
    jid_b=$(submit_chain int8 "$time_b" "${dep[@]}" | tail -1)
  fi

  if (( dry )); then
    log "dry run -- nothing submitted (manifest kept at $manifest)"
    return 0
  fi

  echo "$jid_a $jid_b $manifest" >> "$JOB_DIR/submitted.log"
  log "logs: $LOG_DIR/arks-{fp16,int8}-<jobid>_<task>.out"
  log "watch: squeue -u \$USER    collect: $0 collect"
}

# ---- collect --------------------------------------------------------------

do_collect() {
  ensure_python
  local out="$ROOT/results_summary.csv"
  OUTPUT_DIR="$OUTPUT_DIR" OUT="$out" "$PYTHON_BIN" - <<'PY'
import csv, json, os, glob
root, out = os.environ["OUTPUT_DIR"], os.environ["OUT"]
rows = []
for jf in sorted(glob.glob(os.path.join(root, "*", "seed_*", "ppl", "*", "*", "inference_*.json"))):
    parts = jf.split(os.sep)
    corpus, expdir = parts[-3], parts[-2]
    model, seed = parts[-6], parts[-5]
    try:
        data = json.load(open(jf))
    except Exception:
        continue
    for rec in (data if isinstance(data, list) else [data]):
        if not isinstance(rec, dict) or rec.get("ppl") is None:
            continue
        rows.append({
            "model": model, "seed": seed, "experiment": expdir, "corpus": corpus,
            "ppl": rec.get("ppl"), "dense_ppl": rec.get("dense_ppl"),
            "ppl_increase_percent": rec.get("ppl_increase_percent"),
            "ppl_tokens": rec.get("ppl_tokens"), "source": jf,
        })
if not rows:
    print("no perplexity rows found under", root); raise SystemExit(0)
rows.sort(key=lambda r: (r["model"], r["experiment"], r["corpus"]))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} row(s) -> {out}\n")
print(f"{'model':<22}{'experiment':<26}{'corpus':<12}{'ppl':>10}{'dense':>10}")
for r in rows:
    d = r["dense_ppl"]
    print(f"{r['model']:<22}{r['experiment']:<26}{r['corpus']:<12}{r['ppl']:>10.4f}"
          f"{(f'{d:.4f}' if isinstance(d,(int,float)) else '-'):>10}")
PY
}

# ---- status ---------------------------------------------------------------

do_status() {
  squeue -u "$USER" -o "%.12i %.14j %.9T %.11M %.11l %.6D %R" || true
  echo
  local f
  for f in $(ls -t "$LOG_DIR"/arks-*.out 2>/dev/null | head -4); do
    echo "===== $f"; tail -n 12 "$f"; echo
  done
}

# ---------------------------------------------------------------------------

cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)    do_setup ;;
  download) do_download "$@" ;;
  verify)   do_verify "$@" ;;
  submit)   do_submit "$@" ;;
  smoke)    do_submit "${@}" facebook/opt-125m ;;
  worker)   do_worker "$@" ;;
  collect)  do_collect ;;
  status)   do_status ;;
  help|-h|--help) sed -n '2,60p' "$0" | sed 's/^# \?//' ;;
  *) die "unknown command: $cmd (try: $0 help)" ;;
esac
