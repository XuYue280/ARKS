#!/bin/bash
# =============================================================================
# SeedLM OPT benchmark — Trillium / Alliance Canada SLURM runner
#
# Stages (dependency chain):
#   1) store          — one job per model (write original_model/*.npy)
#   2) compress_mid   — s2/s3/s4/s5a param jobs (after store)
#   3) s5b_prep       — reconstruct quant weights (after s2 / compress_mid)
#   4) s5b            — SeedLM+quant param jobs (after s5b_prep)
#   5) load           — inference/PPL param jobs (after all compress)
#   6) merge          — merge results/server_temp shards -> results/{model}/...
#
# Globals (OPT_MODELS, DEFAULT_SPARSITIES, K_CLUSTER_TILE_FRACTION,
# QUANTIZATION_BITS) are read from SeedLM/bench_compress_and_store.py.
#
# Example usage:
#   ./run.sh setup
#       Create .venv and install requirements.txt
#
#   ./run.sh submit
#       Generate job lists + sbatch all stages with dependencies
#
#   ./run.sh submit --dry-run
#       Print planned sbatch commands without submitting
#
#   ./run.sh gen-jobs
#       Only write job JSON lists under results/server_temp/jobs/
#
#   ./run.sh merge
#       Merge server_temp compress + inference shards into final JSON
#
#   squeue -u $USER
#   scancel <jobid>
#
# Environment overrides:
#   ROOT, ACCOUNT, PARTITION, QOS, PYTHON_BIN, HF_SCRATCH_CACHE
#   TIME_STORE / TIME_COMPRESS / TIME_LOAD / TIME_MERGE
#   MEM_STORE / MEM_COMPRESS / MEM_LOAD
# =============================================================================

#SBATCH --job-name=seedlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100g
#SBATCH --time=23:59:00
#SBATCH --output=logs/seedlm-%x-%A-%a.out
#SBATCH --error=logs/seedlm-%x-%A-%a.err
# Optional Trillium / Alliance overrides (set via submit flags or env):
# #SBATCH --account=def-xxxx
# #SBATCH --partition=compute
# #SBATCH --qos=normal
# Load jobs may request a GPU (see submit_array for load phase).
module load StdEnv/2023
module load gcc/12.3
module load arrow

set -euo pipefail

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
JOBS_DIR="$ROOT/results/server_temp/jobs"
LOG_DIR="$ROOT/logs"
COMPRESS_PY="$ROOT/SeedLM/bench_compress_and_store.py"
LOAD_PY="$ROOT/SeedLM/bench_load_and_inference.py"
SLURM_JOBS_PY="$ROOT/SeedLM/slurm_jobs.py"

TIME_STORE="${TIME_STORE:-23:59:00}"
TIME_COMPRESS="${TIME_COMPRESS:-23:59:00}"
TIME_LOAD="${TIME_LOAD:-23:59:00}"
TIME_MERGE="${TIME_MERGE:-23:59:00}"
MEM_STORE="${MEM_STORE:-64G}"
MEM_COMPRESS="${MEM_COMPRESS:-64G}"
MEM_LOAD="${MEM_LOAD:-128G}"
# Nibi requires an explicit GPU type (cannot use bare --gres=gpu:1).
GPU_TYPE="${GPU_TYPE:-h100}"
GPU_COUNT="${GPU_COUNT:-1}"

load_cluster_modules() {
  if type module &>/dev/null; then
    module load StdEnv/2023 python gcc cuda 2>/dev/null || true
    module load arrow 2>/dev/null || true
  fi
}

setup_venv() {
  echo "[setup] creating venv at $VENV"
  load_cluster_modules
  python3 -m venv "$VENV"
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r "$ROOT/requirements.txt"
  echo "[setup] done — activate with: source $VENV/bin/activate"
}

ensure_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN"
    echo "Run first: $ROOT/run.sh setup"
    exit 1
  fi
}

setup_hf_cache() {
  local cache_root="${HF_SCRATCH_CACHE:-${SCRATCH:-$ROOT}/.cache/huggingface}"
  mkdir -p "$cache_root/hub" "$cache_root/datasets"
  export HF_HOME="$cache_root"
  export HF_HUB_CACHE="$cache_root/hub"
  export HUGGINGFACE_HUB_CACHE="$cache_root/hub"
  export TRANSFORMERS_CACHE="$cache_root"
  export HF_DATASETS_CACHE="$cache_root/datasets"
  echo "[run] HF cache=$cache_root"
}

gen_jobs() {
  load_cluster_modules
  ensure_python
  mkdir -p "$JOBS_DIR"
  "$PYTHON_BIN" "$SLURM_JOBS_PY" gen store --out "$JOBS_DIR/store.json"
  "$PYTHON_BIN" "$SLURM_JOBS_PY" gen compress_mid --out "$JOBS_DIR/compress_mid.json"
  "$PYTHON_BIN" "$SLURM_JOBS_PY" gen s5b_prep --out "$JOBS_DIR/s5b_prep.json"
  "$PYTHON_BIN" "$SLURM_JOBS_PY" gen s5b --out "$JOBS_DIR/s5b.json"
  "$PYTHON_BIN" "$SLURM_JOBS_PY" gen load --out "$JOBS_DIR/load.json"
}

job_count() {
  "$PYTHON_BIN" -c "import json; print(len(json.load(open('$1'))))"
}

run_worker() {
  # Invoked as: PHASE=store|compress_mid|s5b_prep|s5b|load run.sh worker
  local phase="${SEEDLM_PHASE:-}"
  if [[ -z "$phase" ]]; then
    echo "SEEDLM_PHASE is required for worker"
    exit 1
  fi
  local jobs_file="$JOBS_DIR/${phase}.json"
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID missing"
    exit 1
  fi
  load_cluster_modules
  setup_hf_cache
  ensure_python
  mkdir -p "$LOG_DIR"

  local argv
  if [[ "$phase" == "load" ]]; then
    argv=$("$PYTHON_BIN" "$SLURM_JOBS_PY" job-argv --jobs "$jobs_file" --index "$SLURM_ARRAY_TASK_ID" --stage load)
    echo "[worker] load $argv"
    # shellcheck disable=SC2086
    exec "$PYTHON_BIN" "$LOAD_PY" $argv ${SEEDLM_EXTRA_ARGS:-}
  else
    argv=$("$PYTHON_BIN" "$SLURM_JOBS_PY" job-argv --jobs "$jobs_file" --index "$SLURM_ARRAY_TASK_ID" --stage compress)
    echo "[worker] compress/$phase $argv"
    # shellcheck disable=SC2086
    exec "$PYTHON_BIN" "$COMPRESS_PY" $argv ${SEEDLM_EXTRA_ARGS:-}
  fi
}

submit_array() {
  local phase="$1"
  local jobs_file="$2"
  local time_lim="$3"
  local mem="$4"
  local dep="${5:-}"
  local extra_sbatch="${6:-}"

  local n
  n=$(job_count "$jobs_file")
  if (( n <= 0 )); then
    echo "[submit] skip empty phase=$phase"
    echo ""
    return 0
  fi
  local last=$((n - 1))
  local cmd=(sbatch --parsable
    --job-name="seedlm-${phase}"
    --array="0-${last}"
    --time="$time_lim"
    --mem="$mem"
    --export=ALL,SEEDLM_PHASE="${phase}",SEEDLM_EXTRA_ARGS="${SEEDLM_EXTRA_ARGS:-}"
    --output="$LOG_DIR/seedlm-${phase}-%A-%a.out"
    --error="$LOG_DIR/seedlm-${phase}-%A-%a.err"
  )
  if [[ -n "${ACCOUNT:-}" ]]; then cmd+=(--account="$ACCOUNT"); fi
  if [[ -n "${PARTITION:-}" ]]; then cmd+=(--partition="$PARTITION"); fi
  if [[ -n "${QOS:-}" ]]; then cmd+=(--qos="$QOS"); fi
  if [[ -n "$dep" ]]; then cmd+=(--dependency="$dep"); fi
  # shellcheck disable=SC2206
  if [[ -n "$extra_sbatch" ]]; then cmd+=($extra_sbatch); fi
  cmd+=("$ROOT/run.sh" worker)

  echo "[submit] ${cmd[*]}" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN_JOBID_${phase}"
    return 0
  fi
  "${cmd[@]}"
}

do_submit() {
  load_cluster_modules
  ensure_python
  mkdir -p "$LOG_DIR" "$JOBS_DIR"
  gen_jobs

  local id_store id_mid id_prep id_s5b id_load
  id_store=$(submit_array store "$JOBS_DIR/store.json" "$TIME_STORE" "$MEM_STORE")
  echo "[submit] store job=$id_store" >&2

  local dep_store=""
  if [[ -n "$id_store" && "$id_store" != DRY_RUN* ]]; then
    dep_store="afterok:${id_store}"
  fi
  id_mid=$(submit_array compress_mid "$JOBS_DIR/compress_mid.json" "$TIME_COMPRESS" "$MEM_COMPRESS" "$dep_store")
  echo "[submit] compress_mid job=$id_mid" >&2

  local dep_mid=""
  if [[ -n "$id_mid" && "$id_mid" != DRY_RUN* ]]; then
    dep_mid="afterok:${id_mid}"
  elif [[ -n "$dep_store" ]]; then
    dep_mid="$dep_store"
  fi
  id_prep=$(submit_array s5b_prep "$JOBS_DIR/s5b_prep.json" "$TIME_COMPRESS" "$MEM_COMPRESS" "$dep_mid")
  echo "[submit] s5b_prep job=$id_prep" >&2

  local dep_prep=""
  if [[ -n "$id_prep" && "$id_prep" != DRY_RUN* ]]; then
    dep_prep="afterok:${id_prep}"
  elif [[ -n "$dep_mid" ]]; then
    dep_prep="$dep_mid"
  fi
  id_s5b=$(submit_array s5b "$JOBS_DIR/s5b.json" "$TIME_COMPRESS" "$MEM_COMPRESS" "$dep_prep")
  echo "[submit] s5b job=$id_s5b" >&2

  local dep_s5b=""
  if [[ -n "$id_s5b" && "$id_s5b" != DRY_RUN* ]]; then
    dep_s5b="afterok:${id_s5b}"
  elif [[ -n "$dep_prep" ]]; then
    dep_s5b="$dep_prep"
  fi
  # Load may use 1 GPU for evaluate_ppl only (compress is CPU).
  # Nibi: must name the GPU type, e.g. --gres=gpu:h100:1
  id_load=$(submit_array load "$JOBS_DIR/load.json" "$TIME_LOAD" "$MEM_LOAD" "$dep_s5b" "--gres=gpu:${GPU_TYPE}:${GPU_COUNT}")
  echo "[submit] load job=$id_load" >&2

  local dep_load=""
  if [[ -n "$id_load" && "$id_load" != DRY_RUN* ]]; then
    dep_load="afterok:${id_load}"
  elif [[ -n "$dep_s5b" ]]; then
    dep_load="$dep_s5b"
  fi

  local merge_cmd=(sbatch --parsable
    --job-name="seedlm-merge"
    --time="$TIME_MERGE"
    --mem=8G
    --export=ALL
    --output="$LOG_DIR/seedlm-merge-%j.out"
    --error="$LOG_DIR/seedlm-merge-%j.err"
  )
  if [[ -n "${ACCOUNT:-}" ]]; then merge_cmd+=(--account="$ACCOUNT"); fi
  if [[ -n "${PARTITION:-}" ]]; then merge_cmd+=(--partition="$PARTITION"); fi
  if [[ -n "${QOS:-}" ]]; then merge_cmd+=(--qos="$QOS"); fi
  if [[ -n "$dep_load" ]]; then merge_cmd+=(--dependency="$dep_load"); fi
  merge_cmd+=("$ROOT/run.sh" merge)

  echo "[submit] ${merge_cmd[*]}" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[submit] dry-run complete" >&2
    return 0
  fi
  local id_merge
  id_merge=$("${merge_cmd[@]}")
  echo "[submit] merge job=$id_merge" >&2
  echo "[submit] done. Monitor with: squeue -u \$USER" >&2
}

do_merge() {
  ensure_python
  "$PYTHON_BIN" "$SLURM_JOBS_PY" merge compress
  "$PYTHON_BIN" "$SLURM_JOBS_PY" merge inference
}

usage() {
  sed -n '1,40p' "$0" | sed 's/^# \\?//'
}

cmd="${1:-}"
case "$cmd" in
  setup)
    setup_venv
    ;;
  gen-jobs)
    gen_jobs
    ;;
  submit)
    shift || true
    if [[ "${1:-}" == "--dry-run" ]]; then
      DRY_RUN=1
    fi
    do_submit
    ;;
  worker)
    run_worker
    ;;
  merge)
    do_merge
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac
