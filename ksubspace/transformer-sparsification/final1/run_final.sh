#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-facebook/opt-125m}"
MODEL_TAG="${MODEL_TAG:-${MODEL##*/}}"
INPUT_DIR="${INPUT_DIR:-SeedLM/model/${MODEL_TAG}}"

RUN_ADAPTIVE_FP16="${RUN_ADAPTIVE_FP16:-1}"
RUN_ADAPTIVE_INT8="${RUN_ADAPTIVE_INT8:-1}"
RUN_VQ_NOCALIB="${RUN_VQ_NOCALIB:-1}"
RUN_VQ_CALIBRATED="${RUN_VQ_CALIBRATED:-1}"
RUN_ADAPTIVE_SPARSITY="${RUN_ADAPTIVE_SPARSITY:-1}"

K="${K:-512}"
BASE_RANK="${BASE_RANK:-4}"
MAX_RANK="${MAX_RANK:-6}"
EXTRA_BUDGET_MB="${EXTRA_BUDGET_MB:-1.88}"
C="${C:-8}"
TILE_WIDTH="${TILE_WIDTH:-1}"

SAMPLE_SIZE="${SAMPLE_SIZE:-50000}"
ITERATIONS="${ITERATIONS:-15}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
SELECTION_POOL="${SELECTION_POOL:-100}"
SEED="${SEED:-0}"

RANK_CALIBRATION_SAMPLES="${RANK_CALIBRATION_SAMPLES:-4}"
RANK_CALIBRATION_SEQLEN="${RANK_CALIBRATION_SEQLEN:-256}"

CODEBOOK_SIZE="${CODEBOOK_SIZE:-65536}"
VQ_STAGES="${VQ_STAGES:-1}"
VQ_SAMPLE_SIZE="${VQ_SAMPLE_SIZE:-0}"

case "$CODEBOOK_SIZE" in
  16) VQ_COEFFICIENT_BITS=4 ;;
  256) VQ_COEFFICIENT_BITS=8 ;;
  65536) VQ_COEFFICIENT_BITS=16 ;;
  *)
    echo "Unsupported CODEBOOK_SIZE=$CODEBOOK_SIZE. Use 16, 256, or 65536." >&2
    exit 1
    ;;
esac

CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-32}"
CALIBRATION_SEQLEN="${CALIBRATION_SEQLEN:-2048}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-10}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-32}"
CALIBRATION_EARLY_STOP="${CALIBRATION_EARLY_STOP:-3}"
CALIBRATION_STEPS="${CALIBRATION_STEPS:-0}"
CALIBRATION_LR="${CALIBRATION_LR:-3e-5}"
CALIBRATION_VALIDATION_SAMPLES="${CALIBRATION_VALIDATION_SAMPLES:-8}"

SENSITIVE_COEFF_SPARSITY="${SENSITIVE_COEFF_SPARSITY:-0.20}"
ROBUST_COEFF_SPARSITY="${ROBUST_COEFF_SPARSITY:-0.35}"
SENSITIVE_MATRICES="${SENSITIVE_MATRICES:-}"
COEFF_SPARSITY_OVERRIDES="${COEFF_SPARSITY_OVERRIDES:-}"
SPARSITY_SOURCE="${SPARSITY_SOURCE:-fp16}"

DEVICE="${DEVICE:-cuda}"
INFERENCE_SEQLEN="${INFERENCE_SEQLEN:-2048}"
MAX_PPL_SAMPLES="${MAX_PPL_SAMPLES:-0}"

FINAL_ROOT="${FINAL_ROOT:-SeedLM/data/SeedLM_compressed/${MODEL_TAG}/final}"
FP16_ROOT="${FP16_ROOT:-${FINAL_ROOT}/adaptive_baseline_fp16}"
INT8_ROOT="${INT8_ROOT:-${FINAL_ROOT}/adaptive_baseline_int8}"
VQ_NOCALIB_ROOT="${VQ_NOCALIB_ROOT:-${FINAL_ROOT}/adaptive_vq_nocalib}"
VQ_CALIB_ROOT="${VQ_CALIB_ROOT:-${FINAL_ROOT}/adaptive_vq_calibrated}"
SPARSITY_ROOT="${SPARSITY_ROOT:-${FINAL_ROOT}/adaptive_sparsity}"
SPARSITY_RESULT_DIR="${SPARSITY_RESULT_DIR:-${SPARSITY_ROOT}/results}"

mkdir -p "$FINAL_ROOT" "$SPARSITY_RESULT_DIR"

run_base() {
  local run_root="$1"
  local representation_mode="$2"
  local coefficient_bits="$3"
  local calibration_mode="$4"

  env \
    MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" INPUT_DIR="$INPUT_DIR" \
    K="$K" BASE_RANK="$BASE_RANK" MAX_RANK="$MAX_RANK" \
    EXTRA_BUDGET_MB="$EXTRA_BUDGET_MB" \
    C="$C" TILE_WIDTH="$TILE_WIDTH" \
    SAMPLE_SIZE="$SAMPLE_SIZE" ITERATIONS="$ITERATIONS" \
    BATCH_SIZE="$BATCH_SIZE" SELECTION_POOL="$SELECTION_POOL" SEED="$SEED" \
    RANK_CALIBRATION_SAMPLES="$RANK_CALIBRATION_SAMPLES" \
    RANK_CALIBRATION_SEQLEN="$RANK_CALIBRATION_SEQLEN" \
    REPRESENTATION_MODE="$representation_mode" \
    COEFFICIENT_BITS="$coefficient_bits" \
    CODEBOOK_SIZE="$CODEBOOK_SIZE" \
    VQ_STAGES="$VQ_STAGES" \
    VQ_SAMPLE_SIZE="$VQ_SAMPLE_SIZE" \
    CALIBRATION_MODE="$calibration_mode" \
    CALIBRATION_SAMPLES="$CALIBRATION_SAMPLES" \
    CALIBRATION_SEQLEN="$CALIBRATION_SEQLEN" \
    CALIBRATION_EPOCHS="$CALIBRATION_EPOCHS" \
    CALIBRATION_BATCH_SIZE="$CALIBRATION_BATCH_SIZE" \
    CALIBRATION_EARLY_STOP="$CALIBRATION_EARLY_STOP" \
    CALIBRATION_STEPS="$CALIBRATION_STEPS" \
    CALIBRATION_LR="$CALIBRATION_LR" \
    CALIBRATION_VALIDATION_SAMPLES="$CALIBRATION_VALIDATION_SAMPLES" \
    TILE_MODE=fixed \
    DEVICE="$DEVICE" \
    INFERENCE_SEQLEN="$INFERENCE_SEQLEN" \
    MAX_PPL_SAMPLES="$MAX_PPL_SAMPLES" \
    RUN_ROOT="$run_root" \
    "$SCRIPT_DIR/run_base_pipeline.sh"
}

if [[ "$RUN_ADAPTIVE_FP16" == "1" ]]; then
  echo "[1/5] Adaptive rank + scalar FP16"
  run_base "$FP16_ROOT" scalar 16 none
fi

if [[ "$RUN_ADAPTIVE_INT8" == "1" ]]; then
  echo "[2/5] Adaptive rank + scalar INT8"
  run_base "$INT8_ROOT" scalar 8 none
fi

if [[ "$RUN_VQ_NOCALIB" == "1" ]]; then
  echo "[3/5] Adaptive rank + VQ, no calibration"
  run_base "$VQ_NOCALIB_ROOT" vq "$VQ_COEFFICIENT_BITS" none
fi

if [[ "$RUN_VQ_CALIBRATED" == "1" ]]; then
  echo "[4/5] Adaptive rank + VQ + calibration"
  run_base "$VQ_CALIB_ROOT" vq "$VQ_COEFFICIENT_BITS" calibrated
fi

if [[ "$RUN_ADAPTIVE_SPARSITY" == "1" ]]; then
  echo "[5/5] Adaptive rank + adaptive coefficient sparsity"

  if [[ "$SPARSITY_SOURCE" == "fp16" ]]; then
    SOURCE_ROOT="$FP16_ROOT"
    SOURCE_BITS=16
    [[ -d "$SOURCE_ROOT/artifacts" ]] || run_base "$FP16_ROOT" scalar 16 none
  elif [[ "$SPARSITY_SOURCE" == "int8" ]]; then
    SOURCE_ROOT="$INT8_ROOT"
    SOURCE_BITS=8
    [[ -d "$SOURCE_ROOT/artifacts" ]] || run_base "$INT8_ROOT" scalar 8 none
  else
    echo "SPARSITY_SOURCE must be fp16 or int8." >&2
    exit 1
  fi

  SENSITIVITY_JSON="${SENSITIVITY_JSON:-${SOURCE_ROOT}/rank_allocation/selected_first_order_ranks.json}"

  ARGS=(
    --model "$MODEL"
    --artifact-root "$SOURCE_ROOT/artifacts"
    --adaptive-coeff-pruning
    --sensitivity-json "$SENSITIVITY_JSON"
    --sensitive-coeff-sparsity "$SENSITIVE_COEFF_SPARSITY"
    --robust-coeff-sparsity "$ROBUST_COEFF_SPARSITY"
    --coefficient-value-bits "$SOURCE_BITS"
    --seqlen "$INFERENCE_SEQLEN"
    --max-ppl-samples "$MAX_PPL_SAMPLES"
    --device "$DEVICE"
    --skip-direct-weight
    --output-csv "$SPARSITY_RESULT_DIR/adaptive_sparsity.csv"
  )

  if [[ -n "$SENSITIVE_MATRICES" ]]; then
    read -ra TMP <<< "$SENSITIVE_MATRICES"
    for x in "${TMP[@]}"; do ARGS+=(--sensitive-matrix "$x"); done
  fi

  if [[ -n "$COEFF_SPARSITY_OVERRIDES" ]]; then
    read -ra TMP <<< "$COEFF_SPARSITY_OVERRIDES"
    for x in "${TMP[@]}"; do ARGS+=(--coeff-sparsity-override "$x"); done
  fi

  python "$SCRIPT_DIR/compare_magnitude_pruning.py" "${ARGS[@]}"
fi

echo
echo "Done."
echo "FP16 baseline:      $FP16_ROOT"
echo "INT8 baseline:      $INT8_ROOT"
echo "VQ no calibration:  $VQ_NOCALIB_ROOT"
echo "VQ calibrated:      $VQ_CALIB_ROOT"
echo "Adaptive sparsity:  $SPARSITY_RESULT_DIR"
