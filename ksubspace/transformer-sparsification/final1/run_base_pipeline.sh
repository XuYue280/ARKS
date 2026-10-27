#!/usr/bin/env bash
set -euo pipefail

# Run from the transformer-sparsification repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL="${MODEL:-facebook/opt-125m}"
MODEL_TAG="${MODEL_TAG:-${MODEL##*/}}"
INPUT_DIR="${INPUT_DIR:-SeedLM/model/${MODEL_TAG}}"

# Supported experiment axes:
#   CALIBRATION_MODE=none|calibrated
#   TILE_MODE=fixed|adaptive
CALIBRATION_MODE="${CALIBRATION_MODE:-calibrated}"
TILE_MODE="${TILE_MODE:-fixed}"

# vq stores one or more codeword IDs plus coefficient codebooks. scalar stores
# the quantized coefficient values directly and has no coefficient codebook.
REPRESENTATION_MODE="${REPRESENTATION_MODE:-vq}"

# ---------------------------------------------------------------------------
# K-subspaces configuration
# ---------------------------------------------------------------------------

K="${K:-512}"
BASE_RANK="${BASE_RANK:-4}"
MAX_RANK="${MAX_RANK:-6}"
EXTRA_BUDGET_MB="${EXTRA_BUDGET_MB:-1.88}"

TILE_WIDTH="${TILE_WIDTH:-1}"

# Fixed mode reproduces the previous all-8x1 experiment. Adaptive mode uses
# 16x1 by default and assigns activation-sensitive matrices to 8x1.
SMALL_TILE_C="${SMALL_TILE_C:-8}"
LARGE_TILE_C="${LARGE_TILE_C:-16}"
if [[ "$TILE_MODE" == "adaptive" ]]; then
  C="$LARGE_TILE_C"
else
  C="${C:-8}"
fi

# Use only one of these adaptive-tile selection controls. If none is set, the
# allocator assigns the most-sensitive 25 percent of matrices to SMALL_TILE_C.
TILE_SENSITIVE_COUNT="${TILE_SENSITIVE_COUNT:-}"
TILE_SENSITIVE_FRACTION="${TILE_SENSITIVE_FRACTION:-}"
TILE_EXTRA_BUDGET_MB="${TILE_EXTRA_BUDGET_MB:-}"

SAMPLE_SIZE="${SAMPLE_SIZE:-50000}"
ITERATIONS="${ITERATIONS:-15}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
SELECTION_POOL="${SELECTION_POOL:-100}"
SEED="${SEED:-0}"

# ---------------------------------------------------------------------------
# Coefficient representation
# ---------------------------------------------------------------------------

CODEBOOK_SIZE="${CODEBOOK_SIZE:-65536}"
VQ_STAGES="${VQ_STAGES:-1}"

# 0 means use all coefficient vectors.
VQ_SAMPLE_SIZE="${VQ_SAMPLE_SIZE:-0}"

# VQ requires ceil(log2(codebook size)) bits per codeword ID. Scalar mode uses
# direct 8-bit coefficient storage unless explicitly set to 4 or 16 bits.
if [[ "$REPRESENTATION_MODE" == "scalar" ]]; then
  COEFFICIENT_BITS="${COEFFICIENT_BITS:-8}"
elif [[ -n "${COEFFICIENT_BITS:-}" ]]; then
  COEFFICIENT_BITS="$COEFFICIENT_BITS"
else
  COEFFICIENT_BITS=0
  value=$((CODEBOOK_SIZE - 1))

  while (( value > 0 )); do
    COEFFICIENT_BITS=$((COEFFICIENT_BITS + 1))
    value=$((value >> 1))
  done
fi

# The Python script currently accepts only 4, 8, or 16 bits.
if [[ "$COEFFICIENT_BITS" != "4" &&
      "$COEFFICIENT_BITS" != "8" &&
      "$COEFFICIENT_BITS" != "16" ]]; then
  echo "Unsupported coefficient bits: $COEFFICIENT_BITS" >&2
  echo "The codebook size must require 4, 8, or 16 bits." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Rank-allocation calibration
# ---------------------------------------------------------------------------

RANK_CALIBRATION_SAMPLES="${RANK_CALIBRATION_SAMPLES:-4}"
RANK_CALIBRATION_SEQLEN="${RANK_CALIBRATION_SEQLEN:-256}"

# ---------------------------------------------------------------------------
# Codebook calibration
# ---------------------------------------------------------------------------

CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-32}"
CALIBRATION_SEQLEN="${CALIBRATION_SEQLEN:-2048}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-10}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-32}"
CALIBRATION_EARLY_STOP="${CALIBRATION_EARLY_STOP:-3}"
# Optional optimizer-step cap. Zero processes complete epochs.
CALIBRATION_STEPS="${CALIBRATION_STEPS:-0}"
CALIBRATION_LR="${CALIBRATION_LR:-3e-5}"
CALIBRATION_VALIDATION_SAMPLES="${CALIBRATION_VALIDATION_SAMPLES:-8}"
if [[ -n "${CALIBRATION_REASSIGN_EVERY:-}" ]]; then
  CALIBRATION_REASSIGN_EVERY="$CALIBRATION_REASSIGN_EVERY"
else
  # AQLM block fine-tuning freezes code IDs.
  CALIBRATION_REASSIGN_EVERY=0
fi
CALIBRATION_REASSIGN_BATCH_SIZE="${CALIBRATION_REASSIGN_BATCH_SIZE:-4096}"
CALIBRATION_MAX_COEFFICIENT_MSE_RATIO="${CALIBRATION_MAX_COEFFICIENT_MSE_RATIO:-1.01}"
CALIBRATION_SEQUENTIAL_INPUTS="${CALIBRATION_SEQUENTIAL_INPUTS:-1}"
CALIBRATION_PPL_GUARD_SAMPLES="${CALIBRATION_PPL_GUARD_SAMPLES:-4}"
CALIBRATION_PPL_GUARD_SEQLEN="${CALIBRATION_PPL_GUARD_SEQLEN:-2048}"
CALIBRATION_PPL_GUARD_SPLIT="${CALIBRATION_PPL_GUARD_SPLIT:-validation}"
CALIBRATION_PPL_GUARD_TOLERANCE="${CALIBRATION_PPL_GUARD_TOLERANCE:-0.0}"
CALIBRATION_KEEP_BACKUP="${CALIBRATION_KEEP_BACKUP:-0}"

# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------

DEVICE="${DEVICE:-cuda}"
INFERENCE_SEQLEN="${INFERENCE_SEQLEN:-2048}"
MAX_PPL_SAMPLES="${MAX_PPL_SAMPLES:-0}"

# Set to 0 to detect the number of matrices automatically.
EXPECTED_MATRICES="${EXPECTED_MATRICES:-0}"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

if [[ "$TILE_MODE" == "adaptive" ]]; then
  TILE_RUN_TAG="adaptive_C${LARGE_TILE_C}-to-C${SMALL_TILE_C}"
else
  TILE_RUN_TAG="fixed_C${C}"
fi

if [[ "$CALIBRATION_MODE" == "calibrated" ]]; then
  CALIBRATION_RUN_TAG="calibrated"
else
  CALIBRATION_RUN_TAG="nocalib"
fi

if [[ "$REPRESENTATION_MODE" == "scalar" ]]; then
  REPRESENTATION_RUN_TAG="scalar_B${COEFFICIENT_BITS}"
else
  REPRESENTATION_RUN_TAG="VQ${VQ_STAGES}x${CODEBOOK_SIZE}_allcoeff"
fi

DEFAULT_RUN_NAME="block_${CALIBRATION_RUN_TAG}_${TILE_RUN_TAG}_K${K}_R${BASE_RANK}-${MAX_RANK}_W${TILE_WIDTH}_${REPRESENTATION_RUN_TAG}"

RUN_ROOT="${RUN_ROOT:-SeedLM/data/SeedLM_compressed/${MODEL_TAG}/${DEFAULT_RUN_NAME}}"
PROFILE_ROOT="${PROFILE_ROOT:-${RUN_ROOT}/rank_profile}"
ALLOCATION_DIR="${ALLOCATION_DIR:-${RUN_ROOT}/rank_allocation}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RUN_ROOT}/artifacts}"
RESULT_DIR="${RESULT_DIR:-${RUN_ROOT}/results}"

PROFILE_DIR="${PROFILE_DIR:-${PROFILE_ROOT}/sp_1.00_K${K}_R${MAX_RANK}_B8_C${C}_W${TILE_WIDTH}}"
SMALL_PROFILE_ROOT="${SMALL_PROFILE_ROOT:-${PROFILE_ROOT}/small_C${SMALL_TILE_C}}"
LARGE_PROFILE_ROOT="${LARGE_PROFILE_ROOT:-${PROFILE_ROOT}/large_C${LARGE_TILE_C}}"
SMALL_PROFILE_DIR="${SMALL_PROFILE_DIR:-${SMALL_PROFILE_ROOT}/sp_1.00_K${K}_R${MAX_RANK}_B8_C${SMALL_TILE_C}_W${TILE_WIDTH}}"
LARGE_PROFILE_DIR="${LARGE_PROFILE_DIR:-${LARGE_PROFILE_ROOT}/sp_1.00_K${K}_R${MAX_RANK}_B8_C${LARGE_TILE_C}_W${TILE_WIDTH}}"
OVERRIDES_FILE="${ALLOCATION_DIR}/rank_override_arguments.txt"
TILE_OVERRIDES_FILE="${ALLOCATION_DIR}/tile_override_arguments.txt"

# ---------------------------------------------------------------------------
# Stage-control flags
# ---------------------------------------------------------------------------

SKIP_ALLOCATION="${SKIP_ALLOCATION:-0}"
SKIP_COMPRESSION="${SKIP_COMPRESSION:-0}"
SKIP_CODEBOOK_CALIBRATION="${SKIP_CODEBOOK_CALIBRATION:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"

if [[ "$CALIBRATION_MODE" == "none" || "$REPRESENTATION_MODE" == "scalar" ]]; then
  SKIP_CODEBOOK_CALIBRATION=1
fi

# ---------------------------------------------------------------------------
# Initial validation
# ---------------------------------------------------------------------------

if [[ "$CALIBRATION_MODE" != "none" && "$CALIBRATION_MODE" != "calibrated" ]]; then
  echo "CALIBRATION_MODE must be 'none' or 'calibrated'." >&2
  exit 1
fi

if [[ "$REPRESENTATION_MODE" != "vq" && "$REPRESENTATION_MODE" != "scalar" ]]; then
  echo "REPRESENTATION_MODE must be 'vq' or 'scalar'." >&2
  exit 1
fi

if [[ "$REPRESENTATION_MODE" == "scalar" && "$CALIBRATION_MODE" != "none" ]]; then
  echo "Scalar coefficients have no codebook to calibrate." >&2
  echo "Use CALIBRATION_MODE=none with REPRESENTATION_MODE=scalar." >&2
  exit 1
fi

if [[ "$TILE_MODE" != "fixed" && "$TILE_MODE" != "adaptive" ]]; then
  echo "TILE_MODE must be 'fixed' or 'adaptive'." >&2
  exit 1
fi

tile_selector_count=0
[[ -n "$TILE_SENSITIVE_COUNT" ]] && tile_selector_count=$((tile_selector_count + 1))
[[ -n "$TILE_SENSITIVE_FRACTION" ]] && tile_selector_count=$((tile_selector_count + 1))
[[ -n "$TILE_EXTRA_BUDGET_MB" ]] && tile_selector_count=$((tile_selector_count + 1))
if (( tile_selector_count > 1 )); then
  echo "Set only one of TILE_SENSITIVE_COUNT, TILE_SENSITIVE_FRACTION, or TILE_EXTRA_BUDGET_MB." >&2
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Missing model weight directory: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p \
  "$RUN_ROOT" \
  "$PROFILE_ROOT" \
  "$ALLOCATION_DIR" \
  "$ARTIFACT_ROOT" \
  "$RESULT_DIR"

echo "Model:                    $MODEL"
echo "Model tag:                $MODEL_TAG"
echo "Input directory:          $INPUT_DIR"
echo "Run root:                 $RUN_ROOT"
echo "Calibration mode:         $CALIBRATION_MODE"
echo "Tile mode:                $TILE_MODE"
echo "Coefficient representation: $REPRESENTATION_MODE"
echo "K-subspaces:              $K"
echo "Base/max rank:            $BASE_RANK/$MAX_RANK"
echo "Tile:                     ${C}x${TILE_WIDTH}"
if [[ "$TILE_MODE" == "adaptive" ]]; then
  echo "Sensitive/robust tiles:   ${SMALL_TILE_C}x${TILE_WIDTH} / ${LARGE_TILE_C}x${TILE_WIDTH}"
fi
if [[ "$REPRESENTATION_MODE" == "vq" ]]; then
  echo "Coefficient codebooks:    ${VQ_STAGES}x${CODEBOOK_SIZE}"
  echo "Coefficient ID bits:      $COEFFICIENT_BITS"
  echo "Coefficient VQ sample:    $VQ_SAMPLE_SIZE (0 means all)"
else
  echo "Coefficient storage:      direct scalar INT${COEFFICIENT_BITS}"
  echo "Coefficient codebooks:    none"
fi
echo "Calibration samples:      $CALIBRATION_SAMPLES"
echo "Calibration sequence:     $CALIBRATION_SEQLEN"
echo "Calibration epochs:       $CALIBRATION_EPOCHS"
echo "Calibration batch:        $CALIBRATION_BATCH_SIZE"
echo "Calibration early stop:   $CALIBRATION_EARLY_STOP"
echo "Optimizer-step cap:       $CALIBRATION_STEPS (0 means none)"
echo "Calibration validation:   $CALIBRATION_VALIDATION_SAMPLES samples"
echo "Code-ID reassignment:     every $CALIBRATION_REASSIGN_EVERY steps"
echo "Sequential block inputs:  $CALIBRATION_SEQUENTIAL_INPUTS"
echo "Held-out PPL guard:       $CALIBRATION_PPL_GUARD_SAMPLES x $CALIBRATION_PPL_GUARD_SEQLEN"

# ---------------------------------------------------------------------------
# Stage 1: rank profiling and allocation
# ---------------------------------------------------------------------------

if [[ "$SKIP_ALLOCATION" != "1" ]]; then
  profile_common_args=(
    --model "$MODEL"
    --input-dir "$INPUT_DIR"
    --weight-scope all
    --k "$K"
    --ranks "$MAX_RANK"
    --coefficient-bits 8
    --coefficient-encoding scalar
    --tile-width "$TILE_WIDTH"
    --sparsity 1.0
    --sample-size "$SAMPLE_SIZE"
    --iterations "$ITERATIONS"
    --batch-size "$BATCH_SIZE"
    --selection-pool "$SELECTION_POOL"
    --seed "$SEED"
  )

  allocation_tile_args=()
  if [[ "$TILE_MODE" == "adaptive" ]]; then
    echo "[1/5] Building ${SMALL_TILE_C}x${TILE_WIDTH} and ${LARGE_TILE_C}x${TILE_WIDTH} rank-${MAX_RANK} profiles"
    python "$SCRIPT_DIR/bench_compress_minibatch_vq.py" \
      "${profile_common_args[@]}" \
      --output-root "$SMALL_PROFILE_ROOT" \
      --C "$SMALL_TILE_C"
    python "$SCRIPT_DIR/bench_compress_minibatch_vq.py" \
      "${profile_common_args[@]}" \
      --output-root "$LARGE_PROFILE_ROOT" \
      --C "$LARGE_TILE_C"

    allocation_profile_dir="$SMALL_PROFILE_DIR"
    allocation_tile_args=(
      --large-tile-artifact-dir "$LARGE_PROFILE_DIR"
      --small-tile-c "$SMALL_TILE_C"
      --large-tile-c "$LARGE_TILE_C"
    )
    if [[ -n "$TILE_SENSITIVE_COUNT" ]]; then
      allocation_tile_args+=(--tile-sensitive-count "$TILE_SENSITIVE_COUNT")
    elif [[ -n "$TILE_SENSITIVE_FRACTION" ]]; then
      allocation_tile_args+=(--tile-sensitive-fraction "$TILE_SENSITIVE_FRACTION")
    elif [[ -n "$TILE_EXTRA_BUDGET_MB" ]]; then
      allocation_tile_args+=(--tile-extra-budget-mb "$TILE_EXTRA_BUDGET_MB")
    fi
  else
    echo "[1/5] Building fixed ${C}x${TILE_WIDTH} rank-${MAX_RANK} profile"
    python "$SCRIPT_DIR/bench_compress_minibatch_vq.py" \
      "${profile_common_args[@]}" \
      --output-root "$PROFILE_ROOT" \
      --C "$C"
    allocation_profile_dir="$PROFILE_DIR"
  fi

  python "$SCRIPT_DIR/first_order_rank_allocator.py" \
    --model "$MODEL" \
    --max-rank-artifact-dir "$allocation_profile_dir" \
    --base-rank "$BASE_RANK" \
    --max-rank "$MAX_RANK" \
    --extra-budget-mb "$EXTRA_BUDGET_MB" \
    --calibration-samples "$RANK_CALIBRATION_SAMPLES" \
    --seqlen "$RANK_CALIBRATION_SEQLEN" \
    --weight-scope all \
    --device "$DEVICE" \
    --seed "$SEED" \
    --output-dir "$ALLOCATION_DIR" \
    "${allocation_tile_args[@]}"
else
  echo "[1/5] Skipping rank profiling and allocation"
fi

if [[ ! -s "$OVERRIDES_FILE" ]]; then
  echo "Missing rank-override file: $OVERRIDES_FILE" >&2
  exit 1
fi

read -r -a rank_args < "$OVERRIDES_FILE"

tile_args=()
if [[ "$TILE_MODE" == "adaptive" ]]; then
  if [[ ! -s "$TILE_OVERRIDES_FILE" ]]; then
    echo "Missing tile-override file: $TILE_OVERRIDES_FILE" >&2
    exit 1
  fi
  read -r -a tile_args < "$TILE_OVERRIDES_FILE"
fi

# ---------------------------------------------------------------------------
# Arguments shared by compression and inference
# ---------------------------------------------------------------------------

common_args=(
  --model "$MODEL"
  --sparsity 1.0
  --k "$K"
  --ranks "$BASE_RANK"
  --coefficient-bits "$COEFFICIENT_BITS"
  --coefficient-encoding "$REPRESENTATION_MODE"
  --C "$C"
  --tile-width "$TILE_WIDTH"
  "${rank_args[@]}"
  "${tile_args[@]}"
)

if [[ "$REPRESENTATION_MODE" == "vq" ]]; then
  common_args+=(
    --coefficient-codebook-size "$CODEBOOK_SIZE"
    --coefficient-vq-stages "$VQ_STAGES"
  )
fi

# ---------------------------------------------------------------------------
# Stage 2: coefficient codebook construction
# ---------------------------------------------------------------------------

if [[ "$SKIP_COMPRESSION" != "1" ]]; then
  compression_representation_args=()
  if [[ "$REPRESENTATION_MODE" == "vq" ]]; then
    echo "[2/5] Building coefficient codebooks with MiniBatch K-means"
    compression_representation_args+=(
      --coefficient-vq-sample-size "$VQ_SAMPLE_SIZE"
    )
  else
    echo "[2/5] Compressing with direct scalar coefficients and no codebook"
  fi

  python "$SCRIPT_DIR/bench_compress_minibatch_vq.py" \
    "${common_args[@]}" \
    --input-dir "$INPUT_DIR" \
    --output-root "$ARTIFACT_ROOT" \
    --weight-scope all \
    --sample-size "$SAMPLE_SIZE" \
    "${compression_representation_args[@]}" \
    --iterations "$ITERATIONS" \
    --batch-size "$BATCH_SIZE" \
    --selection-pool "$SELECTION_POOL" \
    --seed "$SEED"
else
  echo "[2/5] Skipping coefficient compression"
fi

# ---------------------------------------------------------------------------
# Locate the generated artifact
# ---------------------------------------------------------------------------

if [[ "$REPRESENTATION_MODE" == "vq" ]]; then
  ARTIFACT_PATTERN="sp_1.00_K${K}_R${BASE_RANK}*_B${COEFFICIENT_BITS}_C${C}*_W${TILE_WIDTH}_CVQ${VQ_STAGES}x${CODEBOOK_SIZE}*"
else
  # The scalar directory ends after W${TILE_WIDTH}; this excludes VQ runs that
  # share the same rank, bit width, and tile configuration.
  ARTIFACT_PATTERN="sp_1.00_K${K}_R${BASE_RANK}*_B${COEFFICIENT_BITS}_C${C}*_W${TILE_WIDTH}"
fi

mapfile -t candidates < <(
  find "$ARTIFACT_ROOT" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -name "$ARTIFACT_PATTERN"
)

if [[ "${#candidates[@]}" -ne 1 ]]; then
  echo "Expected exactly one matching artifact; found ${#candidates[@]}" >&2
  printf '%s\n' "${candidates[@]}" >&2
  exit 1
fi

CONFIG_DIR="${candidates[0]}"

matrix_count=$(
  find "$CONFIG_DIR" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -name 'layer_*' |
  wc -l
)

if [[ "$matrix_count" -eq 0 ]]; then
  echo "No matrix artifacts were found in $CONFIG_DIR" >&2
  exit 1
fi

if [[ "$EXPECTED_MATRICES" -gt 0 &&
      "$matrix_count" -ne "$EXPECTED_MATRICES" ]]; then
  echo "Expected $EXPECTED_MATRICES matrix artifacts; found $matrix_count" >&2
  exit 1
fi

echo "Detected matrix artifacts: $matrix_count"

# ---------------------------------------------------------------------------
# Stage 3: block-output calibration
# ---------------------------------------------------------------------------

if [[ "$SKIP_CODEBOOK_CALIBRATION" != "1" ]]; then
  if find "$CONFIG_DIR" \
    -name metadata.json \
    -type f \
    -exec grep -l '"coefficient_codebook_calibration"' {} + |
    grep -q .; then

    echo "These artifacts were already block-calibrated." >&2
    echo "Use a new RUN_ROOT instead of calibrating them twice." >&2
    exit 1
  fi

  echo "[3/5] Tuning coefficient codebooks with block-output loss"

  calibration_safety_args=()
  if [[ "$CALIBRATION_SEQUENTIAL_INPUTS" == "1" ]]; then
    calibration_safety_args+=(--sequential-inputs)
  else
    calibration_safety_args+=(--no-sequential-inputs)
  fi
  if [[ "$CALIBRATION_KEEP_BACKUP" == "1" ]]; then
    calibration_safety_args+=(--keep-backup)
  fi

  python "$SCRIPT_DIR/calibrate_codebooks_block_output.py" \
    --model "$MODEL" \
    --artifact-dir "$CONFIG_DIR" \
    --samples "$CALIBRATION_SAMPLES" \
    --seqlen "$CALIBRATION_SEQLEN" \
    --epochs "$CALIBRATION_EPOCHS" \
    --batch-size "$CALIBRATION_BATCH_SIZE" \
    --early-stop "$CALIBRATION_EARLY_STOP" \
    --steps "$CALIBRATION_STEPS" \
    --lr "$CALIBRATION_LR" \
    --seed "$SEED" \
    --validation-samples "$CALIBRATION_VALIDATION_SAMPLES" \
    --reassign-every "$CALIBRATION_REASSIGN_EVERY" \
    --reassignment-batch-size "$CALIBRATION_REASSIGN_BATCH_SIZE" \
    --max-coefficient-mse-ratio "$CALIBRATION_MAX_COEFFICIENT_MSE_RATIO" \
    --ppl-guard-samples "$CALIBRATION_PPL_GUARD_SAMPLES" \
    --ppl-guard-seqlen "$CALIBRATION_PPL_GUARD_SEQLEN" \
    --ppl-guard-split "$CALIBRATION_PPL_GUARD_SPLIT" \
    --ppl-guard-tolerance "$CALIBRATION_PPL_GUARD_TOLERANCE" \
    --device "$DEVICE" \
    --output-csv "$RESULT_DIR/block_codebook_calibration.csv" \
    "${calibration_safety_args[@]}"
else
  if [[ "$REPRESENTATION_MODE" == "scalar" ]]; then
    echo "[3/5] Scalar mode: no coefficient codebook exists to calibrate"
  elif [[ "$CALIBRATION_MODE" == "none" ]]; then
    echo "[3/5] No-calibration mode: leaving coefficient codebooks unchanged"
  else
    echo "[3/5] Skipping coefficient codebook calibration"
  fi
fi

# ---------------------------------------------------------------------------
# Stage 4: full-model perplexity
# ---------------------------------------------------------------------------

if [[ "$SKIP_INFERENCE" != "1" ]]; then
  echo "[4/5] Evaluating full-model WikiText-2 perplexity"

  python "$SCRIPT_DIR/bench_inference_minibatch_vq.py" \
    "${common_args[@]}" \
    --artifact-root "$ARTIFACT_ROOT" \
    --original-dir "$INPUT_DIR" \
    --result-dir "$RESULT_DIR" \
    --device "$DEVICE" \
    --seqlen "$INFERENCE_SEQLEN" \
    --batch-size 1 \
    --max-ppl-samples "$MAX_PPL_SAMPLES" \
    --targets all
else
  echo "[4/5] Skipping inference"
fi

echo "[5/5] Complete"
echo "Model:                $MODEL"
echo "Calibration mode:     $CALIBRATION_MODE"
echo "Tile mode:            $TILE_MODE"
echo "Representation mode:  $REPRESENTATION_MODE"
echo "Artifact directory:   $CONFIG_DIR"
echo "Results:              $RESULT_DIR"

#cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification
#source .venv/bin/activate
#
#chmod +x ./block-aqlm-calibration-safe/run_opt125_block_aqlm_calibration.sh
#
#SKIP_ALLOCATION=1 \
#MODEL="facebook/opt-125m" \
#MODEL_TAG="opt-125m" \
#INPUT_DIR="SeedLM/model/opt-125m" \
#CODEBOOK_SIZE=65536 \
#VQ_STAGES=1 \
#COEFFICIENT_BITS=16 \
#VQ_SAMPLE_SIZE=0 \
#CALIBRATION_SAMPLES=32 \
#CALIBRATION_SEQLEN=2048 \
#CALIBRATION_EPOCHS=10 \
#CALIBRATION_LR=1e-4 \
#RUN_ROOT="SeedLM/data/SeedLM_compressed/opt-125m/calibrated_vq1x65536_allcoeff_seq2048" \
#./block-aqlm-calibration-safe/run_opt125_block_aqlm_calibration.sh
