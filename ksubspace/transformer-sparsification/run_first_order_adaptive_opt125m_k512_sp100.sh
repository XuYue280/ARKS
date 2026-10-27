#!/usr/bin/env bash

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="facebook/opt-125m"
INPUT_DIR="SeedLM/model/opt-125m"

COMPRESS_SCRIPT="SeedLM/k-space/different-rank/bench_compress_and_store_ksubspaces.py"
ALLOCATOR_SCRIPT="SeedLM/k-space/different-rank/first_order_rank_allocator.py"
INFERENCE_SCRIPT="SeedLM/k-space/different-rank/bench_load_and_inference_ksubspaces_with_dense.py"

K=512
BASE_RANK=4
MAX_RANK=6
SPARSITY=1.0

# This is approximately the storage required to upgrade all 12 OPT v_proj
# matrices from Rank 4 to Rank 6 at K=512 and sparsity=1.0.
EXTRA_BUDGET_MB=1.88

EXPECTED_MATRICES=72

PROFILE_ROOT="SeedLM/data/SeedLM_compressed/opt-125m/first_order_profile_K512_sp100"
PROFILE_DIR="${PROFILE_ROOT}/sp_1.00_K512_R6_B8_C8_W1"

ALLOCATION_DIR="results/opt-125m/first_order_full_K512_sp100"
OVERRIDE_FILE="${ALLOCATION_DIR}/rank_override_arguments.txt"

ADAPTIVE_ROOT="SeedLM/data/SeedLM_compressed/opt-125m/first_order_adaptive_K512_sp100"
INFERENCE_DIR="results/opt-125m/inference_first_order_K512_sp100"

mkdir -p logs
LOG_FILE="logs/first_order_opt125m_K512_sp100_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Start time: $(date)"
echo "Repository: $SCRIPT_DIR"
echo "Log file: $LOG_FILE"
echo "Model: $MODEL"
echo "K: $K"
echo "Tile coverage: $SPARSITY"
echo "Candidate ranks: $BASE_RANK and $MAX_RANK"
echo "Extra rank budget: $EXTRA_BUDGET_MB MiB"

for required_file in "$COMPRESS_SCRIPT" "$ALLOCATOR_SCRIPT" "$INFERENCE_SCRIPT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Required script does not exist: $required_file"
        exit 1
    fi
done

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Original model directory does not exist: $INPUT_DIR"
    exit 1
fi

echo
echo "[1/4] Checking the maximum-rank profile"

METADATA_COUNT=0
if [[ -d "$PROFILE_DIR" ]]; then
    METADATA_COUNT=$(find "$PROFILE_DIR" -name metadata.json -type f | wc -l)
fi

if [[ "$METADATA_COUNT" -eq "$EXPECTED_MATRICES" ]]; then
    echo "Complete Rank-$MAX_RANK profile already exists. Skipping this step."
else
    echo "Found $METADATA_COUNT of $EXPECTED_MATRICES profile artifacts."
    echo "Creating a complete Rank-$MAX_RANK profile..."

    time python "$COMPRESS_SCRIPT" \
      --model "$MODEL" \
      --input-dir "$INPUT_DIR" \
      --output-root "$PROFILE_ROOT" \
      --weight-scope all \
      --k "$K" \
      --ranks "$MAX_RANK" \
      --coefficient-bits 8 \
      --C 8 \
      --tile-width 1 \
      --sparsity "$SPARSITY" \
      --sample-size 50000 \
      --iterations 15 \
      --batch-size 2048 \
      --selection-pool 100 \
      --selection-seed 42 \
      --seed 0
fi

METADATA_COUNT=$(find "$PROFILE_DIR" -name metadata.json -type f | wc -l)
if [[ "$METADATA_COUNT" -ne "$EXPECTED_MATRICES" ]]; then
    echo "ERROR: Rank-$MAX_RANK profile is incomplete."
    echo "Expected $EXPECTED_MATRICES metadata files, but found $METADATA_COUNT."
    exit 1
fi

echo
echo "[2/4] Running first-order rank allocation"

if [[ -s "$OVERRIDE_FILE" ]]; then
    echo "Existing rank allocation found. Skipping this step: $OVERRIDE_FILE"
else
    time python "$ALLOCATOR_SCRIPT" \
      --model "$MODEL" \
      --max-rank-artifact-dir "$PROFILE_DIR" \
      --base-rank "$BASE_RANK" \
      --max-rank "$MAX_RANK" \
      --extra-budget-mb "$EXTRA_BUDGET_MB" \
      --calibration-samples 4 \
      --seqlen 256 \
      --weight-scope all \
      --output-dir "$ALLOCATION_DIR" \
      --device cuda
fi

if [[ ! -s "$OVERRIDE_FILE" ]]; then
    echo "ERROR: Rank override file was not created: $OVERRIDE_FILE"
    exit 1
fi

read -r -a RANK_OVERRIDES < "$OVERRIDE_FILE"
SELECTED_COUNT=$(grep -o -- "--rank-override" "$OVERRIDE_FILE" | wc -l)

if [[ "$SELECTED_COUNT" -eq 0 ]]; then
    echo "ERROR: The allocator did not select any Rank-$MAX_RANK matrices."
    exit 1
fi

echo "Matrices selected for Rank $MAX_RANK: $SELECTED_COUNT"
echo "Matrices remaining at Rank $BASE_RANK: $((EXPECTED_MATRICES - SELECTED_COUNT))"

echo
echo "[3/4] Creating the final adaptive compressed model"

time python "$COMPRESS_SCRIPT" \
  --model "$MODEL" \
  --input-dir "$INPUT_DIR" \
  --output-root "$ADAPTIVE_ROOT" \
  --weight-scope all \
  --k "$K" \
  --ranks "$BASE_RANK" \
  "${RANK_OVERRIDES[@]}" \
  --coefficient-bits 8 \
  --C 8 \
  --tile-width 1 \
  --sparsity "$SPARSITY" \
  --sample-size 50000 \
  --iterations 15 \
  --batch-size 2048 \
  --selection-pool 100 \
  --selection-seed 42 \
  --seed 0

echo
echo "[4/4] Computing full WikiText-2 perplexity"

mkdir -p "$INFERENCE_DIR"

time python "$INFERENCE_SCRIPT" \
  --model "$MODEL" \
  --original-dir "$INPUT_DIR" \
  --artifact-root "$ADAPTIVE_ROOT" \
  --result-dir "$INFERENCE_DIR" \
  --targets all \
  --sparsity "$SPARSITY" \
  --k "$K" \
  --ranks "$BASE_RANK" \
  "${RANK_OVERRIDES[@]}" \
  --coefficient-bits 8 \
  --C 8 \
  --tile-width 1 \
  --device cuda \
  --seqlen 2048 \
  --batch-size 1 \
  --max-ppl-samples 0

echo
echo "Pipeline completed successfully."
echo "End time: $(date)"
echo "Rank allocation results: $ALLOCATION_DIR"
echo "Compressed artifacts: $ADAPTIVE_ROOT"
echo "Inference results: $INFERENCE_DIR"
echo "Complete log: $LOG_FILE"
