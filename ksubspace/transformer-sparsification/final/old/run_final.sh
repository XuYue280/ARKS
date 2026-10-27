#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-facebook/opt-125m}"
MODEL_TAG="${MODEL_TAG:-${MODEL##*/}}"
K="${K:-512}"
BASE_RANK="${BASE_RANK:-4}"
MAX_RANK="${MAX_RANK:-6}"
EXTRA_BUDGET_MB="${EXTRA_BUDGET_MB:-1.88}"
C="${C:-8}"
TILE_WIDTH="${TILE_WIDTH:-1}"
SPARSITIES="${SPARSITIES:-0.25 0.50 0.75}"
RUN_ROOT="${RUN_ROOT:-SeedLM/data/SeedLM_compressed/${MODEL_TAG}/final/arks_scalar}"
RESULT_DIR="${RESULT_DIR:-SeedLM/data/SeedLM_compressed/${MODEL_TAG}/final/results}"
mkdir -p "$RESULT_DIR"

# Build the same ARKS adaptive-rank representation used previously, but scalar INT8
# coefficients, because Stage 2 is now magnitude pruning rather than VQ.
MODEL="$MODEL" MODEL_TAG="$MODEL_TAG" K="$K" BASE_RANK="$BASE_RANK" MAX_RANK="$MAX_RANK" \
EXTRA_BUDGET_MB="$EXTRA_BUDGET_MB" C="$C" TILE_WIDTH="$TILE_WIDTH" \
REPRESENTATION_MODE=scalar COEFFICIENT_BITS=8 CALIBRATION_MODE=none TILE_MODE=fixed \
RUN_ROOT="$RUN_ROOT" \
"$SCRIPT_DIR/run_base_pipeline.sh"

python "$SCRIPT_DIR/compare_magnitude_pruning.py" \
  --model "$MODEL" \
  --artifact-root "$RUN_ROOT/artifacts" \
  --sparsities $SPARSITIES \
  --seqlen "${INFERENCE_SEQLEN:-2048}" \
  --max-ppl-samples "${MAX_PPL_SAMPLES:-0}" \
  --device "${DEVICE:-cuda}" \
  --output-csv "$RESULT_DIR/magnitude_comparison.csv"
