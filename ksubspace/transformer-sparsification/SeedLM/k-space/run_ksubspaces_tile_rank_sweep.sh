#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification"
PYTHON_BIN="/root/.virtualenvs/transformer-sparsification/bin/python"
MODEL="facebook/opt-125m"
ARTIFACT_ROOT="data/SeedLM_compressed/opt-125m/kspace_tile_sweep"

SPARSITY="0.8"
K="256"
COEFFICIENT_BITS="8"
SELECTION_POOL="100"
SELECTION_SEED="42"
CLUSTER_SEED="42"
ITERATIONS="15"
CLUSTER_BATCH_SIZE="2048"

DEVICE="cuda"
SEQLEN="2048"
PPL_BATCH_SIZE="1"

# Use 8 for a quick screening run. Change to 0 for full WikiText-2.
MAX_PPL_SAMPLES="0"

cd "$PROJECT_ROOT"

run_case() {
    local tile_size="$1"
    shift
    local ranks=("$@")

    echo
    echo "Compressing C=${tile_size}, ranks=${ranks[*]}"

    "$PYTHON_BIN" SeedLM/k-space/bench_compress_and_store_ksubspaces.py \
        --model "$MODEL" \
        --sparsity "$SPARSITY" \
        --k "$K" \
        --ranks "${ranks[@]}" \
        --coefficient-bits "$COEFFICIENT_BITS" \
        --C "$tile_size" \
        --selection-pool "$SELECTION_POOL" \
        --selection-seed "$SELECTION_SEED" \
        --seed "$CLUSTER_SEED" \
        --iterations "$ITERATIONS" \
        --batch-size "$CLUSTER_BATCH_SIZE" \
        --output-root "$ARTIFACT_ROOT"

    echo
    echo "Evaluating C=${tile_size}, ranks=${ranks[*]}"

    "$PYTHON_BIN" SeedLM/k-space/bench_load_and_inference_ksubspaces.py \
        --model "$MODEL" \
        --sparsity "$SPARSITY" \
        --k "$K" \
        --ranks "${ranks[@]}" \
        --coefficient-bits "$COEFFICIENT_BITS" \
        --C "$tile_size" \
        --artifact-root "$ARTIFACT_ROOT" \
        --device "$DEVICE" \
        --seqlen "$SEQLEN" \
        --batch-size "$PPL_BATCH_SIZE" \
        --max-ppl-samples "$MAX_PPL_SAMPLES"
}

run_case 4 1 2
run_case 8 2 4
run_case 16 4 8
run_case 32 8 16
run_case 64 16 32

echo
echo "Tile/rank sweep completed successfully."

#cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification
#
#chmod +x run_ksubspaces_tile_rank_sweep.sh
#
#./run_ksubspaces_tile_rank_sweep.sh
