# final: ARKS quantization/sparsification experiments

This folder preserves the previous adaptive-rank K-subspaces pipeline and adds the experiment requested for magnitude pruning.

## What `run_final.sh` does
1. Builds the existing ARKS representation with fixed 8x1 tiles, K-subspaces, and adaptive rank 4/6.
2. Uses scalar INT8 coefficients, with no coefficient VQ/calibration, so Stage 2 can be replaced by coefficient magnitude pruning.
3. Evaluates the dense model.
4. Evaluates direct magnitude pruning of decoder linear weights.
5. Evaluates magnitude pruning of ARKS coefficients at the same pruning fractions.
6. Writes one CSV containing PPL and sparsity information.

## OPT-125M
```bash
MODEL=facebook/opt-125m MODEL_TAG=opt-125m K=512 \
BASE_RANK=4 MAX_RANK=6 EXTRA_BUDGET_MB=1.88 \
SPARSITIES="0.25 0.50 0.75" \
./final/run_final.sh
```

## Llama-3.2-1B
Adjust `EXTRA_BUDGET_MB` to the budget used in your Llama experiment.
```bash
MODEL=meta-llama/Llama-3.2-1B MODEL_TAG=Llama-3.2-1B K=256 \
BASE_RANK=4 MAX_RANK=6 EXTRA_BUDGET_MB=64.13 \
SPARSITIES="0.25 0.50 0.75" \
./final/run_final.sh
```

The output is under:
`SeedLM/data/SeedLM_compressed/<MODEL_TAG>/final/results/magnitude_comparison.csv`

`MAX_PPL_SAMPLES=0` evaluates all available WikiText-2 chunks, matching the existing runner convention.
