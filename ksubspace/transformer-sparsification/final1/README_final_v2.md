# ARKS final experiment runner

This folder contains the final ARKS pipeline for OPT and Llama experiments.

## Supported experiments

The master runner can execute five experiments independently.

| Experiment | Coefficients | VQ | Calibration |
|---|---|---|---|
| Adaptive rank baseline | FP16 scalar | No | No |
| Adaptive rank + INT8 | INT8 scalar | No | No |
| Adaptive rank + VQ | VQ | Yes | No |
| Adaptive rank + calibrated VQ | VQ | Yes | Yes |
| Adaptive rank + adaptive sparsity | FP16 or INT8 scalar | No | No |

The first two experiments separate adaptive rank from scalar coefficient
quantization. The VQ experiments evaluate codebook-based coefficient
quantization. The sparsity experiment applies different coefficient sparsities
to sensitive and robust matrices.

## Required files

```text
final/
├── bench_compress_minibatch_vq.py
├── bench_inference_minibatch_vq.py
├── compare_magnitude_pruning.py
├── first_order_rank_allocator.py
├── ksubspaces_artifacts.py
├── run_base_pipeline.sh
└── run_final.sh
```

`compare_magnitude_pruning.py` must support:

```text
--adaptive-coeff-pruning
--sensitivity-json
--sensitive-coeff-sparsity
--robust-coeff-sparsity
--coeff-sparsity-override
```

## Default OPT-125M run

From the repository root:

```bash
cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification
source .venv/bin/activate
chmod +x final/run_final.sh
```

Run all five experiments:

```bash
MODEL=facebook/opt-125m \
MODEL_TAG=opt-125m \
K=512 \
BASE_RANK=4 \
MAX_RANK=6 \
EXTRA_BUDGET_MB=1.88 \
CODEBOOK_SIZE=65536 \
VQ_STAGES=1 \
SENSITIVE_COEFF_SPARSITY=0.20 \
ROBUST_COEFF_SPARSITY=0.35 \
MAX_PPL_SAMPLES=0 \
./final/run_final.sh
```

This produces:

```text
adaptive_baseline_fp16
adaptive_baseline_int8
adaptive_vq_nocalib
adaptive_vq_calibrated
adaptive_sparsity
```

## Run selected experiments

Only FP16 and INT8 baselines:

```bash
RUN_ADAPTIVE_FP16=1 \
RUN_ADAPTIVE_INT8=1 \
RUN_VQ_NOCALIB=0 \
RUN_VQ_CALIBRATED=0 \
RUN_ADAPTIVE_SPARSITY=0 \
./final/run_final.sh
```

Only adaptive sparsity:

```bash
RUN_ADAPTIVE_FP16=0 \
RUN_ADAPTIVE_INT8=0 \
RUN_VQ_NOCALIB=0 \
RUN_VQ_CALIBRATED=0 \
RUN_ADAPTIVE_SPARSITY=1 \
SENSITIVE_COEFF_SPARSITY=0.20 \
ROBUST_COEFF_SPARSITY=0.35 \
./final/run_final.sh
```

If the required scalar artifact does not exist, the runner builds it first.

## VQ configurations

### 1x65536

```bash
CODEBOOK_SIZE=65536 \
VQ_STAGES=1 \
./final/run_final.sh
```

### 3x256

```bash
CODEBOOK_SIZE=256 \
VQ_STAGES=3 \
./final/run_final.sh
```

`1x65536` and `3x256` are coefficient VQ settings. They are not scalar INT8.

## Adaptive coefficient sparsity

Example:

```bash
SENSITIVE_COEFF_SPARSITY=0.20 \
ROBUST_COEFF_SPARSITY=0.30 \
./final/run_final.sh
```

The notation `20/30` means:

```text
20% coefficient sparsity for sensitive matrices
30% coefficient sparsity for robust matrices
```

For the current OPT-125M allocation, the expected split is 9 sensitive
matrices and 63 robust matrices.

A matrix-specific override can be supplied with:

```bash
COEFF_SPARSITY_OVERRIDES="layer_00_fc1=0.10 layer_04_self_attn_v_proj=0.15" \
./final/run_final.sh
```

By default, adaptive sparsity uses the FP16 scalar representation:

```bash
SPARSITY_SOURCE=fp16
```

To apply sparsity after scalar INT8 quantization:

```bash
SPARSITY_SOURCE=int8 \
./final/run_final.sh
```

## Full WikiText-2 evaluation

```bash
MAX_PPL_SAMPLES=0
```

A small smoke test can use:

```bash
MAX_PPL_SAMPLES=8
```

## Output structure

```text
SeedLM/data/SeedLM_compressed/<MODEL_TAG>/final/
├── adaptive_baseline_fp16/
├── adaptive_baseline_int8/
├── adaptive_vq_nocalib/
├── adaptive_vq_calibrated/
└── adaptive_sparsity/
    └── results/
        ├── adaptive_sparsity.csv
        └── adaptive_sparsity_adaptive_per_matrix.csv
```

Each baseline/VQ directory contains its own rank profile, rank allocation,
artifacts, and inference results.

## What to report

For comparisons, use the same storage definition for every method. Report:

- WikiText-2 PPL
- artifact storage and total model storage
- compression time and inference time
- effective coefficient sparsity when applicable

Do not compare ARKS coefficient-only storage directly with another baseline's
total compressed model size.

## Interpretation

`adaptive_baseline_fp16` isolates the adaptive-rank representation.

`adaptive_baseline_int8` measures the additional effect of direct scalar
8-bit coefficient quantization.

`adaptive_vq_nocalib` measures coefficient codebook quantization without
calibration.

`adaptive_vq_calibrated` measures the same representation after codebook
calibration.

`adaptive_sparsity` measures sensitivity-aware coefficient pruning, with
different sparsity levels for sensitive and robust matrices.
