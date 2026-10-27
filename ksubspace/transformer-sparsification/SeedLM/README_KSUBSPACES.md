# SeedLM K-subspaces: actual compression and inference

Keep the existing K-lines files. Copy these new files into `SeedLM/`:

- `ksubspaces_artifacts.py`
- `bench_compress_and_store_ksubspaces.py`
- `bench_load_and_inference_ksubspaces.py`

## 1. Compress and save real artifacts

The command below trains with every recovered tile because `--sample-size 0`.

```bash
cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 \
/root/.virtualenvs/transformer-sparsification/bin/python \
-m SeedLM.bench_compress_and_store_ksubspaces \
--model facebook/opt-125m \
--sparsity 0.5 \
--k 128 \
--ranks 1 2 4 \
--C 8 \
--sample-size 0 \
--iterations 15 \
--batch-size 2048
```

Artifacts are saved under:

```text
data/SeedLM_compressed/opt-125m/ksubspaces/
├── sp_0.50_K128_R1/
├── sp_0.50_K128_R2/
└── sp_0.50_K128_R4/
```

Each layer directory contains:

```text
structured_pruning.npy
subspace_bases.npy
nearest_subspace_ids.npy
tile_coefficients.npy
metadata.json
```

Compression CSV and JSON summaries are placed in:

```text
data/SeedLM_compressed/opt-125m/ksubspaces/benchmark_results/
```

## 2. Load, reconstruct, and measure PPL

Smoke test:

```bash
cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 \
/root/.virtualenvs/transformer-sparsification/bin/python \
-m SeedLM.bench_load_and_inference_ksubspaces \
--model facebook/opt-125m \
--sparsity 0.5 \
--k 128 \
--ranks 1 2 4 \
--device cuda \
--max-ppl-samples 8
```

Full WikiText-2 PPL:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 \
/root/.virtualenvs/transformer-sparsification/bin/python \
-m SeedLM.bench_load_and_inference_ksubspaces \
--model facebook/opt-125m \
--sparsity 0.5 \
--k 128 \
--ranks 1 2 4 \
--device cuda \
--max-ppl-samples 0
```

Inference CSV and JSON summaries are saved under:

```text
results/opt-125m/ksubspaces/
```

They report PPL, all-FC reconstruction MSE, nonzero MSE, relative Frobenius
error, actual artifact storage, total storage including uncompressed static
parameters, compression/load/reconstruction/inference time, peak CPU RSS, and
peak CUDA allocated and reserved memory.

## Fast one-layer smoke test

Use `--max-layers 1` in both commands. PPL from this mode is a hybrid-model
smoke test because only the selected layer is reconstructed; use all layers for
the final comparison.
