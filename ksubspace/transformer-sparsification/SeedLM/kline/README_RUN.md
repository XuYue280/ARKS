# SeedLM K-lines experiment

Copy the four Python files into the existing `SeedLM/` directory:

```bash
cp SeedLM_cpu_klines.py SeedLM/SeedLM_cpu_klines.py
cp bench_compress_and_store_klines.py SeedLM/bench_compress_and_store_klines.py
cp bench_load_and_inference_klines_metrics.py SeedLM/bench_load_and_inference_klines_metrics.py
cp analyze_cluster_permutation_rank_improved.py SeedLM/analyze_cluster_permutation_rank_improved.py
```

The original project files are not replaced.

## Smoke test compression

```bash
cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
/root/.virtualenvs/transformer-sparsification/bin/python \
-m SeedLM.bench_compress_and_store_klines \
--model facebook/opt-125m \
--section s5a \
--sparsity 0.5 \
--k-cluster-values 128 \
--clustering-method klines \
--clustering-sample-size 10000 \
--klines-iterations 5
```

## Short PPL and reconstruction-error test

```bash
/root/.virtualenvs/transformer-sparsification/bin/python \
-m SeedLM.bench_load_and_inference_klines_metrics \
--model facebook/opt-125m \
--section s5a \
--sparsity 0.5 \
--k-cluster-values 128 \
--clustering-method klines \
--device cuda \
--max-ppl-samples 8
```

For full WikiText-2, use `--max-ppl-samples 0`.

## Improved permutation and rank analysis

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/root/.virtualenvs/transformer-sparsification/bin/python \
-m SeedLM.analyze_cluster_permutation_rank_improved \
--weight data/original_model/opt-125m/layer_00_fc1_weight.npy \
--artifact-dir data/SeedLM_compressed/opt-125m/klines/sp_0.50_K128/layer_00_fc1 \
--sparsity 0.5 \
--k 128 \
--clustering-method klines \
--C 8 \
--P 3 \
--seed-bits 16 \
--top-clusters 12 \
--output-dir results/cluster_rank_analysis/klines_layer_00_fc1
```

The K-lines representation stores one signed float16 coefficient per matrix
tile in the current reference implementation. This increases storage relative
to K-means++. Storage optimization should be evaluated only after accuracy and
rank behavior are validated.
