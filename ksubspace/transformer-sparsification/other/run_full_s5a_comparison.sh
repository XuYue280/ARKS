#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

PY=/root/.virtualenvs/transformer-sparsification/bin/python
MODEL=facebook/opt-125m
MODEL_TAG=opt-125m
SPARSITY=0.5
K=128
CLUSTERING_SAMPLE_SIZE=50000
# Zero means the complete WikiText-2 test stream.
PPL_SAMPLES=0
TAG=$(date +%Y%m%d_%H%M%S)

echo "Experiment tag: ${TAG}"

echo "[1/7] Compress structured-pruning baseline"
"$PY" -m SeedLM.bench_compress_and_store_clustering \
  --model "$MODEL" \
  --section s4 \
  --sparsity "$SPARSITY" \
  --server-temp \
  --job-tag "${TAG}_s4_compress"

echo "[2/7] Compress Random SeedLM"
"$PY" -m SeedLM.bench_compress_and_store_clustering \
  --model "$MODEL" \
  --section s5a \
  --sparsity "$SPARSITY" \
  --k-cluster-values "$K" \
  --clustering-method random \
  --server-temp \
  --job-tag "${TAG}_random_compress"

echo "[3/7] Compress K-means++ SeedLM"
"$PY" -m SeedLM.bench_compress_and_store_clustering \
  --model "$MODEL" \
  --section s5a \
  --sparsity "$SPARSITY" \
  --k-cluster-values "$K" \
  --clustering-method kmeans++ \
  --clustering-sample-size "$CLUSTERING_SAMPLE_SIZE" \
  --server-temp \
  --job-tag "${TAG}_kmeanspp_compress"

echo "[4/7] Evaluate dense baseline on complete WikiText-2"
"$PY" -m SeedLM.bench_load_and_inference_clustering_metrics \
  --model "$MODEL" \
  --section s1 \
  --device cuda \
  --max-ppl-samples "$PPL_SAMPLES" \
  --server-temp \
  --job-tag "${TAG}_dense"

echo "[5/7] Evaluate structured pruning on complete WikiText-2"
"$PY" -m SeedLM.bench_load_and_inference_clustering_metrics \
  --model "$MODEL" \
  --section s4 \
  --sparsity "$SPARSITY" \
  --device cuda \
  --max-ppl-samples "$PPL_SAMPLES" \
  --server-temp \
  --job-tag "${TAG}_s4"

echo "[6/7] Evaluate Random SeedLM on complete WikiText-2"
"$PY" -m SeedLM.bench_load_and_inference_clustering_metrics \
  --model "$MODEL" \
  --section s5a \
  --sparsity "$SPARSITY" \
  --k-cluster-values "$K" \
  --clustering-method random \
  --device cuda \
  --max-ppl-samples "$PPL_SAMPLES" \
  --server-temp \
  --job-tag "${TAG}_random"

echo "[7/7] Evaluate K-means++ SeedLM on complete WikiText-2"
"$PY" -m SeedLM.bench_load_and_inference_clustering_metrics \
  --model "$MODEL" \
  --section s5a \
  --sparsity "$SPARSITY" \
  --k-cluster-values "$K" \
  --clustering-method kmeans++ \
  --device cuda \
  --max-ppl-samples "$PPL_SAMPLES" \
  --server-temp \
  --job-tag "${TAG}_kmeanspp"

export TAG SPARSITY K PPL_SAMPLES MODEL_TAG

"$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

tag = os.environ["TAG"]
sp = float(os.environ["SPARSITY"])
k = int(os.environ["K"])
samples = int(os.environ["PPL_SAMPLES"])
model_tag = os.environ["MODEL_TAG"]

inference_dir = (
    Path("results/server_temp/CPU_inference_result") / model_tag
)
compression_dir = (
    Path("results/server_temp/CPU_compress_result") / model_tag
)

experiments = [
    {
        "configuration": "Dense",
        "method": "dense",
        "inference": inference_dir / f"{tag}_dense.JSON",
        "compression": None,
    },
    {
        "configuration": "Structured pruning",
        "method": "structured_pruning",
        "inference": inference_dir / f"{tag}_s4.JSON",
        "compression": compression_dir / f"{tag}_s4_compress.JSON",
    },
    {
        "configuration": "Random SeedLM",
        "method": "random",
        "inference": inference_dir / f"{tag}_random.JSON",
        "compression": compression_dir / f"{tag}_random_compress.JSON",
    },
    {
        "configuration": "K-means++ SeedLM",
        "method": "kmeans++",
        "inference": inference_dir / f"{tag}_kmeanspp.JSON",
        "compression": compression_dir / f"{tag}_kmeanspp_compress.JSON",
    },
]


def one_record(path: Path, required_key: str):
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    matches = [record for record in records if required_key in record]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {required_key} record in {path}; found {len(matches)}"
        )
    return matches[0]


rows = []
for experiment in experiments:
    inference = one_record(experiment["inference"], "ppl")
    compression = (
        one_record(experiment["compression"], "compress_timings")
        if experiment["compression"] is not None
        else None
    )
    error = inference.get("reconstruction_error", {})
    storage = inference.get("storage_mb", {})
    inference_times = inference.get("load_and_inference_time", {})
    compression_times = (
        compression.get("compress_timings", {}) if compression else {}
    )
    dense = experiment["method"] == "dense"

    rows.append({
        "configuration": experiment["configuration"],
        "method": experiment["method"],
        "sparsity": 0.0 if dense else sp,
        "k": k if "SeedLM" in experiment["configuration"] else "",
        "ppl": inference["ppl"],
        "mse_all_weights": 0.0 if dense else error.get("mse_all_weights", ""),
        "mse_nnz": 0.0 if dense else error.get("mse_nonzero_weights", ""),
        "relative_frobenius_error": (
            0.0 if dense else error.get("relative_frobenius_error", "")
        ),
        "storage_mb": storage.get("total", ""),
        "compression_seconds": compression_times.get("total", ""),
        "load_and_inference_seconds": inference_times.get("total", ""),
        "inference_seconds": inference_times.get("inference_time", ""),
        "reconstruction_metrics_seconds": inference_times.get(
            "reconstruction_metrics", ""
        ),
        "ppl_scope": "full_wikitext2" if samples == 0 else f"{samples}_chunks",
    })

output = Path("results") / f"complete_full_comparison_{tag}.csv"
with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nFinal full-dataset comparison")
print("=" * 110)
for row in rows:
    mse = float(row["mse_all_weights"])
    relative = float(row["relative_frobenius_error"])
    print(
        f"{row['configuration']:<24} "
        f"PPL={float(row['ppl']):>12.4f}  "
        f"MSE={mse:.8e}  "
        f"RelativeError={relative:.8f}  "
        f"Storage={row['storage_mb']} MB"
    )
print("=" * 110)
print(f"CSV saved to: {output.resolve()}")
PY

echo "Done."
