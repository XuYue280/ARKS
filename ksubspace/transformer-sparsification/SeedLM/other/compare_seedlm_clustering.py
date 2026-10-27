#!/usr/bin/env python3
"""Compare center-selection methods on SeedLM pruned Cx1 weight tiles.

Run from the repository root after the OPT components have been stored.
This script does not modify the model or saved SeedLM artifacts.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus

from SeedLM.other.SeedLM_cpu import (
    _build_precomputed,
    _pruned_col_matrix_indices_from_sparsity,
    _weight_col_matrices_flat,
    select_best_seed_numba,
)


METHODS = ("random", "kmeans++", "minibatch", "kmedoids")


def sample_rows(x: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= limit:
        return x
    return x[rng.choice(len(x), size=limit, replace=False)]


def centers_random(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=k, replace=False)].astype(np.float32)


def centers_kmeans_pp(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    centers, _ = kmeans_plusplus(x, n_clusters=k, random_state=seed)
    return centers.astype(np.float32)


def centers_minibatch(x: np.ndarray, k: int, seed: int, batch_size: int) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=3,
        batch_size=min(batch_size, len(x)),
        max_iter=50,
        random_state=seed,
        reassignment_ratio=0.01,
    )
    model.fit(x)
    return model.cluster_centers_.astype(np.float32)


def centers_kmedoids(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError as exc:
        raise RuntimeError(
            "kmedoids requires scikit-learn-extra: pip install scikit-learn-extra"
        ) from exc
    model = KMedoids(
        n_clusters=k,
        metric="euclidean",
        method="pam",
        init="k-medoids++",
        random_state=seed,
    )
    model.fit(x)
    return model.cluster_centers_.astype(np.float32)


def nearest_error_chunked(x: np.ndarray, centers: np.ndarray, chunk: int) -> tuple[float, np.ndarray]:
    total = 0.0
    ids = np.empty(len(x), dtype=np.int32)
    c2 = np.sum(centers * centers, axis=1)[None, :]
    for start in range(0, len(x), chunk):
        xb = x[start:start + chunk]
        d2 = np.sum(xb * xb, axis=1)[:, None] + c2 - 2.0 * (xb @ centers.T)
        np.maximum(d2, 0.0, out=d2)
        local_ids = np.argmin(d2, axis=1)
        ids[start:start + len(xb)] = local_ids
        total += d2[np.arange(len(xb)), local_ids].sum(dtype=np.float64)
    return total / (len(x) * x.shape[1]), ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True, type=Path,
                   help="Stored layer weight, e.g. data/original_model/opt-125m/layer_00_fc1_weight.npy")
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--sample-size", type=int, default=50000,
                   help="Training sample for kmeans methods")
    p.add_argument("--medoid-sample-size", type=int, default=5000,
                   help="Smaller sample used by PAM K-medoids")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--distance-chunk", type=int, default=8192)
    p.add_argument("--seed", type=int, default=25565)
    p.add_argument("--selection-pool", type=int, default=100)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--C", type=int, default=8)
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--seed-bits", type=int, default=16)
    p.add_argument("--output", type=Path, default=Path("clustering_comparison.csv"))
    args = p.parse_args()

    # Stored Hugging Face FC weights are [out, in]; SeedLM operates on the transpose.
    w = np.load(args.weight).T
    n, all_tiles, _ = _weight_col_matrices_flat(w, args.C, np.float32)
    pruned_ids = _pruned_col_matrix_indices_from_sparsity(
        n, args.sparsity, args.selection_pool, args.selection_seed
    )
    x = all_tiles[pruned_ids].astype(np.float32, copy=False)
    if not 1 <= args.k <= len(x):
        raise ValueError(f"k must be in [1, {len(x)}], got {args.k}")

    rng = np.random.default_rng(args.seed)
    train = sample_rows(x, args.sample_size, rng)
    medoid_train = sample_rows(x, args.medoid_sample_size, rng)
    _, us_stack, pinv_stack = _build_precomputed(args.C, args.P, args.seed_bits, args.seed)

    rows = []
    for method in args.methods:
        t0 = time.perf_counter()
        if method == "random":
            centers = centers_random(train, args.k, args.seed)
        elif method == "kmeans++":
            centers = centers_kmeans_pp(train, args.k, args.seed)
        elif method == "minibatch":
            centers = centers_minibatch(train, args.k, args.seed, args.batch_size)
        else:
            if args.k > len(medoid_train):
                raise ValueError("k exceeds medoid sample size")
            centers = centers_kmedoids(medoid_train, args.k, args.seed)
        clustering_seconds = time.perf_counter() - t0

        raw_mse, raw_ids = nearest_error_chunked(x, centers, args.distance_chunk)
        t1 = time.perf_counter()
        _, _, reconstructed = select_best_seed_numba(centers, us_stack, pinv_stack)
        seedlm_fit_seconds = time.perf_counter() - t1
        seedlm_mse, _ = nearest_error_chunked(x, reconstructed, args.distance_chunk)

        row = {
            "weight": str(args.weight), "method": method,
            "sparsity": args.sparsity, "k": args.k,
            "n_pruned_tiles": len(x), "training_sample": len(train),
            "clustering_seconds": clustering_seconds,
            "seedlm_fit_seconds": seedlm_fit_seconds,
            "raw_center_mse": raw_mse, "seedlm_reconstruction_mse": seedlm_mse,
            "occupied_clusters": int(np.unique(raw_ids).size),
        }
        rows.append(row)
        print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()



# --weight /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification/data/original_model/opt-125m/layer_00_fc1_weight.npy
# --sparsity 0.5
# --k 16
# --methods random kmeans++ minibatch
# --sample-size 10000
# --output /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification/results/clustering_smoke.csv


# --weight /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification/data/original_model/opt-125m/layer_00_fc1_weight.npy
# --sparsity 0.5
# --k 16
# --methods random
# --sample-size 2000
# --distance-chunk 1024
# --seed-bits 8
# --output /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification/results/random_smoke.csv


