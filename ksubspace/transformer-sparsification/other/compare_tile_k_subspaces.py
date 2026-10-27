#!/usr/bin/env python3
"""Compare linear K-subspaces of several ranks on SeedLM pruned tiles.

This is a layer-level screening experiment. It uses the same Cx1 tiles and
structured-pruning indices as SeedLM, then fits a union of rank-r linear
subspaces. It reports reconstruction error and estimated recovery metadata.
It does not modify model artifacts or compute model perplexity.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from SeedLM.SeedLM_cpu import (
    _pruned_col_matrix_indices_from_sparsity,
    _weight_col_matrices_flat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True, type=Path)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--C", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-pool", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--coefficient-bytes",
        type=int,
        default=2,
        help="Storage estimate per coefficient; 2 corresponds to float16.",
    )
    parser.add_argument(
        "--basis-bytes",
        type=int,
        default=2,
        help="Storage estimate per basis value; 2 corresponds to float16.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tile_k_subspaces"),
    )
    return parser.parse_args()


def normalized_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def initial_labels(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Direction-aware initialization shared across all tested ranks."""
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=3,
        batch_size=min(4096, len(x)),
        max_iter=100,
        reassignment_ratio=0.0,
        random_state=seed,
    )
    return model.fit_predict(normalized_rows(x)).astype(np.int32)


def update_bases(
    x: np.ndarray,
    labels: np.ndarray,
    k: int,
    rank: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Estimate one orthonormal rank-r basis per cluster using PCA/SVD."""
    dimension = x.shape[1]
    bases = np.zeros((k, dimension, rank), dtype=np.float32)
    empty = 0
    for cluster_id in range(k):
        members = x[labels == cluster_id].astype(np.float64, copy=False)
        if len(members) == 0:
            empty += 1
            # A random orthonormal basis is only a temporary placeholder.
            random_matrix = rng.normal(size=(dimension, rank))
            q, _ = np.linalg.qr(random_matrix)
            bases[cluster_id] = q[:, :rank].astype(np.float32)
            continue
        _, _, vh = np.linalg.svd(members, full_matrices=False)
        available = min(rank, vh.shape[0])
        bases[cluster_id, :, :available] = vh[:available].T.astype(np.float32)
        if available < rank:
            # Complete a deficient basis without changing its existing span.
            candidate = np.concatenate(
                [
                    bases[cluster_id, :, :available],
                    rng.normal(size=(dimension, rank - available)).astype(np.float32),
                ],
                axis=1,
            )
            q, _ = np.linalg.qr(candidate)
            bases[cluster_id] = q[:, :rank].astype(np.float32)
    return bases, empty


def assign_subspaces(
    x: np.ndarray,
    bases: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Assign each vector to the subspace with minimum projection residual."""
    labels = np.empty(len(x), dtype=np.int32)
    residual_sum = 0.0
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].astype(np.float32, copy=False)
        # projection[b, k, r] = x_b^T U_k
        projection = np.einsum("bd,kdr->bkr", xb, bases, optimize=True)
        captured = np.sum(projection * projection, axis=2)
        chosen = np.argmax(captured, axis=1)
        labels[start : start + len(xb)] = chosen
        norm2 = np.sum(xb * xb, axis=1)
        residual = norm2 - captured[np.arange(len(xb)), chosen]
        residual_sum += np.maximum(residual, 0.0).sum(dtype=np.float64)
    return labels, residual_sum


def fit_k_subspaces(
    train: np.ndarray,
    initial: np.ndarray,
    k: int,
    rank: int,
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    rng = np.random.default_rng(seed + rank)
    labels = initial.copy()
    completed_iterations = 0
    empty = 0
    for iteration in range(iterations):
        bases, empty = update_bases(train, labels, k, rank, rng)
        new_labels, _ = assign_subspaces(train, bases, batch_size)
        completed_iterations = iteration + 1
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
    bases, empty = update_bases(train, labels, k, rank, rng)
    return bases, labels, completed_iterations, empty


def evaluate(
    x: np.ndarray,
    bases: np.ndarray,
    batch_size: int,
) -> tuple[float, float, float, int]:
    labels, residual_sum = assign_subspaces(x, bases, batch_size)
    total_energy = float(np.sum(x.astype(np.float64) ** 2))
    mse = residual_sum / x.size
    relative_error = math.sqrt(residual_sum / max(total_energy, 1e-30))
    captured_energy = 1.0 - residual_sum / max(total_energy, 1e-30)
    return mse, relative_error, captured_energy, int(np.unique(labels).size)


def storage_estimate(
    n_tiles: int,
    k: int,
    dimension: int,
    rank: int,
    coefficient_bytes: int,
    basis_bytes: int,
) -> dict[str, float]:
    id_bits = max(1, math.ceil(math.log2(max(k, 2))))
    ids = n_tiles * id_bits / 8
    coefficients = n_tiles * rank * coefficient_bytes
    bases = k * dimension * rank * basis_bytes
    mib = 1024**2
    return {
        "cluster_ids_mb": ids / mib,
        "coefficients_mb": coefficients / mib,
        "raw_bases_mb": bases / mib,
        "recovery_metadata_mb": (ids + coefficients + bases) / mib,
    }


def save_plot(rows: list[dict], output: Path) -> None:
    ranks = [row["rank"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    panels = [
        ("reconstruction_mse", "Tile reconstruction MSE", "lower is better"),
        ("captured_energy", "Energy captured", "higher is better"),
        ("recovery_metadata_mb", "Estimated recovery metadata", "lower is better"),
    ]
    for axis, (key, title, direction) in zip(axes, panels):
        values = [row[key] for row in rows]
        bars = axis.bar([str(rank) for rank in ranks], values, color="#4C78A8")
        axis.set_title(f"{title}\n{direction}")
        axis.set_xlabel("Subspace rank")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            text = f"{value:.3e}" if key == "reconstruction_mse" else f"{value:.4f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("K-subspaces trade-off on SeedLM pruned tiles")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if any(rank < 1 or rank > args.C for rank in args.ranks):
        raise ValueError(f"Every rank must be between 1 and C={args.C}")

    # Stored FC weights are [out, in]; SeedLM operates on the transpose.
    weight = np.load(args.weight).T
    n, all_tiles, _ = _weight_col_matrices_flat(weight, args.C, np.float32)
    pruned_ids = _pruned_col_matrix_indices_from_sparsity(
        n,
        args.sparsity,
        args.selection_pool,
        args.selection_seed,
    )
    tiles = all_tiles[pruned_ids].astype(np.float32, copy=False)
    if not 1 <= args.k <= len(tiles):
        raise ValueError(f"k must be in [1, {len(tiles)}]")

    rng = np.random.default_rng(args.seed)
    if len(tiles) > args.sample_size:
        train_ids = rng.choice(len(tiles), args.sample_size, replace=False)
        train = tiles[train_ids]
    else:
        train = tiles

    print(
        f"Weight={args.weight}, pruned tiles={len(tiles)}, "
        f"training sample={len(train)}, C={args.C}, K={args.k}"
    )
    initialization = initial_labels(train, args.k, args.seed)
    rows: list[dict] = []
    for rank in args.ranks:
        started = time.perf_counter()
        bases, _, iterations, empty = fit_k_subspaces(
            train,
            initialization,
            args.k,
            rank,
            args.iterations,
            args.batch_size,
            args.seed,
        )
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        mse, relative_error, captured_energy, occupied = evaluate(
            tiles, bases, args.batch_size
        )
        evaluation_seconds = time.perf_counter() - started
        storage = storage_estimate(
            len(tiles),
            args.k,
            args.C,
            rank,
            args.coefficient_bytes,
            args.basis_bytes,
        )
        row = {
            "weight": str(args.weight),
            "sparsity": args.sparsity,
            "k": args.k,
            "rank": rank,
            "C": args.C,
            "n_pruned_tiles": len(tiles),
            "training_sample": len(train),
            "iterations": iterations,
            "empty_training_clusters": empty,
            "occupied_evaluation_clusters": occupied,
            "fit_seconds": fit_seconds,
            "evaluation_seconds": evaluation_seconds,
            "reconstruction_mse": mse,
            "relative_frobenius_error": relative_error,
            "captured_energy": captured_energy,
            **storage,
        }
        rows.append(row)
        print(row, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tile_k_subspaces_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, args.output_dir / "tile_k_subspaces_comparison.png")
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
