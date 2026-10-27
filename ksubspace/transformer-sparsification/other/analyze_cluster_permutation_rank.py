#!/usr/bin/env python3
"""Visualize tile-cluster reordering and measure cluster-wise rank.

This analyzes one stored Section-5A layer. It does not change model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from SeedLM.other.SeedLM_cpu_clustering import SeedLM_load


def tile_grid(matrix: np.ndarray, c: int) -> np.ndarray:
    """Return [row_block, column, C] tiles without changing within-tile order."""
    rows, cols = matrix.shape
    if rows % c:
        raise ValueError(f"Matrix rows={rows} must be divisible by C={c}")
    return matrix.reshape(rows // c, c, cols).transpose(0, 2, 1)


def alternating_lexicographic_order(
    labels: np.ndarray, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find deterministic row-block/column orders from a categorical label grid.

    This is a scalable seriation heuristic. It alternately lexicographically
    sorts row-block label patterns and column label patterns. Non-pruned tiles
    (-1) are placed after cluster labels during sorting.
    """
    work = labels.astype(np.int64, copy=True)
    missing = int(work.max(initial=-1) + 1)
    work[work < 0] = missing
    row_order = np.arange(work.shape[0])
    col_order = np.arange(work.shape[1])

    for _ in range(iterations):
        row_view = work[np.ix_(row_order, col_order)]
        row_local = np.lexsort(row_view[:, ::-1].T)
        row_order = row_order[row_local]

        col_view = work[np.ix_(row_order, col_order)]
        col_local = np.lexsort(col_view[::-1, :])
        col_order = col_order[col_local]

    return row_order, col_order


def expand_row_blocks(row_block_order: np.ndarray, c: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(block * c, (block + 1) * c) for block in row_block_order]
    )


def rank_metrics(tiles: np.ndarray, tolerance: float) -> dict[str, float | int]:
    """Compute rank statistics for a [members, C] cluster."""
    # Each tile is a column, as requested: T is [C, members].
    t = tiles.astype(np.float64, copy=False).T
    singular = np.linalg.svd(t, full_matrices=False, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        return {
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "rank_95_energy": 0,
            "first_singular_energy_fraction": 0.0,
            "best_rank1_relative_error": 0.0,
        }

    threshold = tolerance * singular[0]
    numerical_rank = int(np.sum(singular > threshold))

    singular_sum = float(singular.sum())
    probabilities = singular / singular_sum
    positive = probabilities > 0
    effective_rank = float(
        np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    )

    energy = singular * singular
    total_energy = float(energy.sum())
    cumulative = np.cumsum(energy) / total_energy
    rank_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    first_fraction = float(energy[0] / total_energy)
    rank1_error = float(math.sqrt(max(0.0, 1.0 - first_fraction)))
    return {
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "rank_95_energy": rank_95,
        "first_singular_energy_fraction": first_fraction,
        "best_rank1_relative_error": rank1_error,
    }


def save_cluster_map(
    original_labels: np.ndarray,
    reordered_labels: np.ndarray,
    output: Path,
) -> None:
    k = int(max(original_labels.max(initial=-1), 0) + 1)
    base = plt.get_cmap("turbo", max(k, 1))
    colors = [(0.82, 0.82, 0.82, 1.0)] + [base(i) for i in range(max(k, 1))]
    cmap = ListedColormap(colors)
    original_plot = original_labels + 1
    reordered_plot = reordered_labels + 1

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(original_plot, aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=k)
    axes[0].set_title("Original tile layout colored by cluster ID")
    axes[0].set_ylabel("Row block (C rows)")
    axes[0].set_xlabel("Weight-matrix column")

    axes[1].imshow(reordered_plot, aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=k)
    axes[1].set_title("After row-block and column permutation")
    axes[1].set_ylabel("Reordered row block")
    axes[1].set_xlabel("Reordered column")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_weight_comparison(
    original_reordered: np.ndarray,
    representative_reordered: np.ndarray,
    reconstructed_reordered: np.ndarray,
    output: Path,
) -> None:
    limit = float(np.percentile(np.abs(original_reordered), 99.5))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0], original_reordered, "Original weights after permutation"),
        (axes[1], representative_reordered,
         "Each cluster replaced by one observed representative tile"),
        (axes[2], reconstructed_reordered,
         "Reconstructed weights after shared-cluster replacement"),
    ):
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest",
                            cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(title)
        axis.set_ylabel("Reordered row")
        axis.set_xlabel("Reordered column")
    fig.colorbar(image, ax=axes, label="Weight value", shrink=0.8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_rank_plot(rows: list[dict[str, float | int]], output: Path) -> None:
    ids = np.array([int(row["cluster_id"]) for row in rows])
    effective = np.array([float(row["effective_rank"]) for row in rows])
    rank95 = np.array([int(row["rank_95_energy"]) for row in rows])
    rank1 = np.array([float(row["best_rank1_relative_error"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             constrained_layout=True)
    axes[0].bar(ids - 0.2, effective, width=0.4, label="Effective rank")
    axes[0].bar(ids + 0.2, rank95, width=0.4, label="Rank for 95% energy")
    axes[0].set_ylabel("Rank")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(ids, rank1, width=0.8, color="#b24745")
    axes[1].set_ylabel("Relative error")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_title("Best rank-1 approximation error")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True, type=Path,
                        help="Original saved Hugging Face FC weight [out, in]")
    parser.add_argument("--artifact-dir", required=True, type=Path,
                        help="Section-5A layer folder containing four .npy artifacts")
    parser.add_argument("--sparsity", required=True, type=float)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--C", type=int, default=8)
    parser.add_argument("--P", type=int, default=3)
    parser.add_argument("--seed-bits", type=int, default=16)
    parser.add_argument("--ordering-iterations", type=int, default=5)
    parser.add_argument("--rank-tolerance", type=float, default=1e-3,
                        help="Relative numerical-rank threshold: tol * largest singular value")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cluster_rank_analysis"))
    args = parser.parse_args()

    stored = np.load(args.weight)
    original = stored.T.astype(np.float32, copy=False)  # SeedLM orientation [in, out]
    rows, cols = original.shape
    reconstructed, _, _ = SeedLM_load(
        rows, cols, args.sparsity, args.k,
        load_path=args.artifact_dir,
        C=args.C, P=args.P, K_bits=args.seed_bits,
    )
    reconstructed = reconstructed.astype(np.float32, copy=False)

    labels_flat = np.load(args.artifact_dir / "nearest_cluster_center_ids.npy")
    row_blocks = rows // args.C
    if labels_flat.size != row_blocks * cols:
        raise ValueError(
            f"Label count={labels_flat.size}, expected {row_blocks * cols}"
        )
    labels = labels_flat.reshape(row_blocks, cols).astype(np.int32, copy=False)
    original_tiles = tile_grid(original, args.C)
    reconstructed_tiles = tile_grid(reconstructed, args.C)
    representative_tiles = original_tiles.copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float | int]] = []
    for cluster_id in range(args.k):
        mask = labels == cluster_id
        members = original_tiles[mask]
        replacements = reconstructed_tiles[mask]
        if members.shape[0] == 0:
            continue
        metrics = rank_metrics(members, args.rank_tolerance)
        mean_tile = members.astype(np.float64).mean(axis=0)
        representative_id = int(
            np.argmin(np.sum((members.astype(np.float64) - mean_tile) ** 2, axis=1))
        )
        representative = members[representative_id]
        representative_tiles[mask] = representative
        representative_difference = (
            members.astype(np.float64) - representative.astype(np.float64)
        )
        difference = members.astype(np.float64) - replacements.astype(np.float64)
        denominator = float(np.sum(members.astype(np.float64) ** 2))
        representative_relative_error = (
            float(np.sqrt(np.sum(representative_difference ** 2) / denominator))
            if denominator > 0 else 0.0
        )
        replacement_relative_error = (
            float(np.sqrt(np.sum(difference ** 2) / denominator))
            if denominator > 0 else 0.0
        )
        metric_rows.append({
            "cluster_id": cluster_id,
            "member_tiles": int(members.shape[0]),
            **metrics,
            "observed_representative_relative_error": representative_relative_error,
            "seedlm_replacement_relative_error": replacement_relative_error,
        })

    csv_path = args.output_dir / "cluster_rank_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    row_block_order, col_order = alternating_lexicographic_order(
        labels, args.ordering_iterations
    )
    row_order = expand_row_blocks(row_block_order, args.C)
    reordered_labels = labels[np.ix_(row_block_order, col_order)]
    original_reordered = original[np.ix_(row_order, col_order)]
    representative_matrix = (
        representative_tiles.transpose(0, 2, 1).reshape(rows, cols)
    )
    representative_reordered = representative_matrix[np.ix_(row_order, col_order)]
    reconstructed_reordered = reconstructed[np.ix_(row_order, col_order)]

    np.savez(
        args.output_dir / "permutation_orders.npz",
        row_block_order=row_block_order,
        row_order=row_order,
        col_order=col_order,
    )
    save_cluster_map(
        labels, reordered_labels, args.output_dir / "tile_cluster_permutation.png"
    )
    save_weight_comparison(
        original_reordered,
        representative_reordered,
        reconstructed_reordered,
        args.output_dir / "reordered_weight_comparison.png",
    )
    save_rank_plot(metric_rows, args.output_dir / "cluster_rank_summary.png")

    print(f"Analyzed matrix in SeedLM orientation: {original.shape}")
    print(f"Tile-label grid: {labels.shape}; clusters reported: {len(metric_rows)}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {args.output_dir / 'tile_cluster_permutation.png'}")
    print(f"Saved: {args.output_dir / 'reordered_weight_comparison.png'}")
    print(f"Saved: {args.output_dir / 'cluster_rank_summary.png'}")
    print(f"Saved: {args.output_dir / 'permutation_orders.npz'}")


if __name__ == "__main__":
    main()
