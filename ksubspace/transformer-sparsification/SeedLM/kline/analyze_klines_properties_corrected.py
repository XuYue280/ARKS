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
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist

from SeedLM.kline.SeedLM_cpu_klines import LFSR_get_Us, SeedLM_load


def tile_grid(matrix: np.ndarray, c: int) -> np.ndarray:
    """Return [row_block, column, C] tiles without changing within-tile order."""
    rows, cols = matrix.shape
    if rows % c:
        raise ValueError(f"Matrix rows={rows} must be divisible by C={c}")
    return matrix.reshape(rows // c, c, cols).transpose(0, 2, 1)


def _histogram_features(labels: np.ndarray, k: int, axis: int) -> np.ndarray:
    items = labels if axis == 0 else labels.T
    features = np.zeros((items.shape[0], k + 1), dtype=np.float32)
    for index, values in enumerate(items):
        valid = values[values >= 0]
        if valid.size:
            features[index, :k] = np.bincount(valid, minlength=k)
        features[index, k] = np.sum(values < 0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def _optimal_order(features: np.ndarray) -> np.ndarray:
    if features.shape[0] <= 2:
        return np.arange(features.shape[0])
    distances = pdist(features, metric="cosine")
    distances[~np.isfinite(distances)] = 1.0
    tree = linkage(distances, method="average")
    tree = optimal_leaf_ordering(tree, distances)
    return leaves_list(tree).astype(np.int64)


def similarity_seriation(labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Order row blocks and columns by cluster-composition similarity."""
    return (
        _optimal_order(_histogram_features(labels, k, axis=0)),
        _optimal_order(_histogram_features(labels, k, axis=1)),
    )


def neighbor_agreement(labels: np.ndarray) -> float:
    horizontal_valid = (labels[:, :-1] >= 0) & (labels[:, 1:] >= 0)
    vertical_valid = (labels[:-1, :] >= 0) & (labels[1:, :] >= 0)
    equal_h = (labels[:, :-1] == labels[:, 1:]) & horizontal_valid
    equal_v = (labels[:-1, :] == labels[1:, :]) & vertical_valid
    valid_count = int(horizontal_valid.sum() + vertical_valid.sum())
    return float((equal_h.sum() + equal_v.sum()) / valid_count) if valid_count else 0.0


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
            "top2_singular_energy_fraction": 0.0,
            "top4_singular_energy_fraction": 0.0,
            "singular_value_1_to_2_ratio": 0.0,
            "best_rank1_relative_error": 0.0,
            **{
                f"normalized_singular_value_{index + 1}": 0.0
                for index in range(8)
            },
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
    normalized_singular = singular / singular[0]
    singular_metrics = {
        f"normalized_singular_value_{index + 1}": (
            float(normalized_singular[index])
            if index < normalized_singular.size else 0.0
        )
        for index in range(8)
    }
    return {
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "rank_95_energy": rank_95,
        "first_singular_energy_fraction": first_fraction,
        "top2_singular_energy_fraction": float(cumulative[min(1, len(cumulative) - 1)]),
        "top4_singular_energy_fraction": float(cumulative[min(3, len(cumulative) - 1)]),
        "singular_value_1_to_2_ratio": (
            float(singular[0] / singular[1])
            if singular.size > 1 and singular[1] > 0 else float("inf")
        ),
        "best_rank1_relative_error": rank1_error,
        **singular_metrics,
    }


def shared_property_metrics(tiles: np.ndarray) -> dict[str, float]:
    """Measure properties shared by the [members, C] tiles in one cluster."""
    x = tiles.astype(np.float64, copy=False)
    center = x.mean(axis=0)
    residual = x - center
    squared_distances = np.sum(residual * residual, axis=1)
    distances = np.sqrt(squared_distances)

    tile_norms = np.linalg.norm(x, axis=1)
    center_norm = float(np.linalg.norm(center))
    valid = tile_norms > 0
    if center_norm > 0 and np.any(valid):
        cosine_to_center = (
            x[valid] @ center / (tile_norms[valid] * center_norm)
        )
        mean_cosine = float(np.mean(cosine_to_center))
        minimum_cosine = float(np.min(cosine_to_center))
        cosine_std = float(np.std(cosine_to_center))
    else:
        mean_cosine = 0.0
        minimum_cosine = 0.0
        cosine_std = 0.0

    mean_norm = float(np.mean(tile_norms))
    return {
        # Distance from each tile to the cluster mean.
        "mean_euclidean_distance_to_center": float(np.mean(distances)),
        "rms_euclidean_distance_to_center": float(np.sqrt(np.mean(squared_distances))),
        "max_euclidean_distance_to_center": float(np.max(distances)),
        "relative_rms_distance_to_center": (
            float(np.sqrt(np.mean(squared_distances)) / mean_norm)
            if mean_norm > 0 else 0.0
        ),
        # Variance averaged over the C tile dimensions.
        "mean_element_variance": float(np.mean(np.var(x, axis=0))),
        "total_variance": float(np.sum(np.var(x, axis=0))),
        # Directional similarity, insensitive to tile magnitude.
        "mean_cosine_similarity_to_center": mean_cosine,
        "minimum_cosine_similarity_to_center": minimum_cosine,
        "cosine_similarity_std": cosine_std,
        "mean_tile_l2_norm": mean_norm,
        "total_tile_energy": float(np.sum(x * x)),
        "tile_l2_norm_cv": (
            float(np.std(tile_norms) / mean_norm) if mean_norm > 0 else 0.0
        ),
    }


def load_stored_directions(
    artifact_dir: Path, c: int, p: int, seed_bits: int
) -> np.ndarray:
    """Reconstruct the K-lines directions actually stored by SeedLM."""
    seeds = np.load(artifact_dir / "seeds.npy")
    ts = np.load(artifact_dir / "ts.npy")
    if seeds.shape[0] != ts.shape[0]:
        raise ValueError("seeds.npy and ts.npy have inconsistent lengths")
    directions = np.zeros((seeds.shape[0], c), dtype=np.float64)
    for cluster_id in range(seeds.shape[0]):
        _, us = LFSR_get_Us(c, p, seed_bits, int(seeds[cluster_id]))
        directions[cluster_id] = (
            us.astype(np.float64) @ ts[cluster_id].astype(np.float64)
        )
    return directions


def klines_property_metrics(
    tiles: np.ndarray,
    stored_direction: np.ndarray,
    stored_coefficients: np.ndarray,
) -> dict[str, float]:
    """Measure alignment and reconstruction properties for one K-lines cluster."""
    x = tiles.astype(np.float64, copy=False)
    direction = stored_direction.astype(np.float64, copy=False)
    direction_norm = float(np.linalg.norm(direction))
    tile_norms = np.linalg.norm(x, axis=1)
    valid = tile_norms > 0

    if direction_norm == 0 or not np.any(valid):
        return {
            "mean_absolute_cosine_to_klines_direction": 0.0,
            "minimum_absolute_cosine_to_klines_direction": 0.0,
            "absolute_cosine_std": 0.0,
            "klines_direction_energy_fraction": 0.0,
            "ideal_line_projection_relative_error": 0.0,
            "mean_tile_projection_relative_error": 0.0,
            "coefficient_relative_rmse": 0.0,
            "coefficient_correlation": float("nan"),
        }

    unit_direction = direction / direction_norm
    dot_unit = x @ unit_direction
    absolute_cosine = np.abs(dot_unit[valid]) / tile_norms[valid]
    ideal_reconstruction = dot_unit[:, None] * unit_direction[None, :]
    ideal_residual = x - ideal_reconstruction
    total_energy = float(np.sum(x * x))
    residual_energy = float(np.sum(ideal_residual * ideal_residual))

    per_tile_projection_error = np.zeros(x.shape[0], dtype=np.float64)
    per_tile_projection_error[valid] = (
        np.linalg.norm(ideal_residual[valid], axis=1) / tile_norms[valid]
    )

    # Optimal coefficient for the non-unit stored direction d:
    # argmin_a ||x - a d||^2 = (x^T d) / (d^T d).
    optimal_coefficients = (x @ direction) / float(direction @ direction)
    stored_coefficients = stored_coefficients.astype(np.float64, copy=False)
    coefficient_error = stored_coefficients - optimal_coefficients
    coefficient_denominator = float(np.sum(optimal_coefficients ** 2))

    return {
        # Use absolute cosine because u and -u describe the same line.
        "mean_absolute_cosine_to_klines_direction": float(
            np.mean(absolute_cosine)
        ),
        "minimum_absolute_cosine_to_klines_direction": float(
            np.min(absolute_cosine)
        ),
        "absolute_cosine_std": float(np.std(absolute_cosine)),
        "klines_direction_energy_fraction": (
            float(np.sum(dot_unit ** 2) / total_energy)
            if total_energy > 0 else 0.0
        ),
        "ideal_line_projection_relative_error": (
            float(math.sqrt(residual_energy / total_energy))
            if total_energy > 0 else 0.0
        ),
        "mean_tile_projection_relative_error": float(
            np.mean(per_tile_projection_error[valid])
        ),
        "coefficient_relative_rmse": (
            float(math.sqrt(np.sum(coefficient_error ** 2)
                            / coefficient_denominator))
            if coefficient_denominator > 0 else 0.0
        ),
        "coefficient_correlation": safe_correlation(
            stored_coefficients, optimal_coefficients
        ),
    }


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save_cluster_map(
    original_labels: np.ndarray,
    reordered_labels: np.ndarray,
    top_clusters: np.ndarray,
    output: Path,
) -> None:
    mapping = {int(cluster): index + 1 for index, cluster in enumerate(top_clusters)}
    def filtered(values: np.ndarray) -> np.ndarray:
        result = np.zeros(values.shape, dtype=np.int16)
        for cluster, display_id in mapping.items():
            result[values == cluster] = display_id
        return result
    original_plot = filtered(original_labels)
    reordered_plot = filtered(reordered_labels)
    n = len(top_clusters)
    base = plt.get_cmap("tab20", max(n, 1))
    colors = [(0.86, 0.86, 0.86, 1.0)] + [base(i) for i in range(max(n, 1))]
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(original_plot, aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=n)
    axes[0].set_title(f"Original layout: {n} largest clusters colored; others gray")
    axes[0].set_ylabel("Row block (C rows)")
    axes[0].set_xlabel("Weight-matrix column")

    axes[1].imshow(reordered_plot, aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=n)
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


def save_original_vs_cluster_rank_plot(
    original_metrics: dict[str, float | int],
    cluster_metrics: dict[str, float | int],
    output: Path,
) -> None:
    labels = [
        "Effective rank",
        "Rank for 95% energy",
        "Rank-1 relative error",
    ]
    original_values = [
        float(original_metrics["effective_rank"]),
        float(original_metrics["rank_95_energy"]),
        float(original_metrics["best_rank1_relative_error"]),
    ]
    cluster_values = [
        float(cluster_metrics["effective_rank"]),
        float(cluster_metrics["rank_95_energy"]),
        float(cluster_metrics["best_rank1_relative_error"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.bar(
            [0, 1],
            [original_values[index], cluster_values[index]],
            color=["#4c78a8", "#f58518"],
        )
        axis.set_xticks([0, 1], ["Original", "K-lines"])
        axis.set_title(labels[index])
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Original unclustered tiles versus K-lines clusters "
        "(weighted aggregate)"
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_property_error_plot(
    rows: list[dict[str, float | int]], output: Path
) -> None:
    """Plot K-lines properties against the actual K-lines reconstruction error."""
    error = np.array([
        float(row["seedlm_replacement_relative_error"]) for row in rows
    ])
    panels = [
        (
            "mean_absolute_cosine_to_klines_direction",
            "Mean absolute cosine to K-lines direction",
        ),
        (
            "ideal_line_projection_relative_error",
            "Ideal line-projection relative error",
        ),
        (
            "klines_direction_energy_fraction",
            "Energy captured by K-lines direction",
        ),
        (
            "coefficient_relative_rmse",
            "Stored-coefficient relative RMSE",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, panels):
        values = np.array([float(row[metric]) for row in rows])
        correlation = safe_correlation(values, error)
        axis.scatter(values, error, s=18, alpha=0.7)
        axis.set_xlabel(label)
        axis.set_ylabel("Actual K-lines reconstruction relative error")
        axis.set_title(f"Pearson r = {correlation:.3f}")
        axis.grid(alpha=0.2)
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
    parser.add_argument("--clustering-method", default="kmeans++",
                        choices=["random", "kmeans++", "minibatch", "kmedoids", "klines"])
    parser.add_argument("--top-clusters", type=int, default=12)
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
        clustering_method=args.clustering_method,
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
    ideal_line_tiles = original_tiles.copy()

    if args.clustering_method != "klines":
        raise ValueError(
            "This corrected property analysis is designed for "
            "--clustering-method klines"
        )
    stored_directions = load_stored_directions(
        args.artifact_dir, args.C, args.P, args.seed_bits
    )
    if stored_directions.shape[0] != args.k:
        raise ValueError(
            f"Stored direction count={stored_directions.shape[0]}, expected K={args.k}"
        )
    coefficient_flat = np.load(args.artifact_dir / "tile_coefficients.npy")
    if coefficient_flat.size != labels_flat.size:
        raise ValueError(
            f"Coefficient count={coefficient_flat.size}, "
            f"expected {labels_flat.size}"
        )
    coefficient_grid = coefficient_flat.reshape(row_blocks, cols)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float | int]] = []
    for cluster_id in range(args.k):
        mask = labels == cluster_id
        members = original_tiles[mask]
        replacements = reconstructed_tiles[mask]
        if members.shape[0] == 0:
            continue
        metrics = rank_metrics(members, args.rank_tolerance)
        shared_metrics = shared_property_metrics(members)
        line_metrics = klines_property_metrics(
            members,
            stored_directions[cluster_id],
            coefficient_grid[mask],
        )
        direction = stored_directions[cluster_id]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 0:
            unit_direction = direction / direction_norm
            ideal_line_tiles[mask] = (
                (members.astype(np.float64) @ unit_direction)[:, None]
                * unit_direction[None, :]
            ).astype(np.float32)
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
            **shared_metrics,
            **line_metrics,
            "observed_representative_relative_error": representative_relative_error,
            "seedlm_replacement_relative_error": replacement_relative_error,
        })

    # Fair baseline: use exactly the same pruned tiles assigned by K-lines,
    # but stack them once without separating them into clusters.
    clustered_mask = labels >= 0
    original_unclustered_tiles = original_tiles[clustered_mask]
    original_unclustered_metrics = rank_metrics(
        original_unclustered_tiles, args.rank_tolerance
    )

    member_counts = np.array(
        [int(row["member_tiles"]) for row in metric_rows], dtype=np.float64
    )
    cluster_energies = np.array(
        [float(row["total_tile_energy"]) for row in metric_rows],
        dtype=np.float64,
    )
    total_members = float(np.sum(member_counts))
    total_cluster_energy = float(np.sum(cluster_energies))

    def member_weighted(metric: str) -> float:
        values = np.array([float(row[metric]) for row in metric_rows])
        return (
            float(np.sum(values * member_counts) / total_members)
            if total_members > 0 else float("nan")
        )

    first_energy_values = np.array([
        float(row["first_singular_energy_fraction"]) for row in metric_rows
    ])
    clustered_first_energy = (
        float(np.sum(first_energy_values * cluster_energies)
              / total_cluster_energy)
        if total_cluster_energy > 0 else float("nan")
    )
    clustered_aggregate_metrics: dict[str, float | int] = {
        "numerical_rank": member_weighted("numerical_rank"),
        "effective_rank": member_weighted("effective_rank"),
        "rank_95_energy": member_weighted("rank_95_energy"),
        "first_singular_energy_fraction": clustered_first_energy,
        "best_rank1_relative_error": (
            math.sqrt(max(0.0, 1.0 - clustered_first_energy))
            if np.isfinite(clustered_first_energy) else float("nan")
        ),
    }

    rank_comparison_rows = []
    for metric in (
        "numerical_rank",
        "effective_rank",
        "rank_95_energy",
        "first_singular_energy_fraction",
        "best_rank1_relative_error",
    ):
        rank_comparison_rows.append({
            "metric": metric,
            "original_unclustered_tiles": original_unclustered_metrics[metric],
            "klines_clusters_weighted_aggregate":
                clustered_aggregate_metrics[metric],
            "clustered_minus_original": (
                float(clustered_aggregate_metrics[metric])
                - float(original_unclustered_metrics[metric])
            ),
        })

    rank_comparison_path = args.output_dir / "original_vs_cluster_rank.csv"
    with rank_comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "original_unclustered_tiles",
                "klines_clusters_weighted_aggregate",
                "clustered_minus_original",
            ],
        )
        writer.writeheader()
        writer.writerows(rank_comparison_rows)

    csv_path = args.output_dir / "cluster_rank_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    row_block_order, col_order = similarity_seriation(labels, args.k)
    row_order = expand_row_blocks(row_block_order, args.C)
    reordered_labels = labels[np.ix_(row_block_order, col_order)]
    locality_before = neighbor_agreement(labels)
    locality_after = neighbor_agreement(reordered_labels)
    original_reordered = original[np.ix_(row_order, col_order)]
    representative_matrix = (
        representative_tiles.transpose(0, 2, 1).reshape(rows, cols)
    )
    ideal_line_matrix = ideal_line_tiles.transpose(0, 2, 1).reshape(rows, cols)
    representative_reordered = representative_matrix[np.ix_(row_order, col_order)]
    reconstructed_reordered = reconstructed[np.ix_(row_order, col_order)]

    np.savez(
        args.output_dir / "permutation_orders.npz",
        row_block_order=row_block_order,
        row_order=row_order,
        col_order=col_order,
    )
    counts = np.bincount(labels[labels >= 0], minlength=args.k)
    top_clusters = np.argsort(counts)[::-1][:min(args.top_clusters, args.k)]
    save_cluster_map(labels, reordered_labels, top_clusters,
                     args.output_dir / "tile_cluster_permutation.png")
    save_weight_comparison(
        original_reordered,
        representative_reordered,
        reconstructed_reordered,
        args.output_dir / "reordered_weight_comparison.png",
    )
    save_rank_plot(metric_rows, args.output_dir / "cluster_rank_summary.png")
    save_original_vs_cluster_rank_plot(
        original_unclustered_metrics,
        clustered_aggregate_metrics,
        args.output_dir / "original_vs_cluster_rank.png",
    )
    save_property_error_plot(
        metric_rows, args.output_dir / "cluster_property_vs_replacement_error.png"
    )

    original64 = original.astype(np.float64)
    reconstructed64 = reconstructed.astype(np.float64)
    representative64 = representative_matrix.astype(np.float64)
    ideal_line64 = ideal_line_matrix.astype(np.float64)
    layer_energy = float(np.sum(original64 * original64))
    layer_elements = int(original.size)
    reconstructed_squared_error = float(
        np.sum((original64 - reconstructed64) ** 2)
    )
    representative_squared_error = float(
        np.sum((original64 - representative64) ** 2)
    )
    ideal_line_squared_error = float(
        np.sum((original64 - ideal_line64) ** 2)
    )

    explanation_properties = [
        "mean_absolute_cosine_to_klines_direction",
        "minimum_absolute_cosine_to_klines_direction",
        "klines_direction_energy_fraction",
        "ideal_line_projection_relative_error",
        "mean_tile_projection_relative_error",
        "coefficient_relative_rmse",
        "coefficient_correlation",
    ]
    seedlm_errors = np.array([
        float(row["seedlm_replacement_relative_error"])
        for row in metric_rows
    ])
    summary_rows: list[dict[str, str | float | int]] = [
        {
            "section": "layer_reconstruction",
            "metric": "matrix_elements",
            "value": layer_elements,
        },
        {
            "section": "layer_reconstruction",
            "metric": "seedlm_reconstruction_mse",
            "value": reconstructed_squared_error / layer_elements,
        },
        {
            "section": "layer_reconstruction",
            "metric": "observed_representative_mse",
            "value": representative_squared_error / layer_elements,
        },
        {
            "section": "layer_reconstruction",
            "metric": "ideal_stored_direction_projection_mse",
            "value": ideal_line_squared_error / layer_elements,
        },
        {
            "section": "layer_reconstruction",
            "metric": "seedlm_relative_frobenius_error",
            "value": (
                math.sqrt(reconstructed_squared_error / layer_energy)
                if layer_energy > 0 else 0.0
            ),
        },
        {
            "section": "layer_reconstruction",
            "metric": "observed_representative_relative_frobenius_error",
            "value": (
                math.sqrt(representative_squared_error / layer_energy)
                if layer_energy > 0 else 0.0
            ),
        },
        {
            "section": "layer_reconstruction",
            "metric": "ideal_stored_direction_projection_relative_frobenius_error",
            "value": (
                math.sqrt(ideal_line_squared_error / layer_energy)
                if layer_energy > 0 else 0.0
            ),
        },
    ]
    for metric in (
        "numerical_rank",
        "effective_rank",
        "rank_95_energy",
        "first_singular_energy_fraction",
        "best_rank1_relative_error",
    ):
        summary_rows.extend([
            {
                "section": "original_unclustered_rank",
                "metric": metric,
                "value": original_unclustered_metrics[metric],
            },
            {
                "section": "klines_cluster_rank_weighted_aggregate",
                "metric": metric,
                "value": clustered_aggregate_metrics[metric],
            },
        ])
    for property_name in explanation_properties:
        values = np.array([float(row[property_name]) for row in metric_rows])
        summary_rows.append({
            "section": "klines_explanation_correlation",
            "metric": f"corr_{property_name}_vs_actual_klines_error",
            "value": safe_correlation(values, seedlm_errors),
        })

    layer_summary_path = args.output_dir / "layer_summary.csv"
    with layer_summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_path = args.output_dir / "permutation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "clustering_method", "neighbor_agreement_before",
            "neighbor_agreement_after", "relative_improvement"
        ])
        writer.writeheader()
        writer.writerow({
            "clustering_method": args.clustering_method,
            "neighbor_agreement_before": locality_before,
            "neighbor_agreement_after": locality_after,
            "relative_improvement": (
                (locality_after - locality_before) / locality_before
                if locality_before > 0 else float("nan")
            ),
        })

    print(f"Analyzed matrix in SeedLM orientation: {original.shape}")
    print(f"Tile-label grid: {labels.shape}; clusters reported: {len(metric_rows)}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {args.output_dir / 'tile_cluster_permutation.png'}")
    print(f"Saved: {args.output_dir / 'reordered_weight_comparison.png'}")
    print(f"Saved: {args.output_dir / 'cluster_rank_summary.png'}")
    print(f"Saved: {args.output_dir / 'original_vs_cluster_rank.png'}")
    print(f"Saved: {rank_comparison_path}")
    print(f"Saved: "
          f"{args.output_dir / 'cluster_property_vs_replacement_error.png'}")
    print(f"Saved: {args.output_dir / 'permutation_orders.npz'}")
    print(f"Layer SeedLM reconstruction MSE: "
          f"{reconstructed_squared_error / layer_elements:.8e}")
    print(f"Saved: {layer_summary_path}")
    print(f"Neighbor agreement: {locality_before:.6f} -> {locality_after:.6f}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
