#!/usr/bin/env python3
"""Compare local rank after row-, column-, and tile-guided permutations.

All methods operate on the same small submatrix. After permutation, the matrix
is divided into fixed-size blocks, and identical local-rank metrics are measured
for every method. Pure permutation preserves all global singular values.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--row-start", type=int, default=0)
    p.add_argument("--column-start", type=int, default=0)
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--columns", type=int, default=64)
    p.add_argument("--local-block-rows", type=int, default=8)
    p.add_argument("--local-block-columns", type=int, default=8)
    p.add_argument("--tile-height", type=int, default=8)
    p.add_argument("--k-rows", type=int, default=8)
    p.add_argument("--k-columns", type=int, default=8)
    p.add_argument("--k-tiles", type=int, default=16)
    p.add_argument("--klines-iterations", type=int, default=20)
    p.add_argument(
        "--exhaustive",
        action="store_true",
        help="Search every row/column permutation; intended only for tiny matrices.",
    )
    p.add_argument(
        "--exhaustive-objective",
        choices=["effective_rank_ratio", "rank_95_ratio", "rank1_relative_error"],
        default="effective_rank_ratio",
    )
    p.add_argument(
        "--max-exhaustive-combinations",
        type=int,
        default=1_000_000,
        help="Safety limit for factorial(row_count) * factorial(column_count).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/small_matrix_permutations")
    return p.parse_args()


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-15)


def cluster_order(samples: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= k <= samples.shape[0]:
        raise ValueError(f"k={k} is invalid for {samples.shape[0]} samples")
    labels = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        random_state=seed,
    ).fit_predict(unit_rows(samples))
    # Sort clusters by their normalized centers, then preserve sample order
    # within a cluster. This makes the result deterministic.
    centers = []
    for cluster_id in range(k):
        center = unit_rows(samples[labels == cluster_id]).mean(axis=0)
        centers.append(center)
    center_order = seriation(np.asarray(centers))
    position = np.empty(k, dtype=np.int64)
    position[center_order] = np.arange(k)
    order = np.argsort(position[labels], kind="stable")
    return labels.astype(np.int32), order.astype(np.int64)


def klines(tiles: np.ndarray, k: int, iterations: int, seed: int) -> np.ndarray:
    """Cluster vectors by their unsigned direction using alternating updates."""
    x = unit_rows(tiles)
    if not 1 <= k <= x.shape[0]:
        raise ValueError(f"k_tiles={k} is invalid for {x.shape[0]} tiles")

    # K-means++ initialization on [u, -u]-invariant outer-product features.
    features = np.einsum("ni,nj->nij", x, x).reshape(x.shape[0], -1)
    init_labels = KMeans(
        n_clusters=k, init="k-means++", n_init=10, random_state=seed
    ).fit_predict(features)
    labels = init_labels
    directions = np.zeros((k, x.shape[1]), dtype=np.float64)

    for _ in range(iterations):
        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if members.size == 0:
                directions[cluster_id] = x[(cluster_id + seed) % x.shape[0]]
            else:
                _, _, vh = np.linalg.svd(members, full_matrices=False)
                directions[cluster_id] = vh[0]
        new_labels = np.argmax(np.abs(x @ directions.T), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels.astype(np.int32)


def seriation(features: np.ndarray) -> np.ndarray:
    if features.shape[0] <= 2:
        return np.arange(features.shape[0], dtype=np.int64)
    distance = pdist(features, metric="cosine")
    distance[~np.isfinite(distance)] = 1.0
    tree = linkage(distance, method="average")
    tree = optimal_leaf_ordering(tree, distance)
    return leaves_list(tree).astype(np.int64)


def tile_guided_orders(
    matrix: np.ndarray, tile_height: int, k: int, iterations: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.shape[0] % tile_height:
        raise ValueError("Selected row count must be divisible by tile height")
    grid = matrix.reshape(
        matrix.shape[0] // tile_height, tile_height, matrix.shape[1]
    ).transpose(0, 2, 1)
    labels = klines(grid.reshape(-1, tile_height), k, iterations, seed).reshape(
        grid.shape[:2]
    )

    row_features = np.stack(
        [np.bincount(row, minlength=k) for row in labels], axis=0
    ).astype(np.float64)
    column_features = np.stack(
        [np.bincount(column, minlength=k) for column in labels.T], axis=0
    ).astype(np.float64)
    row_block_order = seriation(row_features)
    column_order = seriation(column_features)
    row_order = np.concatenate(
        [
            np.arange(block * tile_height, (block + 1) * tile_height)
            for block in row_block_order
        ]
    )
    return row_order, column_order, labels


def metrics(matrix: np.ndarray) -> dict[str, float]:
    singular = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    energy = singular * singular
    energy /= max(float(energy.sum()), 1e-30)
    probability = energy[energy > 0]
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
    rank95 = int(np.searchsorted(np.cumsum(energy), 0.95) + 1)
    first = float(energy[0])
    return {
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / min(matrix.shape),
        "rank_95_energy": rank95,
        "rank_95_ratio": rank95 / min(matrix.shape),
        "first_direction_energy": first,
        "rank1_relative_error": float(np.sqrt(max(0.0, 1.0 - first))),
    }


def local_metrics(
    matrix: np.ndarray, block_rows: int, block_columns: int
) -> list[dict[str, float]]:
    if matrix.shape[0] % block_rows or matrix.shape[1] % block_columns:
        raise ValueError("Submatrix dimensions must be divisible by local block size")
    results = []
    for r in range(0, matrix.shape[0], block_rows):
        for c in range(0, matrix.shape[1], block_columns):
            results.append(metrics(matrix[r : r + block_rows, c : c + block_columns]))
    return results


def summarize(name: str, matrix: np.ndarray, br: int, bc: int) -> dict:
    blocks = local_metrics(matrix, br, bc)
    result = {"method": name, "n_local_blocks": len(blocks)}
    for key in blocks[0]:
        values = np.asarray([row[key] for row in blocks])
        result[f"mean_local_{key}"] = float(values.mean())
        result[f"median_local_{key}"] = float(np.median(values))
    global_result = metrics(matrix)
    for key, value in global_result.items():
        result[f"global_{key}"] = value
    return result


def objective_value(
    matrix: np.ndarray, block_rows: int, block_columns: int, objective: str
) -> float:
    blocks = local_metrics(matrix, block_rows, block_columns)
    return float(np.mean([row[objective] for row in blocks]))


def exhaustive_orders(
    matrix: np.ndarray,
    block_rows: int,
    block_columns: int,
    objective: str,
    max_combinations: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Find exact best column-only, row-only, and joint permutations."""
    import math
    import time

    n_rows, n_columns = matrix.shape
    row_count = math.factorial(n_rows)
    column_count = math.factorial(n_columns)
    joint_count = row_count * column_count
    if joint_count > max_combinations:
        raise ValueError(
            f"Exhaustive joint search needs {joint_count:,} combinations, "
            f"above --max-exhaustive-combinations={max_combinations:,}. "
            "Use a smaller matrix or raise the safety limit explicitly."
        )

    identity_rows = np.arange(n_rows, dtype=np.int64)
    identity_columns = np.arange(n_columns, dtype=np.int64)
    row_permutations = [
        np.asarray(order, dtype=np.int64)
        for order in itertools.permutations(range(n_rows))
    ]
    column_permutations = [
        np.asarray(order, dtype=np.int64)
        for order in itertools.permutations(range(n_columns))
    ]

    started = time.perf_counter()
    best_column_order = identity_columns
    best_column_value = float("inf")
    for column_order in column_permutations:
        value = objective_value(
            matrix[:, column_order], block_rows, block_columns, objective
        )
        if value < best_column_value:
            best_column_value = value
            best_column_order = column_order.copy()

    best_row_order = identity_rows
    best_row_value = float("inf")
    for row_order in row_permutations:
        value = objective_value(
            matrix[row_order, :], block_rows, block_columns, objective
        )
        if value < best_row_value:
            best_row_value = value
            best_row_order = row_order.copy()

    best_joint_rows = identity_rows
    best_joint_columns = identity_columns
    best_joint_value = float("inf")
    for row_order in row_permutations:
        row_reordered = matrix[row_order, :]
        for column_order in column_permutations:
            value = objective_value(
                row_reordered[:, column_order],
                block_rows,
                block_columns,
                objective,
            )
            if value < best_joint_value:
                best_joint_value = value
                best_joint_rows = row_order.copy()
                best_joint_columns = column_order.copy()

    details = {
        "objective": objective,
        "column_permutations_evaluated": column_count,
        "row_permutations_evaluated": row_count,
        "joint_permutations_evaluated": joint_count,
        "best_column_objective": best_column_value,
        "best_row_objective": best_row_value,
        "best_joint_objective": best_joint_value,
        "search_seconds": time.perf_counter() - started,
        "best_column_order": best_column_order,
        "best_row_order": best_row_order,
    }
    return best_joint_rows, best_joint_columns, details


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], path: Path) -> None:
    labels = [row["method"] for row in rows]
    panels = [
        ("mean_local_effective_rank_ratio", "Local effective-rank ratio", False),
        ("mean_local_rank_95_ratio", "Local rank-95 ratio", False),
        ("mean_local_first_direction_energy", "First-direction energy", True),
        ("mean_local_rank1_relative_error", "Rank-1 relative error", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    for axis, (key, title, higher) in zip(axes, panels):
        values = [row[key] for row in rows]
        colors = ["#4C78A8" if name == "original" else "#F58518" for name in labels]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title + (" (higher is better)" if higher else " (lower is better)"))
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Fixed-size local blocks after alternative permutations")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full = np.load(args.weight)
    matrix = full[
        args.row_start : args.row_start + args.rows,
        args.column_start : args.column_start + args.columns,
    ].astype(np.float32)
    if matrix.shape != (args.rows, args.columns):
        raise ValueError(f"Requested submatrix is outside weight shape {full.shape}")

    row_labels, row_order = cluster_order(matrix, args.k_rows, args.seed)
    column_labels, column_order = cluster_order(
        matrix.T, args.k_columns, args.seed + 1
    )
    tile_rows, tile_columns, tile_labels = tile_guided_orders(
        matrix,
        args.tile_height,
        args.k_tiles,
        args.klines_iterations,
        args.seed + 2,
    )
    rng = np.random.default_rng(args.seed)
    random_rows = rng.permutation(matrix.shape[0])
    random_columns = rng.permutation(matrix.shape[1])

    alternatives = {
        "original": matrix,
        "random_both": matrix[np.ix_(random_rows, random_columns)],
        "column_cluster": matrix[:, column_order],
        "row_cluster": matrix[row_order, :],
        "row_column_cluster": matrix[np.ix_(row_order, column_order)],
        "tile_guided": matrix[np.ix_(tile_rows, tile_columns)],
    }
    exhaustive_details = None
    if args.exhaustive:
        exhaustive_rows, exhaustive_columns, exhaustive_details = exhaustive_orders(
            matrix,
            args.local_block_rows,
            args.local_block_columns,
            args.exhaustive_objective,
            args.max_exhaustive_combinations,
        )
        alternatives["exhaustive_best_column"] = matrix[
            :, exhaustive_details["best_column_order"]
        ]
        alternatives["exhaustive_best_row"] = matrix[
            exhaustive_details["best_row_order"], :
        ]
        alternatives["exhaustive_best_both"] = matrix[
            np.ix_(exhaustive_rows, exhaustive_columns)
        ]
    rows = [
        summarize(
            name,
            candidate,
            args.local_block_rows,
            args.local_block_columns,
        )
        for name, candidate in alternatives.items()
    ]
    save_csv(output / "small_matrix_permutation_metrics.csv", rows)
    save_plot(rows, output / "small_matrix_permutation_comparison.png")
    saved_orders = {
        "row_order": row_order,
        "column_order": column_order,
        "tile_row_order": tile_rows,
        "tile_column_order": tile_columns,
        "random_row_order": random_rows,
        "random_column_order": random_columns,
        "row_labels": row_labels,
        "column_labels": column_labels,
        "tile_labels": tile_labels,
    }
    if exhaustive_details is not None:
        saved_orders.update(
            {
                "exhaustive_best_row_order": exhaustive_rows,
                "exhaustive_best_column_order": exhaustive_columns,
                "exhaustive_best_row_only_order":
                    exhaustive_details["best_row_order"],
                "exhaustive_best_column_only_order":
                    exhaustive_details["best_column_order"],
            }
        )
    np.savez(output / "permutation_orders.npz", **saved_orders)
    print(f"Selected submatrix: {matrix.shape}")
    for row in rows:
        print(
            row["method"],
            "local_ER_ratio=",
            f'{row["mean_local_effective_rank_ratio"]:.4f}',
            "local_rank1_error=",
            f'{row["mean_local_rank1_relative_error"]:.4f}',
        )
    if exhaustive_details is not None:
        print(
            "Exhaustive search:",
            f'{exhaustive_details["joint_permutations_evaluated"]:,}',
            "joint combinations in",
            f'{exhaustive_details["search_seconds"]:.2f}s',
        )
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
