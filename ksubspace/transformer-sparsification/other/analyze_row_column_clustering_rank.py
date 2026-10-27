#!/usr/bin/env python3
"""
Analyze rank structure without forming SeedLM tiles.

The experiment treats either every matrix column or every matrix row as one
sample, clusters those samples, and then reorders the weight matrix by cluster
label. It reports:

1. Global rank metrics for the original matrix.
2. Global rank metrics after column, row, and joint row-column permutation.
3. Rank metrics inside each row or column cluster.

Important: a pure row/column permutation cannot change singular values, rank,
effective rank, energy rank, or best rank-1 error. Cluster-local metrics are
therefore the relevant test of whether clustering reveals local low-rank
structure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans


EPS = 1e-30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True, help="Path to a 2-D .npy matrix")
    p.add_argument("--k-columns", type=int, default=32)
    p.add_argument("--k-rows", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument(
        "--normalize-samples",
        action="store_true",
        help="Cluster by direction (cosine geometry) instead of raw magnitude.",
    )
    p.add_argument(
        "--transpose-weight",
        action="store_true",
        help="Transpose the loaded matrix before analysis.",
    )
    p.add_argument("--rank-tolerance", type=float, default=None)
    p.add_argument("--output-dir", default="results/row_column_rank_analysis")
    return p.parse_args()


def singular_values(a: np.ndarray) -> np.ndarray:
    if min(a.shape) == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.svd(a.astype(np.float64, copy=False), compute_uv=False)


def rank_metrics(a: np.ndarray, tolerance: float | None) -> dict[str, float]:
    s = singular_values(a)
    if s.size == 0 or s[0] == 0:
        return {
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "rank_95_energy": 0,
            "first_direction_energy": 0.0,
            "rank1_relative_error": 0.0,
        }

    if tolerance is None:
        tol = max(a.shape) * np.finfo(np.float64).eps * s[0]
    else:
        tol = tolerance

    energy = s * s
    p = energy / max(float(energy.sum()), EPS)
    effective_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    rank95 = int(np.searchsorted(np.cumsum(p), 0.95, side="left") + 1)
    first_energy = float(p[0])
    rank1_error = float(np.sqrt(max(0.0, 1.0 - first_energy)))
    return {
        "numerical_rank": int(np.sum(s > tol)),
        "effective_rank": effective_rank,
        "rank_95_energy": rank95,
        "first_direction_energy": first_energy,
        "rank1_relative_error": rank1_error,
    }


def prepare_samples(samples: np.ndarray, normalize: bool) -> np.ndarray:
    x = samples.astype(np.float32, copy=False)
    if not normalize:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cluster_samples(
    samples: np.ndarray,
    k: int,
    seed: int,
    batch_size: int,
    max_iter: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= k <= samples.shape[0]:
        raise ValueError(f"k={k} must be between 1 and {samples.shape[0]}")
    x = prepare_samples(samples, normalize)
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=seed,
        batch_size=max(batch_size, k),
        max_iter=max_iter,
        reassignment_ratio=0.0,
    )
    labels = model.fit_predict(x)

    # Stable ordering preserves the original order inside each cluster.
    order = np.argsort(labels, kind="stable")
    return labels.astype(np.int32), order.astype(np.int64)


def per_cluster_metrics(
    a: np.ndarray,
    labels: np.ndarray,
    axis: str,
    tolerance: float | None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for cluster_id in range(int(labels.max()) + 1):
        members = np.flatnonzero(labels == cluster_id)
        block = a[:, members] if axis == "column" else a[members, :]
        metrics = rank_metrics(block, tolerance)
        rows.append(
            {
                "axis": axis,
                "cluster_id": cluster_id,
                "n_samples": int(members.size),
                "block_rows": int(block.shape[0]),
                "block_columns": int(block.shape[1]),
                **metrics,
            }
        )
    return rows


def weighted_summary(
    cluster_rows: list[dict[str, float | int | str]], axis: str
) -> dict[str, float | int | str]:
    selected = [r for r in cluster_rows if r["axis"] == axis]
    weights = np.asarray([r["n_samples"] for r in selected], dtype=np.float64)
    weights /= weights.sum()
    out: dict[str, float | int | str] = {
        "configuration": f"{axis}_clusters_weighted",
        "n_clusters": len(selected),
    }
    for key in (
        "numerical_rank",
        "effective_rank",
        "rank_95_energy",
        "first_direction_energy",
        "rank1_relative_error",
    ):
        values = np.asarray([r[key] for r in selected], dtype=np.float64)
        out[key] = float(np.sum(weights * values))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_cluster_summary(
    global_rows: list[dict],
    cluster_summary: list[dict],
    output: Path,
) -> None:
    original = global_rows[0]
    col = cluster_summary[0]
    row = cluster_summary[1]
    labels = ["Original global", "Column clusters", "Row clusters"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    metrics = [
        ("effective_rank", "Effective rank"),
        ("rank_95_energy", "Rank for 95% energy"),
        ("first_direction_energy", "First-direction energy"),
        ("rank1_relative_error", "Rank-1 relative error"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        values = [float(original[key]), float(col[key]), float(row[key])]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(
        "Global original matrix versus weighted rank structure inside clusters"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a = np.load(args.weight)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, received shape {a.shape}")
    if args.transpose_weight:
        a = a.T
    a = a.astype(np.float32, copy=False)

    column_labels, column_order = cluster_samples(
        a.T,
        args.k_columns,
        args.seed,
        args.batch_size,
        args.max_iter,
        args.normalize_samples,
    )
    row_labels, row_order = cluster_samples(
        a,
        args.k_rows,
        args.seed + 1,
        args.batch_size,
        args.max_iter,
        args.normalize_samples,
    )

    matrices = {
        "original": a,
        "column_reordered": a[:, column_order],
        "row_reordered": a[row_order, :],
        "row_and_column_reordered": a[np.ix_(row_order, column_order)],
    }
    global_rows = [
        {
            "configuration": name,
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            **rank_metrics(matrix, args.rank_tolerance),
        }
        for name, matrix in matrices.items()
    ]

    cluster_rows = per_cluster_metrics(
        a, column_labels, "column", args.rank_tolerance
    )
    cluster_rows += per_cluster_metrics(a, row_labels, "row", args.rank_tolerance)
    cluster_summary = [
        weighted_summary(cluster_rows, "column"),
        weighted_summary(cluster_rows, "row"),
    ]

    write_csv(output_dir / "global_rank_metrics.csv", global_rows)
    write_csv(output_dir / "per_cluster_rank_metrics.csv", cluster_rows)
    write_csv(output_dir / "cluster_rank_summary.csv", cluster_summary)
    np.save(output_dir / "column_order.npy", column_order)
    np.save(output_dir / "row_order.npy", row_order)
    np.save(output_dir / "column_labels.npy", column_labels)
    np.save(output_dir / "row_labels.npy", row_labels)

    plot_cluster_summary(
        global_rows,
        cluster_summary,
        output_dir / "row_column_cluster_rank_comparison.png",
    )

    metadata = {
        "weight": args.weight,
        "shape": list(a.shape),
        "k_columns": args.k_columns,
        "k_rows": args.k_rows,
        "normalize_samples": args.normalize_samples,
        "seed": args.seed,
        "mathematical_note": (
            "Row and column permutations preserve singular values; therefore "
            "all four global rank rows should agree up to floating-point error."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nGlobal matrix metrics")
    for row in global_rows:
        print(row)
    print("\nWeighted metrics inside clusters")
    for row in cluster_summary:
        print(row)
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

