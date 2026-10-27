#!/usr/bin/env python3
"""Aggregate per-layer K-lines analysis outputs into one clean CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


RANK_COLUMNS = {
    "numerical_rank": ("original_numerical_rank", "klines_numerical_rank"),
    "effective_rank": ("original_effective_rank", "klines_effective_rank"),
    "rank_95_energy": ("original_rank95", "klines_rank95"),
    "first_singular_energy_fraction": (
        "original_first_direction_energy",
        "klines_first_direction_energy",
    ),
    "best_rank1_relative_error": (
        "original_rank1_error",
        "klines_rank1_error",
    ),
}

SUMMARY_COLUMNS = {
    ("layer_reconstruction", "seedlm_reconstruction_mse"): "actual_klines_mse",
    (
        "layer_reconstruction",
        "ideal_stored_direction_projection_mse",
    ): "ideal_projection_mse",
    (
        "layer_reconstruction",
        "observed_representative_mse",
    ): "fixed_tile_mse",
    (
        "layer_reconstruction",
        "seedlm_relative_frobenius_error",
    ): "actual_klines_relative_error",
    (
        "layer_reconstruction",
        "ideal_stored_direction_projection_relative_frobenius_error",
    ): "ideal_projection_relative_error",
    (
        "layer_reconstruction",
        "observed_representative_relative_frobenius_error",
    ): "fixed_tile_relative_error",
    (
        "klines_explanation_correlation",
        "corr_mean_absolute_cosine_to_klines_direction_vs_actual_klines_error",
    ): "corr_abs_cosine_vs_klines_error",
    (
        "klines_explanation_correlation",
        "corr_mean_tile_projection_relative_error_vs_actual_klines_error",
    ): "corr_projection_error_vs_klines_error",
    (
        "klines_explanation_correlation",
        "corr_coefficient_relative_rmse_vs_actual_klines_error",
    ): "corr_coefficient_rmse_vs_klines_error",
    (
        "klines_explanation_correlation",
        "corr_coefficient_correlation_vs_actual_klines_error",
    ): "corr_coefficient_similarity_vs_klines_error",
}


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_percent_reduction(original: float, new: float) -> float:
    if not math.isfinite(original) or not math.isfinite(new) or original == 0:
        return float("nan")
    return 100.0 * (original - new) / original


def layer_name(folder: Path) -> str:
    match = re.search(r"(layer_\d+_fc[12])", str(folder))
    return match.group(1) if match else folder.name


def read_rank_comparison(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            metric = row["metric"]
            if metric not in RANK_COLUMNS:
                continue
            original_name, klines_name = RANK_COLUMNS[metric]
            result[original_name] = as_float(row["original_unclustered_tiles"])
            result[klines_name] = as_float(
                row["klines_clusters_weighted_aggregate"]
            )
    return result


def read_layer_summary(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            output_name = SUMMARY_COLUMNS.get((row["section"], row["metric"]))
            if output_name:
                result[output_name] = as_float(row["value"])
    return result


def find_result_folders(root: Path) -> list[Path]:
    folders = {
        rank_file.parent
        for rank_file in root.rglob("original_vs_cluster_rank.csv")
        if (rank_file.parent / "layer_summary.csv").is_file()
    }
    named_layer_folders = {
        folder for folder in folders
        if re.search(r"layer_\d+_fc[12]", str(folder))
    }
    # If per-layer subfolders exist, ignore an older result stored directly in
    # the root because its layer identity is ambiguous and may duplicate layer 0.
    if named_layer_folders:
        folders = named_layer_folders
    return sorted(folders, key=lambda path: layer_name(path))


def add_derived_metrics(row: dict[str, float | str]) -> None:
    row["effective_rank_reduction_pct"] = safe_percent_reduction(
        float(row.get("original_effective_rank", float("nan"))),
        float(row.get("klines_effective_rank", float("nan"))),
    )
    row["rank95_reduction_pct"] = safe_percent_reduction(
        float(row.get("original_rank95", float("nan"))),
        float(row.get("klines_rank95", float("nan"))),
    )
    row["first_direction_energy_gain_points"] = 100.0 * (
        float(row.get("klines_first_direction_energy", float("nan")))
        - float(row.get("original_first_direction_energy", float("nan")))
    )
    row["rank1_error_reduction_pct"] = safe_percent_reduction(
        float(row.get("original_rank1_error", float("nan"))),
        float(row.get("klines_rank1_error", float("nan"))),
    )
    row["klines_gap_from_ideal_mse_pct"] = (
        100.0
        * (
            float(row.get("actual_klines_mse", float("nan")))
            - float(row.get("ideal_projection_mse", float("nan")))
        )
        / float(row.get("ideal_projection_mse", float("nan")))
        if float(row.get("ideal_projection_mse", float("nan"))) > 0
        else float("nan")
    )
    row["mse_reduction_vs_fixed_tile_pct"] = safe_percent_reduction(
        float(row.get("fixed_tile_mse", float("nan"))),
        float(row.get("actual_klines_mse", float("nan"))),
    )


def aggregate_row(
    rows: list[dict[str, float | str]],
    fieldnames: list[str],
    name: str,
    function,
) -> dict[str, float | str]:
    output: dict[str, float | str] = {"layer": name}
    for field in fieldnames:
        if field == "layer":
            continue
        values = [
            float(row[field])
            for row in rows
            if isinstance(row.get(field), (int, float))
            and math.isfinite(float(row[field]))
        ]
        output[field] = function(values) if values else float("nan")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/klines_properties_corrected"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/klines_all_layers_results.csv"),
    )
    args = parser.parse_args()

    folders = find_result_folders(args.input_root)
    if not folders:
        raise FileNotFoundError(
            f"No folders containing both original_vs_cluster_rank.csv and "
            f"layer_summary.csv were found under {args.input_root}"
        )

    rows: list[dict[str, float | str]] = []
    for folder in folders:
        row: dict[str, float | str] = {"layer": layer_name(folder)}
        row.update(read_rank_comparison(folder / "original_vs_cluster_rank.csv"))
        row.update(read_layer_summary(folder / "layer_summary.csv"))
        add_derived_metrics(row)
        rows.append(row)

    preferred_order = [
        "layer",
        "original_numerical_rank",
        "klines_numerical_rank",
        "original_effective_rank",
        "klines_effective_rank",
        "effective_rank_reduction_pct",
        "original_rank95",
        "klines_rank95",
        "rank95_reduction_pct",
        "original_first_direction_energy",
        "klines_first_direction_energy",
        "first_direction_energy_gain_points",
        "original_rank1_error",
        "klines_rank1_error",
        "rank1_error_reduction_pct",
        "fixed_tile_mse",
        "ideal_projection_mse",
        "actual_klines_mse",
        "mse_reduction_vs_fixed_tile_pct",
        "klines_gap_from_ideal_mse_pct",
        "fixed_tile_relative_error",
        "ideal_projection_relative_error",
        "actual_klines_relative_error",
        "corr_abs_cosine_vs_klines_error",
        "corr_projection_error_vs_klines_error",
        "corr_coefficient_rmse_vs_klines_error",
        "corr_coefficient_similarity_vs_klines_error",
    ]
    available = {key for row in rows for key in row}
    fieldnames = [name for name in preferred_order if name in available]

    rows.append(
        aggregate_row(rows, fieldnames, "MEAN", statistics.fmean)
    )
    rows.append(
        aggregate_row(rows[:-1], fieldnames, "MEDIAN", statistics.median)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Layers aggregated: {len(folders)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

# cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification
#
# /root/.virtualenvs/transformer-sparsification/bin/python \
# aggregate_klines_layer_results.py \
# --input-root results/klines_properties_corrected \
# --output results/klines_all_layers_results.csv