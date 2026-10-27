#!/usr/bin/env python3
"""Collect the key numbers of one ARKS run directory into summary files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

EXPERIMENT_DIRS = {
    "fp16": "adaptive_baseline_fp16",
    "int8": "adaptive_baseline_int8",
    "vq_nocalib": "adaptive_vq_nocalib",
    "vq_calibrated": "adaptive_vq_calibrated",
    "sparsity": "adaptive_sparsity",
}


def latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def load_stamp(path: Path) -> dict:
    stamp = path / ".arks_stage.json"
    if stamp.is_file():
        try:
            return json.loads(stamp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir

    plan = {}
    plan_path = run / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    allocation = {}
    alloc_path = run / "shared" / "rank_allocation" / "selected_first_order_ranks.json"
    if alloc_path.is_file():
        allocation = json.loads(alloc_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for key, dirname in EXPERIMENT_DIRS.items():
        base = run / dirname
        if not base.is_dir():
            continue
        row = {
            "experiment": key,
            "directory": dirname,
            "ppl": "",
            "dense_ppl": "",
            "ppl_increase_percent": "",
            "artifact_storage_mb": "",
            "total_model_bits_per_parameter": "",
            "codebook_calibration": "",
            "effective_sparsity": "",
            "coefficient_storage_mib": "",
            "compress_seconds": "",
            "calibrate_seconds": "",
            "inference_seconds": "",
            "status": "",
        }
        compress_stamp = load_stamp(base / "artifacts")
        row["compress_seconds"] = compress_stamp.get("elapsed_seconds", "")
        if key == "sparsity":
            csv_path = latest(str(base / "results" / "adaptive_sparsity*.csv"))
            csv_path = csv_path if csv_path and "per_matrix" not in csv_path else latest(
                str(base / "results" / "adaptive_sparsity.csv")
            )
            if csv_path:
                with open(csv_path, newline="", encoding="utf-8") as handle:
                    entries = list(csv.DictReader(handle))
                dense = next((e for e in entries if e["method"] == "dense"), None)
                adaptive = next(
                    (e for e in entries if e["method"].endswith("adaptive")), None
                )
                if dense:
                    row["dense_ppl"] = dense["ppl"]
                if adaptive:
                    row["ppl"] = adaptive["ppl"]
                    row["effective_sparsity"] = adaptive["sparsity"]
                    row["coefficient_storage_mib"] = adaptive["coefficient_storage_mib"]
                    row["inference_seconds"] = adaptive["inference_seconds"]
                    if dense:
                        row["ppl_increase_percent"] = 100.0 * (
                            float(adaptive["ppl"]) / float(dense["ppl"]) - 1.0
                        )
            row["status"] = load_stamp(base / "results").get("status", "")
        else:
            json_path = latest(str(base / "results" / "inference_*.json"))
            if json_path:
                results = json.loads(Path(json_path).read_text(encoding="utf-8"))
                result = results[0] if isinstance(results, list) and results else results
                for field in (
                    "ppl",
                    "dense_ppl",
                    "ppl_increase_percent",
                    "artifact_storage_mb",
                    "total_model_bits_per_parameter",
                    "codebook_calibration",
                    "inference_seconds",
                ):
                    row[field] = result.get(field, "")
            if key == "vq_calibrated":
                row["calibrate_seconds"] = load_stamp(base / "calibration").get(
                    "elapsed_seconds", ""
                )
            row["status"] = load_stamp(base / "results").get("status", "")
        rows.append(row)

    summary = {
        "run_dir": str(run),
        "model": plan.get("MODEL_SUMMARY", {}).get("model"),
        "hardware": plan.get("HARDWARE_SUMMARY"),
        "plan": {k: plan.get(k) for k in ("DEVICE", "WORKERS", "THREADS_PER_WORKER", "LOSS_CHUNK_TOKENS", "MODEL_STAGING", "CALIBRATION_STREAMING")},
        "rank_allocation": {
            "selected_matrix_count": allocation.get("selected_matrix_count"),
            "used_extra_budget_mb": allocation.get("used_extra_budget_mb"),
            "requested_extra_budget_mb": allocation.get("requested_extra_budget_mb"),
        },
        "shared_stage_seconds": {
            "profile": load_stamp(run / "shared" / "rank_profile").get("elapsed_seconds"),
            "allocate": load_stamp(run / "shared" / "rank_allocation").get("elapsed_seconds"),
        },
        "experiments": rows,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with (run / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nRun summary")
    print(f"  model: {summary['model']}")
    print(
        "  rank allocation: "
        f"{summary['rank_allocation']['selected_matrix_count']} matrices at max rank, "
        f"{summary['rank_allocation']['used_extra_budget_mb']} MB used"
    )
    header = f"  {'experiment':<14}{'ppl':>14}{'dense':>12}{'+%':>9}{'MB':>10}{'bits/param':>12}  status"
    print(header)
    for row in rows:
        def num(value, digits):
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return "-"
        print(
            f"  {row['experiment']:<14}{num(row['ppl'], 4):>14}{num(row['dense_ppl'], 4):>12}"
            f"{num(row['ppl_increase_percent'], 2):>9}{num(row['artifact_storage_mb'], 2):>10}"
            f"{num(row['total_model_bits_per_parameter'], 3):>12}  {row['status']}"
        )
    print(f"  saved: {run / 'summary.csv'}")


if __name__ == "__main__":
    main()
