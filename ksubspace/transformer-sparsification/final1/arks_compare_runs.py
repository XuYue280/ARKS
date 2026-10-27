#!/usr/bin/env python3
"""Compare the results of one ARKS run directory with a reference directory.

Both directories use the experiment sub-directory layout
(``adaptive_baseline_fp16``, ``adaptive_baseline_int8``, ``adaptive_vq_nocalib``,
``adaptive_vq_calibrated``, ``adaptive_sparsity``). The reference may be a run
of the original ``run_final.sh`` (rank allocation inside each experiment) or
another ``arks_pipeline.sh`` run (shared rank allocation).

Exit status is 0 when every compared value is within ``--rel-tol`` (relative)
or ``--abs-tol`` (absolute), otherwise 1.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

EXPERIMENTS = [
    "adaptive_baseline_fp16",
    "adaptive_baseline_int8",
    "adaptive_vq_nocalib",
    "adaptive_vq_calibrated",
]
INFERENCE_FIELDS = [
    "ppl",
    "dense_ppl",
    "mse_all_weights",
    "relative_frobenius_error",
    "artifact_storage_mb",
    "total_model_bits_per_parameter",
]


def latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def load_inference(base: Path) -> dict | None:
    path = latest(str(base / "results" / "inference_*.json"))
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) and data else data


def load_allocation(base: Path, run: Path) -> dict | None:
    for candidate in (base / "rank_allocation", run / "shared" / "rank_allocation"):
        path = candidate / "selected_first_order_ranks.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_sparsity(base: Path, name_hint: str | None) -> dict | None:
    candidates = sorted(glob.glob(str(base / "results" / "adaptive_sparsity*.csv")))
    candidates = [c for c in candidates if "per_matrix" not in c]
    if name_hint:
        preferred = [c for c in candidates if name_hint in os.path.basename(c)]
        candidates = preferred or candidates
    if not candidates:
        return None
    with open(candidates[-1], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    adaptive = next((r for r in rows if r["method"].endswith("adaptive")), None)
    dense = next((r for r in rows if r["method"] == "dense"), None)
    return {"file": candidates[-1], "adaptive": adaptive, "dense": dense}


def load_calibration(base: Path) -> list[dict] | None:
    path = base / "results" / "block_codebook_calibration.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Comparator:
    def __init__(self, rel_tol: float, abs_tol: float):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.rows: list[tuple[str, str, str, str, str]] = []
        self.failures = 0

    def number(self, label: str, name: str, a, b) -> None:
        try:
            x = float(a)
            y = float(b)
        except (TypeError, ValueError):
            status = "same" if str(a) == str(b) else "DIFF"
            self.rows.append((label, name, str(a), str(b), status))
            self.failures += status == "DIFF"
            return
        diff = abs(x - y)
        rel = diff / max(abs(x), abs(y), 1e-30)
        if diff == 0:
            status = "exact"
        elif diff <= self.abs_tol or rel <= self.rel_tol:
            status = f"ok (rel {rel:.2e})"
        else:
            status = f"DIFF (rel {rel:.2e})"
            self.failures += 1
        self.rows.append((label, name, repr(x), repr(y), status))

    def report(self) -> None:
        width = max(len(r[0]) + len(r[1]) for r in self.rows) + 3
        for label, name, a, b, status in self.rows:
            print(f"  {(label + ' ' + name):<{width}} {a:>24} {b:>24}  {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    parser.add_argument("--sparsity-reference-name", default=None, help="substring selecting the reference sparsity CSV, e.g. 20_35_fp16")
    args = parser.parse_args()
    run, ref = args.run_dir, args.reference_dir
    cmp = Comparator(args.rel_tol, args.abs_tol)

    print(f"run:       {run}\nreference: {ref}\n")
    allocation_run = load_allocation(run / EXPERIMENTS[0], run)
    allocation_ref = load_allocation(ref / EXPERIMENTS[0], ref)
    if allocation_run and allocation_ref:
        same = allocation_run["rank_overrides"] == allocation_ref["rank_overrides"]
        cmp.rows.append(("allocation", "rank_overrides", f"{len(allocation_run['rank_overrides'])} matrices", f"{len(allocation_ref['rank_overrides'])} matrices", "exact" if same else "DIFF"))
        cmp.failures += not same
        cmp.number("allocation", "calibration_loss", allocation_run["calibration_loss"], allocation_ref["calibration_loss"])
        cmp.number("allocation", "used_extra_budget_mb", allocation_run["used_extra_budget_mb"], allocation_ref["used_extra_budget_mb"])

    for experiment in EXPERIMENTS:
        a = load_inference(run / experiment)
        b = load_inference(ref / experiment)
        if a is None or b is None:
            cmp.rows.append((experiment, "inference", "present" if a else "missing", "present" if b else "missing", "skipped"))
            continue
        for field in INFERENCE_FIELDS:
            cmp.number(experiment, field, a.get(field), b.get(field))
        cmp.number(experiment, "codebook_calibration", a.get("codebook_calibration"), b.get("codebook_calibration"))

    ca = load_calibration(run / "adaptive_vq_calibrated")
    cb = load_calibration(ref / "adaptive_vq_calibrated")
    if ca and cb:
        for ra, rb in zip(ca, cb):
            for field in ("initial_validation_block_mse", "calibrated_validation_block_mse", "epochs_completed", "best_epoch", "block_update_accepted"):
                cmp.number(f"calibration layer {ra['layer']}", field, ra.get(field), rb.get(field))
        for field in ("initial_guard_ppl", "calibrated_guard_ppl", "global_update_accepted"):
            cmp.number("calibration", field, ca[0].get(field), cb[0].get(field))

    sa = load_sparsity(run / "adaptive_sparsity", None)
    sb = load_sparsity(ref / "adaptive_sparsity", args.sparsity_reference_name)
    if sa and sb and sa["adaptive"] and sb["adaptive"]:
        for field in ("ppl", "sparsity", "nnz_fraction", "coefficient_storage_mib"):
            cmp.number("sparsity", field, sa["adaptive"].get(field), sb["adaptive"].get(field))
        if sa["dense"] and sb["dense"]:
            cmp.number("sparsity", "dense_ppl", sa["dense"].get("ppl"), sb["dense"].get("ppl"))

    cmp.report()
    print(f"\n{cmp.failures} value(s) outside tolerance (rel {args.rel_tol:g}, abs {args.abs_tol:g})")
    raise SystemExit(1 if cmp.failures else 0)


if __name__ == "__main__":
    main()
