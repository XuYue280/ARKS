#!/usr/bin/env python3
"""Build the PPL and parameter-compression figures for the ARKS runs.

One pair of figures per evaluation corpus (WikiText-2, PTB, C4), each split by
model family, written to <dataset>_ppl/ and <dataset>_parameter_compression_ratio/.

Perplexities come from arks_runs/<model>/seed_0/ppl/<dataset>/adaptive_baseline_{fp16,int8}.
The parameter reduction ratio is the analytic rho of the paper, evaluated per weight
matrix from the run configuration (k, tile c x t, per-matrix rank) and the model shapes
recorded in model_info.json.

A model with no results yet for a given corpus is skipped and reported, never
silently dropped -- rerun once its jobs land and it joins the figures unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "transformer-sparsification" / "arks_runs"
SEED_DIR = "seed_0"

# (directory stem, display name) -- the eval corpora, one figure pair each.
DATASETS = [
    ("wikitext2", "WikiText-2"),
    ("ptb", "PTB"),
    ("c4", "C4"),
]

# OPT-125M was the smoke-test model. Its PPL is roughly 3x every other OPT, so on
# a shared axis it flattens the 6.7B-66B trend these figures exist to show; a log
# axis only trades that for an unreadable tick set. Flip this to put it back.
INCLUDE_SMOKE_TEST_MODEL = False

# (run directory, display label, family) ordered by model size within a family.
MODELS = [
    *([("opt-125m", "OPT-125M", "opt")] if INCLUDE_SMOKE_TEST_MODEL else []),
    ("opt-6.7b", "OPT-6.7B", "opt"),
    ("opt-13b", "OPT-13B", "opt"),
    ("opt-30b", "OPT-30B", "opt"),
    ("opt-66b", "OPT-66B", "opt"),
    ("Llama-3.2-1B", "Llama-3.2-1B", "llama"),
    ("Llama-3.1-8B", "Llama-3.1-8B", "llama"),
    ("Llama-3.1-70B", "Llama-3.1-70B", "llama"),
]

LAYER_PREFIX = re.compile(r"^layer_\d+_")


def read_experiment(model_dir: Path, dataset: str, experiment: str) -> dict | None:
    """Latest inference record, or None when that run has not landed yet."""
    result_dir = model_dir / SEED_DIR / "ppl" / dataset / f"adaptive_baseline_{experiment}"
    payloads = sorted(result_dir.glob("inference_*.json"))
    if not payloads:
        return None
    records = json.loads(payloads[-1].read_text())
    return records[-1]


def matrix_ranks(record: dict, model_info: dict) -> list[tuple[int, int, int]]:
    """Return (rows_out, cols_in, rank) for every compressed weight matrix."""
    shapes = {entry["suffix"]: entry for entry in model_info["matrix_shapes"]}
    overrides = json.loads(record["rank_overrides"])
    unknown = {LAYER_PREFIX.sub("", name) for name in overrides} - set(shapes)
    if unknown:
        raise KeyError(f"rank override suffixes absent from model_info: {sorted(unknown)}")

    base_rank = int(record["rank"])
    matrices = []
    for layer in range(int(model_info["num_layers"])):
        for suffix, shape in shapes.items():
            rank = overrides.get(f"layer_{layer:02d}_{suffix}", base_rank)
            matrices.append((shape["rows_out"], shape["cols_in"], int(rank)))
    if len(matrices) != int(record["matrices_evaluated"]):
        raise ValueError(
            f"{model_info['model_id']}: enumerated {len(matrices)} matrices but the run "
            f"compressed {record['matrices_evaluated']}"
        )
    return matrices


def reduction_ratio(record: dict, model_info: dict, coefficient_bits: int) -> float:
    """1 - rho, aggregated over all compressed matrices of the model."""
    k = int(record["k"])
    ct = int(record["C"]) * int(record["tile_width"])
    id_bits = math.ceil(math.log2(k))

    compressed = 0.0
    dense = 0.0
    for rows_out, cols_in, rank in matrix_ranks(record, model_info):
        n_dense = rows_out * cols_in
        n_tiles = n_dense / ct
        coeff = n_tiles * rank * (coefficient_bits / 16.0)
        basis = k * ct * rank
        ids = n_tiles * (id_bits / 16.0)
        compressed += coeff + basis + ids
        dense += n_dense
    return 1.0 - compressed / dense


def collect(dataset: str) -> tuple[list[dict], list[str], list[str]]:
    """Rows for every model with at least one run, plus what was skipped / partial."""
    rows: list[dict] = []
    skipped: list[str] = []
    partial: list[str] = []
    for tag, label, family in MODELS:
        model_dir = RUNS / tag
        info_path = model_dir / SEED_DIR / "model_info.json"
        if not info_path.exists():
            skipped.append(f"{label} (no model_info.json)")
            continue
        model_info = json.loads(info_path.read_text())

        fp16 = read_experiment(model_dir, dataset, "fp16")
        int8 = read_experiment(model_dir, dataset, "int8")
        if fp16 is None and int8 is None:
            skipped.append(f"{label} (no runs yet)")
            continue
        if int8 is None:
            partial.append(f"{label} (int8 pending)")
        elif fp16 is None:
            partial.append(f"{label} (fp16 pending)")

        present = fp16 or int8
        dense_ppl = present["dense_ppl"]
        if fp16 is not None and int8 is not None:
            if not math.isclose(dense_ppl, int8["dense_ppl"], rel_tol=1e-9):
                raise ValueError(f"{label}: fp16/int8 runs disagree on the dense baseline PPL")

        dense_linear_mb = model_info["decoder_linear_elements"] * 2 / 1024**2

        def ppl(rec):
            return float("nan") if rec is None else rec["ppl"]

        def reduction(rec, bits):
            return float("nan") if rec is None else 100.0 * reduction_ratio(rec, model_info, bits)

        def measured(rec):
            return (float("nan") if rec is None
                    else 100.0 * (1.0 - rec["artifact_storage_mb"] / dense_linear_mb))

        rows.append(
            {
                "model": label,
                "family": family,
                "run_dir": tag,
                "dataset": dataset,
                "params_linear": model_info["decoder_linear_elements"],
                "k": present["k"],
                "tile": f"{present['C']}x{present['tile_width']}",
                "base_rank": present["rank"],
                "effective_ranks": present["effective_ranks"],
                "dense_ppl": dense_ppl,
                "arks_fp16_ppl": ppl(fp16),
                "arks_int8_ppl": ppl(int8),
                "reduction_fp16_percent": reduction(fp16, 16),
                "reduction_int8_percent": reduction(int8, 8),
                "measured_reduction_fp16_percent": measured(fp16),
                "measured_reduction_int8_percent": measured(int8),
            }
        )
    return rows, skipped, partial


def draw(
    rows: list[dict],
    out_dir: Path,
    family: str,
    title: str,
    ylabel: str,
    series: tuple[tuple[str, str], ...],
    fmt: str,
    log_y: bool = False,
) -> bool:
    subset = [row for row in rows if row["family"] == family]
    if not subset:
        return False
    labels = [row["model"] for row in subset]
    xs = list(range(len(subset)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, (key, label) in enumerate(series):
        ys = [row[key] for row in subset]
        ax.plot(xs, ys, marker="o", label=label)
        # Stagger the labels so overlapping series stay readable.
        offset = 11 if index % 2 == 0 else -17
        for x, y in zip(xs, ys):
            if y != y:  # NaN: this run has not landed yet
                continue
            ax.annotate(
                fmt.format(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=8,
            )

    ax.set_xlim(-0.5, len(subset) - 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.margins(y=0.12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{family}.png", dpi=150)
    plt.close(fig)
    return True


REDUCTION_SERIES = (("reduction_int8_percent", "ARKS INT8"),)

PPL_SERIES = (
    ("dense_ppl", "Dense PPL"),
    ("arks_int8_ppl", "ARKS INT8 PPL"),
)


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for dataset, display in DATASETS:
        rows, skipped, partial = collect(dataset)
        print(f"\n=== {display} ({dataset}) ===")
        if skipped:
            print("  skipped: " + ", ".join(skipped))
        if partial:
            print("  partial: " + ", ".join(partial))
        if not rows:
            print("  no complete runs yet -- no figures written")
            continue

        ratio_dir = ROOT / f"{dataset}_parameter_compression_ratio"
        ppl_dir = ROOT / f"{dataset}_ppl"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        ppl_dir.mkdir(parents=True, exist_ok=True)

        for family, name in (("opt", "OPT"), ("llama", "Llama")):
            draw(
                rows,
                ratio_dir,
                family,
                f"{display}: Compression Reduction across {name} Models",
                "Reduction (%)",
                REDUCTION_SERIES,
                "{:.1f}%",
            )
            draw(
                rows,
                ppl_dir,
                family,
                f"{display}: PPL across {name} Models",
                "Perplexity (PPL)",
                PPL_SERIES,
                "{:.2f}",
            )

        write_csv(rows, ratio_dir / "data.csv")
        write_csv(rows, ppl_dir / "data.csv")

        header = (
            f"  {'model':<16}{'dense':>9}{'fp16':>9}{'int8':>9}{'R_fp16':>9}{'R_int8':>9}"
        )
        print(header)
        for row in rows:
            print(
                f"  {row['model']:<16}{row['dense_ppl']:>9.3f}{row['arks_fp16_ppl']:>9.3f}"
                f"{row['arks_int8_ppl']:>9.3f}{row['reduction_fp16_percent']:>8.2f}%"
                f"{row['reduction_int8_percent']:>8.2f}%".replace("nan", " --")
            )


if __name__ == "__main__":
    main()
