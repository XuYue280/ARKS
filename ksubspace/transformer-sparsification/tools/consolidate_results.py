#!/usr/bin/env python3
"""Consolidate every ARKS inference record into one self-describing JSON file.

Each run writes a 49-column CSV under
arks_runs/<model>/seed_0/ppl/<corpus>/adaptive_baseline_<precision>/inference_*.csv.
Those 49 columns do NOT say which eval corpus or coefficient precision the row
is for -- that lives in the directory path -- so a plain concatenation of the
CSVs is ambiguous. This script recovers the four identity fields from the path
and ships them alongside the 49, with a column dictionary so a reader never has
to guess what `C`, `sparsity` or `effective_ranks` mean.

Output: arks_all_results.json
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "arks_all_results.json"

MODEL_ORDER = ["opt-125m", "opt-6.7b", "opt-13b", "opt-30b", "opt-66b",
               "Llama-3.2-1B", "Llama-3.1-8B", "Llama-3.1-70B"]
CORPUS_ORDER = ["wikitext2", "ptb", "c4"]

# Identity recovered from the directory path -- NOT part of the 49 CSV columns.
IDENTITY = [
    ("model_tag", "string", None, "Run directory name, e.g. 'opt-66b'. Short form of the `model` column."),
    ("eval_corpus", "string", None,
     "Evaluation corpus this row's perplexity was measured on: wikitext2 | ptb | c4. "
     "NOT present in the CSV -- recovered from the ppl/<corpus>/ path segment."),
    ("precision", "string", None,
     "Coefficient precision: 'fp16' (16-bit) or 'int8' (8-bit). Recovered from the "
     "adaptive_baseline_<precision>/ path segment; matches the `coefficient_bits` column."),
    ("seed", "integer", None, "Compression/allocation seed. 0 for every run here."),
]

# (name, type, unit, description) for all 49 CSV columns, in file order.
COLUMNS = [
    # --- configuration and scope (1-23) ---
    ("model", "string", None, "Hugging Face model id, e.g. 'facebook/opt-66b'."),
    ("target", "string", None, "Which weights were compressed. 'all' in every run here."),
    ("codebook_calibration", "string", None,
     "Codebook fine-tuning mode. 'none' in every run -- no gradient training of the "
     "basis was performed, so these results are training-free."),
    ("matrix_types", "string", None,
     "Comma-separated linear-layer suffixes that were compressed (q/k/v/o_proj, MLP projections)."),
    ("selected_layers", "string", None, "Comma-separated decoder layer indices that were compressed."),
    ("rank", "integer", None, "Base rank r applied to every tile before the allocator promotes any."),
    ("rank_overrides", "json-string", None,
     "JSON object mapping 'layer_NN_<suffix>' -> promoted rank, for the matrices the "
     "first-order allocator selected to upgrade. Empty object means nothing was promoted."),
    ("tile_overrides", "json-string", None, "JSON object of per-matrix tile-shape overrides. '{}' in every run here."),
    ("effective_ranks", "string", None, "The distinct ranks actually present after allocation, e.g. '4,6'."),
    ("coefficient_bits", "integer", "bits", "Bits per stored coefficient: 16 for fp16 runs, 8 for int8 runs."),
    ("coefficient_encoding", "string", None, "Coefficient encoding scheme. 'scalar' in every run here."),
    ("coefficient_codebook_size", "integer|empty", None, "VQ codebook size; empty under scalar encoding."),
    ("coefficient_vq_stages", "integer|empty", None, "VQ stage count; empty under scalar encoding."),
    ("vq_stage_overrides", "json-string", None, "Per-matrix VQ stage overrides. '{}' in every run here."),
    ("effective_vq_stages", "string", None, "Distinct VQ stage counts present. '1' in every run here."),
    ("sparsity", "float", None,
     "The --sparsity argument as passed to the benchmark. 1.0 in every run here "
     "(no coefficient sparsification on top of the subspace coding)."),
    ("k", "integer", None, "Number of shared subspace basis vectors (K). 512 in every run here."),
    ("C", "integer", None, "Tile height c."),
    ("tile_width", "integer", None, "Tile width t."),
    ("tile_shape", "string", None, "Tile shape as 'CxT', e.g. '8x1'."),
    ("effective_tile_heights", "string", None, "Distinct tile heights present after any overrides."),
    ("layers_evaluated", "integer", None, "Number of decoder layers compressed."),
    ("matrices_evaluated", "integer", None, "Number of weight matrices compressed."),
    # --- quality (24-34) ---
    ("ppl", "float", None, "Perplexity of the ARKS-compressed model on `eval_corpus`."),
    ("ppl_tokens", "integer", "tokens", "Token count the compressed-model perplexity was averaged over."),
    ("dense_ppl", "float", None,
     "Perplexity of the uncompressed model on the same corpus and tokenization. "
     "The baseline every degradation figure is relative to."),
    ("dense_ppl_tokens", "integer", "tokens", "Token count for the dense baseline. Equals ppl_tokens."),
    ("ppl_increase", "float", None, "ppl - dense_ppl, in absolute perplexity units."),
    ("ppl_increase_percent", "float", "%", "100 * (ppl / dense_ppl - 1)."),
    ("dense_inference_seconds", "float", "s", "Wall-clock of the dense baseline forward pass."),
    ("mse_all_weights", "float", None,
     "Mean squared reconstruction error over every compressed weight element: "
     "sum((W - W_hat)^2) / count."),
    ("mse_nonzero_weights", "float", None, "Same, restricted to elements that are nonzero in the original weight."),
    ("relative_frobenius_error", "float", None,
     "||W - W_hat||_F / ||W||_F, aggregated over all compressed matrices."),
    ("evaluated_weights", "integer", "elements", "Number of weight elements compressed and scored."),
    # --- storage (35-39) ---
    ("artifact_storage_mb", "float", "MiB",
     "On-disk size of the compressed artifacts: shared bases + per-tile coefficients + subspace ids."),
    ("artifact_bits_per_compressed_parameter", "float", "bits",
     "8 * artifact_bytes / compressed_parameter_count. Counts only the compressed matrices."),
    ("static_parameter_storage_mb", "float", "MiB",
     "Size of the parameters left untouched (embeddings, layer norms, lm_head) at their original precision."),
    ("total_model_storage_mb", "float", "MiB", "artifact_storage_mb + static_parameter_storage_mb."),
    ("total_model_bits_per_parameter", "float", "bits",
     "8 * (artifact_bytes + static_bytes) / total_parameter_count. "
     "The parameter retention ratio is rho = this / 16; compression reduction is 1 - rho."),
    # --- timing (40-45) ---
    ("model_load_seconds", "float", "s", "Loading the base model weights."),
    ("artifact_load_seconds", "float", "s", "Loading the compressed artifacts from disk."),
    ("reconstruction_seconds", "float", "s",
     "Rebuilding dense weights from the artifacts. Single-threaded NumPy; on the 66B/70B "
     "models this is ~98% of the stage and dwarfs the GPU forward pass."),
    ("metric_seconds", "float", "s", "Computing the MSE / Frobenius reconstruction metrics."),
    ("inference_seconds", "float", "s", "GPU forward pass for the perplexity evaluation only."),
    ("total_seconds", "float", "s", "Total wall-clock for the inference stage."),
    # --- resource peaks (46-49) ---
    ("peak_cpu_rss_mb", "float", "MiB", "Peak host resident set size of the process, absolute."),
    ("peak_cpu_rss_increase_mb", "float", "MiB", "Peak host RSS measured as an increase over process start."),
    ("peak_cuda_allocated_mb", "float", "MiB", "Peak VRAM actually allocated by PyTorch tensors."),
    ("peak_cuda_reserved_mb", "float", "MiB",
     "Peak VRAM reserved from the driver by the caching allocator. Always >= allocated; "
     "this is the number to use when asking whether a model fits on a GPU."),
]

CAVEATS = [
    "The 49 CSV columns carry no eval-corpus or precision field; both are recovered from the "
    "directory path and exposed here as `eval_corpus` and `precision`. Never join these rows "
    "without them -- a model has 6 rows (3 corpora x 2 precisions) that are otherwise identical "
    "in their configuration columns.",
    "codebook_calibration is 'none' everywhere: no weights and no codebooks were trained. The "
    "rank allocator does run a backward pass, but only to read dL/dW for its importance score -- "
    "there is no optimizer and no parameter update. These are one-shot, post-training results.",
    "Calibration always used wikitext2 train (the first 4 x 256 = 1024 tokens of the joined "
    "split, deterministically, no shuffling), for every model, corpus and precision. wikitext2 "
    "eval is therefore in-domain with calibration; PTB and C4 are held out.",
    "peak_cuda_reserved_mb >= peak_cuda_allocated_mb. Use reserved for capacity questions.",
    "rho (parameter retention) = total_model_bits_per_parameter / 16. Compression reduction is "
    "1 - rho. The two are easy to confuse; state which one a figure plots.",
    "opt-66b and Llama-3.1-70B ran their inference layer-wise (one decoder layer resident on the "
    "GPU at a time) because a reconstructed dense copy does not fit in 80 GB. Verified bitwise "
    "identical to the all-on-GPU path on opt-125m, so the PPL numbers are directly comparable.",
    "Llama-3.1-70B on wikitext2 is the one outlier in the set: +42.95% versus +5.65% on PTB and "
    "+12.53% on C4 for the same model. Unexplained; do not average it away without noting it.",
]


def main() -> None:
    pattern = str(ROOT / "arks_runs" / "*" / "seed_0" / "ppl" / "*" / "*" / "inference_*.csv")
    paths = sorted(glob.glob(pattern))

    numeric = {name for name, typ, _, _ in COLUMNS if typ.startswith(("float", "integer"))}
    rows = []
    for path in paths:
        parts = path.split(os.sep)
        model_tag, corpus, experiment = parts[-6], parts[-3], parts[-2]
        with open(path, newline="", encoding="utf-8") as handle:
            record = next(csv.DictReader(handle))

        row = {
            "model_tag": model_tag,
            "eval_corpus": corpus,
            "precision": "int8" if "int8" in experiment else "fp16",
            "seed": 0,
        }
        for name, typ, _unit, _desc in COLUMNS:
            raw = record.get(name, "")
            if name in numeric and raw not in ("", None):
                try:
                    value = float(raw)
                    if typ.startswith("integer") and value.is_integer():
                        value = int(value)
                except ValueError:
                    value = raw
            else:
                value = raw if raw != "" else None
            row[name] = value
        rows.append(row)

    rows.sort(key=lambda r: (MODEL_ORDER.index(r["model_tag"]),
                             r["precision"],
                             CORPUS_ORDER.index(r["eval_corpus"])))

    expected = len(MODEL_ORDER) * 2 * len(CORPUS_ORDER)
    have = {(r["model_tag"], r["precision"], r["eval_corpus"]) for r in rows}
    missing = [f"{m}/{p}/{c}" for m in MODEL_ORDER for p in ("fp16", "int8")
               for c in CORPUS_ORDER if (m, p, c) not in have]

    document = {
        "schema_version": 1,
        "description": (
            "Every completed ARKS K-subspace compression run, one record per "
            "(model, coefficient precision, eval corpus). Each record carries the 49 columns "
            "the benchmark writes, plus 4 identity fields recovered from the run directory path."
        ),
        "source": "arks_runs/<model_tag>/seed_0/ppl/<eval_corpus>/adaptive_baseline_<precision>/inference_*.csv",
        "experiment": {
            "method": "ARKS -- K-subspaces weight compression with first-order adaptive rank allocation",
            "k_subspaces": 512,
            "base_rank": 4,
            "promoted_rank": 6,
            "calibration_dataset": "wikitext2",
            "calibration_split": "train",
            "calibration_tokens": 1024,
            "calibration_shape": "4 samples x 256 tokens, first 1024 tokens of the joined split, not shuffled",
            "extra_budget_mb": {"opt": 100.0, "llama": 200.0},
            "eval_corpora": {
                "wikitext2": "test split, rows joined with '\\n\\n', full split",
                "ptb": "test split, rows stripped and joined with ' ', full split",
                "c4": "validation split, first 1100 rows joined with ' ', first 256 segments of 2048 tokens",
            },
            "seed": 0,
            "training_free": True,
        },
        "status": {
            "records": len(rows),
            "expected": expected,
            "missing": missing,
        },
        "identity_fields": [
            {"name": n, "type": t, "unit": u, "description": d} for n, t, u, d in IDENTITY
        ],
        "columns": [
            {"index": i, "name": n, "type": t, "unit": u, "description": d}
            for i, (n, t, u, d) in enumerate(COLUMNS, start=1)
        ],
        "caveats": CAVEATS,
        "rows": rows,
    }

    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({len(rows)}/{expected} records, {len(COLUMNS)} columns, {size_kb:.0f} KB)")
    if missing:
        print("missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
