
#!/usr/bin/env python3
"""Compare direct weight magnitude pruning with magnitude pruning of ARKS coefficients.

Supports OPT and Llama and an existing scalar-coefficient ARKS artifact tree.

Two coefficient-pruning modes are supported:

1. Uniform:
   the same coefficient sparsity is applied to every matrix through --sparsities.

2. Adaptive:
   sensitive matrices use --sensitive-coeff-sparsity and all other matrices use
   --robust-coeff-sparsity. Sensitive matrices can be read from the
   "rank_overrides" dictionary in --sensitivity-json and/or supplied explicitly
   with repeated --sensitive-matrix arguments.

The adaptive coefficient-sparsity policy is independent of the rank stored in the
artifact. Therefore it also works with a uniform-rank artifact such as rank 7.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arks_common import dense_ppl_cache_lookup, dense_ppl_cache_store
from bench_inference_minibatch_vq import (
    add_dataset_arguments,
    resolve_dataset,
    test_loader,
    perplexity,
    decoder_linear_records,
    dense_ppl_cache_key,
)
from ksubspaces_artifacts import _load_coefficients, _load_subspace_ids


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--artifact-root", type=Path, required=True)

    # Existing uniform sweep.
    p.add_argument(
        "--sparsities",
        nargs="+",
        type=float,
        default=[],
        help=(
            "Uniform pruning fractions. Example: --sparsities 0.20 0.25 0.50. "
            "Omit when only --adaptive-coeff-pruning is required."
        ),
    )

    # New adaptive coefficient-sparsity policy.
    p.add_argument(
        "--adaptive-coeff-pruning",
        action="store_true",
        help="Run one matrix-adaptive coefficient magnitude-pruning experiment.",
    )
    p.add_argument(
        "--sensitivity-json",
        type=Path,
        default=None,
        help=(
            "Optional selected_first_order_ranks.json. Keys in its rank_overrides "
            "dictionary are treated as sensitive matrices."
        ),
    )
    p.add_argument(
        "--sensitive-matrix",
        action="append",
        default=[],
        metavar="SUFFIX",
        help=(
            "Explicit sensitive matrix name or suffix. Repeat as needed. "
            "This is combined with matrices read from --sensitivity-json."
        ),
    )
    p.add_argument(
        "--sensitive-coeff-sparsity",
        type=float,
        default=0.20,
        help="Coefficient sparsity for sensitive matrices. Default: 0.20.",
    )
    p.add_argument(
        "--robust-coeff-sparsity",
        type=float,
        default=0.50,
        help="Coefficient sparsity for all other matrices. Default: 0.50.",
    )

    # Optional arbitrary per-matrix sparsity overrides.
    p.add_argument(
        "--coeff-sparsity-override",
        action="append",
        default=[],
        metavar="SUFFIX=SPARSITY",
        help=(
            "Override coefficient sparsity for a matching matrix suffix. "
            "Longest matching suffix wins. Example: "
            "--coeff-sparsity-override layer_00_fc1=0.10"
        ),
    )

    p.add_argument("--device", default=None)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-ppl-samples", type=int, default=0)
    add_dataset_arguments(p)
    p.add_argument("--coefficient-value-bits", type=int, default=8)
    p.add_argument("--output-csv", type=Path, required=True)

    # Keep the old direct-weight baseline optional because for an adaptive
    # coefficient policy there is no single matching direct-weight sparsity.
    p.add_argument(
        "--skip-direct-weight",
        action="store_true",
        help="Skip direct weight-magnitude pruning baselines.",
    )
    p.add_argument(
        "--loss-chunk-tokens",
        type=int,
        default=0,
        help="Chunked LM-head/cross-entropy evaluation; 0 keeps the single pass.",
    )
    p.add_argument(
        "--dense-ppl-cache",
        type=Path,
        default=None,
        help="JSON cache of dense perplexities shared with the inference stage.",
    )

    a = p.parse_args()

    values = list(a.sparsities) + [
        a.sensitive_coeff_sparsity,
        a.robust_coeff_sparsity,
    ]
    if any(not 0 <= s < 1 for s in values):
        p.error("all sparsities must be in [0,1)")

    a.coeff_sparsity_overrides = parse_sparsity_overrides(
        a.coeff_sparsity_override, p
    )

    if not a.sparsities and not a.adaptive_coeff_pruning:
        p.error(
            "Nothing to run. Supply --sparsities and/or --adaptive-coeff-pruning."
        )

    if a.sensitivity_json is not None and not a.sensitivity_json.exists():
        p.error(f"Missing --sensitivity-json: {a.sensitivity_json}")

    return a


def parse_sparsity_overrides(values, parser):
    overrides = {}
    for value in values:
        if "=" not in value:
            parser.error(
                f"Invalid --coeff-sparsity-override {value!r}; "
                "use SUFFIX=SPARSITY."
            )
        suffix, raw = (x.strip() for x in value.rsplit("=", 1))
        if not suffix:
            parser.error("Coefficient sparsity override suffix cannot be empty.")
        try:
            sparsity = float(raw)
        except ValueError:
            parser.error(
                f"Invalid sparsity in --coeff-sparsity-override {value!r}."
            )
        if not 0 <= sparsity < 1:
            parser.error(
                f"Override sparsity for {suffix!r} must be in [0,1)."
            )
        overrides[suffix] = sparsity
    return overrides


_ARTIFACT_INDEX = {}


def find_artifact(root, name):
    # Walk the artifact tree once per root instead of once per matrix and
    # policy; the selection rule for each name is unchanged.
    root_key = str(Path(root).resolve())
    if root_key not in _ARTIFACT_INDEX:
        index = {}
        for p in root.rglob("metadata.json"):
            if (p.parent / "tile_coefficients.npy").exists():
                index.setdefault(p.parent.name, []).append(p.parent)
        _ARTIFACT_INDEX[root_key] = index
    hits = list(_ARTIFACT_INDEX[root_key].get(name, []))
    if not hits:
        raise FileNotFoundError(f"No artifact for {name} under {root}")
    hits.sort(key=lambda p: (0 if "artifacts" in p.parts else 1, len(p.parts)))
    return hits[0]


def prune_exact(x, sparsity):
    flat = np.abs(x).reshape(-1)
    nzero = int(round(sparsity * flat.size))
    if nzero <= 0:
        return x.copy(), 0
    if nzero >= flat.size:
        return np.zeros_like(x), flat.size
    idx = np.argpartition(flat, nzero - 1)[:nzero]
    y = x.copy().reshape(-1)
    y[idx] = 0
    return y.reshape(x.shape), nzero


def reconstruct_coeff_pruned(adir, sparsity):
    m = json.loads((adir / "metadata.json").read_text())
    compact = np.load(adir / "structured_pruning.npy")
    bases = np.load(adir / "subspace_bases.npy").astype(np.float32)
    ids = _load_subspace_ids(adir, m)
    coeff = _load_coefficients(adir, m).astype(np.float32)
    coeff2, nzero = prune_exact(coeff, sparsity)

    rows, cols = int(m["rows"]), int(m["columns"])
    C = int(m["C"])
    tw = int(m.get("tile_width", 1))
    rb, cb = rows // C, cols // tw
    n = rb * cb
    dim = C * tw

    tiles = np.empty((n, dim), dtype=np.float16)
    rec = ids >= 0
    tiles[~rec] = compact
    tiles[rec] = np.einsum(
        "ncr,nr->nc",
        bases[ids[rec]],
        coeff2,
        optimize=True,
    ).astype(np.float16)

    W = (
        tiles.reshape(rb, cb, C, tw)
        .transpose(0, 2, 1, 3)
        .reshape(rows, cols)
    )

    return W.T.copy(), m, coeff.size - nzero, coeff.size


def set_direct_pruned(model, records, originals, sparsity):
    for r, w in zip(records, originals):
        wp, _ = prune_exact(w, sparsity)
        r["module"].weight.data.copy_(
            torch.from_numpy(wp).to(
                r["module"].weight.device,
                r["module"].weight.dtype,
            )
        )


def restore(model, records, originals):
    for r, w in zip(records, originals):
        r["module"].weight.data.copy_(
            torch.from_numpy(w).to(
                r["module"].weight.device,
                r["module"].weight.dtype,
            )
        )


def load_sensitive_names(path):
    if path is None:
        return set()

    data = json.loads(path.read_text())
    overrides = data.get("rank_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(
            f"{path}: expected a dictionary named 'rank_overrides'."
        )
    return set(overrides)


def suffix_matches(name, suffixes):
    return any(name == suffix or name.endswith(suffix) for suffix in suffixes)


def override_for(name, overrides):
    matches = [
        (suffix, sparsity)
        for suffix, sparsity in overrides.items()
        if name == suffix or name.endswith(suffix)
    ]
    if not matches:
        return None
    return max(matches, key=lambda x: len(x[0]))[1]


def adaptive_sparsity_for(name, sensitive_names, args):
    explicit = override_for(name, args.coeff_sparsity_overrides)
    if explicit is not None:
        return explicit, "override"

    if suffix_matches(name, sensitive_names):
        return args.sensitive_coeff_sparsity, "sensitive"

    return args.robust_coeff_sparsity, "robust"


def sparse_storage_bytes(nt, nz, coefficient_value_bits):
    # One bit per coefficient plus packed nonzero values.
    mask_bytes = math.ceil(nt / 8)
    value_bytes = math.ceil(nz * coefficient_value_bits / 8)
    return mask_bytes + value_bytes


def evaluate_coefficient_policy(
    model,
    records,
    originals,
    loader,
    dev,
    tok,
    artifact_root,
    policy_name,
    sparsity_fn,
    coefficient_value_bits,
    loss_chunk_tokens=0,
):
    restore(model, records, originals)

    nnzc = 0
    totc = 0
    sparse_bytes = 0
    per_matrix = []

    for r in records:
        name = r["artifact_name"]
        ad = find_artifact(artifact_root, name)
        sparsity, group = sparsity_fn(name)

        W, m, nz, nt = reconstruct_coeff_pruned(ad, sparsity)
        r["module"].weight.data.copy_(
            torch.from_numpy(W).to(dev, r["module"].weight.dtype)
        )

        nnzc += nz
        totc += nt
        sparse_bytes += sparse_storage_bytes(
            nt, nz, coefficient_value_bits
        )

        per_matrix.append(
            {
                "matrix": name,
                "rank": int(m["rank"]),
                "group": group,
                "requested_sparsity": sparsity,
                "nnz_fraction": nz / nt,
            }
        )

    t = time.perf_counter()
    ppl, tokens = perplexity(
        model, loader, dev, tok.pad_token_id, loss_chunk_tokens
    )

    effective_sparsity = 1.0 - nnzc / totc

    row = dict(
        model="",
        method=policy_name,
        sparsity=effective_sparsity,
        ppl=ppl,
        ppl_tokens=tokens,
        inference_seconds=time.perf_counter() - t,
        nnz_fraction=nnzc / totc,
        coefficient_storage_mib=sparse_bytes / 1024**2,
        sensitive_sparsity="",
        robust_sparsity="",
    )
    return row, per_matrix


def main():
    a = parse()
    dev = torch.device(
        a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    loader = test_loader(
        tok,
        a.seqlen,
        a.batch_size,
        a.max_ppl_samples,
        a.model,
        **resolve_dataset(a),
    )

    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=(
            torch.float16 if dev.type == "cuda" else torch.float32
        ),
        low_cpu_mem_usage=True,
    ).to(dev)

    records = decoder_linear_records(model)
    # Keep the pristine weights in the model's own dtype. Copying an fp16
    # weight as fp16 is lossless, and it halves host memory compared with
    # the previous float32 copies; restore() casts back to the same dtype.
    originals = [
        r["module"].weight.detach().cpu().numpy().copy()
        for r in records
    ]

    rows = []

    # Dense.
    cache_key = dense_ppl_cache_key(a, dev)
    cached = dense_ppl_cache_lookup(a.dense_ppl_cache, cache_key)
    if cached is not None:
        ppl = float(cached["ppl"])
        tokens = int(cached["ppl_tokens"])
        dense_seconds = float(cached["inference_seconds"])
        print(f"Reusing cached dense perplexity from {a.dense_ppl_cache}")
    else:
        t = time.perf_counter()
        ppl, tokens = perplexity(
            model,
            loader,
            dev,
            tok.pad_token_id,
            a.loss_chunk_tokens,
        )
        dense_seconds = time.perf_counter() - t
        dense_ppl_cache_store(
            a.dense_ppl_cache,
            cache_key,
            {"ppl": ppl, "ppl_tokens": tokens, "inference_seconds": dense_seconds},
        )
    rows.append(
        dict(
            model=a.model,
            method="dense",
            sparsity=0,
            ppl=ppl,
            ppl_tokens=tokens,
            inference_seconds=dense_seconds,
            nnz_fraction=1.0,
            coefficient_storage_mib="",
            sensitive_sparsity="",
            robust_sparsity="",
        )
    )

    # Existing uniform experiments.
    for s in a.sparsities:
        if not a.skip_direct_weight:
            restore(model, records, originals)
            set_direct_pruned(model, records, originals, s)

            t = time.perf_counter()
            ppl, tokens = perplexity(
                model, loader, dev, tok.pad_token_id, a.loss_chunk_tokens
            )
            nnz = sum(
                np.count_nonzero(
                    r["module"].weight.detach().cpu().numpy()
                )
                for r in records
            )
            total = sum(w.size for w in originals)

            rows.append(
                dict(
                    model=a.model,
                    method="magnitude_weight",
                    sparsity=s,
                    ppl=ppl,
                    ppl_tokens=tokens,
                    inference_seconds=time.perf_counter() - t,
                    nnz_fraction=nnz / total,
                    coefficient_storage_mib="",
                    sensitive_sparsity="",
                    robust_sparsity="",
                )
            )

        row, _ = evaluate_coefficient_policy(
            model=model,
            records=records,
            originals=originals,
            loader=loader,
            dev=dev,
            tok=tok,
            artifact_root=a.artifact_root,
            policy_name="arks_coefficient_magnitude_uniform",
            sparsity_fn=lambda name, s=s: (s, "uniform"),
            coefficient_value_bits=a.coefficient_value_bits,
            loss_chunk_tokens=a.loss_chunk_tokens,
        )
        row["model"] = a.model
        rows.append(row)

        if not a.skip_direct_weight:
            print(rows[-2])
        print(rows[-1])

    # New adaptive coefficient sparsity.
    if a.adaptive_coeff_pruning:
        sensitive_names = load_sensitive_names(
            a.sensitivity_json
        )
        sensitive_names.update(a.sensitive_matrix)

        if not sensitive_names:
            raise ValueError(
                "Adaptive coefficient pruning was requested, but no sensitive "
                "matrices were provided. Use --sensitivity-json and/or "
                "--sensitive-matrix."
            )

        print(
            f"Adaptive coefficient policy: "
            f"{len(sensitive_names)} sensitive selectors, "
            f"sensitive={a.sensitive_coeff_sparsity:.2f}, "
            f"robust={a.robust_coeff_sparsity:.2f}"
        )

        row, per_matrix = evaluate_coefficient_policy(
            model=model,
            records=records,
            originals=originals,
            loader=loader,
            dev=dev,
            tok=tok,
            artifact_root=a.artifact_root,
            policy_name="arks_coefficient_magnitude_adaptive",
            sparsity_fn=lambda name: adaptive_sparsity_for(
                name, sensitive_names, a
            ),
            coefficient_value_bits=a.coefficient_value_bits,
            loss_chunk_tokens=a.loss_chunk_tokens,
        )
        row["model"] = a.model
        row["sensitive_sparsity"] = a.sensitive_coeff_sparsity
        row["robust_sparsity"] = a.robust_coeff_sparsity
        rows.append(row)

        per_matrix_path = a.output_csv.with_name(
            a.output_csv.stem + "_adaptive_per_matrix.csv"
        )
        with per_matrix_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(per_matrix[0].keys()),
            )
            w.writeheader()
            w.writerows(per_matrix)

        print(rows[-1])
        print(f"Saved: {per_matrix_path}")

    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.output_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {a.output_csv}")


if __name__ == "__main__":
    main()
#source .venv/bin/activate
# python final/compare_magnitude_pruning.py \
#   --model facebook/opt-125m \
#   --artifact-root SeedLM/data/SeedLM_compressed/opt-125m/final_rank7/artifacts \
#   --adaptive-coeff-pruning \
#   --sensitivity-json SeedLM/data/SeedLM_compressed/opt-125m/shared_fixed_C8/rank_allocation/selected_first_order_ranks.json \
#   --sensitive-coeff-sparsity 0.25 \
#   --robust-coeff-sparsity 0.30 \
#   --coefficient-value-bits 8 \
#   --seqlen 2048 \
#   --max-ppl-samples 0 \
#   --device cuda \
#   --skip-direct-weight \
#   --output-csv SeedLM/data/SeedLM_compressed/opt-125m/final/results/rank7_adaptive_sparsity_0_37.csv