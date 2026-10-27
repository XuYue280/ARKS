#!/usr/bin/env python3
"""Compress OPT or Llama decoder weights into K-subspaces artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ksubspaces_artifacts import PeakRSS, compress_weight


ROOT = Path(__file__).resolve().parents[2]

ATTENTION_SUFFIXES = (
    "self_attn_q_proj",
    "self_attn_k_proj",
    "self_attn_v_proj",
    "self_attn_out_proj",
    "self_attn_o_proj",
)
MLP_SUFFIXES = (
    "fc1",
    "fc2",
    "mlp_gate_proj",
    "mlp_up_proj",
    "mlp_down_proj",
)


def model_tag(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def parse_rank_overrides(
    values: list[str], tile_dimension: int, parser: argparse.ArgumentParser
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            parser.error(
                f"Invalid --rank-override {value!r}; use SUFFIX=RANK, "
                "for example down_proj=6."
            )
        suffix, raw_rank = (part.strip() for part in value.rsplit("=", 1))
        try:
            rank = int(raw_rank)
        except ValueError:
            parser.error(f"Invalid rank in --rank-override {value!r}.")
        if not suffix:
            parser.error("The suffix in --rank-override cannot be empty.")
        if rank < 1 or rank > tile_dimension:
            parser.error(
                f"Override rank for {suffix!r} must be between 1 and "
                f"C*tile_width={tile_dimension}."
            )
        overrides[suffix] = rank
    return overrides


def parse_tile_overrides(
    values: list[str], parser: argparse.ArgumentParser
) -> dict[str, int]:
    """Parse per-matrix tile-height overrides such as layer_00_fc1=8."""
    overrides: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            parser.error(
                f"Invalid --tile-override {value!r}; use SUFFIX=C, "
                "for example layer_00_fc1=8."
            )
        suffix, raw_c = (part.strip() for part in value.rsplit("=", 1))
        try:
            matrix_c = int(raw_c)
        except ValueError:
            parser.error(f"Invalid tile height in --tile-override {value!r}.")
        if not suffix:
            parser.error("The suffix in --tile-override cannot be empty.")
        if matrix_c < 1:
            parser.error(
                f"Tile height for {suffix!r} must be a positive integer."
            )
        overrides[suffix] = matrix_c
    return overrides


def effective_rank(
    layer_name: str, default_rank: int, overrides: dict[str, int]
) -> int:
    matches = [
        (suffix, rank)
        for suffix, rank in overrides.items()
        if layer_name.endswith(suffix)
    ]
    return max(matches, key=lambda item: len(item[0]))[1] if matches else default_rank


def effective_tile_height(
    layer_name: str, default_c: int, overrides: dict[str, int]
) -> int:
    matches = [
        (suffix, matrix_c)
        for suffix, matrix_c in overrides.items()
        if layer_name.endswith(suffix)
    ]
    return max(matches, key=lambda item: len(item[0]))[1] if matches else default_c


def rank_override_tag(overrides: dict[str, int]) -> str:
    if not overrides:
        return ""

    # Large adaptive allocations may contain one override per matrix. Use a
    # stable digest to avoid exceeding the filesystem filename-length limit.
    payload = json.dumps(
        sorted(overrides.items()),
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    if len(overrides) > 3:
        return f"_MIX-adaptive-{len(overrides)}-{digest}"

    parts = [
        f"{re.sub(r'[^A-Za-z0-9_-]+', '-', suffix)}-{rank}"
        for suffix, rank in sorted(overrides.items())
    ]
    return "_MIX-" + "-".join(parts)


def override_tag(overrides: dict[str, int], label: str) -> str:
    if not overrides:
        return ""
    payload = json.dumps(sorted(overrides.items()), separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"_{label}-adaptive-{len(overrides)}-{digest}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[128],
        help="One or more cluster counts, for example: --k 32 64 128 256",
    )
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument(
        "--rank-override",
        action="append",
        default=[],
        metavar="SUFFIX=RANK",
        help=(
            "Override rank for matching matrix-name suffixes. Repeat as needed; "
            "for example: --ranks 4 --rank-override down_proj=6"
        ),
    )
    p.add_argument(
        "--coefficient-bits",
        nargs="+",
        type=int,
        choices=(4, 8, 16),
        default=[16],
        help="Stored coefficient precision. INT4 is packed two values per byte.",
    )
    p.add_argument(
        "--coefficient-encoding",
        choices=("scalar", "vq"),
        default="scalar",
        help="Scalar quantization or one vector-code ID per coefficient vector.",
    )
    p.add_argument(
        "--coefficient-codebook-size",
        type=int,
        default=256,
        help="Number of coefficient vectors in each matrix-specific VQ codebook.",
    )
    p.add_argument(
        "--coefficient-vq-sample-size",
        type=int,
        default=50000,
        help="Coefficient vectors used to train VQ; 0 uses all vectors.",
    )
    p.add_argument(
        "--coefficient-vq-stages",
        type=int,
        default=1,
        help="Number of sequential residual VQ codebooks.",
    )
    p.add_argument(
        "--vq-stage-override",
        action="append",
        default=[],
        metavar="SUFFIX=STAGES",
        help="Override VQ stage count for matching matrix names.",
    )
    p.add_argument("--C", type=int, default=8)
    p.add_argument(
        "--tile-override",
        action="append",
        default=[],
        metavar="SUFFIX=C",
        help=(
            "Override tile height for matching matrix-name suffixes. Repeat as "
            "needed. Example: use --C 16 and --tile-override layer_00_fc1=8 "
            "to keep a sensitive matrix at 8x1 while robust matrices use 16x1."
        ),
    )
    p.add_argument(
        "--tile-width",
        type=int,
        default=1,
        help="Number of adjacent matrix columns in each C x tile_width tile.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="0 uses every recovered tile for training.",
    )
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--selection-pool",
        type=int,
        default=100,
        help=(
            "Number of C x tile_width tiles in each structured-selection pool. "
            "Use 0 to place all tiles of each weight matrix in one pool."
        ),
    )
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument(
        "--weight-scope",
        choices=("mlp", "attention", "all"),
        default="mlp",
        help="Decoder weight families to compress.",
    )
    p.add_argument(
        "--include-regex",
        default=None,
        help="Optional regex applied to the filename stem without _weight.",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Exact decoder layer indices. Omit to use every layer.",
    )
    p.add_argument("--max-layers", type=int, default=None)
    args = p.parse_args()
    if any(k < 2 for k in args.k):
        p.error("Every --k value must be at least 2.")
    if args.C < 1:
        p.error("--C must be positive.")
    if args.tile_width < 1:
        p.error("--tile-width must be positive.")
    tile_dimension = args.C * args.tile_width
    if any(rank < 1 or rank > tile_dimension for rank in args.ranks):
        p.error(
            "Every rank must be between 1 and "
            f"C*tile_width={tile_dimension}."
        )
    args.k = sorted(set(args.k))
    args.ranks = sorted(set(args.ranks))
    args.rank_overrides = parse_rank_overrides(
        args.rank_override, tile_dimension, p
    )
    args.tile_overrides = parse_tile_overrides(args.tile_override, p)
    args.vq_stage_overrides = parse_rank_overrides(
        args.vq_stage_override, 64, p
    )
    if args.coefficient_encoding == "vq":
        if args.coefficient_vq_stages < 1:
            p.error("--coefficient-vq-stages must be at least 1")
        if args.coefficient_codebook_size < 2:
            p.error("--coefficient-codebook-size must be at least 2")
        required_bits = int(np.ceil(np.log2(args.coefficient_codebook_size)))
        if any(bits != required_bits for bits in args.coefficient_bits):
            p.error(
                "For VQ, --coefficient-bits must equal ceil(log2(codebook size)); "
                f"size {args.coefficient_codebook_size} requires {required_bits} bits"
            )
    return args


def discover_weight_files(
    input_dir: Path,
    weight_scope: str,
    include_regex: str | None,
    layers: list[int] | None,
    max_layers: int | None,
) -> list[Path]:
    """Find supported OPT/Llama decoder matrices saved by store-llm-model.py."""
    allowed = {
        "mlp": MLP_SUFFIXES,
        "attention": ATTENTION_SUFFIXES,
        "all": MLP_SUFFIXES + ATTENTION_SUFFIXES,
    }[weight_scope]
    regex = re.compile(include_regex) if include_regex else None
    selected = []
    for path in sorted(input_dir.glob("layer_*_weight.npy")):
        match = re.fullmatch(r"layer_(\d+)_(.+)_weight\.npy", path.name)
        if not match:
            continue
        layer_index = int(match.group(1))
        matrix_type = match.group(2)
        if not any(matrix_type.endswith(suffix) for suffix in allowed):
            continue
        if layers is not None and layer_index not in layers:
            continue
        if max_layers is not None and layer_index >= max_layers:
            continue
        stem = path.name.removesuffix("_weight.npy")
        if regex and not regex.search(stem):
            continue
        selected.append(path)
    return selected


def main() -> None:
    args = parse_args()
    tag = model_tag(args.model)
    input_dir = args.input_dir or ROOT / "data" / "original_model" / tag
    output_root = args.output_root or (
        ROOT / "data" / "SeedLM_compressed" / tag / "ksubspaces"
    )
    weight_files = discover_weight_files(
        input_dir,
        args.weight_scope,
        args.include_regex,
        args.layers,
        args.max_layers,
    )
    if not weight_files:
        raise FileNotFoundError(
            f"No supported {args.weight_scope} decoder weights in {input_dir}"
        )
    print(
        f"Found {len(weight_files)} {args.weight_scope} matrices in {input_dir}"
    )

    all_rows: list[dict] = []
    started_all = time.perf_counter()
    with PeakRSS() as memory:
        for k in args.k:
            for rank in args.ranks:
                for coefficient_bits in args.coefficient_bits:
                    rank_tag = rank_override_tag(args.rank_overrides)
                    tile_tag = override_tag(args.tile_overrides, "TILE")
                    stage_tag = override_tag(args.vq_stage_overrides, "VQSTAGE")
                    encoding_tag = (
                        f"_CVQ{args.coefficient_vq_stages}x"
                        f"{args.coefficient_codebook_size}{stage_tag}"
                        if args.coefficient_encoding == "vq"
                        else ""
                    )
                    config_dir = (
                        output_root
                        / (
                            f"sp_{args.sparsity:.2f}_K{k}_R{rank}{rank_tag}"
                            f"_B{coefficient_bits}_C{args.C}{tile_tag}"
                            f"_W{args.tile_width}"
                            f"{encoding_tag}"
                        )
                    )
                    print(
                        f"\nK-subspaces: K={k}, rank={rank}, "
                        f"coefficient_bits={coefficient_bits}, "
                        f"matrices={len(weight_files)}, output={config_dir}",
                        flush=True,
                    )
                    for weight_path in weight_files:
                        layer_name = weight_path.stem.removesuffix("_weight")
                        matrix_rank = effective_rank(
                            layer_name, rank, args.rank_overrides
                        )
                        matrix_c = effective_tile_height(
                            layer_name, args.C, args.tile_overrides
                        )
                        matrix_vq_stages = effective_rank(
                            layer_name,
                            args.coefficient_vq_stages,
                            args.vq_stage_overrides,
                        )
                        original = np.load(weight_path, mmap_mode="r")
                        input_dimension = original.shape[1]
                        output_dimension = original.shape[0]
                        matrix_tile_dimension = matrix_c * args.tile_width
                        if matrix_rank > matrix_tile_dimension:
                            raise ValueError(
                                f"{layer_name}: rank={matrix_rank} exceeds "
                                f"C*tile_width={matrix_tile_dimension} for "
                                f"the effective tile {matrix_c}x{args.tile_width}"
                            )
                        if input_dimension % matrix_c:
                            raise ValueError(
                                f"{layer_name}: input dimension "
                                f"{input_dimension} is not divisible by "
                                f"effective C={matrix_c}"
                            )
                        if output_dimension % args.tile_width:
                            raise ValueError(
                                f"{layer_name}: output dimension "
                                f"{output_dimension} is not divisible by "
                                f"tile_width={args.tile_width}"
                            )
                        if args.selection_pool < 0:
                            raise ValueError(
                                "--selection-pool must be 0 or a positive integer"
                            )
                        selection_pool = (
                            original.size // (matrix_c * args.tile_width)
                            if args.selection_pool == 0
                            else args.selection_pool
                        )
                        result = compress_weight(
                            original,
                            config_dir / layer_name,
                            sparsity=args.sparsity,
                            k=k,
                            rank=matrix_rank,
                            C=matrix_c,
                            tile_width=args.tile_width,
                            sample_size=args.sample_size,
                            iterations=args.iterations,
                            batch_size=args.batch_size,
                            seed=args.seed,
                            selection_pool=selection_pool,
                            selection_seed=args.selection_seed,
                            coefficient_bits=coefficient_bits,
                            coefficient_encoding=args.coefficient_encoding,
                            coefficient_codebook_size=args.coefficient_codebook_size,
                            coefficient_vq_sample_size=args.coefficient_vq_sample_size,
                            coefficient_vq_stages=matrix_vq_stages,
                        )
                        row = {
                            "model": args.model,
                            "layer": layer_name,
                            **{
                                key: value
                                for key, value in result.items()
                                if key != "artifact_files"
                            },
                        }
                        row["default_rank"] = rank
                        row["rank_overrides"] = json.dumps(
                            args.rank_overrides, sort_keys=True
                        )
                        row["default_C"] = args.C
                        row["tile_overrides"] = json.dumps(
                            args.tile_overrides, sort_keys=True
                        )
                        row["default_vq_stages"] = args.coefficient_vq_stages
                        row["vq_stage_overrides"] = json.dumps(
                            args.vq_stage_overrides, sort_keys=True
                        )
                        all_rows.append(row)
                        print(
                            f"{layer_name}: "
                            f"rank={matrix_rank}, "
                            f"tile={matrix_c}x{args.tile_width}, "
                            f"vq_stages={matrix_vq_stages}, "
                            f"pre_quant_MSE={result['recovered_tile_mse']:.8e}, "
                            f"storage={result['artifact_bytes']/1024**2:.4f} MiB, "
                            f"time={result['total_compression_seconds']:.2f}s",
                            flush=True,
                        )

    elapsed = time.perf_counter() - started_all
    summary_dir = output_root / "benchmark_results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = summary_dir / f"compression_{timestamp}.csv"
    fieldnames = list(
        dict.fromkeys(
            key
            for row in all_rows
            for key in row
        )
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    # with csv_path.open("w", newline="", encoding="utf-8") as handle:
    #     writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
    #     writer.writeheader()
    #     writer.writerows(all_rows)

    configuration_summaries = []
    for k in args.k:
        for rank in args.ranks:
            for coefficient_bits in args.coefficient_bits:
                selected = [
                    row
                    for row in all_rows
                    if row["k"] == k
                    and row["default_rank"] == rank
                    and row["coefficient_bits"] == coefficient_bits
                ]
                configuration_summaries.append(
                    {
                        "k": k,
                        "rank": rank,
                        "rank_overrides": args.rank_overrides,
                        "coefficient_bits": coefficient_bits,
                        "matrices": len(selected),
                        "artifact_mb": sum(
                            row["artifact_bytes"] for row in selected
                        )
                        / 1024**2,
                        "compression_seconds": sum(
                            row["total_compression_seconds"] for row in selected
                        ),
                        "mean_recovered_tile_mse_before_coefficient_quantization": (
                            float(
                                np.mean(
                                    [
                                        row["recovered_tile_mse"]
                                        for row in selected
                                    ]
                                )
                            )
                        ),
                    }
                )
    summary = {
        "model": args.model,
        "sparsity": args.sparsity,
        "k_values": args.k,
        "weight_scope": args.weight_scope,
        "ranks": args.ranks,
        "rank_overrides": args.rank_overrides,
        "tile_overrides": args.tile_overrides,
        "coefficient_bits": args.coefficient_bits,
        "coefficient_encoding": args.coefficient_encoding,
        "coefficient_codebook_size": args.coefficient_codebook_size,
        "coefficient_vq_sample_size": args.coefficient_vq_sample_size,
        "coefficient_vq_stages": args.coefficient_vq_stages,
        "vq_stage_overrides": args.vq_stage_overrides,
        "C": args.C,
        "tile_width": args.tile_width,
        "tile_shape": [args.C, args.tile_width],
        "effective_tile_heights": sorted(
            {args.C, *args.tile_overrides.values()}
        ),
        "selection_pool_requested": args.selection_pool,
        "selection_pool_mode": (
            "all_tiles_per_weight" if args.selection_pool == 0 else "fixed_size"
        ),
        "sample_size": args.sample_size,
        "all_tiles_used_for_training": args.sample_size <= 0,
        "wallclock_seconds": elapsed,
        "peak_cpu_rss_mb": memory.peak_mb,
        "peak_cpu_rss_increase_mb": memory.increase_mb,
        "configuration_summaries": configuration_summaries,
        "detail_csv": str(csv_path),
    }
    json_path = summary_dir / f"compression_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nCompression summary")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()


# python SeedLM/k-space/different-rank/bench_compress_and_store_ksubspaces.py \
#   --model meta-llama/Llama-3.2-1B \
#   --input-dir SeedLM/model/Llama-3.2-1B \
#   --output-root SeedLM/data/SeedLM_compressed/Llama-3.2-1B/first_order_profile_K128_sp100 \
#   --weight-scope all \
#   --k 256 \
#   --ranks 6 \
#   --coefficient-bits 8 \
#   --C 8 \
#   --tile-width 1 \
#   --sparsity 1.0 \
#   --sample-size 50000 \
#   --iterations 15 \
#   --batch-size 2048 \
#   --selection-pool 100 \
#   --selection-seed 42 \
#   --seed 0
##################################
# python SeedLM/k-space/different-rank/first_order_rank_allocator.py \
#   --model meta-llama/Llama-3.2-1B \
#   --max-rank-artifact-dir SeedLM/data/SeedLM_compressed/Llama-3.2-1B/first_order_profile_K128_sp100/sp_1.00_K256_R6_B8_C8_W1 \
#   --base-rank 4 \
#   --max-rank 6 \
#   --extra-budget-mb 64.13 \
#   --calibration-samples 4 \
#   --seqlen 256 \
#   --weight-scope all \
#   --output-dir results/Llama-3.2-1B/first_order_full_K256_sp100 \
#   --device cuda
# source .venv/bin/activate
#python SeedLM/k-space/different-rank/bench_compress_and_store_ksubspaces.py \
#   --model facebook/opt-125m \
#   --input-dir SeedLM/model/opt-125m \
#   --output-root SeedLM/data/SeedLM_compressed/opt-125m/ksubspaces_K512_C8_W1_mixed_r4_vr6_sp100_512 \
#   --weight-scope all \
#   --k 512 \
#   --ranks 4 \
#   --rank-override v_proj=6 \
#   --coefficient-bits 8 \
#   --C 8 \
#   --tile-width 1 \
#   --sparsity 1.0 \
#   --sample-size 50000 \
#   --iterations 15 \
#   --batch-size 2048 \
#   --selection-pool 100 \
#   --selection-seed 42 \
#   --seed 0

# python SeedLM/k-space/different-rank/bench_compress_and_store_ksubspaces.py \
#   --model meta-llama/Llama-3.2-1B \
#   --input-dir SeedLM/model/Llama-3.2-1B \
#   --output-root SeedLM/data/SeedLM_compressed/Llama-3.2-1B/ksubspaces_K128_C8_W1_mixed_r4_downr6 \
#   --weight-scope all \
#   --k 128 \
#   --ranks 4 \
#   --rank-override down_proj=6 \
#   --coefficient-bits 8 \
#   --C 8 \
#   --tile-width 1 \
#   --sparsity 0.5 \
#   --sample-size 50000 \
#   --iterations 15 \
#   --batch-size 2048 \
#   --selection-pool 100 \
#   --selection-seed 42 \
#   --seed 0
