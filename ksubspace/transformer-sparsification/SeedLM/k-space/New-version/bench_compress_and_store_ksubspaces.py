#!/usr/bin/env python3
"""Compress OPT or Llama decoder weights into K-subspaces artifacts."""

from __future__ import annotations

import argparse
import csv
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
        "--coefficient-bits",
        nargs="+",
        type=int,
        choices=(4, 8, 16),
        default=[16],
        help="Stored coefficient precision. INT4 is packed two values per byte.",
    )
    p.add_argument("--C", type=int, default=8)
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
                    config_dir = (
                        output_root
                        / (
                            f"sp_{args.sparsity:.2f}_K{k}_R{rank}"
                            f"_B{coefficient_bits}_C{args.C}"
                            f"_W{args.tile_width}"
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
                        original = np.load(weight_path, mmap_mode="r")
                        input_dimension = original.shape[1]
                        output_dimension = original.shape[0]
                        if input_dimension % args.C:
                            raise ValueError(
                                f"{layer_name}: input dimension "
                                f"{input_dimension} is not divisible by "
                                f"C={args.C}"
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
                            original.size // (args.C * args.tile_width)
                            if args.selection_pool == 0
                            else args.selection_pool
                        )
                        result = compress_weight(
                            original,
                            config_dir / layer_name,
                            sparsity=args.sparsity,
                            k=k,
                            rank=rank,
                            C=args.C,
                            tile_width=args.tile_width,
                            sample_size=args.sample_size,
                            iterations=args.iterations,
                            batch_size=args.batch_size,
                            seed=args.seed,
                            selection_pool=selection_pool,
                            selection_seed=args.selection_seed,
                            coefficient_bits=coefficient_bits,
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
                        all_rows.append(row)
                        print(
                            f"{layer_name}: "
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
                    and row["rank"] == rank
                    and row["coefficient_bits"] == coefficient_bits
                ]
                configuration_summaries.append(
                    {
                        "k": k,
                        "rank": rank,
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
        "coefficient_bits": args.coefficient_bits,
        "C": args.C,
        "tile_width": args.tile_width,
        "tile_shape": [args.C, args.tile_width],
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

# source .venv/bin/activate

# python SeedLM/k-space/bench_compress_and_store_ksubspaces.py \
#   --model meta-llama/Llama-3.2-1B \
#   --input-dir SeedLM/model/Llama-3.2-1B \
#   --output-root SeedLM/data/SeedLM_compressed/Llama-3.2-1B/ksubspaces_K128_C1_W8_rank2_layers_00_07_15 \
#   --weight-scope all \
#   --layers 0 7 15 \
#   --k 128 \
#   --ranks 2 \
#   --coefficient-bits 8 \
#   --C 1 \
#   --tile-width 4 \
#   --sparsity 0.5 \
#   --sample-size 50000 \
#   --iterations 15 \
#   --batch-size 2048 \
#   --selection-pool 100 \
#   --selection-seed 42 \
#   --seed 0