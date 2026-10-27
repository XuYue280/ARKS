#!/usr/bin/env python3
"""Compress every OPT FC layer into actual K-subspaces artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ksubspaces_artifacts import PeakRSS, compress_weight


ROOT = Path(__file__).resolve().parent.parent


def model_tag(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--k", type=int, default=128)
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
        "--sample-size",
        type=int,
        default=0,
        help="0 uses every recovered tile for training.",
    )
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection-pool", type=int, default=100)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--max-layers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = model_tag(args.model)
    input_dir = args.input_dir or ROOT / "data" / "original_model" / tag
    output_root = args.output_root or (
        ROOT / "data" / "SeedLM_compressed" / tag / "ksubspaces"
    )
    weight_files = sorted(input_dir.glob("layer_*_fc[12]_weight.npy"))
    if args.max_layers is not None:
        weight_files = weight_files[: 2 * args.max_layers]
    if not weight_files:
        raise FileNotFoundError(f"No layer_*_fc[12]_weight.npy files in {input_dir}")

    all_rows: list[dict] = []
    started_all = time.perf_counter()
    with PeakRSS() as memory:
        for rank in args.ranks:
            for coefficient_bits in args.coefficient_bits:
                config_dir = (
                    output_root
                    / (
                        f"sp_{args.sparsity:.2f}_K{args.k}_R{rank}"
                        f"_B{coefficient_bits}"
                    )
                )
                print(
                    f"\nK-subspaces: rank={rank}, "
                    f"coefficient_bits={coefficient_bits}, "
                    f"layers={len(weight_files)}, output={config_dir}",
                    flush=True,
                )
                for weight_path in weight_files:
                    layer_name = weight_path.stem.removesuffix("_weight")
                    original = np.load(weight_path)
                    result = compress_weight(
                        original,
                        config_dir / layer_name,
                        sparsity=args.sparsity,
                        k=args.k,
                        rank=rank,
                        C=args.C,
                        sample_size=args.sample_size,
                        iterations=args.iterations,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        selection_pool=args.selection_pool,
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
    for rank in args.ranks:
        for coefficient_bits in args.coefficient_bits:
            selected = [
                row
                for row in all_rows
                if row["rank"] == rank
                and row["coefficient_bits"] == coefficient_bits
            ]
            configuration_summaries.append(
                {
                    "rank": rank,
                    "coefficient_bits": coefficient_bits,
                    "layers": len(selected),
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
        "k": args.k,
        "ranks": args.ranks,
        "coefficient_bits": args.coefficient_bits,
        "C": args.C,
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
