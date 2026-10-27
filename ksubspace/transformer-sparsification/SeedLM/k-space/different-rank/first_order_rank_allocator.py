#!/usr/bin/env python3
"""Allocate K-subspaces ranks with one max-rank fit and calibration gradients."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ksubspaces_artifacts import _load_coefficients


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--max-rank-artifact-dir", type=Path, required=True)
    p.add_argument("--base-rank", type=int, default=4)
    p.add_argument("--max-rank", type=int, default=6)
    p.add_argument("--extra-budget-mb", type=float, required=True)
    p.add_argument("--calibration-dataset", default="Salesforce/wikitext")
    p.add_argument("--calibration-config", default="wikitext-2-raw-v1")
    p.add_argument("--calibration-split", default="train")
    p.add_argument("--calibration-samples", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--weight-scope", choices=("mlp", "attention", "all"), default="all")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.base_rank < 1 or args.max_rank <= args.base_rank:
        p.error("Require 1 <= base-rank < max-rank")
    if args.extra_budget_mb < 0:
        p.error("--extra-budget-mb cannot be negative")
    if args.calibration_samples < 1 or args.seqlen < 2:
        p.error("Calibration samples must be positive and seqlen at least 2")
    return args


def records_for_model(model) -> list[dict]:
    records = []
    model_type = getattr(model.config, "model_type", "")
    if model_type == "opt" or hasattr(model.model, "decoder"):
        layers = model.model.decoder.layers
        for i, layer in enumerate(layers):
            modules = (
                ("q_proj", "attention", layer.self_attn.q_proj, "self_attn_q_proj"),
                ("k_proj", "attention", layer.self_attn.k_proj, "self_attn_k_proj"),
                ("v_proj", "attention", layer.self_attn.v_proj, "self_attn_v_proj"),
                ("output_proj", "attention", layer.self_attn.out_proj, "self_attn_out_proj"),
                ("fc1", "mlp", layer.fc1, "fc1"),
                ("fc2", "mlp", layer.fc2, "fc2"),
            )
            for kind, family, module, suffix in modules:
                records.append({"layer_index": i, "kind": kind, "family": family,
                                "module": module, "artifact_name": f"layer_{i:02d}_{suffix}"})
        return records
    if model_type == "llama" or hasattr(model.model, "layers"):
        layers = model.model.layers
        for i, layer in enumerate(layers):
            modules = (
                ("q_proj", "attention", layer.self_attn.q_proj, "self_attn_q_proj"),
                ("k_proj", "attention", layer.self_attn.k_proj, "self_attn_k_proj"),
                ("v_proj", "attention", layer.self_attn.v_proj, "self_attn_v_proj"),
                ("output_proj", "attention", layer.self_attn.o_proj, "self_attn_o_proj"),
                ("gate_proj", "mlp", layer.mlp.gate_proj, "mlp_gate_proj"),
                ("up_proj", "mlp", layer.mlp.up_proj, "mlp_up_proj"),
                ("down_proj", "mlp", layer.mlp.down_proj, "mlp_down_proj"),
            )
            for kind, family, module, suffix in modules:
                records.append({"layer_index": i, "kind": kind, "family": family,
                                "module": module, "artifact_name": f"layer_{i:02d}_{suffix}"})
        return records
    raise ValueError(f"Unsupported model_type={model_type!r}")


def select_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    return [r for r in records
            if (args.max_layers is None or r["layer_index"] < args.max_layers)
            and (args.weight_scope == "all" or r["family"] == args.weight_scope)]


def calibration_tensor(tokenizer, args: argparse.Namespace) -> torch.Tensor:
    ds = load_dataset(args.calibration_dataset, args.calibration_config,
                      split=args.calibration_split)
    tokens = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    needed = args.calibration_samples * args.seqlen
    if tokens.numel() < needed:
        raise ValueError(f"Need {needed} tokens, found {tokens.numel()}")
    return tokens[:needed].reshape(args.calibration_samples, args.seqlen)


def reconstruct_at_rank(artifact_dir: Path, target_rank: int) -> tuple[np.ndarray, dict]:
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    stored_rank = int(metadata["rank"])
    if target_rank < 1 or target_rank > stored_rank:
        raise ValueError(f"{artifact_dir}: target rank {target_rank}, stored rank {stored_rank}")
    compact = np.load(artifact_dir / "structured_pruning.npy")
    bases = np.load(artifact_dir / "subspace_bases.npy").astype(np.float32)[:, :, :target_rank]
    ids = np.load(artifact_dir / "nearest_subspace_ids.npy").astype(np.int32)
    coefficients = _load_coefficients(artifact_dir, metadata)[:, :target_rank]
    rows, columns = int(metadata["rows"]), int(metadata["columns"])
    C = int(metadata["C"])
    tile_width = int(metadata.get("tile_width", 1))
    row_blocks, column_blocks = rows // C, columns // tile_width
    n_tiles = row_blocks * column_blocks
    tiles = np.empty((n_tiles, C * tile_width), dtype=np.float16)
    recovered = ids >= 0
    tiles[~recovered] = compact
    selected_bases = bases[ids[recovered]]
    tiles[recovered] = np.einsum("ncr,nr->nc", selected_bases, coefficients,
                                  optimize=True).astype(np.float16)
    W_in_out = (tiles.reshape(row_blocks, column_blocks, C, tile_width)
                .transpose(0, 2, 1, 3).reshape(rows, columns))
    return W_in_out, metadata


def estimated_extra_bytes(metadata: dict, base_rank: int, max_rank: int) -> int:
    dr = max_rank - base_rank
    n = int(metadata["n_recovered_tiles"])
    k = int(metadata["k"])
    d = int(metadata["tile_dimension"])
    bits = int(metadata["coefficient_bits"])
    coefficient_bytes = int(np.ceil(n * dr * bits / 8))
    basis_bytes = k * d * dr * 2  # FP16 bases
    scale_bytes = dr * 4 if bits in (4, 8) else 0
    return coefficient_bytes + basis_bytes + scale_bytes


def gradient_calibration(model, records: list[dict], tokenizer, args, device) -> tuple[float, float]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for record in records:
        record["module"].weight.requires_grad_(True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    input_ids = calibration_tensor(tokenizer, args)
    started = time.perf_counter()
    total_loss = 0.0
    model.zero_grad(set_to_none=True)
    for sample in input_ids:
        sample = sample.unsqueeze(0).to(device)
        output = model(input_ids=sample, labels=sample, use_cache=False)
        loss = output.loss / args.calibration_samples
        total_loss += float(output.loss.detach())
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return total_loss / args.calibration_samples, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    total_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    records = select_records(records_for_model(model), args)
    if not records:
        raise ValueError("No matrices selected")
    print(f"Calibration backward for {len(records)} matrices...", flush=True)
    calibration_loss, backward_seconds = gradient_calibration(
        model, records, tokenizer, args, device
    )
    print(f"Calibration loss={calibration_loss:.6f}, time={backward_seconds:.2f}s")

    rows = []
    for index, record in enumerate(records, 1):
        started = time.perf_counter()
        name = record["artifact_name"]
        artifact_dir = args.max_rank_artifact_dir / name
        W4_in_out, metadata = reconstruct_at_rank(artifact_dir, args.base_rank)
        W6_in_out, _ = reconstruct_at_rank(artifact_dir, args.max_rank)
        gradient = record["module"].weight.grad
        if gradient is None:
            raise RuntimeError(f"Missing gradient for {name}")
        delta = torch.from_numpy(np.ascontiguousarray(W4_in_out.T - W6_in_out.T))
        signed = 0.0
        absolute = 0.0
        chunk = 512
        gradient_cpu = gradient.detach().float().cpu()
        for start in range(0, gradient_cpu.shape[0], chunk):
            product = gradient_cpu[start:start + chunk] * delta[start:start + chunk].float()
            signed += float(product.sum())
            absolute += float(product.abs().sum())
        extra_bytes = estimated_extra_bytes(metadata, args.base_rank, args.max_rank)
        extra_mib = extra_bytes / 1024**2
        rows.append({
            "artifact_name": name, "layer_index": record["layer_index"],
            "matrix_kind": record["kind"], "family": record["family"],
            "base_rank": args.base_rank, "candidate_rank": args.max_rank,
            "signed_first_order_gain": signed,
            "absolute_first_order_importance": absolute,
            "extra_bytes": extra_bytes, "extra_mib": extra_mib,
            "importance_per_mib": absolute / max(extra_mib, 1e-30),
            "scoring_seconds": time.perf_counter() - started,
            "selected": False,
        })
        print(f"[{index}/{len(records)}] {name}: importance/MiB="
              f"{rows[-1]['importance_per_mib']:.6e}", flush=True)
        del W4_in_out, W6_in_out, delta, gradient_cpu

    budget = int(args.extra_budget_mb * 1024**2)
    used = 0
    for row in sorted(rows, key=lambda x: x["importance_per_mib"], reverse=True):
        if used + row["extra_bytes"] <= budget:
            row["selected"] = True
            used += row["extra_bytes"]
    ranked = sorted(rows, key=lambda x: x["importance_per_mib"], reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "first_order_rank_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader(); writer.writerows(ranked)
    overrides = {r["artifact_name"]: args.max_rank for r in rows if r["selected"]}
    summary = {
        "model": args.model, "calibration_loss": calibration_loss,
        "calibration_backward_seconds": backward_seconds,
        "base_rank": args.base_rank, "max_rank": args.max_rank,
        "requested_extra_budget_mb": args.extra_budget_mb,
        "used_extra_budget_mb": used / 1024**2,
        "selected_matrix_count": len(overrides), "rank_overrides": overrides,
        "total_wallclock_seconds": time.perf_counter() - total_started,
    }
    json_path = args.output_dir / "selected_first_order_ranks.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    command_path = args.output_dir / "rank_override_arguments.txt"
    command_path.write_text(" ".join(
        f"--rank-override {name}={args.max_rank}" for name in overrides
    ) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {csv_path}\nSaved: {json_path}\nSaved: {command_path}")


if __name__ == "__main__":
    main()
