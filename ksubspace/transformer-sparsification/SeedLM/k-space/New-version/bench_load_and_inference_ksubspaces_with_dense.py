#!/usr/bin/env python3
"""Evaluate K-subspaces artifacts on selected OPT or Llama decoder weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ksubspaces_artifacts import PeakRSS, load_weight



ROOT = Path(__file__).resolve().parents[2]


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
    p.add_argument(
        "--C",
        type=int,
        default=8,
        help="Tile height. It must match the value used during compression.",
    )
    p.add_argument(
        "--tile-width",
        type=int,
        default=1,
        help=(
            "Number of adjacent matrix columns per tile. It must match the "
            "value used during compression."
        ),
    )
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument(
        "--coefficient-bits",
        nargs="+",
        type=int,
        choices=(4, 8, 16),
        default=[16],
    )
    p.add_argument("--artifact-root", type=Path, default=None)
    p.add_argument("--original-dir", type=Path, default=None)
    p.add_argument("--result-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-ppl-samples", type=int, default=8)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Exact decoder layer indices. Omit to evaluate every layer.",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        choices=(
            "mlp",
            "fc1",
            "fc2",
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "output_proj",
            "attention",
            "all",
        ),
        default=[
            "mlp",
            "q_proj",
            "k_proj",
            "v_proj",
            "output_proj",
            "attention",
            "all",
        ],
        help="Independent replacement scenarios for which PPL is reported.",
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
    args.targets = list(dict.fromkeys(args.targets))
    return args


def test_loader(tokenizer, seqlen: int, batch_size: int, maximum: int | None):
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    tokenized = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    chunks = tokenized.input_ids.numel() // seqlen
    input_ids = tokenized.input_ids[0, : chunks * seqlen].reshape(chunks, seqlen)
    attention = tokenized.attention_mask[0, : chunks * seqlen].reshape(
        chunks, seqlen
    )
    if maximum is not None and maximum > 0:
        input_ids = input_ids[:maximum]
        attention = attention[:maximum]

    class Tokens(Dataset):
        def __len__(self):
            return len(input_ids)

        def __getitem__(self, index):
            return {
                "input_ids": input_ids[index],
                "attention_mask": attention[index],
            }

    return DataLoader(Tokens(), batch_size=batch_size)


@torch.no_grad()
def perplexity(model, loader, device, pad_token_id):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="none", ignore_index=pad_token_id or -100
    )
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits[:, :-1, :]
        labels = batch["input_ids"][:, 1:]
        nll = loss_fn(logits.permute(0, 2, 1), labels)
        mask = (
            labels != pad_token_id
            if pad_token_id is not None
            else torch.ones_like(labels, dtype=torch.bool)
        )
        total_nll += float((nll * mask).sum())
        total_tokens += int(mask.sum())
    if not total_tokens:
        raise ValueError("No PPL tokens were evaluated")
    return math.exp(total_nll / total_tokens), total_tokens


def error_state():
    return {
        "error": 0.0,
        "original": 0.0,
        "error_nnz": 0.0,
        "count": 0,
        "count_nnz": 0,
    }


def add_error(state, original: np.ndarray, reconstructed: np.ndarray):
    a = original.astype(np.float32, copy=False)
    b = reconstructed.astype(np.float32, copy=False)
    difference = a.astype(np.float64) - b.astype(np.float64)
    nonzero = a != 0
    state["error"] += float(np.sum(difference * difference))
    state["original"] += float(np.sum(a.astype(np.float64) ** 2))
    state["error_nnz"] += float(np.sum(difference[nonzero] ** 2))
    state["count"] += a.size
    state["count_nnz"] += int(nonzero.sum())


def finish_error(state):
    return {
        "mse_all_weights": state["error"] / state["count"],
        "mse_nonzero_weights": state["error_nnz"] / state["count_nnz"],
        "relative_frobenius_error": math.sqrt(
            state["error"] / state["original"]
        ),
        "evaluated_weights": state["count"],
    }


def decoder_linear_records(model) -> list[dict]:
    """Return a common description of supported OPT or Llama linear layers."""
    records: list[dict] = []
    model_type = getattr(model.config, "model_type", "")

    if model_type == "opt" or hasattr(model.model, "decoder"):
        layers = model.model.decoder.layers
        for index, layer in enumerate(layers):
            modules = (
                ("q_proj", "attention", layer.self_attn.q_proj, "self_attn_q_proj"),
                ("k_proj", "attention", layer.self_attn.k_proj, "self_attn_k_proj"),
                ("v_proj", "attention", layer.self_attn.v_proj, "self_attn_v_proj"),
                (
                    "output_proj",
                    "attention",
                    layer.self_attn.out_proj,
                    "self_attn_out_proj",
                ),
                ("fc1", "mlp", layer.fc1, "fc1"),
                ("fc2", "mlp", layer.fc2, "fc2"),
            )
            for kind, family, module, file_suffix in modules:
                records.append(
                    {
                        "layer": index,
                        "kind": kind,
                        "family": family,
                        "module": module,
                        "artifact_name": f"layer_{index:02d}_{file_suffix}",
                    }
                )
        return records

    if model_type == "llama" or hasattr(model.model, "layers"):
        layers = model.model.layers
        for index, layer in enumerate(layers):
            modules = (
                ("q_proj", "attention", layer.self_attn.q_proj, "self_attn_q_proj"),
                ("k_proj", "attention", layer.self_attn.k_proj, "self_attn_k_proj"),
                ("v_proj", "attention", layer.self_attn.v_proj, "self_attn_v_proj"),
                (
                    "output_proj",
                    "attention",
                    layer.self_attn.o_proj,
                    "self_attn_o_proj",
                ),
                ("gate_proj", "mlp", layer.mlp.gate_proj, "mlp_gate_proj"),
                ("up_proj", "mlp", layer.mlp.up_proj, "mlp_up_proj"),
                ("down_proj", "mlp", layer.mlp.down_proj, "mlp_down_proj"),
            )
            for kind, family, module, file_suffix in modules:
                records.append(
                    {
                        "layer": index,
                        "kind": kind,
                        "family": family,
                        "module": module,
                        "artifact_name": f"layer_{index:02d}_{file_suffix}",
                    }
                )
        return records

    raise ValueError(
        f"Unsupported model_type={model_type!r}; expected OPT or Llama"
    )


def select_records(
    records: list[dict],
    target: str,
    layers: list[int] | None,
    max_layers: int | None,
) -> list[dict]:
    selected = []
    for record in records:
        layer_index = int(record["layer"])
        if layers is not None and layer_index not in layers:
            continue
        if max_layers is not None and layer_index >= max_layers:
            continue
        if target == "all":
            matches = True
        elif target in ("mlp", "attention"):
            matches = record["family"] == target
        else:
            matches = record["kind"] == target
        if matches:
            selected.append(record)
    return selected


def static_parameter_bytes(model, compressed_records: list[dict]) -> int:
    compressed_parameter_ids = {
        id(record["module"].weight) for record in compressed_records
    }
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if id(parameter) not in compressed_parameter_ids
    )


def validate_artifact_configuration(
    artifact_dir: Path,
    *,
    C: int,
    tile_width: int,
    k: int,
    rank: int,
    coefficient_bits: int,
) -> None:
    """Fail early when an artifact does not match the requested experiment."""
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing artifact metadata {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "C": C,
        "tile_width": tile_width,
        "k": k,
        "rank": rank,
        "coefficient_bits": coefficient_bits,
    }
    actual = {
        "C": int(metadata["C"]),
        "tile_width": int(metadata.get("tile_width", 1)),
        "k": int(metadata["k"]),
        "rank": int(metadata["rank"]),
        "coefficient_bits": int(metadata.get("coefficient_bits", 16)),
    }
    mismatches = [
        f"{key}: requested={expected[key]}, artifact={actual[key]}"
        for key in expected
        if expected[key] != actual[key]
    ]
    if mismatches:
        raise ValueError(
            f"Artifact configuration mismatch in {artifact_dir}: "
            + "; ".join(mismatches)
        )


def main() -> None:
    args = parse_args()
    tag = model_tag(args.model)
    artifact_root = args.artifact_root or (
        ROOT / "data" / "SeedLM_compressed" / tag / "ksubspaces"
    )
    original_dir = args.original_dir or ROOT / "data" / "original_model" / tag
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = None if args.skip_ppl else test_loader(
        tokenizer, args.seqlen, args.batch_size, args.max_ppl_samples
    )

    # Evaluate the unmodified model once using exactly the same loader and
    # precision used for every compressed scenario. Keeping this evaluation
    # outside the configuration and target loops avoids repeating it.
    dense_ppl = None
    dense_tokens = 0
    dense_inference_seconds = 0.0
    if loader is not None:
        print("\nEvaluating dense baseline...")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        dense_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.float32
            ),
            low_cpu_mem_usage=True,
        ).to(device)

        dense_started = time.perf_counter()
        dense_ppl, dense_tokens = perplexity(
            dense_model,
            loader,
            device,
            tokenizer.pad_token_id,
        )
        dense_inference_seconds = time.perf_counter() - dense_started

        print(
            json.dumps(
                {
                    "model": args.model,
                    "target": "dense",
                    "ppl": dense_ppl,
                    "ppl_tokens": dense_tokens,
                    "inference_seconds": dense_inference_seconds,
                },
                indent=2,
            )
        )

        del dense_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = []
    for k in args.k:
        for rank in args.ranks:
            for coefficient_bits in args.coefficient_bits:
                config_dir = artifact_root / (
                    f"sp_{args.sparsity:.2f}_K{k}_R{rank}"
                    f"_B{coefficient_bits}_C{args.C}"
                    f"_W{args.tile_width}"
                )
                if not config_dir.exists():
                    width_one_dir = artifact_root / (
                        f"sp_{args.sparsity:.2f}_K{k}_R{rank}"
                        f"_B{coefficient_bits}_C{args.C}"
                    )
                    legacy_dir = artifact_root / (
                        f"sp_{args.sparsity:.2f}_K{k}_R{rank}"
                    )
                    if args.tile_width == 1 and width_one_dir.exists():
                        config_dir = width_one_dir
                    elif (
                        args.tile_width == 1
                        and coefficient_bits == 16
                        and legacy_dir.exists()
                    ):
                        config_dir = legacy_dir
                    else:
                        raise FileNotFoundError(
                            f"Missing {config_dir}; run compression for K={k}, "
                            f"rank={rank}, coefficient_bits={coefficient_bits}, "
                            f"C={args.C}, tile_width={args.tile_width} first"
                        )

                for target in args.targets:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)

                    scenario_started = time.perf_counter()
                    with PeakRSS() as memory:
                        model_load_started = time.perf_counter()
                        model = AutoModelForCausalLM.from_pretrained(
                            args.model,
                            torch_dtype=(
                                torch.float16
                                if device.type == "cuda"
                                else torch.float32
                            ),
                            low_cpu_mem_usage=True,
                        ).to(device)
                        model_load_seconds = (
                            time.perf_counter() - model_load_started
                        )

                        records = decoder_linear_records(model)
                        selected = select_records(
                            records,
                            target,
                            args.layers,
                            args.max_layers,
                        )
                        if not selected:
                            print(
                                f"[skip] target={target} is unavailable for "
                                f"model_type={model.config.model_type}"
                            )
                            del model
                            continue

                        static_bytes = static_parameter_bytes(model, selected)
                        artifact_bytes = 0
                        artifact_load_seconds = 0.0
                        reconstruction_seconds = 0.0
                        metric_seconds = 0.0
                        errors = error_state()

                        for record in selected:
                            layer_name = record["artifact_name"]
                            artifact_dir = config_dir / layer_name
                            if not artifact_dir.exists():
                                raise FileNotFoundError(
                                    f"Missing artifact {artifact_dir}. Compress "
                                    f"with --weight-scope all and matching layers."
                                )
                            validate_artifact_configuration(
                                artifact_dir,
                                C=args.C,
                                tile_width=args.tile_width,
                                k=k,
                                rank=rank,
                                coefficient_bits=coefficient_bits,
                            )

                            reconstructed_in_out, timings = load_weight(
                                artifact_dir
                            )
                            artifact_bytes += timings["artifact_bytes"]
                            artifact_load_seconds += timings["load_seconds"]
                            reconstruction_seconds += timings[
                                "reconstruction_seconds"
                            ]

                            metric_started = time.perf_counter()
                            original = np.load(
                                original_dir / f"{layer_name}_weight.npy",
                                mmap_mode="r",
                            )
                            reconstructed_out_in = reconstructed_in_out.T
                            add_error(errors, original, reconstructed_out_in)
                            metric_seconds += (
                                time.perf_counter() - metric_started
                            )

                            module = record["module"]
                            module.weight.data = torch.from_numpy(
                                reconstructed_out_in
                            ).to(
                                device=module.weight.device,
                                dtype=module.weight.dtype,
                            )

                        reconstruction_error = finish_error(errors)
                        inference_seconds = 0.0
                        ppl = None
                        tokens = 0
                        if loader is not None:
                            inference_started = time.perf_counter()
                            ppl, tokens = perplexity(
                                model,
                                loader,
                                device,
                                tokenizer.pad_token_id,
                            )
                            inference_seconds = (
                                time.perf_counter() - inference_started
                            )

                        cuda_allocated = (
                            torch.cuda.max_memory_allocated(device) / 1024**2
                            if device.type == "cuda"
                            else 0.0
                        )
                        cuda_reserved = (
                            torch.cuda.max_memory_reserved(device) / 1024**2
                            if device.type == "cuda"
                            else 0.0
                        )
                        evaluated_layers = sorted(
                            {int(record["layer"]) for record in selected}
                        )
                        result = {
                            "model": args.model,
                            "target": target,
                            "matrix_types": ",".join(
                                sorted({record["kind"] for record in selected})
                            ),
                            "selected_layers": ",".join(
                                str(index) for index in evaluated_layers
                            ),
                            "rank": rank,
                            "coefficient_bits": coefficient_bits,
                            "sparsity": args.sparsity,
                            "k": k,
                            "C": args.C,
                            "tile_width": args.tile_width,
                            "tile_shape": f"{args.C}x{args.tile_width}",
                            "layers_evaluated": len(evaluated_layers),
                            "matrices_evaluated": len(selected),
                            "ppl": ppl,
                            "ppl_tokens": tokens,
                            "dense_ppl": dense_ppl,
                            "dense_ppl_tokens": dense_tokens,
                            "ppl_increase": (
                                ppl - dense_ppl
                                if ppl is not None and dense_ppl is not None
                                else None
                            ),
                            "ppl_increase_percent": (
                                100.0 * (ppl / dense_ppl - 1.0)
                                if ppl is not None and dense_ppl is not None
                                else None
                            ),
                            "dense_inference_seconds": dense_inference_seconds,
                            **reconstruction_error,
                            "artifact_storage_mb": artifact_bytes / 1024**2,
                            "static_parameter_storage_mb": static_bytes / 1024**2,
                            "total_model_storage_mb": (
                                artifact_bytes + static_bytes
                            )
                            / 1024**2,
                            "model_load_seconds": model_load_seconds,
                            "artifact_load_seconds": artifact_load_seconds,
                            "reconstruction_seconds": reconstruction_seconds,
                            "metric_seconds": metric_seconds,
                            "inference_seconds": inference_seconds,
                            "total_seconds": (
                                time.perf_counter() - scenario_started
                            ),
                            "peak_cpu_rss_mb": 0.0,
                            "peak_cpu_rss_increase_mb": 0.0,
                            "peak_cuda_allocated_mb": cuda_allocated,
                            "peak_cuda_reserved_mb": cuda_reserved,
                        }
                    result["peak_cpu_rss_mb"] = memory.peak_mb
                    result["peak_cpu_rss_increase_mb"] = memory.increase_mb
                    results.append(result)
                    print("\nK-subspaces inference result")
                    print(json.dumps(result, indent=2))
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    result_dir = args.result_dir or ROOT / "results" / tag / "ksubspaces"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = result_dir / f"inference_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    json_path = result_dir / f"inference_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

# python SeedLM/k-space/bench_load_and_inference_ksubspaces_with_dense.py \
#   --model meta-llama/Llama-3.2-1B \
#   --original-dir SeedLM/model/Llama-3.2-1B \
#   --artifact-root SeedLM/data/SeedLM_compressed/Llama-3.2-1B/ksubspaces_K128_C1_W8_rank2_layers_00_07_15 \
#   --result-dir results/Llama-3.2-1B/ksubspaces_K128_C1_W8_rank2_layers_00_07_15 \
#   --layers 0 7 15 \
#   --k 128 \
#   --ranks 2 \
#   --coefficient-bits 8 \
#   --C 1 \
#   --tile-width 4 \
#   --sparsity 0.5 \
#   --targets mlp gate_proj up_proj down_proj q_proj k_proj v_proj output_proj attention all \
#   --device cuda \
#   --seqlen 2048 \
#   --batch-size 1 \
#   --max-ppl-samples 8