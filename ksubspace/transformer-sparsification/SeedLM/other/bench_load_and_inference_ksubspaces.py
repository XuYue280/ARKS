#!/usr/bin/env python3
"""Load real K-subspaces artifacts, reconstruct OPT FC weights, and measure PPL."""

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

from SeedLM.ksubspaces_artifacts import PeakRSS, load_weight


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
    )
    p.add_argument("--artifact-root", type=Path, default=None)
    p.add_argument("--original-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-ppl-samples", type=int, default=8)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--max-layers", type=int, default=None)
    return p.parse_args()


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


def static_parameter_bytes(model) -> int:
    total = 0
    for name, parameter in model.named_parameters():
        is_compressed_fc_weight = name.endswith("fc1.weight") or name.endswith(
            "fc2.weight"
        )
        if not is_compressed_fc_weight:
            total += parameter.numel() * parameter.element_size()
    return total


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

    results = []
    for rank in args.ranks:
        for coefficient_bits in args.coefficient_bits:
            config_dir = artifact_root / (
                f"sp_{args.sparsity:.2f}_K{args.k}_R{rank}"
                f"_B{coefficient_bits}"
            )
            if not config_dir.exists():
                legacy_dir = artifact_root / (
                    f"sp_{args.sparsity:.2f}_K{args.k}_R{rank}"
                )
                if coefficient_bits == 16 and legacy_dir.exists():
                    config_dir = legacy_dir
                else:
                    raise FileNotFoundError(
                        f"Missing {config_dir}; run compression for rank "
                        f"{rank}, coefficient_bits={coefficient_bits} first"
                    )
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            rank_started = time.perf_counter()
            with PeakRSS() as memory:
                model_load_started = time.perf_counter()
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=(
                        torch.float16 if device.type == "cuda" else torch.float32
                    ),
                    low_cpu_mem_usage=True,
                ).to(device)
                model_load_seconds = time.perf_counter() - model_load_started
                static_bytes = static_parameter_bytes(model)

                artifact_bytes = 0
                artifact_load_seconds = 0.0
                reconstruction_seconds = 0.0
                metric_seconds = 0.0
                errors = error_state()
                layers = list(enumerate(model.model.decoder.layers))
                if args.max_layers is not None:
                    layers = layers[: args.max_layers]

                for layer_index, layer in layers:
                    for fc_name in ("fc1", "fc2"):
                        layer_name = f"layer_{layer_index:02d}_{fc_name}"
                        reconstructed_in_out, timings = load_weight(
                            config_dir / layer_name
                        )
                        artifact_bytes += timings["artifact_bytes"]
                        artifact_load_seconds += timings["load_seconds"]
                        reconstruction_seconds += timings[
                            "reconstruction_seconds"
                        ]

                        metric_started = time.perf_counter()
                        original = np.load(
                            original_dir / f"{layer_name}_weight.npy"
                        )
                        reconstructed_out_in = reconstructed_in_out.T
                        add_error(errors, original, reconstructed_out_in)
                        metric_seconds += time.perf_counter() - metric_started

                        fc = getattr(layer, fc_name)
                        fc.weight.data = torch.from_numpy(
                            reconstructed_out_in
                        ).to(device=fc.weight.device, dtype=fc.weight.dtype)

                reconstruction_error = finish_error(errors)
                inference_seconds = 0.0
                ppl = None
                tokens = 0
                if loader is not None:
                    inference_started = time.perf_counter()
                    ppl, tokens = perplexity(
                        model, loader, device, tokenizer.pad_token_id
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
                result = {
                    "model": args.model,
                    "rank": rank,
                    "coefficient_bits": coefficient_bits,
                    "sparsity": args.sparsity,
                    "k": args.k,
                    "layers_evaluated": len(layers),
                    "ppl": ppl,
                    "ppl_tokens": tokens,
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
                    "total_seconds": time.perf_counter() - rank_started,
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

    result_dir = ROOT / "results" / tag / "ksubspaces"
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
