#!/usr/bin/env python3
"""Evaluate K-subspaces artifacts on selected OPT or Llama decoder weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
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

    # Keep this logic identical to the compression script so both stages
    # resolve the same mixed-rank artifact directory.
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
    p.add_argument(
        "--tile-override",
        action="append",
        default=[],
        metavar="SUFFIX=C",
        help=(
            "Override tile height for matching matrix-name suffixes. These "
            "options must exactly match compression."
        ),
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
    )
    p.add_argument(
        "--coefficient-encoding",
        choices=("scalar", "vq"),
        default="scalar",
    )
    p.add_argument("--coefficient-codebook-size", type=int, default=256)
    p.add_argument("--coefficient-vq-stages", type=int, default=1)
    p.add_argument(
        "--vq-stage-override",
        action="append",
        default=[],
        metavar="SUFFIX=STAGES",
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
        required_bits = int(math.ceil(math.log2(args.coefficient_codebook_size)))
        if any(bits != required_bits for bits in args.coefficient_bits):
            p.error(
                "For VQ, coefficient bits must equal ceil(log2(codebook size)); "
                f"size {args.coefficient_codebook_size} requires {required_bits} bits"
            )
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
    coefficient_encoding: str,
    coefficient_codebook_size: int,
    coefficient_vq_stages: int,
) -> dict:
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
        "coefficient_encoding": coefficient_encoding,
    }
    actual = {
        "C": int(metadata["C"]),
        "tile_width": int(metadata.get("tile_width", 1)),
        "k": int(metadata["k"]),
        "rank": int(metadata["rank"]),
        "coefficient_bits": int(metadata.get("coefficient_bits", 16)),
        "coefficient_encoding": metadata.get("coefficient_encoding", "scalar"),
    }
    if coefficient_encoding == "vq":
        expected["coefficient_codebook_size"] = coefficient_codebook_size
        expected["coefficient_vq_stages"] = coefficient_vq_stages
        actual["coefficient_codebook_size"] = int(
            metadata.get("coefficient_codebook_size", -1)
        )
        actual["coefficient_vq_stages"] = int(
            metadata.get("coefficient_vq_stages", 1)
        )
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
    return metadata


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
                rank_tag = rank_override_tag(args.rank_overrides)
                tile_tag = override_tag(args.tile_overrides, "TILE")
                stage_tag = override_tag(args.vq_stage_overrides, "VQSTAGE")
                encoding_tag = (
                    f"_CVQ{args.coefficient_vq_stages}x"
                    f"{args.coefficient_codebook_size}{stage_tag}"
                    if args.coefficient_encoding == "vq"
                    else ""
                )
                config_dir = artifact_root / (
                    f"sp_{args.sparsity:.2f}_K{k}_R{rank}{rank_tag}"
                    f"_B{coefficient_bits}_C{args.C}{tile_tag}"
                    f"_W{args.tile_width}"
                    f"{encoding_tag}"
                )
                if (
                    not config_dir.exists()
                    and not args.rank_overrides
                    and not args.tile_overrides
                ):
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
                elif not config_dir.exists():
                    raise FileNotFoundError(
                        f"Missing adaptive artifacts at {config_dir}; run "
                        "compression with the same rank, tile, and VQ "
                        "override options first."
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
                        calibrated_matrix_count = 0
                        calibration_markers: set[str] = set()
                        errors = error_state()

                        for record in selected:
                            layer_name = record["artifact_name"]
                            matrix_rank = effective_rank(
                                layer_name, rank, args.rank_overrides
                            )
                            matrix_c = effective_tile_height(
                                layer_name, args.C, args.tile_overrides
                            )
                            if matrix_rank > matrix_c * args.tile_width:
                                raise ValueError(
                                    f"{layer_name}: rank={matrix_rank} exceeds "
                                    f"the effective tile dimension "
                                    f"{matrix_c * args.tile_width}"
                                )
                            matrix_vq_stages = effective_rank(
                                layer_name,
                                args.coefficient_vq_stages,
                                args.vq_stage_overrides,
                            )
                            artifact_dir = config_dir / layer_name
                            if not artifact_dir.exists():
                                raise FileNotFoundError(
                                    f"Missing artifact {artifact_dir}. Compress "
                                    f"with --weight-scope all and matching layers."
                                )
                            artifact_metadata = validate_artifact_configuration(
                                artifact_dir,
                                C=matrix_c,
                                tile_width=args.tile_width,
                                k=k,
                                rank=matrix_rank,
                                coefficient_bits=coefficient_bits,
                                coefficient_encoding=args.coefficient_encoding,
                                coefficient_codebook_size=args.coefficient_codebook_size,
                                coefficient_vq_stages=matrix_vq_stages,
                            )
                            calibration_marker = artifact_metadata.get(
                                "coefficient_codebook_calibration"
                            )
                            if calibration_marker:
                                calibrated_matrix_count += 1
                                calibration_markers.add(str(calibration_marker))

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
                        compressed_parameter_count = sum(
                            int(record["module"].weight.numel())
                            for record in selected
                        )
                        total_parameter_count = sum(
                            int(parameter.numel())
                            for parameter in model.parameters()
                        )
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
                            "codebook_calibration": (
                                next(iter(calibration_markers))
                                if (
                                    calibrated_matrix_count == len(selected)
                                    and len(calibration_markers) == 1
                                )
                                else (
                                    "none"
                                    if calibrated_matrix_count == 0
                                    else "mixed"
                                )
                            ),
                            "matrix_types": ",".join(
                                sorted({record["kind"] for record in selected})
                            ),
                            "selected_layers": ",".join(
                                str(index) for index in evaluated_layers
                            ),
                            "rank": rank,
                            "rank_overrides": json.dumps(
                                args.rank_overrides, sort_keys=True
                            ),
                            "tile_overrides": json.dumps(
                                args.tile_overrides, sort_keys=True
                            ),
                            "effective_ranks": ",".join(
                                str(value)
                                for value in sorted(
                                    {
                                        effective_rank(
                                            record["artifact_name"],
                                            rank,
                                            args.rank_overrides,
                                        )
                                        for record in selected
                                    }
                                )
                            ),
                            "coefficient_bits": coefficient_bits,
                            "coefficient_encoding": args.coefficient_encoding,
                            "coefficient_codebook_size": (
                                args.coefficient_codebook_size
                                if args.coefficient_encoding == "vq"
                                else None
                            ),
                            "coefficient_vq_stages": (
                                args.coefficient_vq_stages
                                if args.coefficient_encoding == "vq"
                                else None
                            ),
                            "vq_stage_overrides": json.dumps(
                                args.vq_stage_overrides, sort_keys=True
                            ),
                            "effective_vq_stages": ",".join(
                                str(value)
                                for value in sorted({
                                    effective_rank(
                                        record["artifact_name"],
                                        args.coefficient_vq_stages,
                                        args.vq_stage_overrides,
                                    )
                                    for record in selected
                                })
                            ),
                            "sparsity": args.sparsity,
                            "k": k,
                            "C": args.C,
                            "tile_width": args.tile_width,
                            "tile_shape": (
                                f"{args.C}x{args.tile_width}"
                                if not args.tile_overrides
                                else "mixed"
                            ),
                            "effective_tile_heights": ",".join(
                                str(value)
                                for value in sorted({
                                    effective_tile_height(
                                        record["artifact_name"],
                                        args.C,
                                        args.tile_overrides,
                                    )
                                    for record in selected
                                })
                            ),
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
                            "artifact_bits_per_compressed_parameter": (
                                8.0 * artifact_bytes
                                / max(compressed_parameter_count, 1)
                            ),
                            "static_parameter_storage_mb": static_bytes / 1024**2,
                            "total_model_storage_mb": (
                                artifact_bytes + static_bytes
                            )
                            / 1024**2,
                            "total_model_bits_per_parameter": (
                                8.0 * (artifact_bytes + static_bytes)
                                / max(total_parameter_count, 1)
                            ),
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



# MODEL="facebook/opt-125m" \
# MODEL_TAG="opt-125m" \
# INPUT_DIR="SeedLM/model/opt-125m" \
# K=512 \
# BASE_RANK=6 \
# MAX_RANK=7 \
# EXTRA_BUDGET_MB=1.88 \
# C=8 \
# TILE_WIDTH=1 \
# CODEBOOK_SIZE=256 \
# VQ_STAGES=3 \
# COEFFICIENT_BITS=8 \
# SAMPLE_SIZE=50000 \
# VQ_SAMPLE_SIZE=0 \
# EXPECTED_MATRICES=72 \
# SKIP_CODEBOOK_CALIBRATION=1 \
# RUN_ROOT="SeedLM/data/SeedLM_compressed/opt-125m/nocalib_vq3x256_allcoeff" \
# ./block-aqlm-calibration-safe/run_block_aqlm_calibration.sh

# MODEL="meta-llama/Llama-3.2-1B" \
# MODEL_TAG="Llama-3.2-1B" \
# INPUT_DIR="SeedLM/model/Llama-3.2-1B" \
# K=512 \
# BASE_RANK=6 \
# MAX_RANK=7 \
# EXTRA_BUDGET_MB=64.13 \
# C=8 \
# TILE_WIDTH=1 \
# CODEBOOK_SIZE=512 \
# VQ_STAGES=3 \
# COEFFICIENT_BITS=16 \
# SAMPLE_SIZE=50000 \
# VQ_SAMPLE_SIZE=0 \
# EXPECTED_MATRICES=112 \
# SKIP_CODEBOOK_CALIBRATION=1 \
# RUN_ROOT="SeedLM/data/SeedLM_compressed/Llama-3.2-1B/nocalib_vq3x256_allcoeff" \
# ./block-aqlm-calibration-safe/run_block_aqlm_calibration.sh