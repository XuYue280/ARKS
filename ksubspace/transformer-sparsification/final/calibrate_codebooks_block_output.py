#!/usr/bin/env python3
"""AQLM-aligned, validation-safe block calibration for coefficient VQ.

The calibration proceeds block by block. Previous accepted blocks are applied
to the model before inputs for the next block are captured. For the current
block, the dense target is evaluated on exactly the same propagated input as
the compressed prediction. Only one block's inputs are retained at a time.

Training uses sequence-level gradient accumulation, epoch-level validation,
early stopping, an optional coefficient-MSE guard, and a final held-out PPL
guard with automatic rollback. K-subspace bases and subspace IDs remain fixed.
Coefficient codebooks and, when explicitly requested, coefficient code IDs are
optimized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

from ksubspaces_artifacts import (
    _load_subspace_ids,
    _pack_unsigned,
    _unpack_unsigned,
    load_weight,
)


CALIBRATION_MARKER = "aqlm_aligned_streaming_v3"

OPT_MATRIX_SPECS = (
    ("self_attn_q_proj", "self_attn.q_proj.weight"),
    ("self_attn_k_proj", "self_attn.k_proj.weight"),
    ("self_attn_v_proj", "self_attn.v_proj.weight"),
    ("self_attn_out_proj", "self_attn.out_proj.weight"),
    ("fc1", "fc1.weight"),
    ("fc2", "fc2.weight"),
)

LLAMA_MATRIX_SPECS = (
    ("self_attn_q_proj", "self_attn.q_proj.weight"),
    ("self_attn_k_proj", "self_attn.k_proj.weight"),
    ("self_attn_v_proj", "self_attn.v_proj.weight"),
    ("self_attn_o_proj", "self_attn.o_proj.weight"),
    ("mlp_gate_proj", "mlp.gate_proj.weight"),
    ("mlp_up_proj", "mlp.up_proj.weight"),
    ("mlp_down_proj", "mlp.down_proj.weight"),
)


def model_layout(model) -> tuple[Any, tuple[tuple[str, str], ...], str]:
    """Return decoder layers, artifact/parameter mappings, and model family."""
    model_type = getattr(model.config, "model_type", "")
    if model_type == "opt" or hasattr(model.model, "decoder"):
        return model.model.decoder.layers, OPT_MATRIX_SPECS, "opt"
    if model_type == "llama" or hasattr(model.model, "layers"):
        return model.model.layers, LLAMA_MATRIX_SPECS, "llama"
    raise ValueError(
        f"Unsupported model_type={model_type!r}; expected OPT or Llama."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seqlen", type=int, default=256)
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum complete passes over the block-calibration training set.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Sequences accumulated before each optimizer update. Each sequence "
            "is forwarded separately to keep peak GPU memory low."
        ),
    )
    p.add_argument(
        "--early-stop",
        type=int,
        default=3,
        help="Stop after this many epochs without an accepted improvement; 0 disables.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=0,
        help=(
            "Deprecated optimizer-step cap. Zero uses epochs. A positive value "
            "stops after at most this many optimizer updates."
        ),
    )
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="Held-out samples. Zero selects min(8, max(1, samples // 4)).",
    )
    p.add_argument(
        "--sequential-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture each block after applying previous compressed blocks. The "
            "dense target is recomputed on that same propagated input."
        ),
    )
    p.add_argument(
        "--reassign-every",
        type=int,
        default=0,
        help=(
            "Greedily reassign residual-VQ code IDs every N optimizer updates. "
            "Zero freezes IDs, matching AQLM block fine-tuning."
        ),
    )
    p.add_argument("--reassignment-batch-size", type=int, default=4096)
    p.add_argument("--max-reassignment-codebook-size", type=int, default=4096)
    p.add_argument(
        "--max-coefficient-mse-ratio",
        type=float,
        default=1.0,
        help=(
            "Accept a block checkpoint only if its coefficient MSE is at most "
            "this multiple of the no-calibration value."
        ),
    )
    p.add_argument(
        "--ppl-guard-samples",
        type=int,
        default=4,
        help="Held-out full-model PPL chunks; 0 disables the global guard.",
    )
    p.add_argument("--ppl-guard-seqlen", type=int, default=2048)
    p.add_argument("--ppl-guard-split", default="validation")
    p.add_argument(
        "--ppl-guard-tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed fractional PPL increase over no calibration.",
    )
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Rollback backup outside the measured artifact tree.",
    )
    p.add_argument("--keep-backup", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--output-csv", type=Path, default=None)
    args = p.parse_args()

    if args.samples < 2 or args.seqlen < 2 or args.lr <= 0:
        p.error(
            "samples and seqlen must be at least 2; lr must be positive"
        )
    if args.epochs < 1 or args.batch_size < 1 or args.early_stop < 0:
        p.error("epochs and batch-size must be positive; early-stop cannot be negative")
    if args.steps < 0:
        p.error("--steps cannot be negative")
    if args.validation_samples == 0:
        args.validation_samples = min(8, max(1, args.samples // 4))
    if args.validation_samples < 1 or args.validation_samples >= args.samples:
        p.error("--validation-samples must be in [1, samples - 1]")
    if args.gradient_clip < 0:
        p.error("--gradient-clip cannot be negative")
    if (
        args.reassign_every < 0
        or args.reassignment_batch_size < 1
        or args.max_reassignment_codebook_size < 2
    ):
        p.error("reassign-every cannot be negative; batch size must be positive")
    if args.max_coefficient_mse_ratio <= 0:
        p.error("--max-coefficient-mse-ratio must be positive")
    if args.ppl_guard_samples < 0 or args.ppl_guard_seqlen < 2:
        p.error("PPL guard samples cannot be negative; seqlen must be at least 2")
    if args.ppl_guard_tolerance < 0:
        p.error("--ppl-guard-tolerance cannot be negative")
    return args


def token_batches(
    tokenizer,
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    samples: int,
    seqlen: int,
    seed: int,
) -> torch.Tensor:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(value for value in dataset["text"] if value.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.numel() < seqlen:
        raise ValueError(
            f"Need at least {seqlen} tokens from {split}; found {ids.numel()}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    max_start = ids.numel() - seqlen
    starts = torch.randint(
        0,
        max_start + 1,
        (samples,),
        generator=generator,
    )
    return torch.stack(
        [ids[start : start + seqlen] for start in starts.tolist()]
    )


def cpu_copy(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Copy nested tensors to CPU while preserving shared tensor references."""
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        key = id(value)
        if key not in memo:
            memo[key] = value.detach().cpu().clone()
        return memo[key]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item, memo) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item, memo) for item in value]
    if isinstance(value, dict):
        return {key: cpu_copy(item, memo) for key, item in value.items()}
    return value


def to_device(
    value: Any,
    device: torch.device,
    memo: dict[int, Any] | None = None,
) -> Any:
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        key = id(value)
        if key not in memo:
            memo[key] = value.to(device)
        return memo[key]
    if isinstance(value, tuple):
        return tuple(to_device(item, device, memo) for item in value)
    if isinstance(value, list):
        return [to_device(item, device, memo) for item in value]
    if isinstance(value, dict):
        return {
            key: to_device(item, device, memo)
            for key, item in value.items()
        }
    return value


class BlockInputCaptured(RuntimeError):
    """Internal control-flow exception used to stop a model forward early."""


def split_hidden_input(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[torch.Tensor, tuple[Any, ...], dict[str, Any], str]:
    """Separate hidden states from token-independent block arguments."""
    template_args = list(args)
    template_kwargs = dict(kwargs)
    if template_args and torch.is_tensor(template_args[0]):
        hidden = template_args[0]
        template_args[0] = None
        location = "args"
    elif torch.is_tensor(template_kwargs.get("hidden_states")):
        hidden = template_kwargs["hidden_states"]
        template_kwargs["hidden_states"] = None
        location = "kwargs"
    else:
        raise ValueError("Could not locate block hidden states")
    return hidden, tuple(template_args), template_kwargs, location


def static_signature(value: Any) -> Any:
    """Describe nested static inputs without retaining another tensor copy."""
    if torch.is_tensor(value):
        return ("tensor", tuple(value.shape), str(value.dtype))
    if isinstance(value, tuple):
        return ("tuple", tuple(static_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(static_signature(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (key, static_signature(item))
                for key, item in sorted(value.items())
            ),
        )
    return ("value", type(value).__name__, repr(value))


def capture_from_template(
    template_args: tuple[Any, ...],
    template_kwargs: dict[str, Any],
    hidden: torch.Tensor,
    hidden_location: str,
) -> dict[str, Any]:
    args = list(template_args)
    kwargs = dict(template_kwargs)
    if hidden_location == "args":
        args[0] = hidden
    else:
        kwargs["hidden_states"] = hidden
    return {"args": tuple(args), "kwargs": kwargs}


def collect_block_inputs(
    model,
    block,
    tokens: torch.Tensor,
    device: torch.device,
    layer_index: int,
) -> list[dict[str, Any]]:
    """Capture one block's propagated inputs and stop before running it.

    Only hidden states vary between fixed-length causal-LM samples. Static
    masks, positions, and rotary embeddings are copied once and shared across
    all captures. This keeps 1024 x 2048 calibration feasible in CPU memory.
    """
    hidden_inputs: list[torch.Tensor] = []
    template_args: tuple[Any, ...] | None = None
    template_kwargs: dict[str, Any] | None = None
    template_signature: Any = None
    hidden_location: str | None = None

    def pre_hook(_module, args, kwargs):
        nonlocal template_args, template_kwargs
        nonlocal template_signature, hidden_location
        hidden, static_args, static_kwargs, location = split_hidden_input(
            args, kwargs
        )
        current_signature = static_signature((static_args, static_kwargs))
        if template_args is None:
            memo: dict[int, Any] = {}
            template_args = cpu_copy(static_args, memo)
            template_kwargs = cpu_copy(static_kwargs, memo)
            template_signature = current_signature
            hidden_location = location
        elif (
            current_signature != template_signature
            or location != hidden_location
        ):
            raise ValueError(
                "Block static-input structure changed across fixed-length "
                f"samples in layer {layer_index}"
            )
        hidden_inputs.append(
            hidden.detach().to(device="cpu", copy=True).contiguous()
        )
        raise BlockInputCaptured

    handle = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            for sample_index, sample in enumerate(tokens, 1):
                try:
                    model(
                        input_ids=sample.unsqueeze(0).to(device),
                        use_cache=False,
                    )
                except BlockInputCaptured:
                    pass
                else:
                    raise RuntimeError(
                        f"Layer {layer_index} input hook did not run"
                    )
                if sample_index == 1 or sample_index % 32 == 0:
                    print(
                        f"Captured block {layer_index} input "
                        f"{sample_index}/{len(tokens)}",
                        flush=True,
                    )
    finally:
        handle.remove()

    if (
        template_args is None
        or template_kwargs is None
        or hidden_location is None
        or len(hidden_inputs) != len(tokens)
    ):
        raise RuntimeError(f"Incomplete input capture for layer {layer_index}")
    hidden_gib = (
        sum(tensor.numel() * tensor.element_size() for tensor in hidden_inputs)
        / (1024 ** 3)
    )
    print(
        f"Block {layer_index} retained hidden inputs: {hidden_gib:.2f} GiB",
        flush=True,
    )
    static_device_memo: dict[int, Any] = {}
    template_args = to_device(
        template_args, device, static_device_memo
    )
    template_kwargs = to_device(
        template_kwargs, device, static_device_memo
    )
    return [
        capture_from_template(
            template_args,
            template_kwargs,
            hidden,
            hidden_location,
        )
        for hidden in hidden_inputs
    ]


def unpack_code_ids(artifact: Path, metadata: dict) -> np.ndarray:
    stored = np.load(artifact / "tile_coefficients.npy")
    count = int(metadata["coefficient_code_count"])
    bits = int(metadata["coefficient_code_bits"])
    stages = int(metadata.get("coefficient_vq_stages", 1))
    return _unpack_unsigned(stored, count, bits).reshape(-1, stages)


class CompressedMatrix:
    def __init__(
        self,
        artifact: Path,
        parameter_name: str,
        dense_weight_out_in: torch.Tensor,
        device: torch.device,
    ):
        self.artifact = artifact
        self.parameter_name = parameter_name
        self.device = device
        self.metadata_path = artifact / "metadata.json"
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("coefficient_quantization") != "vector_kmeans_v1":
            raise ValueError(f"Expected residual coefficient VQ in {artifact}")
        if float(self.metadata["sparsity"]) != 1.0:
            raise ValueError("Block calibration currently requires sparsity=1.0")

        subspace_ids = _load_subspace_ids(artifact, self.metadata)
        if np.any(subspace_ids < 0):
            raise ValueError(f"Non-recovered tiles found in {artifact}")
        code_ids = unpack_code_ids(artifact, self.metadata)
        if len(code_ids) != len(subspace_ids):
            raise ValueError(f"ID count mismatch in {artifact}")

        basis_array = np.load(artifact / "subspace_bases.npy").astype(np.float32)
        self.bases_cpu = torch.from_numpy(basis_array)
        self.bases = self.bases_cpu.to(device)
        self.subspace_ids_cpu = torch.from_numpy(subspace_ids.astype(np.int64))
        self.subspace_ids = self.subspace_ids_cpu.to(device)
        self.code_ids = torch.from_numpy(code_ids.astype(np.int64)).to(device)
        initial = np.load(artifact / "coefficient_codebook.npy").astype(np.float32)
        self.codebook = torch.nn.Parameter(torch.from_numpy(initial).to(device))
        self.target_coefficients = self._project_dense_coefficients(
            dense_weight_out_in
        )

    def _project_dense_coefficients(
        self,
        dense_weight_out_in: torch.Tensor,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        rows = int(self.metadata["rows"])
        columns = int(self.metadata["columns"])
        c = int(self.metadata["C"])
        width = int(self.metadata.get("tile_width", 1))
        dense_in_out = (
            dense_weight_out_in.detach().to(device="cpu", dtype=torch.float32).T
        ).contiguous()
        if tuple(dense_in_out.shape) != (rows, columns):
            raise ValueError(
                f"Dense shape {tuple(dense_in_out.shape)} does not match "
                f"stored shape {(rows, columns)} in {self.artifact}"
            )
        tiles = (
            dense_in_out.reshape(rows // c, c, columns // width, width)
            .permute(0, 2, 1, 3)
            .reshape(-1, c * width)
            .contiguous()
        )
        targets = torch.empty(
            (len(tiles), self.codebook.shape[-1]), dtype=torch.float32
        )
        for start in range(0, len(tiles), batch_size):
            end = min(start + batch_size, len(tiles))
            selected = self.bases_cpu[self.subspace_ids_cpu[start:end]]
            targets[start:end] = torch.einsum(
                "ndr,nd->nr", selected, tiles[start:end]
            )
        return targets

    def coefficients(self) -> torch.Tensor:
        coefficients = torch.zeros(
            (len(self.code_ids), self.codebook.shape[-1]),
            dtype=self.codebook.dtype,
            device=self.codebook.device,
        )
        for stage in range(self.code_ids.shape[1]):
            coefficients = coefficients + self.codebook[
                stage, self.code_ids[:, stage]
            ]
        return coefficients

    def weight(self) -> torch.Tensor:
        tiles = torch.einsum(
            "ndr,nr->nd",
            self.bases[self.subspace_ids],
            self.coefficients(),
        )
        rows = int(self.metadata["rows"])
        columns = int(self.metadata["columns"])
        c = int(self.metadata["C"])
        width = int(self.metadata.get("tile_width", 1))
        weight_in_out = (
            tiles.reshape(rows // c, columns // width, c, width)
            .permute(0, 2, 1, 3)
            .reshape(rows, columns)
        )
        return weight_in_out.T

    @torch.no_grad()
    def coefficient_error_stats(self, batch_size: int = 65536) -> tuple[float, int]:
        squared_error = 0.0
        value_count = 0
        for start in range(0, len(self.code_ids), batch_size):
            end = min(start + batch_size, len(self.code_ids))
            reconstructed = torch.zeros(
                (end - start, self.codebook.shape[-1]),
                dtype=self.codebook.dtype,
                device=self.device,
            )
            for stage in range(self.code_ids.shape[1]):
                reconstructed += self.codebook[
                    stage, self.code_ids[start:end, stage]
                ]
            target = self.target_coefficients[start:end].to(self.device)
            difference = reconstructed.float() - target
            squared_error += float(torch.sum(difference * difference))
            value_count += int(difference.numel())
        return squared_error, value_count

    @torch.no_grad()
    def reassign_code_ids(self, batch_size: int) -> None:
        """Greedy residual-VQ reassignment toward dense projected coefficients."""
        new_ids = torch.empty(
            self.code_ids.shape, dtype=torch.int64, device="cpu"
        )
        for start in range(0, len(self.code_ids), batch_size):
            end = min(start + batch_size, len(self.code_ids))
            residual = self.target_coefficients[start:end].to(self.device)
            for stage in range(self.code_ids.shape[1]):
                codebook = self.codebook[stage]
                distances = (
                    torch.sum(residual * residual, dim=1, keepdim=True)
                    + torch.sum(codebook * codebook, dim=1).unsqueeze(0)
                    - 2.0 * residual @ codebook.T
                )
                ids = torch.argmin(distances, dim=1)
                new_ids[start:end, stage] = ids.cpu()
                residual = residual - codebook[ids]
        self.code_ids = new_ids.to(self.device)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.codebook.detach().clone(),
            self.code_ids.detach().cpu().clone(),
        )

    @torch.no_grad()
    def restore(self, snapshot: tuple[torch.Tensor, torch.Tensor]) -> None:
        codebook, code_ids = snapshot
        self.codebook.copy_(codebook)
        self.code_ids = code_ids.to(self.device)

    def save(
        self,
        args: argparse.Namespace,
        layer_index: int,
        metrics: dict[str, float | int | bool],
    ) -> None:
        np.save(
            self.artifact / "coefficient_codebook.npy",
            self.codebook.detach().cpu().numpy().astype(np.float16),
        )
        bits = int(self.metadata["coefficient_code_bits"])
        code_ids = self.code_ids.detach().cpu().numpy().astype(np.int32)
        np.save(
            self.artifact / "tile_coefficients.npy",
            _pack_unsigned(code_ids, bits),
        )
        self.metadata.update(
            {
                "coefficient_codebook_calibration": CALIBRATION_MARKER,
                "coefficient_codebook_calibration_codes_frozen": (
                    args.reassign_every == 0
                ),
                "coefficient_codebook_calibration_bases_frozen": True,
                "coefficient_codebook_calibration_sequential_inputs": (
                    args.sequential_inputs
                ),
                "coefficient_codebook_calibration_target_policy": (
                    "dense_current_block_on_propagated_input"
                ),
                "coefficient_codebook_calibration_streaming_blocks": True,
                "coefficient_codebook_calibration_block": layer_index,
                "coefficient_codebook_calibration_epochs": args.epochs,
                "coefficient_codebook_calibration_epochs_completed": int(
                    metrics["epochs_completed"]
                ),
                "coefficient_codebook_calibration_best_epoch": int(
                    metrics["best_epoch"]
                ),
                "coefficient_codebook_calibration_optimizer_steps": int(
                    metrics["optimizer_steps"]
                ),
                "coefficient_codebook_calibration_batch_size": (
                    args.batch_size
                ),
                "coefficient_codebook_calibration_early_stop": (
                    args.early_stop
                ),
                "coefficient_codebook_calibration_lr": args.lr,
                "coefficient_codebook_calibration_samples": args.samples,
                "coefficient_codebook_calibration_validation_samples": (
                    args.validation_samples
                ),
                "coefficient_codebook_calibration_seqlen": args.seqlen,
                "coefficient_codebook_calibration_reassign_every": (
                    args.reassign_every
                ),
                "coefficient_codebook_calibration_max_coefficient_mse_ratio": (
                    args.max_coefficient_mse_ratio
                ),
                "coefficient_codebook_calibration_accepted": bool(
                    metrics["block_update_accepted"]
                ),
                "coefficient_codebook_calibration_initial_validation_loss": (
                    float(metrics["initial_validation_block_mse"])
                ),
                "coefficient_codebook_calibration_final_validation_loss": (
                    float(metrics["calibrated_validation_block_mse"])
                ),
                "coefficient_codebook_calibration_initial_coefficient_mse": (
                    float(metrics["initial_coefficient_mse"])
                ),
                "coefficient_codebook_calibration_final_coefficient_mse": (
                    float(metrics["calibrated_coefficient_mse"])
                ),
            }
        )
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )


def block_output(
    block,
    replacements: dict[str, torch.Tensor],
    capture: dict,
    device: torch.device,
) -> torch.Tensor:
    device_memo: dict[int, Any] = {}
    block_args = to_device(capture["args"], device, device_memo)
    block_kwargs = to_device(capture["kwargs"], device, device_memo)
    result = functional_call(
        block,
        replacements,
        args=block_args,
        kwargs=block_kwargs,
        strict=False,
    )
    return result[0] if isinstance(result, (tuple, list)) else result


@torch.no_grad()
def dense_block_output(
    block,
    capture: dict,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate the dense current block on the propagated block input."""
    device_memo: dict[int, Any] = {}
    block_args = to_device(capture["args"], device, device_memo)
    block_kwargs = to_device(capture["kwargs"], device, device_memo)
    result = block(*block_args, **block_kwargs)
    dense = result[0] if isinstance(result, (tuple, list)) else result
    return dense.detach()


def replacement_weights(block, matrices) -> dict[str, torch.Tensor]:
    block_dtype = next(block.parameters()).dtype
    return {
        matrix.parameter_name: matrix.weight().to(dtype=block_dtype)
        for matrix in matrices
    }


def mean_block_loss(block, matrices, captures, device) -> float:
    total = 0.0
    with torch.no_grad():
        replacements = replacement_weights(block, matrices)
        for capture in captures:
            target = dense_block_output(block, capture, device)
            prediction = block_output(block, replacements, capture, device)
            total += float(F.mse_loss(prediction.float(), target.float()))
    return total / len(captures)


def aggregate_coefficient_mse(matrices) -> float:
    squared_error = 0.0
    value_count = 0
    for matrix in matrices:
        matrix_error, matrix_count = matrix.coefficient_error_stats()
        squared_error += matrix_error
        value_count += matrix_count
    return squared_error / max(value_count, 1)


@torch.no_grad()
def apply_matrices_to_block(block, matrices) -> None:
    """Commit the restored best compressed weights for downstream capture."""
    for matrix in matrices:
        parameter = block.get_parameter(matrix.parameter_name)
        parameter.copy_(matrix.weight().to(dtype=parameter.dtype))


def tune_block(
    block,
    matrices,
    train_captures,
    validation_captures,
    args,
    device,
    layer_index,
) -> dict:
    parameters = [matrix.codebook for matrix in matrices]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.95))
    initial_train_loss = mean_block_loss(block, matrices, train_captures, device)
    initial_validation_loss = mean_block_loss(
        block, matrices, validation_captures, device
    )
    initial_coefficient_mse = aggregate_coefficient_mse(matrices)
    best_validation_loss = initial_validation_loss
    best_states = [matrix.snapshot() for matrix in matrices]
    accepted = False
    best_epoch = 0
    epochs_without_improvement = 0
    optimizer_steps = 0
    epochs_completed = 0
    started = time.perf_counter()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + layer_index)
    stop_for_step_cap = False

    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(
            len(train_captures), generator=generator
        ).tolist()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        for batch_start in range(0, len(order), args.batch_size):
            if args.steps > 0 and optimizer_steps >= args.steps:
                stop_for_step_cap = True
                break
            batch_indices = order[
                batch_start : batch_start + args.batch_size
            ]
            optimizer.zero_grad(set_to_none=True)
            for capture_index in batch_indices:
                capture = train_captures[capture_index]
                target = dense_block_output(block, capture, device)
                replacements = replacement_weights(block, matrices)
                prediction = block_output(
                    block, replacements, capture, device
                )
                loss = F.mse_loss(prediction.float(), target.float())
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Non-finite block loss in layer {layer_index}"
                    )
                (loss / len(batch_indices)).backward()
                epoch_loss_sum += float(loss.detach())
                epoch_sample_count += 1
                del target, prediction, replacements, loss

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, args.gradient_clip
                )
            optimizer.step()
            optimizer_steps += 1

            if (
                args.reassign_every > 0
                and optimizer_steps % args.reassign_every == 0
            ):
                for matrix in matrices:
                    matrix.reassign_code_ids(
                        args.reassignment_batch_size
                    )

        if epoch_sample_count == 0:
            break
        epochs_completed = epoch
        validation_loss = mean_block_loss(
            block, matrices, validation_captures, device
        )
        coefficient_mse = aggregate_coefficient_mse(matrices)
        coefficient_guard = (
            coefficient_mse
            <= initial_coefficient_mse * args.max_coefficient_mse_ratio
        )
        validation_improved = validation_loss < best_validation_loss
        checkpoint_accepted = validation_improved and coefficient_guard
        if checkpoint_accepted:
            accepted = True
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_states = [matrix.snapshot() for matrix in matrices]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"layer={layer_index} epoch={epoch}/{args.epochs} "
            f"optimizer_steps={optimizer_steps} "
            f"train_loss={epoch_loss_sum / epoch_sample_count:.8e} "
            f"validation_loss={validation_loss:.8e} "
            f"coefficient_mse={coefficient_mse:.8e} "
            f"coefficient_guard={coefficient_guard} "
            f"checkpoint_accepted={checkpoint_accepted}",
            flush=True,
        )

        if stop_for_step_cap:
            break
        if (
            args.early_stop > 0
            and epochs_without_improvement >= args.early_stop
        ):
            print(
                f"layer={layer_index} early_stop after epoch={epoch}",
                flush=True,
            )
            break

    for matrix, state in zip(matrices, best_states):
        matrix.restore(state)
    final_train_loss = mean_block_loss(block, matrices, train_captures, device)
    final_validation_loss = mean_block_loss(
        block, matrices, validation_captures, device
    )
    final_coefficient_mse = aggregate_coefficient_mse(matrices)
    row = {
        "layer": layer_index,
        "block_update_accepted": accepted,
        "initial_train_block_mse": initial_train_loss,
        "calibrated_train_block_mse": final_train_loss,
        "initial_validation_block_mse": initial_validation_loss,
        "calibrated_validation_block_mse": final_validation_loss,
        "validation_improvement_percent": (
            100.0
            * (initial_validation_loss - final_validation_loss)
            / max(initial_validation_loss, 1e-30)
        ),
        "initial_coefficient_mse": initial_coefficient_mse,
        "calibrated_coefficient_mse": final_coefficient_mse,
        "coefficient_mse_ratio": (
            final_coefficient_mse / max(initial_coefficient_mse, 1e-30)
        ),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "optimizer_steps": optimizer_steps,
        "batch_size": args.batch_size,
        "rejection_reason": (
            "accepted"
            if accepted
            else "no_validation_checkpoint_passed_both_guards"
        ),
        "seconds": time.perf_counter() - started,
    }
    for matrix in matrices:
        matrix.save(args, layer_index, row)
    return row


def create_artifact_backup(artifact_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_dir}. Restore or remove it before "
            "starting another calibration run."
        )
    layer_dirs = sorted(artifact_dir.glob("layer_*"))
    if not layer_dirs:
        raise FileNotFoundError(f"No layer artifacts found in {artifact_dir}")
    for layer_dir in layer_dirs:
        destination = backup_dir / layer_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in (
            "coefficient_codebook.npy",
            "tile_coefficients.npy",
            "metadata.json",
        ):
            source = layer_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing calibration artifact: {source}")
            shutil.copy2(source, destination / filename)


def restore_artifact_backup(artifact_dir: Path, backup_dir: Path) -> None:
    for layer_backup in sorted(backup_dir.glob("layer_*")):
        destination = artifact_dir / layer_backup.name
        for source in layer_backup.iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name)


def load_dense_model(model_name: str, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def apply_artifacts_to_model(model, artifact_dir: Path, device: torch.device) -> None:
    layers, matrix_specs, _ = model_layout(model)
    for layer_index, block in enumerate(layers):
        for artifact_suffix, parameter_name in matrix_specs:
            matrix_dir = artifact_dir / f"layer_{layer_index:02d}_{artifact_suffix}"
            reconstructed_in_out, _ = load_weight(matrix_dir)
            reconstructed_out_in = torch.from_numpy(reconstructed_in_out.T).to(
                device=device,
                dtype=block.get_parameter(parameter_name).dtype,
            )
            block.get_parameter(parameter_name).copy_(reconstructed_out_in)


@torch.no_grad()
def model_perplexity(
    model,
    tokens: torch.Tensor,
    device: torch.device,
) -> float:
    total_nll = 0.0
    total_tokens = 0
    for sample in tokens:
        input_ids = sample.unsqueeze(0).to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        total_nll += float(
            F.cross_entropy(
                logits.transpose(1, 2).float(), labels, reduction="sum"
            )
        )
        total_tokens += int(labels.numel())
    return math.exp(total_nll / total_tokens)


def write_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    output = args.output_csv or args.artifact_dir / "block_codebook_calibration.csv"
    backup_dir = args.backup_dir or (
        args.artifact_dir.parent.parent
        / "calibration_backups"
        / args.artifact_dir.name
    )
    backup_created = False
    calibration_committed = False
    rows: list[dict] = []

    try:
        create_artifact_backup(args.artifact_dir, backup_dir)
        backup_created = True
        guard_tokens = None
        initial_guard_ppl = None
        if args.ppl_guard_samples > 0:
            guard_tokens = token_batches(
                tokenizer,
                dataset_name=args.dataset,
                dataset_config=args.dataset_config,
                split=args.ppl_guard_split,
                samples=args.ppl_guard_samples,
                seqlen=args.ppl_guard_seqlen,
                seed=args.seed + 10000,
            )
            guard_model = load_dense_model(args.model, device)
            apply_artifacts_to_model(guard_model, args.artifact_dir, device)
            initial_guard_ppl = model_perplexity(guard_model, guard_tokens, device)
            print(
                f"No-calibration held-out PPL guard: {initial_guard_ppl:.8f}",
                flush=True,
            )
            del guard_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model = load_dense_model(args.model, device)
        layers, matrix_specs, model_family = model_layout(model)
        print(
            f"AQLM-aligned block calibration: family={model_family}; "
            f"layers={len(layers)}; matrices_per_block={len(matrix_specs)}; "
            f"train_samples={args.samples - args.validation_samples}; "
            f"validation_samples={args.validation_samples}; "
            f"epochs={args.epochs}; batch_size={args.batch_size}; "
            f"early_stop={args.early_stop}",
            flush=True,
        )

        calibration_tokens = token_batches(
            tokenizer,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            samples=args.samples,
            seqlen=args.seqlen,
            seed=args.seed,
        )
        train_count = args.samples - args.validation_samples

        for layer_index, block in enumerate(layers):
            print(
                f"Capturing propagated inputs for transformer block "
                f"{layer_index}",
                flush=True,
            )
            block_captures = collect_block_inputs(
                model,
                block,
                calibration_tokens,
                device,
                layer_index,
            )
            train_captures = block_captures[:train_count]
            validation_captures = block_captures[train_count:]

            matrices = []
            for artifact_suffix, parameter_name in matrix_specs:
                artifact = (
                    args.artifact_dir
                    / f"layer_{layer_index:02d}_{artifact_suffix}"
                )
                if not (artifact / "metadata.json").is_file():
                    raise FileNotFoundError(f"Missing artifact: {artifact}")
                matrices.append(
                    CompressedMatrix(
                        artifact,
                        parameter_name,
                        block.get_parameter(parameter_name),
                        device,
                    )
                )
            if args.reassign_every > 0:
                largest_codebook = max(
                    int(matrix.codebook.shape[1]) for matrix in matrices
                )
                if largest_codebook > args.max_reassignment_codebook_size:
                    raise ValueError(
                        f"Codebook size {largest_codebook} is too large for "
                        "exhaustive code-ID reassignment. Use "
                        "--reassign-every 0 or increase "
                        "--max-reassignment-codebook-size explicitly."
                    )

            print(f"Calibrating transformer block {layer_index}", flush=True)
            row = tune_block(
                block,
                matrices,
                train_captures,
                validation_captures,
                args,
                device,
                layer_index,
            )
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

            if args.sequential_inputs:
                apply_matrices_to_block(block, matrices)

            block_captures.clear()
            del train_captures, validation_captures
            del block_captures, matrices
            if device.type == "cuda":
                torch.cuda.empty_cache()

        final_guard_ppl = None
        global_accepted = True
        if guard_tokens is not None and initial_guard_ppl is not None:
            apply_artifacts_to_model(model, args.artifact_dir, device)
            final_guard_ppl = model_perplexity(model, guard_tokens, device)
            allowed_ppl = initial_guard_ppl * (1.0 + args.ppl_guard_tolerance)
            global_accepted = (
                math.isfinite(final_guard_ppl) and final_guard_ppl <= allowed_ppl
            )
            print(
                f"Calibrated held-out PPL guard: {final_guard_ppl:.8f}; "
                f"allowed={allowed_ppl:.8f}; accepted={global_accepted}",
                flush=True,
            )

        for row in rows:
            row["initial_guard_ppl"] = initial_guard_ppl
            row["calibrated_guard_ppl"] = final_guard_ppl
            row["global_update_accepted"] = global_accepted

        if not global_accepted:
            print(
                "Held-out PPL worsened. Restoring every no-calibration "
                "codebook, code ID, and metadata file.",
                flush=True,
            )
            restore_artifact_backup(args.artifact_dir, backup_dir)
        else:
            calibration_committed = True

        write_rows(rows, output)
        print(f"Saved: {output}")
    except Exception:
        print(
            "Calibration failed. Restoring the no-calibration artifacts.",
            flush=True,
        )
        if backup_created:
            restore_artifact_backup(args.artifact_dir, backup_dir)
        raise
    finally:
        if (
            backup_created
            and backup_dir.exists()
            and (calibration_committed or not args.keep_backup)
        ):
            shutil.rmtree(backup_dir)


if __name__ == "__main__":
    main()
