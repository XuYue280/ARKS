#!/usr/bin/env python3
"""Export decoder linear weights of an OPT or Llama causal LM to .npy files.

The files follow the layout consumed by the compression stage:

    layer_NN_<suffix>_weight.npy   float16, HuggingFace [out, in] orientation
    layer_NN_<suffix>_bias.npy     float16, when the module has a bias

``<suffix>`` is ``self_attn_{q,k,v,out}_proj``/``fc1``/``fc2`` for OPT and
``self_attn_{q,k,v,o}_proj``/``mlp_{gate,up,down}_proj`` for Llama-style
models. A ``manifest.json`` records every exported tensor with its shape,
dtype and SHA-256 digest so later stages can verify the export.

The model is loaded on the CPU in its checkpoint precision and cast to
float16, which is lossless for the fp16/bf16-free checkpoints used here and
matches the original export used for the published OPT-125M results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def decoder_modules(model) -> list[tuple[int, str, torch.nn.Module]]:
    model_type = getattr(model.config, "model_type", "")
    entries: list[tuple[int, str, torch.nn.Module]] = []
    if model_type == "opt" or hasattr(model.model, "decoder"):
        for index, layer in enumerate(model.model.decoder.layers):
            entries += [
                (index, "self_attn_q_proj", layer.self_attn.q_proj),
                (index, "self_attn_k_proj", layer.self_attn.k_proj),
                (index, "self_attn_v_proj", layer.self_attn.v_proj),
                (index, "self_attn_out_proj", layer.self_attn.out_proj),
                (index, "fc1", layer.fc1),
                (index, "fc2", layer.fc2),
            ]
        return entries
    if model_type == "llama" or hasattr(model.model, "layers"):
        for index, layer in enumerate(model.model.layers):
            entries += [
                (index, "self_attn_q_proj", layer.self_attn.q_proj),
                (index, "self_attn_k_proj", layer.self_attn.k_proj),
                (index, "self_attn_v_proj", layer.self_attn.v_proj),
                (index, "self_attn_o_proj", layer.self_attn.o_proj),
                (index, "mlp_gate_proj", layer.mlp.gate_proj),
                (index, "mlp_up_proj", layer.mlp.up_proj),
                (index, "mlp_down_proj", layer.mlp.down_proj),
            ]
        return entries
    raise ValueError(f"Unsupported model_type={model_type!r}; expected OPT or Llama")


def sha256_of_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Stored dtype. float16 matches the historical export format.",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="Optional existing export directory; every common file is compared bitwise.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = AutoConfig.from_pretrained(args.model)
    print(
        f"Exporting {args.model}: model_type={config.model_type}, "
        f"layers={getattr(config, 'num_hidden_layers', '?')}, "
        f"hidden={getattr(config, 'hidden_size', '?')}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()
    numpy_dtype = np.float16 if args.dtype == "float16" else np.float32

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "model_type": config.model_type,
        "dtype": args.dtype,
        "files": {},
    }
    mismatches = 0
    entries = decoder_modules(model)
    for index, suffix, module in entries:
        tensors = {"weight": module.weight}
        if getattr(module, "bias", None) is not None:
            tensors["bias"] = module.bias
        for kind, tensor in tensors.items():
            array = tensor.detach().cpu().numpy().astype(numpy_dtype, copy=False)
            name = f"layer_{index:02d}_{suffix}_{kind}.npy"
            path = args.output_dir / name
            tmp = path.with_suffix(".npy.tmp")
            with open(tmp, "wb") as handle:  # np.save would append ".npy"
                np.save(handle, array)
            os.replace(tmp, path)
            manifest["files"][name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": sha256_of_array(array),
            }
            if args.verify_against is not None:
                reference = args.verify_against / name
                if reference.is_file():
                    other = np.load(reference)
                    same = (
                        other.dtype == array.dtype
                        and other.shape == array.shape
                        and np.array_equal(other, array)
                    )
                    if not same:
                        mismatches += 1
                        print(f"MISMATCH {name}", flush=True)
        if index == 0 or (index + 1) % 8 == 0:
            print(f"  exported layer {index}", flush=True)

    manifest["matrix_count"] = sum(
        1 for name in manifest["files"] if name.endswith("_weight.npy")
    )
    manifest["layers"] = 1 + max(index for index, _, _ in entries)
    manifest["seconds"] = time.perf_counter() - started
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Exported {manifest['matrix_count']} weight matrices "
        f"({len(manifest['files'])} tensors) to {args.output_dir} in "
        f"{manifest['seconds']:.1f}s",
        flush=True,
    )
    if args.verify_against is not None:
        print(f"Verification against {args.verify_against}: {mismatches} mismatching files")
        if mismatches:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
