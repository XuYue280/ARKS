#!/usr/bin/env python3
"""Ensure a HuggingFace causal LM is available in the local cache.

Downloads only what the pipeline needs (configuration, tokenizer files and
the model weights, preferring safetensors) and prints the resolved snapshot
directory. Gated models such as Llama use the token stored by
``huggingface-cli login`` or the HF_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--info-json", type=Path, default=None)
    args = parser.parse_args()

    from huggingface_hub import list_repo_files, snapshot_download

    # HF_HUB_OFFLINE is how a network-less compute node says "use the cache the
    # login node already populated". list_repo_files() is a live Hub API call and
    # would fail there, so resolve the snapshot locally and read the file list
    # off disk instead. Everything downstream only needs `files` to decide
    # safetensors-vs-bin, which the cached snapshot answers just as well.
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() not in ("", "0", "false", "no")
    if offline:
        path = snapshot_download(
            args.model,
            revision=args.revision,
            local_files_only=True,
        )
        files = sorted(os.listdir(path))
    else:
        files = list_repo_files(args.model, revision=args.revision)
        has_safetensors = any(name.endswith(".safetensors") for name in files)
        patterns = ["*.json", "*.txt", "*.model", "*.tiktoken", "tokenizer*", "*.py"]
        patterns.append("*.safetensors" if has_safetensors else "*.bin")
        if has_safetensors:
            patterns.append("*.safetensors.index.json")
        else:
            patterns.append("*.bin.index.json")
        path = snapshot_download(
            args.model,
            revision=args.revision,
            allow_patterns=patterns,
            token=os.environ.get("HF_TOKEN") or True,
        )
    has_safetensors = any(name.endswith(".safetensors") for name in files)
    weight_files = sorted(
        name for name in os.listdir(path)
        if name.endswith(".safetensors") or name.endswith(".bin")
    )
    if not weight_files:
        raise SystemExit(f"No weight files were downloaded for {args.model} into {path}")
    info = {
        "model": args.model,
        "snapshot_path": path,
        "weight_files": weight_files,
        "weight_format": "safetensors" if has_safetensors else "pytorch_bin",
        "weight_bytes": sum(os.path.getsize(os.path.join(path, f)) for f in weight_files),
    }
    if args.info_json is not None:
        args.info_json.parent.mkdir(parents=True, exist_ok=True)
        args.info_json.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
