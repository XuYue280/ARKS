
"""Shared helpers for the ARKS pipeline: token caching, dense-PPL caching,
hardware detection, and model-layout inspection.

Every helper is opt-in and result-preserving: when a cache directory is not
configured, the callers behave exactly as they did before these helpers
existed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

import torch

TOKEN_CACHE_ENV = "ARKS_TOKEN_CACHE"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _tokenizer_signature(tokenizer) -> dict[str, Any]:
    try:
        import transformers

        version = transformers.__version__
    except Exception:  # pragma: no cover - defensive
        version = "unknown"
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "transformers": version,
    }


def dataset_text_column(dataset, requested: str = "auto") -> str:
    """Pick the text column: WikiText and C4 use ``text``, PTB uses ``sentence``."""
    if requested and requested != "auto":
        return requested
    for name in ("text", "sentence"):
        if name in dataset.column_names:
            return name
    raise ValueError(
        f"Could not find a text column in {dataset.column_names}; pass the column name explicitly"
    )


# Per-corpus evaluation text assembly.
#
# The published reference perplexities (OPT-125M: WikiText-2 27.65, PTB 38.99,
# C4 26.56) follow the GPTQ / SparseGPT convention, and that convention is not
# the same for every corpus: WikiText-2 joins the whole test split on blank
# lines, PTB joins *stripped* test lines on single spaces, and C4 joins only
# the first 1100 validation records on single spaces. Joining every corpus the
# WikiText-2 way (the previous behaviour, still reachable as "default") gives
# PTB 31.53 and C4 25.62 instead. Measured on facebook/opt-125m, seqlen 2048.
EVAL_RECIPES = {
    "wikitext2": {"separator": "\n\n", "strip_rows": False, "max_rows": None},
    "ptb": {"separator": " ", "strip_rows": True, "max_rows": None},
    "c4": {"separator": " ", "strip_rows": False, "max_rows": 1100},
    "default": {"separator": "\n\n", "strip_rows": False, "max_rows": None},
}


def eval_recipe(name: str | None) -> dict:
    """Return the text-assembly spec for ``name`` (unknown names -> default)."""
    return EVAL_RECIPES.get(name or "default", EVAL_RECIPES["default"])


def build_eval_text(rows, recipe: str | None = "default") -> str:
    """Assemble the evaluation string for one corpus under ``recipe``."""
    spec = eval_recipe(recipe)
    if spec["max_rows"] is not None:
        rows = rows[: spec["max_rows"]]
    if spec["strip_rows"]:
        rows = [row.strip() for row in rows]
    return spec["separator"].join(rows)


# Known evaluation/calibration corpora. ``data_files`` limits C4 to one
# validation shard (the usual GPTQ/AQLM convention) instead of the full split.
DATASET_PRESETS = {
    "wikitext2": {"name": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "data_files": None},
    # Recent versions of ``datasets`` no longer execute dataset loading
    # scripts, so load the original Penn Treebank text files directly.
    "ptb": {
        "name": "text",
        "config": None,
        "data_files": {
            "train": (
                "https://raw.githubusercontent.com/wojzaremba/"
                "lstm/master/data/ptb.train.txt"
            ),
            "validation": (
                "https://raw.githubusercontent.com/wojzaremba/"
                "lstm/master/data/ptb.valid.txt"
            ),
            "test": (
                "https://raw.githubusercontent.com/wojzaremba/"
                "lstm/master/data/ptb.test.txt"
            ),
        },
    },
    "c4": {
        "name": "allenai/c4",
        "config": "en",
        "data_files": {
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00000-of-00008.json.gz",
        },
    },
}


def cached_token_ids(
    tokenizer,
    *,
    model_id: str,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    variant: str,
    build_text,
    cache_dir: str | os.PathLike | None = None,
    data_files=None,
) -> torch.Tensor:
    """Return the 1-D token id tensor for a dataset split, cached on disk.

    ``build_text`` receives the loaded dataset and returns the exact string
    that the caller used to tokenize before caching existed, so the cached
    ids are identical to the uncached ones. ``variant`` distinguishes callers
    that build different strings from the same split.
    """
    from datasets import load_dataset

    root = cache_dir if cache_dir is not None else os.environ.get(TOKEN_CACHE_ENV)
    signature = _tokenizer_signature(tokenizer)
    key = {
        "model_id": model_id,
        "dataset": dataset_name,
        "config": dataset_config,
        "split": split,
        "variant": variant,
        "data_files": data_files,
        **signature,
    }
    if root:
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        base = (
            Path(root)
            / f"{_safe_name(model_id)}__{_safe_name(dataset_name)}__"
            f"{_safe_name(dataset_config or 'none')}__{split}__{variant}__{digest}"
        )
        # Concatenate, never Path.with_suffix(): _safe_name keeps ".", so a
        # model id like facebook/opt-6.7b or meta-llama/Llama-3.1-8B makes
        # with_suffix() treat ".7b__...__<digest>" as the existing suffix and
        # replace it -- every corpus for that model collapses onto one file
        # (facebook_opt-6.pt). The key check keeps that correct but uncacheable,
        # which breaks the offline compute-node path.
        tensor_path = Path(f"{base}.pt")
        meta_path = Path(f"{base}.json")
        if tensor_path.is_file() and meta_path.is_file():
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
                if stored.get("key") == key:
                    ids = torch.load(tensor_path, map_location="cpu")
                    if ids.numel() == int(stored.get("numel", -1)):
                        print(
                            f"[token-cache] reused {tensor_path.name} "
                            f"({ids.numel()} tokens)",
                            flush=True,
                        )
                        return ids
            except Exception as error:  # pragma: no cover - cache corruption
                print(f"[token-cache] ignoring {base}: {error}", flush=True)

    if data_files:
        files = data_files.get(split, data_files) if isinstance(data_files, dict) else data_files
        # ``no_checks``: pinning one C4 shard makes the recorded split sizes
        # disagree with the full-dataset metadata, which otherwise raises
        # ExpectedMoreSplitsError before anything is tokenized.
        dataset = load_dataset(
            dataset_name, dataset_config, split=split,
            data_files={split: files}, verification_mode="no_checks",
        )
    else:
        dataset = load_dataset(
            dataset_name, dataset_config, split=split,
            verification_mode="no_checks",
        )
    started = time.perf_counter()
    ids = tokenizer(build_text(dataset), return_tensors="pt").input_ids[0]
    elapsed = time.perf_counter() - started
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        tmp = Path(f"{tensor_path}.tmp")
        torch.save(ids.clone(), tmp)
        os.replace(tmp, tensor_path)
        meta_path.write_text(
            json.dumps(
                {"key": key, "numel": int(ids.numel()), "tokenize_seconds": elapsed},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[token-cache] stored {tensor_path.name} ({ids.numel()} tokens, "
            f"{elapsed:.1f}s)",
            flush=True,
        )
    return ids


# ---------------------------------------------------------------------------
# Dense perplexity cache
# ---------------------------------------------------------------------------


def dense_ppl_cache_lookup(path: Path | None, key: dict[str, Any]) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    wanted = json.dumps(key, sort_keys=True)
    for entry in entries if isinstance(entries, list) else []:
        if json.dumps(entry.get("key"), sort_keys=True) == wanted:
            return entry
    return None


def dense_ppl_cache_store(path: Path | None, key: dict[str, Any], value: dict) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list = []
    if path.is_file():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []
    wanted = json.dumps(key, sort_keys=True)
    entries = [
        e for e in entries if json.dumps(e.get("key"), sort_keys=True) != wanted
    ]
    entries.append({"key": key, **value, "stored_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Hardware and model inspection (used by the Bash entry point)
# ---------------------------------------------------------------------------


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, _, rest = line.partition(":")
            number = rest.strip().split()[0]
            values[name] = int(number) * 1024
    except Exception:
        pass
    return values


def hardware_info() -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except Exception:
        affinity = cpu_count
    memory = _meminfo()
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_logical": cpu_count,
        "cpu_usable": affinity,
        "ram_total_bytes": memory.get("MemTotal", 0),
        "ram_available_bytes": memory.get("MemAvailable", 0),
        "swap_total_bytes": memory.get("SwapTotal", 0),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            info["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "vram_total_bytes": int(total),
                    "vram_free_bytes": int(free),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return info


def model_info(model_id: str) -> dict[str, Any]:
    """Describe a causal LM from its configuration without loading weights."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    model_type = getattr(config, "model_type", "")
    hidden = int(getattr(config, "hidden_size", 0))
    layers = int(getattr(config, "num_hidden_layers", 0))
    heads = int(getattr(config, "num_attention_heads", 1))
    kv_heads = int(getattr(config, "num_key_value_heads", heads) or heads)
    head_dim = int(getattr(config, "head_dim", hidden // max(heads, 1)) or hidden // max(heads, 1))
    vocab = int(getattr(config, "vocab_size", 0))
    if model_type == "opt":
        family = "opt"
        ffn = int(getattr(config, "ffn_dim", 4 * hidden))
        matrices = [
            ("self_attn_q_proj", hidden, hidden),
            ("self_attn_k_proj", hidden, hidden),
            ("self_attn_v_proj", hidden, hidden),
            ("self_attn_out_proj", hidden, hidden),
            ("fc1", ffn, hidden),
            ("fc2", hidden, ffn),
        ]
        max_positions = int(getattr(config, "max_position_embeddings", 2048))
    elif model_type in ("llama", "mistral", "qwen2", "gemma", "gemma2"):
        family = "llama"
        ffn = int(getattr(config, "intermediate_size", 4 * hidden))
        matrices = [
            ("self_attn_q_proj", heads * head_dim, hidden),
            ("self_attn_k_proj", kv_heads * head_dim, hidden),
            ("self_attn_v_proj", kv_heads * head_dim, hidden),
            ("self_attn_o_proj", hidden, heads * head_dim),
            ("mlp_gate_proj", ffn, hidden),
            ("mlp_up_proj", ffn, hidden),
            ("mlp_down_proj", hidden, ffn),
        ]
        max_positions = int(getattr(config, "max_position_embeddings", 2048))
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r}; the pipeline supports OPT "
            "and Llama-style decoders."
        )
    per_layer = [
        {"suffix": suffix, "rows_out": out, "cols_in": inp, "elements": out * inp}
        for suffix, out, inp in matrices
    ]
    decoder_elements = layers * sum(m["elements"] for m in per_layer)
    largest = max(m["elements"] for m in per_layer)
    dtype_bytes = 2
    try:
        dtype_bytes = torch.tensor([], dtype=getattr(torch, str(config.torch_dtype).replace("torch.", ""))).element_size() if getattr(config, "torch_dtype", None) else 2
    except Exception:
        dtype_bytes = 2
    return {
        "model_id": model_id,
        "model_type": model_type,
        "family": family,
        "num_layers": layers,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "max_position_embeddings": max_positions,
        "matrices_per_layer": len(per_layer),
        "matrix_shapes": per_layer,
        "expected_matrices": layers * len(per_layer),
        "decoder_linear_elements": decoder_elements,
        "largest_matrix_elements": largest,
        "checkpoint_dtype_bytes": dtype_bytes,
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", True)),
    }


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hardware")
    model = sub.add_parser("model")
    model.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "hardware":
        print(json.dumps(hardware_info(), indent=2))
    elif args.command == "model":
        print(json.dumps(model_info(args.model), indent=2))


if __name__ == "__main__":
    _cli()

