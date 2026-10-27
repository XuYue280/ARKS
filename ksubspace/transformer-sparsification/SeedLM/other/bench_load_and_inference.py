
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from SeedLM.other.SeedLM_cpu import (
    SeedLM_load,
    quantization_load,
    structured_pruning_load,
)
from SeedLM.other.bench_compress_and_store import (
    # shared experiment knobs (keep compress / load configs in sync)
    OPT_MODELS,
    DEFAULT_SPARSITIES,
    K_CLUSTER_TILE_FRACTION,
    C,
    P,
    K_BITS,
    SELECTION_POOL,
    SELECTION_SEED,
    # formatting / accumulation helpers
    _format_timings,
    _accumulate_timings,
    _accumulate_storage,
    _finalize_total,
    _finalize_timings_wallclock,
    _format_storage,
    # directory paths
    _model_tag,
    _opt_store_dir,
    _seedlm_compressed_dir,
    _structured_pruning_dir,
    _magnitude_pruning_dir,
    _seedlm_quantized_compressed_dir,
    _quantized_model_dir,
    _k_sizes_from_opt_dir,
)
_BENCH_LOG: list[dict] = []
_BENCH_LOG_PATH: Path | None = None
_CURRENT_SECTION = "N/A"


def _results_load_dir(model_id: str) -> Path:
    return _ROOT / "results" / _model_tag(model_id) / "CPU_inference_result"


def _server_temp_inference_dir(model_id: str) -> Path:
    return _ROOT / "results" / "server_temp" / "CPU_inference_result" / _model_tag(model_id)


def _init_bench_log(
    model_id: str,
    *,
    server_temp: bool = False,
    job_tag: str | None = None,
) -> Path:
    global _BENCH_LOG, _BENCH_LOG_PATH, _CURRENT_SECTION
    if server_temp:
        if not job_tag:
            raise ValueError("job_tag is required when server_temp=True")
        out_dir = _server_temp_inference_dir(model_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        _BENCH_LOG_PATH = out_dir / f"{job_tag}.JSON"
    else:
        out_dir = _results_load_dir(model_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        _BENCH_LOG_PATH = out_dir / f"{ts}.JSON"
    _BENCH_LOG = []
    _CURRENT_SECTION = "Initialization"
    return _BENCH_LOG_PATH


def _log(**fields: object) -> None:
    if _BENCH_LOG_PATH is None:
        return
    _BENCH_LOG.append({"section": _CURRENT_SECTION, **fields})
    with open(_BENCH_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(_BENCH_LOG, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_token_id: int | None,
) -> float:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id or -100)
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits[:, :-1, :]
        labels = batch["input_ids"][:, 1:]
        nll = loss_fn(logits.permute(0, 2, 1), labels)
        mask = labels != pad_token_id if pad_token_id is not None else torch.ones_like(labels, dtype=torch.bool)
        total_nll += (nll * mask).sum().item()
        total_tokens += mask.sum().item()
    if total_tokens == 0:
        raise ValueError("No tokens evaluated for perplexity.")
    return math.exp(total_nll / total_tokens)


def test_dataset_loader(
    tokenizer,
    seqlen: int,
    batch_size: int,
    max_samples: int | None,
) -> DataLoader:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokenized = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    n_chunks = tokenized.input_ids.numel() // seqlen
    input_ids = tokenized.input_ids[0, : n_chunks * seqlen].reshape(n_chunks, seqlen)
    attn_mask = tokenized.attention_mask[0, : n_chunks * seqlen].reshape(n_chunks, seqlen)
    if max_samples is not None and max_samples > 0:
        input_ids = input_ids[:max_samples]
        attn_mask = attn_mask[:max_samples]

    class _Set(Dataset):
        def __len__(self):
            return input_ids.shape[0]

        def __getitem__(self, idx):
            return {"input_ids": input_ids[idx], "attention_mask": attn_mask[idx]}

    return DataLoader(_Set(), batch_size=batch_size)


def _load_param_from_npy(
    timings: dict[str, float],
    storage_mb: dict[str, float],
    path: Path,
    param: torch.nn.Parameter,
    timing_key: str,
    storage_key: str,
) -> None:
    t0 = time.perf_counter()
    arr = np.load(path)
    param.data = torch.from_numpy(arr).to(device=param.device, dtype=param.dtype)
    _accumulate_timings(timings, {timing_key: time.perf_counter() - t0})
    _accumulate_storage(storage_mb, {storage_key: arr.nbytes})


def _load_static_opt_components(
    model: torch.nn.Module,
    opt_dir: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load attention heads and all biases once (shared across sections)."""
    attn_names = ("q_proj", "k_proj", "v_proj", "out_proj")
    timings: dict[str, float] = {}
    storage_mb: dict[str, float] = {}

    for layer_i, layer in enumerate(model.model.decoder.layers):
        for attn_name in attn_names:
            attn_mod = getattr(layer.self_attn, attn_name)
            _load_param_from_npy(
                timings,
                storage_mb,
                opt_dir / f"layer_{layer_i:02d}_self_attn_{attn_name}_weight.npy",
                attn_mod.weight,
                "load_attention",
                "attention",
            )

            bias_path = opt_dir / f"layer_{layer_i:02d}_self_attn_{attn_name}_bias.npy"
            if attn_mod.bias is not None and bias_path.exists():
                _load_param_from_npy(
                    timings,
                    storage_mb,
                    bias_path,
                    attn_mod.bias,
                    "load_bias",
                    "bias",
                )

        for fc_name in ("fc1", "fc2"):
            fc = getattr(layer, fc_name)
            bias_path = opt_dir / f"layer_{layer_i:02d}_{fc_name}_bias.npy"
            if fc.bias is not None and bias_path.exists():
                _load_param_from_npy(
                    timings,
                    storage_mb,
                    bias_path,
                    fc.bias,
                    "load_bias",
                    "bias",
                )

    return timings, storage_mb


def _section_1_baseline(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int | None,
    model_id: str,
    args: argparse.Namespace,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 1: Baseline (dense model)"
    print("\n============== Section 1: Baseline (dense model) ==============", flush=True)

    load_and_inference_time: dict[str, float] = {}
    storage_mb: dict[str, float] = {}
    t_section_total = time.perf_counter()

    for layer_i, layer in enumerate(model.model.decoder.layers):
        for fc_name in ("fc1", "fc2"):
            fc = getattr(layer, fc_name)
            path = opt_dir / f"layer_{layer_i:02d}_{fc_name}_weight.npy"
            t0 = time.perf_counter()
            arr = np.load(path)
            fc.weight.data = torch.from_numpy(arr).to(
                device=fc.weight.device, dtype=fc.weight.dtype
            )
            _accumulate_timings(
                load_and_inference_time, {"load_fc1_fc2": time.perf_counter() - t0}
            )
            _accumulate_storage(storage_mb, {"fc1_fc2_weights": arr.nbytes})

    _finalize_total(storage_mb)

    t0 = time.perf_counter()
    ppl = evaluate_ppl(model, loader, device, pad_id)
    _accumulate_timings(
        load_and_inference_time,
        {"inference_time": time.perf_counter() - t0},
    )
    _finalize_timings_wallclock(load_and_inference_time, t_section_total)

    print(
        f"{_format_timings('Load and inference time', load_and_inference_time)}\n"
        f"PPL: {ppl:.4f},\n"
        f"{_format_storage('Storage', storage_mb)}\n",
        flush=True,
    )
    _log(
        ppl=ppl,
        load_and_inference_time=load_and_inference_time,
        storage_mb=storage_mb,
    )


def _section_2_quantization(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int | None,
    model_id: str,
    args: argparse.Namespace,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 2: Quantization Load and Inference"
    print("\n============== Section 2: Quantization Load and Inference ==============", flush=True)

    for bits in QUANTIZATION_BITS:
        quant_dir = _quantized_model_dir(model_id, bits)
        load_and_inference_time: dict[str, float] = {}
        storage_mb: dict[str, float] = {}
        t_section_total = time.perf_counter()

        for layer_i, layer in enumerate(model.model.decoder.layers):
            for fc_name in ("fc1", "fc2"):
                fc = getattr(layer, fc_name)
                row, col = fc.weight.shape
                w_q, load_timings, load_bytes = quantization_load(
                    row,
                    col,
                    bits,
                    load_path=quant_dir,
                    load_lut_name=(
                        f"layer_{layer_i:02d}_{fc_name}_{bits}bits_quantization_LUT.npy"
                    ),
                    load_quantization_index_name=(
                        f"layer_{layer_i:02d}_{fc_name}_{bits}bits_quantization_index.npy"
                    ),
                )
                fc.weight.data = torch.from_numpy(w_q).to(
                    device=fc.weight.device, dtype=fc.weight.dtype
                )
                _accumulate_timings(load_and_inference_time, load_timings)
                _accumulate_storage(storage_mb, load_bytes)

        _finalize_total(storage_mb)

        t0 = time.perf_counter()
        ppl = evaluate_ppl(model, loader, device, pad_id)
        _accumulate_timings(
            load_and_inference_time,
            {"inference_time": time.perf_counter() - t0},
        )
        _finalize_timings_wallclock(load_and_inference_time, t_section_total)

        print(
            f"Quantization bits = {bits},\n"
            f"{_format_timings('Load and inference time', load_and_inference_time)}\n"
            f"PPL: {ppl:.4f},\n"
            f"{_format_storage('Storage', storage_mb)}\n",
            flush=True,
        )
        _log(
            quantization_bits=bits,
            ppl=ppl,
            load_and_inference_time=load_and_inference_time,
            storage_mb=storage_mb,
        )


def _section_3_magnitude_pruning(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int | None,
    model_id: str,
    args: argparse.Namespace,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 3: Magnitude Pruning Load and Inference"
    print("\n============== Section 3: Magnitude Pruning Load and Inference ==============", flush=True)

    for sp in sparsities:
        if sp == 0.0:
            continue
        magnitude_dir = _magnitude_pruning_dir(model_id, sparsity=sp)
        load_and_inference_time: dict[str, float] = {}
        storage_mb: dict[str, float] = {}
        t_section_total = time.perf_counter()

        for layer_i, layer in enumerate(model.model.decoder.layers):
            for fc_name in ("fc1", "fc2"):
                fc = getattr(layer, fc_name)
                path = magnitude_dir / f"layer_{layer_i:02d}_{fc_name}_magnitude_pruning.npy"
                t0 = time.perf_counter()
                arr = np.load(path)
                fc.weight.data = torch.from_numpy(arr).to(
                    device=fc.weight.device, dtype=fc.weight.dtype
                )
                _accumulate_timings(
                    load_and_inference_time,
                    {"load_magnitude_pruned": time.perf_counter() - t0},
                )
                _accumulate_storage(storage_mb, {"magnitude_pruned": arr.nbytes})

        _finalize_total(storage_mb)

        t0 = time.perf_counter()
        ppl = evaluate_ppl(model, loader, device, pad_id)
        _accumulate_timings(
            load_and_inference_time,
            {"inference_time": time.perf_counter() - t0},
        )
        _finalize_timings_wallclock(load_and_inference_time, t_section_total)

        print(
            f"Sparsity = {sp:.2f},\n"
            f"{_format_timings('Load and inference time', load_and_inference_time)}\n"
            f"PPL: {ppl:.4f},\n"
            f"{_format_storage('Storage', storage_mb)}\n",
            flush=True,
        )
        _log(
            sparsity=sp,
            ppl=ppl,
            load_and_inference_time=load_and_inference_time,
            storage_mb=storage_mb,
        )


def _section_4_structured_pruning(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int | None,
    model_id: str,
    args: argparse.Namespace,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 4: Structured Pruning Load and Inference"
    print("\n============== Section 4: Structured Pruning Load and Inference ==============", flush=True)

    for sp in sparsities:
        if sp == 0.0:
            continue
        structured_dir_sp = _structured_pruning_dir(model_id, sparsity=sp)
        load_and_inference_time: dict[str, float] = {}
        storage_mb: dict[str, float] = {}
        t_section_total = time.perf_counter()

        for layer_i, layer in enumerate(model.model.decoder.layers):
            for fc_name in ("fc1", "fc2"):
                fc = getattr(layer, fc_name)
                out_f, in_f = fc.weight.shape
                w_out, load_timings, load_bytes = structured_pruning_load(
                    in_f,
                    out_f,
                    sp,
                    SELECTION_POOL,
                    SELECTION_SEED,
                    C=C,
                    load_path=structured_dir_sp,
                    load_compact_name=(
                        f"layer_{layer_i:02d}_{fc_name}_structured_pruning.npy"
                    ),
                )
                fc.weight.data = torch.from_numpy(w_out.T).to(
                    device=fc.weight.device, dtype=fc.weight.dtype
                )
                _accumulate_timings(load_and_inference_time, load_timings)
                _accumulate_storage(storage_mb, load_bytes)

        _finalize_total(storage_mb)

        t0 = time.perf_counter()
        ppl = evaluate_ppl(model, loader, device, pad_id)
        _accumulate_timings(
            load_and_inference_time,
            {"inference_time": time.perf_counter() - t0},
        )
        _finalize_timings_wallclock(load_and_inference_time, t_section_total)

        print(
            f"Sparsity = {sp:.2f},\n"
            f"{_format_timings('Load and inference time', load_and_inference_time)}\n"
            f"PPL: {ppl:.4f},\n"
            f"{_format_storage('Storage', storage_mb)}\n",
            flush=True,
        )
        _log(
            sparsity=sp,
            ppl=ppl,
            load_and_inference_time=load_and_inference_time,
            storage_mb=storage_mb,
        )


def _section_5a_seedlm_k_cluster(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int | None,
    model_id: str,
    args: argparse.Namespace,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 5A: Structured Pruning + SeedLM K-Cluster Load and Inference"
    print(
        "\n============== Section 5A: Structured Pruning + SeedLM K-Cluster Load and Inference ==============",
        flush=True,
    )

    for sp in sparsities:
        if sp == 0.0:
            continue
        for k in k_sizes:
            load_and_inference_time: dict[str, float] = {}
            storage_mb: dict[str, float] = {}
            t_section_total = time.perf_counter()
            seedlm_cfg_dir = _seedlm_compressed_dir(model_id, sparsity=sp, k_cluster=k)

            for layer_i, layer in enumerate(model.model.decoder.layers):
                for fc_name in ("fc1", "fc2"):
                    fc = getattr(layer, fc_name)
                    out_f, in_f = fc.weight.shape
                    layer_load_path = seedlm_cfg_dir / f"layer_{layer_i:02d}_{fc_name}"
                    w_out, load_timings, load_bytes = SeedLM_load(
                        in_f,
                        out_f,
                        sp,
                        k,
                        load_path=layer_load_path,
                        C=C,
                        P=P,
                        K_bits=K_BITS,
                    )
                    fc.weight.data = torch.from_numpy(w_out.T).to(
                        device=fc.weight.device, dtype=fc.weight.dtype
                    )
                    _accumulate_timings(load_and_inference_time, load_timings)
                    _accumulate_storage(storage_mb, load_bytes)

            _finalize_total(storage_mb)

            t0 = time.perf_counter()
            ppl = evaluate_ppl(model, loader, device, pad_id)
            _accumulate_timings(
                load_and_inference_time,
                {"inference_time": time.perf_counter() - t0},
            )
            _finalize_timings_wallclock(load_and_inference_time, t_section_total)

            print(
                f"Sparsity = {sp:.2f},\n"
                f"SeedLM_K_cluster = {k},\n"
                f"C = {C},\n"
                f"P = {P},\n"
                f"K = {K_BITS},\n"
                f"{_format_timings('Load and inference time', load_and_inference_time)}\n"
                f"PPL: {ppl:.4f},\n"
                f"{_format_storage('Storage', storage_mb)}\n",
                flush=True,
            )
            _log(
                sparsity=sp,
                SeedLM_K_cluster=k,
                C=str(C),
                P=str(P),
                K=str(K_BITS),
                ppl=ppl,
                load_and_inference_time=load_and_inference_time,
                storage_mb=storage_mb,
            )


def _section_5b_seedlm_k_cluster_with_quantization(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int | None,
    model_id: str,
    args: argparse.Namespace,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = (
        "Section 5B: Structured Pruning + SeedLM K-Cluster with Quantization Load and Inference"
    )
    print(
        "\n============== Section 5B: Structured Pruning + SeedLM K-Cluster with Quantization Load and Inference ==============",
        flush=True,
    )

    for bits in QUANTIZATION_BITS:
        for sp in sparsities:
            if sp == 0.0:
                continue
            for k in k_sizes:
                load_and_inference_time: dict[str, float] = {}
                storage_mb: dict[str, float] = {}
                t_section_total = time.perf_counter()
                quant_store_dir = _seedlm_quantized_compressed_dir(
                    model_id, bits=bits, sparsity=sp, k_cluster=k
                )

                for layer_i, layer in enumerate(model.model.decoder.layers):
                    for fc_name in ("fc1", "fc2"):
                        fc = getattr(layer, fc_name)
                        out_f, in_f = fc.weight.shape
                        layer_load_path = quant_store_dir / f"layer_{layer_i:02d}_{fc_name}"
                        w_out, load_timings, load_bytes = SeedLM_load(
                            in_f,
                            out_f,
                            sp,
                            k,
                            load_path=layer_load_path,
                            C=C,
                            P=P,
                            K_bits=K_BITS,
                        )
                        fc.weight.data = torch.from_numpy(w_out.T).to(
                            device=fc.weight.device, dtype=fc.weight.dtype
                        )
                        _accumulate_timings(load_and_inference_time, load_timings)
                        _accumulate_storage(storage_mb, load_bytes)

                _finalize_total(storage_mb)

                t0 = time.perf_counter()
                ppl = evaluate_ppl(model, loader, device, pad_id)
                _accumulate_timings(
                    load_and_inference_time,
                    {"inference_time": time.perf_counter() - t0},
                )
                _finalize_timings_wallclock(load_and_inference_time, t_section_total)

                print(
                    f"Sparsity = {sp:.2f},\n"
                    f"Quantization bits = {bits},\n"
                    f"SeedLM_K_cluster = {k},\n"
                    f"C = {C},\n"
                    f"P = {P},\n"
                    f"K = {K_BITS},\n"
                    f"{_format_timings('Load and inference time', load_and_inference_time)}\n"
                    f"PPL: {ppl:.4f},\n"
                    f"{_format_storage('Storage', storage_mb)}\n",
                    flush=True,
                )
                _log(
                    sparsity=sp,
                    quantization_bits=bits,
                    SeedLM_K_cluster=k,
                    C=str(C),
                    P=str(P),
                    K=str(K_BITS),
                    ppl=ppl,
                    load_and_inference_time=load_and_inference_time,
                    storage_mb=storage_mb,
                )


# section 1: baseline dense
# section 2: quantization load and inference
# section 3: magnitude pruning load and inference
# section 4: structured pruning load and inference
# section 5A: structured pruning + SeedLM K-cluster load and inference
# section 5B: structured pruning + SeedLM K-cluster with quantization load and inference
def benchmark_cpu_load_and_inference() -> None:
    parser = argparse.ArgumentParser(description="OPT layer-wise SeedLM load and inference (WikiText-2 PPL).")
    parser.add_argument(
        "--model",
        default=None,
        choices=OPT_MODELS,
        help="Run a single OPT model (default: all models in OPT_MODELS).",
    )
    parser.add_argument(
        "--section",
        default=None,
        choices=["s1", "s2", "s3", "s4", "s5a", "s5b", "all"],
        help="Run only one load section (default: all).",
    )
    parser.add_argument(
        "--sparsity",
        nargs="+",
        type=float,
        default=None,
        help="Sparsity fractions in [0, 1], e.g. 0.0 0.1 0.5 1.0",
    )
    parser.add_argument(
        "--k-cluster-fraction",
        nargs="+",
        type=float,
        default=None,
        help="K cluster as fraction of tile count, e.g. 0.01 0.04 (default: K_CLUSTER_TILE_FRACTION)",
    )
    parser.add_argument(
        "--quantization-bits",
        nargs="+",
        type=int,
        default=None,
        help="Quantization bit-widths, e.g. 3 4 (default: QUANTIZATION_BITS).",
    )
    parser.add_argument(
        "--server-temp",
        action="store_true",
        help="Write JSON under results/server_temp/CPU_inference_result/{model}/.",
    )
    parser.add_argument(
        "--job-tag",
        default=None,
        help="Filename stem for server-temp JSON (required with --server-temp).",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto; PPL uses this device)")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-ppl-samples", type=int, default=None, help="Cap WikiText-2 chunks for faster runs.")
    args = parser.parse_args()

    global QUANTIZATION_BITS

    section = args.section or "all"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sparsities = args.sparsity if args.sparsity is not None else list(DEFAULT_SPARSITIES)
    models_to_run = [args.model] if args.model else list(OPT_MODELS)
    fracs = args.k_cluster_fraction if args.k_cluster_fraction is not None else list(K_CLUSTER_TILE_FRACTION)
    quant_bits = (
        args.quantization_bits if args.quantization_bits is not None else list(QUANTIZATION_BITS)
    )

    _quant_bits_backup = list(QUANTIZATION_BITS)
    QUANTIZATION_BITS = quant_bits

    try:
        for model_id in models_to_run:
            opt_dir = _opt_store_dir(model_id)
            seedlm_dir = _seedlm_compressed_dir(model_id)
            seedlm_quantized_dir = _seedlm_quantized_compressed_dir(model_id)
            structured_dir = _structured_pruning_dir(model_id)

            use_temp = bool(args.server_temp) or (section != "all")
            job_tag = args.job_tag
            if use_temp and not job_tag:
                job_tag = f"{section}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            log_path = _init_bench_log(
                model_id,
                server_temp=use_temp,
                job_tag=job_tag if use_temp else None,
            )
            print(f"Benchmark results for {model_id} -> {log_path}", flush=True)

            print(f"Loading {model_id} on {device} ...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            # Weights load/reconstruct on CPU; move to device for evaluate_ppl.
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)
            model.eval()

            loader = test_dataset_loader(tokenizer, args.seqlen, args.batch_size, args.max_ppl_samples)
            pad_id = model.config.pad_token_id

            k_sizes = _k_sizes_from_opt_dir(opt_dir, fracs)

            static_timings, static_storage_mb = _load_static_opt_components(model, opt_dir)
            _finalize_total(static_storage_mb)
            _CURRENT_SECTION = "Initialization"
            print(
                "\n============== Initialization: Static OPT components (attention + bias) ==============",
                flush=True,
            )
            print(
                f"{_format_timings('Static load time', static_timings)}\n"
                f"{_format_storage('Static storage', static_storage_mb)}\n",
                flush=True,
            )
            # Only log static init on full runs or s1 (keeps merged JSON equivalent for split jobs
            # by attaching static init only to s1 baseline job).
            if section in ("s1", "all"):
                _log(
                    static_load_timings=static_timings,
                    static_storage_mb=static_storage_mb,
                )

            section_kwargs = dict(
                model=model,
                loader=loader,
                device=device,
                pad_id=pad_id,
                model_id=model_id,
                args=args,
                sparsities=sparsities,
                k_sizes=k_sizes,
                opt_dir=opt_dir,
                structured_dir=structured_dir,
                seedlm_dir=seedlm_dir,
                seedlm_quantized_dir=seedlm_quantized_dir,
            )

            if section in ("s1", "all"):
                _section_1_baseline(**section_kwargs)
            if section in ("s2", "all"):
                _section_2_quantization(**section_kwargs)
            if section in ("s3", "all"):
                _section_3_magnitude_pruning(**section_kwargs)
            if section in ("s4", "all"):
                _section_4_structured_pruning(**section_kwargs)
            if section in ("s5a", "all"):
                _section_5a_seedlm_k_cluster(**section_kwargs)
            if section in ("s5b", "all"):
                _section_5b_seedlm_k_cluster_with_quantization(**section_kwargs)
    finally:
        QUANTIZATION_BITS = _quant_bits_backup


if __name__ == "__main__":
    benchmark_cpu_load_and_inference()
