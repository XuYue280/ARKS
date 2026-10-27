
"""Layer-wise OPT MLP benchmark: structured column tile pruning + SeedLM recovery."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from SeedLM.kline.SeedLM_cpu_klines import (
    SeedLM_compress,
    get_uniform_quantization_level,
    magnitude_pruning,
    quantization_compress,
    reconstruct_quantized_weight,
    structured_pruning_compress,
)

'''
run this benchmark from root folder: "/benchmark"

python SeedLM/bench_compress_and_store.py
'''

'''
Global variables setting here.
'''

# ON THE SERVER, FIRST RUN [opt-1.3b, opt-2.7b, opt-6.7b]
OPT_MODELS = [
    "facebook/opt-125m", # success
    "facebook/opt-350m", # success
    #"facebook/opt-1.3b",
    #"facebook/opt-2.7b",
    #"facebook/opt-6.7b",
    # "facebook/opt-13b",
    # "facebook/opt-30b",
    # "facebook/opt-66b",
]

# [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# [0.5] smoke test
DEFAULT_SPARSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# K cluster = fraction of tile count in weight matrix; each fraction yields one K value
K_CLUSTER_TILE_FRACTION = [0.01, 0.04]
C = 8
P = 3
K_BITS = 16    # small number for smoke test
M = 4
INIT_SEED = 25565
SELECTION_POOL = 100
SELECTION_SEED = 42
# quantization parameters
# [4,5,6,7,8]
QUANTIZATION_BITS = [4]
QUANTIZATION_UPPER_BOUND = 1.0
QUANTIZATION_LOWER_BOUND = -1.0


# TODO: storage path change, each model with different paramaters need to be individual folder 
# 例如"SeedLM_compressed"下需要将不同sparsity，K{cluster number}保存在独立文件夹下。一个文件夹里需要只存放当前section 一组parameters的store result
# Layout:
#   magnitude_pruning/{model}/sp_{sp}/layer_XX_fcY_magnitude_pruning.npy
#   structured_pruning/{model}/sp_{sp}/layer_XX_fcY_structured_pruning.npy
#   SeedLM_compressed/{model}/sp_{sp}_K{k}/layer_XX_fcY/{artifacts}
#   SeedLM_compressed_quantized/{model}/{bits}bits_quantization/sp_{sp}_K{k}/layer_XX_fcY/{artifacts}
#   quantized_model/{model}/{bits}bits_quantization/  (already per-bits)


def _model_tag(model_id: str) -> str:
    """HuggingFace id -> short tag, e.g. facebook/opt-125m -> opt-125m."""
    return model_id.rsplit("/", 1)[-1]


def _sp_dir_tag(sparsity: float) -> str:
    return f"sp_{sparsity:.2f}"


def _sp_k_dir_tag(sparsity: float, k_cluster: int) -> str:
    return f"sp_{sparsity:.2f}_K{k_cluster}"


def _opt_store_dir(model_id: str) -> Path:
    return _ROOT / "data" / "original_model" / _model_tag(model_id)


def _seedlm_compressed_dir(
    model_id: str,
    sparsity: float | None = None,
    k_cluster: int | None = None,
    clustering_method: str | None = None,
) -> Path:
    base = _ROOT / "data" / "SeedLM_compressed" / _model_tag(model_id)
    if clustering_method is not None:
        base = base / clustering_method
    if sparsity is None:
        return base
    if k_cluster is None:
        raise ValueError("k_cluster is required when sparsity is set for SeedLM path.")
    return base / _sp_k_dir_tag(sparsity, k_cluster)


def _structured_pruning_dir(
    model_id: str,
    sparsity: float | None = None,
) -> Path:
    base = _ROOT / "data" / "structured_pruning" / _model_tag(model_id)
    if sparsity is None:
        return base
    return base / _sp_dir_tag(sparsity)


def _magnitude_pruning_dir(
    model_id: str,
    sparsity: float | None = None,
) -> Path:
    base = _ROOT / "data" / "magnitude_pruning" / _model_tag(model_id)
    if sparsity is None:
        return base
    return base / _sp_dir_tag(sparsity)


def _seedlm_quantized_compressed_dir(
    model_id: str,
    bits: int | None = None,
    sparsity: float | None = None,
    k_cluster: int | None = None,
) -> Path:
    base = _ROOT / "data" / "SeedLM_compressed_quantized" / _model_tag(model_id)
    if bits is not None:
        base = base / f"{bits}bits_quantization"
    if sparsity is None:
        return base
    if k_cluster is None:
        raise ValueError("k_cluster is required when sparsity is set for SeedLM+quant path.")
    return base / _sp_k_dir_tag(sparsity, k_cluster)


def _quantized_model_dir(model_id: str, bits: int) -> Path:
    return _ROOT / "data" / "quantized_model" / _model_tag(model_id) / f"{bits}bits_quantization"


def _results_dir(model_id: str) -> Path:
    return _ROOT / "results" / _model_tag(model_id) / "CPU_compress_result"


def _server_temp_compress_dir(model_id: str) -> Path:
    return _ROOT / "results" / "server_temp" / "CPU_compress_result" / _model_tag(model_id)


def _count_layers_from_opt_dir(opt_dir: Path) -> int:
    return len(sorted(opt_dir.glob("layer_*_fc1_weight.npy")))


def _k_sizes_from_opt_dir(opt_dir: Path, fracs: list[float]) -> list[int]:
    w = np.load(opt_dir / "layer_00_fc2_weight.npy").T
    num_tiles = (w.shape[0] // C) * w.shape[1]
    return [max(1, int(round(num_tiles * frac))) for frac in fracs]


def _format_runtime(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600}h {(total % 3600) // 60}m {total % 60}s"


def _format_timings(label: str, timings: dict[str, float]) -> str:
    lines = [f"{label}:"]
    for key, value in timings.items():
        if key == "total":
            continue
        lines.append(f"\t{key} = {_format_runtime(value)},")
    if "total" in timings:
        lines.append(f"total = {_format_runtime(timings['total'])},")
    return "\n".join(lines)


_MB = 1.0 / (1024 ** 2)


def _format_storage(label: str, storage_mb: dict[str, float]) -> str:
    lines = [f"{label}:"]
    for key, size_mb in storage_mb.items():
        if key == "total":
            continue
        lines.append(f"\t{key} = {size_mb:.4f} MB,")
    if "total" in storage_mb:
        lines.append(f"total = {storage_mb['total']:.4f} MB,")
    return "\n".join(lines)


def _accumulate_timings(total: dict[str, float], part: dict[str, float]) -> None:
    """Merge per-layer / inline timings into section total."""
    for key, value in part.items():
        total[key] = total.get(key, 0.0) + value


def _accumulate_storage(
    total: dict[str, float],
    part: dict[str, float] | dict[str, int],
) -> None:
    """Merge storage in MB (accepts MB dict or bytes dict from SeedLM_cpu)."""
    for key, value in part.items():
        mb = value * _MB if isinstance(value, int) else value
        total[key] = total.get(key, 0.0) + mb


def _finalize_total(values: dict[str, float]) -> None:
    '''
    total in MB = sum of all values in storage_mb dict
    '''
    values["total"] = sum(v for k, v in values.items() if k != "total")


def _finalize_timings_wallclock(timings: dict[str, float], t_start: float) -> None:
    '''
    total in seconds = time between t_start and the time call this function
    '''
    timings["total"] = time.perf_counter() - t_start


_BENCH_LOG: list[dict] = []
_BENCH_LOG_PATH: Path | None = None
_CURRENT_SECTION = "N/A"


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
        out_dir = _server_temp_compress_dir(model_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        _BENCH_LOG_PATH = out_dir / f"{job_tag}.JSON"
    else:
        out_dir = _results_dir(model_id)
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


# Store opt model in .npy format, later use this for memory cost measurement.
def store_opt_components(
    model: torch.nn.Module,
    model_id: str,
    out_dir: Path | str | None = None,
) -> tuple[Path, dict[str, float], dict[str, float]]:
    """Store OPT self-attn projections and per-layer fc1/fc2 weights/biases as .npy.

    Returns (out_dir, storage_timings, storage_mb).
    """
    storage_timings: dict[str, float] = {
        "extract_attention": 0.0,
        "save_attention": 0.0,
        "extract_bias": 0.0,
        "save_bias": 0.0,
        "extract_fc1_fc2": 0.0,
        "save_fc1_fc2": 0.0,
        "total": 0.0,
    }
    storage_mb: dict[str, float] = {
        "attention": 0.0,
        "bias": 0.0,
        "fc1_fc2_weights": 0.0,
        "total": 0.0,
    }

    t_total = time.perf_counter()

    if out_dir is None:
        out_dir = _opt_store_dir(model_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = model.model.decoder.layers
    attn_names = ("q_proj", "k_proj", "v_proj", "out_proj")
    paths: list[Path] = []

    for i, layer in enumerate(layers):
        for attn_name in attn_names:
            mod = getattr(layer.self_attn, attn_name)
            wpath = out_dir / f"layer_{i:02d}_self_attn_{attn_name}_weight.npy"

            t0 = time.perf_counter()
            w_arr = mod.weight.detach().float().cpu().numpy().astype(np.float16)
            _accumulate_timings(
                storage_timings,
                {"extract_attention": time.perf_counter() - t0},
            )

            t0 = time.perf_counter()
            np.save(wpath, w_arr)
            _accumulate_timings(
                storage_timings,
                {"save_attention": time.perf_counter() - t0},
            )
            _accumulate_storage(storage_mb, {"attention": w_arr.nbytes})
            paths.append(wpath)

            if mod.bias is not None:
                bpath = out_dir / f"layer_{i:02d}_self_attn_{attn_name}_bias.npy"

                t0 = time.perf_counter()
                b_arr = mod.bias.detach().float().cpu().numpy().astype(np.float16)
                _accumulate_timings(
                    storage_timings,
                    {"extract_bias": time.perf_counter() - t0},
                )

                t0 = time.perf_counter()
                np.save(bpath, b_arr)
                _accumulate_timings(
                    storage_timings,
                    {"save_bias": time.perf_counter() - t0},
                )
                _accumulate_storage(storage_mb, {"bias": b_arr.nbytes})
                paths.append(bpath)

        for fc_name in ("fc1", "fc2"):
            fc = getattr(layer, fc_name)
            wpath = out_dir / f"layer_{i:02d}_{fc_name}_weight.npy"

            t0 = time.perf_counter()
            w_arr = fc.weight.detach().float().cpu().numpy().astype(np.float16)
            _accumulate_timings(
                storage_timings,
                {"extract_fc1_fc2": time.perf_counter() - t0},
            )

            t0 = time.perf_counter()
            np.save(wpath, w_arr)
            _accumulate_timings(
                storage_timings,
                {"save_fc1_fc2": time.perf_counter() - t0},
            )
            _accumulate_storage(storage_mb, {"fc1_fc2_weights": w_arr.nbytes})
            paths.append(wpath)

            if fc.bias is not None:
                bpath = out_dir / f"layer_{i:02d}_{fc_name}_bias.npy"

                t0 = time.perf_counter()
                b_arr = fc.bias.detach().float().cpu().numpy().astype(np.float16)
                _accumulate_timings(
                    storage_timings,
                    {"extract_bias": time.perf_counter() - t0},
                )

                t0 = time.perf_counter()
                np.save(bpath, b_arr)
                _accumulate_timings(
                    storage_timings,
                    {"save_bias": time.perf_counter() - t0},
                )
                _accumulate_storage(storage_mb, {"bias": b_arr.nbytes})
                paths.append(bpath)

    _finalize_timings_wallclock(storage_timings, t_total)
    _finalize_total(storage_mb)

    print(
        f"[store_opt_components] model={model_id}, layers={len(layers)}, "
        f"files={len(paths)}, dir={out_dir}\n"
        f"{_format_timings('Store time', storage_timings)}\n"
        f"{_format_storage('Store storage', storage_mb)}\n",
        flush=True,
    )
    return out_dir, storage_timings, storage_mb


# Shared section-function argument names (same for every section after Section 1):
#   model_id, args, n_layers, sparsities, k_sizes,
#   opt_dir, structured_dir, seedlm_dir, seedlm_quantized_dir


def _section_1_store_opt_components(
    export_model: torch.nn.Module,
    model_id: str,
    opt_dir: Path,
) -> tuple[Path, int]:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 1: Store OPT model family components"
    print(
        "\n============== Section 1: Store OPT model family components ==============",
        flush=True,
    )
    opt_dir, storage_timings, storage_mb = store_opt_components(
        export_model, model_id, opt_dir
    )
    n_layers = len(export_model.model.decoder.layers)
    _log(
        model_id=model_id,
        storage_path=str(opt_dir),
        storage_timings=storage_timings,
        storage_mb=storage_mb,
    )
    return opt_dir, n_layers


def _section_2_quantization_compression(
    model_id: str,
    args: argparse.Namespace,
    n_layers: int,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 2: Quantization Compression and Store"
    print("\n============== Section 2: Quantization Compression and Store ==============", flush=True)
    for bits in QUANTIZATION_BITS:
        compress_timings_total: dict[str, float] = {}
        storage_mb_total: dict[str, float] = {}
        quant_dir = _quantized_model_dir(model_id, bits)
        quant_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        quantization_level = get_uniform_quantization_level(
            QUANTIZATION_UPPER_BOUND, QUANTIZATION_LOWER_BOUND, bits
        )
        _accumulate_timings(
            compress_timings_total,
            {"get_quantization_level": time.perf_counter() - t0},
        )

        for layer_i in range(n_layers):
            for fc_name in ("fc1", "fc2"):
                load_name = f"layer_{layer_i:02d}_{fc_name}_weight.npy"
                storage_LUT_name = f"layer_{layer_i:02d}_{fc_name}_{bits}bits_quantization_LUT.npy"
                storage_quantization_index_name = (
                    f"layer_{layer_i:02d}_{fc_name}_{bits}bits_quantization_index.npy"
                )

                compress_timings, storage_bytes = quantization_compress(
                    quantization_upper_bound=QUANTIZATION_UPPER_BOUND,
                    quantization_lower_bound=QUANTIZATION_LOWER_BOUND,
                    bits=bits,
                    load_path=opt_dir,
                    load_name=load_name,
                    storage_path=quant_dir,
                    storage_LUT_name=storage_LUT_name,
                    storage_quantization_index_name=storage_quantization_index_name,
                    quantization_level=quantization_level,
                )
                _accumulate_timings(compress_timings_total, compress_timings)
                _accumulate_storage(storage_mb_total, storage_bytes)

        _finalize_total(compress_timings_total)
        _finalize_total(storage_mb_total)
        print(
            f"Quantization bits = {bits},\n"
            f"{_format_timings('Compression time', compress_timings_total)}\n"
            f"{_format_storage('Storage', storage_mb_total)}\n",
            flush=True,
        )
        _log(
            quantization_bits=bits,
            compress_timings=compress_timings_total,
            storage_mb=storage_mb_total,
        )


def _section_3_magnitude_pruning_compression(
    model_id: str,
    args: argparse.Namespace,
    n_layers: int,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 3: Magnitude Pruning Compression and Store"
    print("\n============== Section 3: Magnitude Pruning Compression and Store ==============", flush=True)

    for sp in sparsities:
        if sp == 0.0:
            continue
        magnitude_dir = _magnitude_pruning_dir(model_id, sparsity=sp)
        magnitude_dir.mkdir(parents=True, exist_ok=True)
        compress_timings_total: dict[str, float] = {}
        storage_mb_total: dict[str, float] = {}

        for layer_i in range(n_layers):
            for fc_name in ("fc1", "fc2"):
                load_name = f"layer_{layer_i:02d}_{fc_name}_weight.npy"
                storage_name = f"layer_{layer_i:02d}_{fc_name}_magnitude_pruning.npy"

                t0 = time.perf_counter()
                W = np.load(opt_dir / load_name)
                _accumulate_timings(
                    compress_timings_total,
                    {"load_from_local": time.perf_counter() - t0},
                )

                t0 = time.perf_counter()
                W_pruned = magnitude_pruning(W, sp)
                _accumulate_timings(
                    compress_timings_total,
                    {"magnitude_pruning": time.perf_counter() - t0},
                )

                t0 = time.perf_counter()
                W_saved = W_pruned.astype(np.float16)
                np.save(magnitude_dir / storage_name, W_saved)
                _accumulate_timings(
                    compress_timings_total,
                    {"save_to_local": time.perf_counter() - t0},
                )
                _accumulate_storage(
                    storage_mb_total,
                    {"magnitude_pruned": W_saved.nbytes},
                )

        _finalize_total(compress_timings_total)
        _finalize_total(storage_mb_total)
        print(
            f"Sparsity = {sp:.2f},\n"
            f"{_format_timings('Compression time', compress_timings_total)}\n"
            f"{_format_storage('Storage', storage_mb_total)}\n",
            flush=True,
        )
        _log(
            sparsity=sp,
            compress_timings=compress_timings_total,
            storage_mb=storage_mb_total,
        )


def _section_4_structured_pruning_compression(
    model_id: str,
    args: argparse.Namespace,
    n_layers: int,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 4: Structured Pruning Compression and Store"
    print("\n============== Section 4: Structured Pruning Compression and Store ==============", flush=True)
    for sp in sparsities:
        if sp == 0.0:
            continue
        structured_dir_sp = _structured_pruning_dir(model_id, sparsity=sp)
        structured_dir_sp.mkdir(parents=True, exist_ok=True)
        compress_timings_total: dict[str, float] = {}
        storage_mb_total: dict[str, float] = {}

        for layer_i in range(n_layers):
            for fc_name in ("fc1", "fc2"):
                load_name = f"layer_{layer_i:02d}_{fc_name}_weight.npy"
                storage_name = f"layer_{layer_i:02d}_{fc_name}_structured_pruning.npy"

                compress_timings, storage_bytes = structured_pruning_compress(
                    sp,
                    SELECTION_POOL,
                    SELECTION_SEED,
                    load_path=opt_dir,
                    load_name=load_name,
                    storage_path=structured_dir_sp,
                    storage_name=storage_name,
                    C=C,
                )
                _accumulate_timings(compress_timings_total, compress_timings)
                _accumulate_storage(storage_mb_total, storage_bytes)

        _finalize_total(compress_timings_total)
        _finalize_total(storage_mb_total)
        print(
            f"Sparsity = {sp:.2f},\n"
            f"{_format_timings('Compression time', compress_timings_total)}\n"
            f"{_format_storage('Storage', storage_mb_total)}\n",
            flush=True,
        )
        _log(
            sparsity=sp,
            compress_timings=compress_timings_total,
            storage_mb=storage_mb_total,
        )


def _section_5a_seedlm_k_cluster_compression(
    model_id: str,
    args: argparse.Namespace,
    n_layers: int,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 5A: Structured Pruning + SeedLM K-Cluster Recovery"
    print(
        "\n============== Section 5A: Structured Pruning + SeedLM K-Cluster Recovery ==============",
        flush=True,
    )
    for sp in sparsities:
        if sp == 0.0:
            continue
        for k in k_sizes:
            compress_timings_total: dict[str, float] = {}
            storage_mb_total: dict[str, float] = {}
            method = args.clustering_method
            seedlm_cfg_dir = _seedlm_compressed_dir(
                model_id, sparsity=sp, k_cluster=k, clustering_method=method
            )

            for layer_i in range(n_layers):
                for fc_name in ("fc1", "fc2"):
                    load_name = f"layer_{layer_i:02d}_{fc_name}_weight.npy"
                    layer_storage_path = seedlm_cfg_dir / f"layer_{layer_i:02d}_{fc_name}"

                    compress_timings, storage_bytes = SeedLM_compress(
                        sp,
                        k,
                        INIT_SEED,
                        SELECTION_POOL,
                        SELECTION_SEED,
                        C=C,
                        P=P,
                        K_bits=K_BITS,
                        clustering_method=method,
                        clustering_sample_size=args.clustering_sample_size,
                        medoid_sample_size=args.medoid_sample_size,
                        clustering_batch_size=args.clustering_batch_size,
                        klines_iterations=args.klines_iterations,
                        load_path=opt_dir,
                        load_name=load_name,
                        storage_path=layer_storage_path,
                    )

                    _accumulate_timings(compress_timings_total, compress_timings)
                    _accumulate_storage(storage_mb_total, storage_bytes)

            _finalize_total(compress_timings_total)
            _finalize_total(storage_mb_total)
            print(
                f"Sparsity = {sp:.2f},\n"
                f"SeedLM_K_cluster = {k},\n"
                f"Clustering method = {method},\n"
                f"C = {C},\n"
                f"P = {P},\n"
                f"K = {K_BITS},\n"
                f"{_format_timings('Compression time', compress_timings_total)}\n"
                f"{_format_storage('Storage', storage_mb_total)}\n",
                flush=True,
            )
            _log(
                sparsity=sp,
                SeedLM_K_cluster=k,
                clustering_method=method,
                C=str(C),
                P=str(P),
                K=str(K_BITS),
                compress_timings=compress_timings_total,
                storage_mb=storage_mb_total,
            )


def _section_5b_quantization_before_seedlm(
    model_id: str,
    args: argparse.Namespace,
    n_layers: int,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 5B: Quantization - before SeedLM"
    print("\n============== Section 5B: Quantization - before SeedLM ==============", flush=True)
    for bits in QUANTIZATION_BITS:
        quant_dir = _quantized_model_dir(model_id, bits)

        quant_prep_timings_total: dict[str, float] = {}
        quant_prep_storage_mb_total: dict[str, float] = {}

        for layer_i in range(n_layers):
            for fc_name in ("fc1", "fc2"):
                t0 = time.perf_counter()
                w_np = np.load(opt_dir / f"layer_{layer_i:02d}_{fc_name}_weight.npy")
                q_idx = np.load(
                    quant_dir
                    / f"layer_{layer_i:02d}_{fc_name}_{bits}bits_quantization_index.npy"
                )
                lut = np.load(
                    quant_dir
                    / f"layer_{layer_i:02d}_{fc_name}_{bits}bits_quantization_LUT.npy"
                )
                _accumulate_timings(
                    quant_prep_timings_total,
                    {"load_quantization": time.perf_counter() - t0},
                )

                t0 = time.perf_counter()
                row, col = w_np.shape
                w_q = reconstruct_quantized_weight(row, col, q_idx, lut, bits)
                _accumulate_timings(
                    quant_prep_timings_total,
                    {"reconstruct_quantized_weight": time.perf_counter() - t0},
                )

                t0 = time.perf_counter()
                quant_load_name = f"layer_{layer_i:02d}_{fc_name}_quant_weight.npy"
                w_q_saved = w_q.astype(np.float16)
                np.save(quant_dir / quant_load_name, w_q_saved)
                _accumulate_timings(
                    quant_prep_timings_total,
                    {"save_quant_weight": time.perf_counter() - t0},
                )
                _accumulate_storage(
                    quant_prep_storage_mb_total,
                    {"quantization_weight": w_q_saved.nbytes},
                )

        _finalize_total(quant_prep_timings_total)
        _finalize_total(quant_prep_storage_mb_total)
        print(
            f"Quantization bits = {bits},\n"
            f"{_format_timings('Quant weight reconstruction time', quant_prep_timings_total)}\n"
            f"{_format_storage('Quant weight storage', quant_prep_storage_mb_total)}\n",
            flush=True,
        )
        _log(
            quantization_bits=bits,
            quant_prep_timings=quant_prep_timings_total,
            storage_mb=quant_prep_storage_mb_total,
        )


def _section_5b_seedlm_k_cluster_with_quantization(
    model_id: str,
    args: argparse.Namespace,
    n_layers: int,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = (
        "Section 5B: Structured Pruning + SeedLM K-Cluster Recovery with Quantization"
    )
    print(
        "\n============== Section 5B: Structured Pruning + SeedLM K-Cluster Recovery with Quantization ==============",
        flush=True,
    )
    for bits in QUANTIZATION_BITS:
        quant_dir = _quantized_model_dir(model_id, bits)

        for sp in sparsities:
            if sp == 0.0:
                continue
            for k in k_sizes:
                compress_timings_total: dict[str, float] = {}
                storage_mb_total: dict[str, float] = {}
                seedlm_store_dir = _seedlm_quantized_compressed_dir(
                    model_id, bits=bits, sparsity=sp, k_cluster=k
                )
                seedlm_store_dir.mkdir(parents=True, exist_ok=True)

                for layer_i in range(n_layers):
                    for fc_name in ("fc1", "fc2"):
                        quant_load_name = f"layer_{layer_i:02d}_{fc_name}_quant_weight.npy"
                        layer_storage_path = seedlm_store_dir / f"layer_{layer_i:02d}_{fc_name}"
                        compress_timings, storage_bytes = SeedLM_compress(
                            sp,
                            k,
                            INIT_SEED,
                            SELECTION_POOL,
                            SELECTION_SEED,
                            C=C,
                            P=P,
                            K_bits=K_BITS,
                            load_path=quant_dir,
                            load_name=quant_load_name,
                            storage_path=layer_storage_path,
                        )

                        _accumulate_timings(compress_timings_total, compress_timings)
                        _accumulate_storage(storage_mb_total, storage_bytes)

                _finalize_total(compress_timings_total)
                _finalize_total(storage_mb_total)
                print(
                    f"Sparsity = {sp:.2f},\n"
                    f"Quantization bits = {bits},\n"
                    f"SeedLM_K_cluster = {k},\n"
                    f"C = {C},\n"
                    f"P = {P},\n"
                    f"K = {K_BITS},\n"
                    f"{_format_timings('Compression time', compress_timings_total)}\n"
                    f"{_format_storage('Storage', storage_mb_total)}\n",
                    flush=True,
                )
                _log(
                    sparsity=sp,
                    quantization_bits=bits,
                    SeedLM_K_cluster=k,
                    C=str(C),
                    P=str(P),
                    K=str(K_BITS),
                    compress_timings=compress_timings_total,
                    storage_mb=storage_mb_total,
                )


def _section_6_per_tile_seedlm(
    model_id: str,
    args: argparse.Namespace,
    n_layers: int,
    sparsities: list[float],
    k_sizes: list[int],
    opt_dir: Path,
    structured_dir: Path,
    seedlm_dir: Path,
    seedlm_quantized_dir: Path,
) -> None:
    global _CURRENT_SECTION
    _CURRENT_SECTION = "Section 6: Structured Pruning + SeedLM Recovery"
    print("\n============== Section 6: Structured Pruning + SeedLM Recovery ==============", flush=True)
    print("Skipped (per-tile SeedLM recovery without K clustering not implemented yet).", flush=True)
    _log(note="Skipped (per-tile SeedLM recovery without K clustering not implemented yet).")


# main cpu benchmark here
def benchmark_cpu_compress() -> None:
    # benchmark CPU:
    # section 0: store OPT model components
    # section 1: baseline - dense
    # section 2: quantization - float16, float4, add this later
    # section 3: magnitude pruning
    # section 4: structured pruning - set column tile to 0
    # section 5A: structured pruning + SeedLM recovery with K clustering
    # section 5B: structured pruning + SeedLM recovery with K clustering and quantization
    # section 6: structured pruning + SeedLM recovery - Not Implemented yet
    # DO NOT delete above prompts

    parser = argparse.ArgumentParser(description="OPT layer-wise SeedLM compress and store.")
    parser.add_argument(
        "--model",
        default=None,
        choices=OPT_MODELS,
        help="Run a single OPT model (default: all models in OPT_MODELS).",
    )
    parser.add_argument(
        "--section",
        default=None,
        choices=["store", "s2", "s3", "s4", "s5a", "s5b_prep", "s5b", "s6", "all"],
        help="Run only one compress section (default: all).",
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
        "--k-cluster-values",
        nargs="+",
        type=int,
        default=None,
        help="Exact K values; overrides --k-cluster-fraction, e.g. 16 32 64.",
    )
    parser.add_argument(
        "--quantization-bits",
        nargs="+",
        type=int,
        default=None,
        help="Quantization bit-widths, e.g. 3 4 (default: QUANTIZATION_BITS).",
    )
    parser.add_argument(
        "--clustering-method",
        choices=["random", "kmeans++", "minibatch", "kmedoids", "klines"],
        default="random",
        help="Center-selection method used by Section 5A (default: random).",
    )
    parser.add_argument("--clustering-sample-size", type=int, default=50000)
    parser.add_argument("--medoid-sample-size", type=int, default=5000)
    parser.add_argument("--clustering-batch-size", type=int, default=4096)
    parser.add_argument("--klines-iterations", type=int, default=10)
    parser.add_argument(
        "--server-temp",
        action="store_true",
        help="Write JSON under results/server_temp/CPU_compress_result/{model}/.",
    )
    parser.add_argument(
        "--job-tag",
        default=None,
        help="Filename stem for server-temp JSON (required with --server-temp).",
    )
    parser.add_argument("--device", default=None, help="Unused for compress (CPU). Kept for CLI compatibility.")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-ppl-samples", type=int, default=None, help="Unused for compress.")
    args = parser.parse_args()

    global QUANTIZATION_BITS

    section = args.section or "all"
    sparsities = args.sparsity if args.sparsity is not None else list(DEFAULT_SPARSITIES)
    quant_bits = (
        args.quantization_bits if args.quantization_bits is not None else list(QUANTIZATION_BITS)
    )
    models_to_run = [args.model] if args.model else list(OPT_MODELS)
    fracs = args.k_cluster_fraction if args.k_cluster_fraction is not None else list(K_CLUSTER_TILE_FRACTION)

    # Temporarily restrict global QUANTIZATION_BITS for section helpers that close over it.
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

            if section in ("store", "all"):
                print(f"Loading {model_id} for component export ...", flush=True)
                export_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                opt_dir, n_layers = _section_1_store_opt_components(export_model, model_id, opt_dir)
                del export_model
            else:
                if not opt_dir.exists():
                    raise FileNotFoundError(
                        f"OPT store dir missing: {opt_dir}. Run --section store first."
                    )
                n_layers = _count_layers_from_opt_dir(opt_dir)

            k_sizes = (
                args.k_cluster_values
                if args.k_cluster_values is not None
                else _k_sizes_from_opt_dir(opt_dir, fracs)
            )

            section_kwargs = dict(
                model_id=model_id,
                args=args,
                n_layers=n_layers,
                sparsities=sparsities,
                k_sizes=k_sizes,
                opt_dir=opt_dir,
                structured_dir=structured_dir,
                seedlm_dir=seedlm_dir,
                seedlm_quantized_dir=seedlm_quantized_dir,
            )

            if section in ("s2", "all"):
                _section_2_quantization_compression(**section_kwargs)
            if section in ("s3", "all"):
                _section_3_magnitude_pruning_compression(**section_kwargs)
            if section in ("s4", "all"):
                _section_4_structured_pruning_compression(**section_kwargs)
            if section in ("s5a", "all"):
                _section_5a_seedlm_k_cluster_compression(**section_kwargs)
            if section in ("s5b_prep", "all"):
                _section_5b_quantization_before_seedlm(**section_kwargs)
            if section in ("s5b", "all"):
                if section == "s5b":
                    # Single-job s5b assumes prep already done (or run prep first on local all).
                    pass
                _section_5b_seedlm_k_cluster_with_quantization(**section_kwargs)
            if section in ("s6", "all"):
                _section_6_per_tile_seedlm(**section_kwargs)
    finally:
        QUANTIZATION_BITS = _quant_bits_backup


if __name__ == "__main__":
    benchmark_cpu_compress()
