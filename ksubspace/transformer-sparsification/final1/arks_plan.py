#!/usr/bin/env python3
"""Choose execution settings for one ARKS run from hardware and model facts.

Only *execution* knobs are chosen automatically: worker processes, BLAS
threads, logit chunking, where the dense model is staged during codebook
calibration, and whether the streaming block-wise calibration is used. All
of these are result-preserving up to floating-point reduction order. Knobs
that define the experiment (sample counts, sequence lengths, epochs, ranks,
codebook sizes) are never changed here; when they do not fit the detected
hardware the plan reports a warning and, for hard limits, an error.

The plan is printed as ``KEY=VALUE`` lines for the Bash entry point and also
stored as JSON with the reason for every choice.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GIB = 1024**3


def fmt_bytes(value: float) -> str:
    return f"{value / GIB:.2f} GiB"


def choose(args, hardware: dict, model: dict) -> dict:
    reasons: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []
    plan: dict[str, object] = {}

    cores = int(hardware.get("cpu_usable") or hardware.get("cpu_logical") or 1)
    ram_total = int(hardware.get("ram_total_bytes") or 0)
    ram_avail = int(hardware.get("ram_available_bytes") or ram_total)
    gpus = hardware.get("gpus") or []
    gpu = gpus[0] if gpus else None
    vram_total = int(gpu["vram_total_bytes"]) if gpu else 0
    vram_free = int(gpu["vram_free_bytes"]) if gpu else 0

    layers = int(model["num_layers"])
    hidden = int(model["hidden_size"])
    vocab = int(model["vocab_size"])
    matrices = int(model["expected_matrices"])
    largest = int(model["largest_matrix_elements"])
    per_layer_elements = sum(int(m["elements"]) for m in model["matrix_shapes"])
    decoder_elements = int(model["decoder_linear_elements"])
    embedding_elements = vocab * hidden * (1 if model.get("tie_word_embeddings", True) else 2)
    model_params = decoder_elements + embedding_elements
    model_bytes_fp16 = model_params * 2
    plan["ESTIMATED_PARAMS"] = model_params
    plan["ESTIMATED_MODEL_FP16_BYTES"] = model_bytes_fp16

    # ---------------------------------------------------------------- device
    if args.device in ("cuda", "cpu"):
        device = args.device
        reasons["DEVICE"] = "user override"
    elif gpu is not None:
        device = "cuda"
        reasons["DEVICE"] = f"CUDA device detected: {gpu['name']} ({fmt_bytes(vram_total)})"
    else:
        device = "cpu"
        reasons["DEVICE"] = "no CUDA device detected; GPU stages run on the CPU in float32"
        warnings.append("No GPU detected. Allocation, calibration and inference will be very slow on CPU.")
    plan["DEVICE"] = device

    # --------------------------------------------------------- CPU workers
    # Per-worker host memory for compression: several float32 copies of the
    # matrix being compressed, the float64 energy temporary, the K-subspaces
    # training sample and the coefficient VQ arrays, plus interpreter overhead.
    per_worker_bytes = 32 * largest + 512 * 1024**2
    # The K-subspaces fit and the coefficient k-means are bit-for-bit
    # reproducible for a fixed BLAS/OpenMP thread count and independent of
    # the number of worker processes. One thread per worker therefore gives
    # results that do not depend on the machine's core count, while
    # "legacy" (library default threads, one process) reproduces runs made
    # with the original sequential scripts on the same machine.
    legacy_threads = args.threads_per_worker == 0 and args.legacy_threads
    if args.threads_per_worker > 0:
        threads = args.threads_per_worker
        reasons["THREADS_PER_WORKER"] = "user override"
    elif legacy_threads:
        threads = 0
        reasons["THREADS_PER_WORKER"] = (
            "legacy mode: library default thread count (reproduces the original "
            "sequential scripts bit-for-bit on this machine)"
        )
    else:
        threads = 1
        reasons["THREADS_PER_WORKER"] = (
            "single-threaded BLAS per worker keeps compression results independent "
            "of the core count; parallelism comes from worker processes"
        )
    if args.workers > 0:
        workers = args.workers
        reasons["WORKERS"] = "user override"
    elif legacy_threads:
        workers = 1
        reasons["WORKERS"] = "legacy thread mode oversubscribes the CPU with more than one process"
    else:
        by_cores = max(1, (cores - 2) // max(threads, 1))
        by_ram = max(1, int(0.7 * ram_avail / per_worker_bytes)) if ram_avail else by_cores
        workers = max(1, min(matrices, by_cores, by_ram, 48))
        reasons["WORKERS"] = (
            f"min(matrices={matrices}, (cores-2)/threads={by_cores}, "
            f"0.7*available_ram/{fmt_bytes(per_worker_bytes)} per worker={by_ram}, 48)"
        )
    plan["WORKERS"] = workers
    plan["THREADS_PER_WORKER"] = threads
    plan["COMPRESSION_RAM_ESTIMATE_BYTES"] = workers * per_worker_bytes
    if ram_avail and workers * per_worker_bytes > 0.9 * ram_avail:
        warnings.append(
            f"Compression may exceed available RAM: {workers} workers x "
            f"{fmt_bytes(per_worker_bytes)} vs {fmt_bytes(ram_avail)} available."
        )

    # ------------------------------------------------ logits / inference
    seqlen = int(args.inference_seqlen)
    logits_bytes = seqlen * vocab * 2
    # Full-sequence path keeps roughly the logits, a log-softmax copy and the
    # float32 upcast alive; activations add about 30% of the model size.
    inference_need = model_bytes_fp16 + 4 * logits_bytes + int(0.3 * model_bytes_fp16) + 512 * 1024**2
    plan["INFERENCE_VRAM_ESTIMATE_BYTES"] = inference_need
    if args.loss_chunk_tokens not in ("auto", None):
        chunk = int(args.loss_chunk_tokens)
        reasons["LOSS_CHUNK_TOKENS"] = "user override"
    elif device == "cuda" and inference_need > 0.9 * vram_free:
        chunk = 512
        reasons["LOSS_CHUNK_TOKENS"] = (
            f"full-logit evaluation needs about {fmt_bytes(inference_need)} but "
            f"{fmt_bytes(vram_free)} VRAM is free; chunking bounds logits to 512 positions"
        )
    else:
        chunk = 0
        reasons["LOSS_CHUNK_TOKENS"] = (
            f"full-logit evaluation ({fmt_bytes(inference_need)}) fits in "
            f"{fmt_bytes(vram_free) if device == 'cuda' else 'host memory'}; original single pass kept"
        )
    plan["LOSS_CHUNK_TOKENS"] = chunk
    layerwise = getattr(args, "model_parallel", "none") == "layerwise"
    plan["MODEL_PARALLEL"] = "layerwise" if layerwise else "none"
    if device == "cuda" and model_bytes_fp16 * 1.15 > vram_total and not layerwise:
        errors.append(
            f"The fp16 model ({fmt_bytes(model_bytes_fp16)}) does not fit in "
            f"{fmt_bytes(vram_total)} of VRAM for full-model perplexity evaluation. "
            "Use a larger GPU, --model-parallel layerwise, or --device cpu."
        )
    elif device == "cuda" and layerwise:
        reasons["MODEL_PARALLEL"] = (
            f"layerwise: the fp16 model ({fmt_bytes(model_bytes_fp16)}) is streamed "
            f"one decoder layer at a time, so perplexity needs about "
            f"{fmt_bytes(model_bytes_fp16 // max(1, layers) + 4 * logits_bytes)} "
            f"instead of {fmt_bytes(inference_need)}"
        )

    # ------------------------------------------- rank allocator memory
    largest_layer_grad = 2 * per_layer_elements
    allocator_need = model_bytes_fp16 + largest_layer_grad + 2 * args.rank_calibration_seqlen * vocab * 4 + 512 * 1024**2
    plan["ALLOCATOR_VRAM_ESTIMATE_BYTES"] = allocator_need
    # layerwise implies the model does not fit one card, so the allocator is
    # sharded too and the budget is every GPU's VRAM, not just gpus[0]'s.
    allocator_vram = vram_free * max(1, len(gpus)) if layerwise else vram_free
    if device == "cuda" and allocator_need > 0.95 * allocator_vram:
        warnings.append(
            f"Rank allocation may exceed VRAM even with gradient offload "
            f"({fmt_bytes(allocator_need)} needed, {fmt_bytes(allocator_vram)} free"
            f"{' across ' + str(len(gpus)) + ' GPUs' if layerwise else ''})."
        )

    # ------------------------------------ codebook calibration staging
    cal_samples = int(args.calibration_samples)
    cal_seqlen = int(args.calibration_seqlen)
    capture_bytes = cal_samples * cal_seqlen * hidden * 2
    plan["CALIBRATION_CAPTURE_RAM_BYTES"] = 2 * capture_bytes  # inputs + targets
    if ram_avail and 2 * capture_bytes > 0.6 * ram_avail:
        warnings.append(
            f"Calibration captures need about {fmt_bytes(2 * capture_bytes)} of host RAM "
            f"({cal_samples} x {cal_seqlen} x {hidden} fp16, inputs and targets) but only "
            f"{fmt_bytes(ram_avail)} is available. Reduce --calibration-samples or "
            "--calibration-seqlen explicitly if the run is killed."
        )
    # Per-block working set on the GPU: reconstructed weights (fp32 + fp16),
    # the basis gather kept by autograd (rank x fp32 per element), gradients
    # of the codebooks, and one sequence of activations.
    max_rank = int(args.max_rank)
    block_working = per_layer_elements * (4 + 2 + 4 * max_rank) + 8 * cal_seqlen * hidden * 4 * 4
    plan["CALIBRATION_BLOCK_VRAM_ESTIMATE_BYTES"] = block_working
    if args.model_staging in ("cuda", "cpu"):
        staging = args.model_staging
        reasons["MODEL_STAGING"] = "user override"
    elif device != "cuda":
        staging = "cpu"
        reasons["MODEL_STAGING"] = "CPU device"
    elif model_bytes_fp16 + block_working + 512 * 1024**2 <= 0.9 * vram_free:
        staging = "cuda"
        reasons["MODEL_STAGING"] = (
            f"dense model ({fmt_bytes(model_bytes_fp16)}) plus block working set "
            f"({fmt_bytes(block_working)}) fits in {fmt_bytes(vram_free)} free VRAM"
        )
    else:
        staging = "cpu"
        reasons["MODEL_STAGING"] = (
            f"dense model ({fmt_bytes(model_bytes_fp16)}) plus block working set "
            f"({fmt_bytes(block_working)}) exceeds {fmt_bytes(vram_free)} free VRAM; "
            "the model stays in host memory and one block at a time is moved to the GPU"
        )
    plan["MODEL_STAGING"] = staging
    if args.calibration_streaming in ("on", "off"):
        streaming = args.calibration_streaming
        reasons["CALIBRATION_STREAMING"] = "user override"
    else:
        streaming = "on"
        reasons["CALIBRATION_STREAMING"] = (
            "streaming propagates each block's output to the next block once per "
            "sample (linear in depth) instead of re-running the model prefix for "
            f"every block (quadratic); {layers} layers"
        )
    plan["CALIBRATION_STREAMING"] = streaming
    if staging == "cpu" and streaming == "off":
        errors.append(
            "MODEL_STAGING=cpu requires CALIBRATION_STREAMING=on; the non-streaming "
            "calibration keeps the whole model on the GPU."
        )
    if device == "cuda" and block_working + 512 * 1024**2 > 0.95 * vram_free:
        warnings.append(
            f"A single calibration block needs about {fmt_bytes(block_working)} of VRAM; "
            f"only {fmt_bytes(vram_free)} is free."
        )

    plan["REASONS"] = reasons
    plan["WARNINGS"] = warnings
    plan["ERRORS"] = errors
    plan["HARDWARE_SUMMARY"] = {
        "cpu_usable": cores,
        "ram_total": fmt_bytes(ram_total),
        "ram_available": fmt_bytes(ram_avail),
        "gpu": gpu["name"] if gpu else None,
        "vram_total": fmt_bytes(vram_total) if gpu else None,
        "vram_free": fmt_bytes(vram_free) if gpu else None,
    }
    plan["MODEL_SUMMARY"] = {
        "model": model["model_id"],
        "family": model["family"],
        "layers": layers,
        "hidden": hidden,
        "vocab": vocab,
        "matrices": matrices,
        "estimated_params": model_params,
        "fp16_model": fmt_bytes(model_bytes_fp16),
    }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-json", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads-per-worker", type=int, default=0)
    parser.add_argument(
        "--legacy-threads",
        action="store_true",
        help="Use the BLAS library's default thread count in one process.",
    )
    parser.add_argument("--loss-chunk-tokens", default="auto")
    parser.add_argument("--model-staging", default="auto")
    parser.add_argument("--calibration-streaming", default="auto")
    parser.add_argument("--model-parallel", default="none")
    parser.add_argument("--inference-seqlen", type=int, default=2048)
    parser.add_argument("--rank-calibration-seqlen", type=int, default=256)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--max-rank", type=int, default=6)
    args = parser.parse_args()

    hardware = json.loads(args.hardware_json.read_text(encoding="utf-8"))
    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    plan = choose(args, hardware, model)
    args.plan_json.parent.mkdir(parents=True, exist_ok=True)
    args.plan_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    for key in (
        "DEVICE",
        "WORKERS",
        "THREADS_PER_WORKER",
        "LOSS_CHUNK_TOKENS",
        "MODEL_STAGING",
        "CALIBRATION_STREAMING",
    ):
        print(f"{key}={plan[key]}")
    print(f"PLAN_WARNINGS={len(plan['WARNINGS'])}")
    print(f"PLAN_ERRORS={len(plan['ERRORS'])}")


if __name__ == "__main__":
    main()
