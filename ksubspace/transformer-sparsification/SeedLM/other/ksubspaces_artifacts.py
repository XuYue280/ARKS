#!/usr/bin/env python3
"""Actual on-disk K-subspaces artifacts for SeedLM Cx1 tiles."""

from __future__ import annotations

import json
import os
import resource
import threading
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _pruned_col_matrix_indices_from_sparsity(
    num_col_matrices: int,
    sparsity: float,
    selection_pool: int = 100,
    selection_seed: int = 42,
) -> np.ndarray:
    """Same deterministic structured-pruning selection used by SeedLM."""
    if selection_pool <= 0:
        raise ValueError("selection_pool must be positive")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    rng = np.random.default_rng(selection_seed)
    full_pools, remainder = divmod(num_col_matrices, selection_pool)
    count_full = min(
        selection_pool, max(0, int(round(selection_pool * sparsity)))
    )
    selected: list[np.ndarray] = []
    for pool_index in range(full_pools):
        if count_full:
            base = pool_index * selection_pool
            selected.append(
                base
                + rng.choice(selection_pool, size=count_full, replace=False)
            )
    if remainder:
        count_remainder = min(
            remainder, max(0, int(round(remainder * sparsity)))
        )
        if count_remainder:
            selected.append(
                full_pools * selection_pool
                + rng.choice(remainder, size=count_remainder, replace=False)
            )
    return (
        np.concatenate(selected).astype(np.int64)
        if selected
        else np.empty(0, dtype=np.int64)
    )


def _weight_col_matrices_flat(
    weight: np.ndarray,
    C: int,
    dtype: np.dtype = np.float16,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Convert [input, output] weights into SeedLM Cx1 tile order."""
    rows, columns = weight.shape
    if rows % C:
        raise ValueError(f"Weight rows={rows} must be divisible by C={C}")
    blocks_per_column = rows // C
    tile_count = blocks_per_column * columns
    tiles = (
        weight.reshape(blocks_per_column, C, columns)
        .transpose(2, 0, 1)
        .reshape(tile_count, C)
        .astype(dtype, copy=False)
    )
    reconstruction_order = (
        (np.arange(tile_count) % columns) * blocks_per_column
        + (np.arange(tile_count) // columns)
    )
    return tile_count, tiles, reconstruction_order


class PeakRSS:
    """Sample process RSS in a background thread."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.start_rss = _current_rss_bytes()
        self.peak_rss = self.start_rss

        def sample():
            while not self._stop.is_set():
                self.peak_rss = max(self.peak_rss, _current_rss_bytes())
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_rss = max(self.peak_rss, _current_rss_bytes())

    @property
    def peak_mb(self) -> float:
        return self.peak_rss / 1024**2

    @property
    def increase_mb(self) -> float:
        return max(0, self.peak_rss - self.start_rss) / 1024**2


def _current_rss_bytes() -> int:
    """Current resident memory without a third-party dependency."""
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        return int(peak * 1024 if peak < 10**10 else peak)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _initialize(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=3,
        batch_size=min(4096, len(x)),
        max_iter=100,
        reassignment_ratio=0.0,
        random_state=seed,
    )
    return model.fit_predict(_unit_rows(x)).astype(np.int32)


def _update(
    x: np.ndarray,
    labels: np.ndarray,
    k: int,
    rank: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    c = x.shape[1]
    bases = np.empty((k, c, rank), dtype=np.float32)
    empty = 0
    for cluster_id in range(k):
        members = x[labels == cluster_id].astype(np.float64, copy=False)
        if len(members) == 0:
            empty += 1
            q, _ = np.linalg.qr(rng.normal(size=(c, rank)))
            bases[cluster_id] = q[:, :rank]
            continue
        _, _, vh = np.linalg.svd(members, full_matrices=False)
        available = min(rank, len(vh))
        candidate = vh[:available].T
        if available < rank:
            candidate = np.concatenate(
                [candidate, rng.normal(size=(c, rank - available))], axis=1
            )
        q, _ = np.linalg.qr(candidate)
        bases[cluster_id] = q[:, :rank].astype(np.float32)
    return bases, empty


def _assign(
    x: np.ndarray,
    bases: np.ndarray,
    batch_size: int,
    return_coefficients: bool,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    labels = np.empty(len(x), dtype=np.int32)
    coefficients = (
        np.empty((len(x), bases.shape[2]), dtype=np.float32)
        if return_coefficients
        else None
    )
    residual_sum = 0.0
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].astype(np.float32, copy=False)
        projection = np.einsum("bd,kdr->bkr", xb, bases, optimize=True)
        captured = np.sum(projection * projection, axis=2)
        chosen = np.argmax(captured, axis=1)
        labels[start : start + len(xb)] = chosen
        if coefficients is not None:
            coefficients[start : start + len(xb)] = projection[
                np.arange(len(xb)), chosen
            ]
        residual = np.sum(xb * xb, axis=1) - captured[np.arange(len(xb)), chosen]
        residual_sum += np.maximum(residual, 0).sum(dtype=np.float64)
    return labels, coefficients, residual_sum


def fit_subspaces(
    tiles: np.ndarray,
    k: int,
    rank: int,
    sample_size: int,
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    if sample_size > 0 and len(tiles) > sample_size:
        train = tiles[rng.choice(len(tiles), sample_size, replace=False)]
    else:
        train = tiles
    labels = _initialize(train, k, seed)
    empty = 0
    completed = 0
    for iteration in range(iterations):
        bases, empty = _update(train, labels, k, rank, rng)
        new_labels, _, _ = _assign(train, bases, batch_size, False)
        completed = iteration + 1
        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels
    bases, empty = _update(train, labels, k, rank, rng)
    return bases, {
        "training_sample": len(train),
        "iterations": completed,
        "empty_training_clusters": empty,
    }


def _id_dtype(k: int):
    if k <= np.iinfo(np.int16).max:
        return np.int16
    return np.int32


def _quantize_coefficients(
    coefficients: np.ndarray,
    bits: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Encode coefficients as FP16, symmetric INT8, or packed symmetric INT4."""
    if bits == 16:
        return {
            "tile_coefficients.npy": coefficients.astype(np.float16)
        }, {
            "coefficient_shape": list(coefficients.shape),
            "coefficient_quantization": "fp16",
        }
    if bits not in (4, 8):
        raise ValueError("coefficient_bits must be one of 4, 8, or 16")

    qmax = (1 << (bits - 1)) - 1
    maximum = np.max(np.abs(coefficients), axis=0).astype(np.float32)
    scales = np.where(maximum > 0, maximum / qmax, 1.0).astype(np.float32)
    quantized = np.clip(
        np.rint(coefficients / scales),
        -qmax,
        qmax,
    ).astype(np.int8)
    arrays = {"coefficient_scales.npy": scales}
    metadata = {
        "coefficient_shape": list(coefficients.shape),
        "coefficient_quantization": f"symmetric_int{bits}_per_rank",
    }
    if bits == 8:
        arrays["tile_coefficients.npy"] = quantized
        return arrays, metadata

    # Store two signed 4-bit values per byte. Values are shifted from
    # [-7, 7] to [1, 15]; zero remains represented exactly by nibble 8.
    nibbles = (quantized.reshape(-1).astype(np.int16) + 8).astype(np.uint8)
    if len(nibbles) % 2:
        nibbles = np.pad(nibbles, (0, 1), constant_values=8)
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    arrays["tile_coefficients.npy"] = packed
    metadata["coefficient_packed_values"] = int(coefficients.size)
    return arrays, metadata


def _load_coefficients(artifact_dir: Path, metadata: dict) -> np.ndarray:
    bits = int(metadata.get("coefficient_bits", 16))
    stored = np.load(artifact_dir / "tile_coefficients.npy")
    if bits == 16:
        return stored.astype(np.float32)
    if bits not in (4, 8):
        raise ValueError(f"Unsupported coefficient_bits={bits}")

    scales = np.load(
        artifact_dir / "coefficient_scales.npy"
    ).astype(np.float32)
    shape = tuple(int(value) for value in metadata["coefficient_shape"])
    if bits == 8:
        quantized = stored.astype(np.int8, copy=False).reshape(shape)
    else:
        packed = stored.astype(np.uint8, copy=False).reshape(-1)
        nibbles = np.empty(packed.size * 2, dtype=np.uint8)
        nibbles[0::2] = packed & 0x0F
        nibbles[1::2] = packed >> 4
        value_count = int(np.prod(shape))
        quantized = (
            nibbles[:value_count].astype(np.int16) - 8
        ).astype(np.int8).reshape(shape)
    return quantized.astype(np.float32) * scales


def compress_weight(
    original_weight_out_in: np.ndarray,
    output_dir: Path,
    *,
    sparsity: float,
    k: int,
    rank: int,
    C: int = 8,
    sample_size: int = 0,
    iterations: int = 15,
    batch_size: int = 2048,
    seed: int = 42,
    selection_pool: int = 100,
    selection_seed: int = 42,
    coefficient_bits: int = 16,
) -> dict:
    started = time.perf_counter()
    W = original_weight_out_in.T.astype(np.float32, copy=False)
    rows, columns = W.shape
    if rows % C:
        raise ValueError(f"Input dimension {rows} must be divisible by C={C}")

    n_tiles, all_tiles, cm_idx = _weight_col_matrices_flat(W, C, np.float32)
    pruned_ids = _pruned_col_matrix_indices_from_sparsity(
        n_tiles, sparsity, selection_pool, selection_seed
    )
    pruned_mask_cm = np.zeros(n_tiles, dtype=bool)
    pruned_mask_cm[pruned_ids] = True
    tiles = all_tiles[pruned_ids]

    fit_started = time.perf_counter()
    bases, fit_info = fit_subspaces(
        tiles, k, rank, sample_size, iterations, batch_size, seed
    )
    fit_seconds = time.perf_counter() - fit_started

    assign_started = time.perf_counter()
    ids_pruned, coefficients_pruned, residual_sum = _assign(
        tiles, bases, batch_size, True
    )
    assign_seconds = time.perf_counter() - assign_started

    ids_cm = np.full(n_tiles, -1, dtype=np.int32)
    ids_cm[pruned_ids] = ids_pruned
    coefficients_cm = np.zeros((n_tiles, rank), dtype=np.float32)
    coefficients_cm[pruned_ids] = coefficients_pruned

    # Convert to reconstruction traversal: row block first, then column.
    ids_ordered = ids_cm[cm_idx]
    coefficients_ordered = coefficients_cm[cm_idx]
    ordered_pruned = ids_ordered >= 0
    compact = all_tiles[cm_idx[~ordered_pruned]].astype(np.float16)
    saved_coefficients = coefficients_ordered[ordered_pruned]
    coefficient_arrays, coefficient_metadata = _quantize_coefficients(
        saved_coefficients, coefficient_bits
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "structured_pruning.npy": compact,
        "subspace_bases.npy": bases.astype(np.float16),
        "nearest_subspace_ids.npy": ids_ordered.astype(_id_dtype(k)),
        **coefficient_arrays,
    }
    for name, array in arrays.items():
        np.save(output_dir / name, array)

    total_energy = float(np.sum(tiles.astype(np.float64) ** 2))
    metadata = {
        "format": "seedlm_k_subspaces_v2",
        "rows": rows,
        "columns": columns,
        "C": C,
        "sparsity": sparsity,
        "k": k,
        "rank": rank,
        "coefficient_bits": coefficient_bits,
        **coefficient_metadata,
        "selection_pool": selection_pool,
        "selection_seed": selection_seed,
        "n_tiles": n_tiles,
        "n_recovered_tiles": len(tiles),
        "training_sample": fit_info["training_sample"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    file_bytes = {
        name: (output_dir / name).stat().st_size for name in [*arrays, "metadata.json"]
    }
    return {
        **metadata,
        **fit_info,
        "fit_seconds": fit_seconds,
        "assignment_seconds": assign_seconds,
        "total_compression_seconds": time.perf_counter() - started,
        "recovered_tile_mse": residual_sum / tiles.size,
        "recovered_tile_relative_error": np.sqrt(
            residual_sum / max(total_energy, 1e-30)
        ),
        "artifact_bytes": int(sum(file_bytes.values())),
        "artifact_files": file_bytes,
    }


def load_weight(artifact_dir: Path) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    arrays_started = time.perf_counter()
    compact = np.load(artifact_dir / "structured_pruning.npy")
    bases = np.load(artifact_dir / "subspace_bases.npy").astype(np.float32)
    ids = np.load(artifact_dir / "nearest_subspace_ids.npy").astype(np.int32)
    coefficients = _load_coefficients(artifact_dir, metadata)
    load_seconds = time.perf_counter() - arrays_started

    rows = int(metadata["rows"])
    columns = int(metadata["columns"])
    C = int(metadata["C"])
    n_tiles = rows // C * columns
    if len(ids) != n_tiles:
        raise ValueError(f"ID count {len(ids)} does not match expected {n_tiles}")

    reconstruction_started = time.perf_counter()
    tiles = np.empty((n_tiles, C), dtype=np.float16)
    recovered = ids >= 0
    if int(recovered.sum()) != len(coefficients):
        raise ValueError("Coefficient count does not match recovered tile count")
    tiles[~recovered] = compact
    selected_bases = bases[ids[recovered]]
    reconstructed = np.einsum(
        "ncr,nr->nc", selected_bases, coefficients, optimize=True
    )
    tiles[recovered] = reconstructed.astype(np.float16)
    W = (
        tiles.reshape(rows // C, columns, C)
        .transpose(0, 2, 1)
        .reshape(rows, columns)
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    file_bytes = sum(path.stat().st_size for path in artifact_dir.iterdir() if path.is_file())
    return W, {
        "load_seconds": load_seconds,
        "reconstruction_seconds": reconstruction_seconds,
        "total_load_seconds": time.perf_counter() - started,
        "artifact_bytes": file_bytes,
    }
