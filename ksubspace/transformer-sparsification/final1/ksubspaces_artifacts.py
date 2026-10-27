#!/usr/bin/env python3


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


def _weight_tiles_flat(
    weight: np.ndarray,
    C: int,
    tile_width: int = 1,
    dtype: np.dtype = np.float16,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Convert [input, output] weights into column-major C x tile_width tiles.

    Tiles are returned as flattened vectors of length C * tile_width. The
    returned reconstruction order converts column-block-major tile order into
    row-block-major order, which is used when rebuilding the full matrix.
    """
    if C <= 0:
        raise ValueError("C must be positive")
    if tile_width <= 0:
        raise ValueError("tile_width must be positive")
    rows, columns = weight.shape
    if rows % C:
        raise ValueError(f"Weight rows={rows} must be divisible by C={C}")
    if columns % tile_width:
        raise ValueError(
            f"Weight columns={columns} must be divisible by "
            f"tile_width={tile_width}"
        )
    row_blocks = rows // C
    column_blocks = columns // tile_width
    tile_count = row_blocks * column_blocks
    tiles = (
        weight.reshape(row_blocks, C, column_blocks, tile_width)
        .transpose(2, 0, 1, 3)
        .reshape(tile_count, C * tile_width)
        .astype(dtype, copy=False)
    )
    reconstruction_order = (
        (np.arange(tile_count) % column_blocks) * row_blocks
        + (np.arange(tile_count) // column_blocks)
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
    # Group member rows once with a stable sort instead of scanning the whole
    # label array K times. Within each cluster the rows keep their original
    # increasing order, so ``members`` is identical to ``x[labels == id]``.
    order = np.argsort(labels, kind="stable")
    boundaries = np.searchsorted(labels[order], np.arange(k + 1))
    for cluster_id in range(k):
        member_rows = order[boundaries[cluster_id] : boundaries[cluster_id + 1]]
        members = x[member_rows].astype(np.float64, copy=False)
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
    if len(train) < k:
        raise ValueError(
            f"K={k} requires at least {k} training tiles, but only "
            f"{len(train)} are available. Increase --sample-size or reduce K."
        )
    if rank < 1 or rank > train.shape[1]:
        raise ValueError(
            f"rank={rank} must be between 1 and flattened tile dimension "
            f"{train.shape[1]}"
        )
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


def _pack_unsigned(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack fixed-width unsigned integers into an LSB-first byte stream."""
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    if bits < 1 or bits > 32:
        raise ValueError("bits must be between 1 and 32")
    limit = 1 << bits
    if values.size and np.any(values >= limit):
        raise ValueError(f"A value does not fit in {bits} bits")

    starts = np.arange(values.size, dtype=np.uint64) * bits
    byte_indices = (starts // 8).astype(np.int64)
    offsets = starts % 8
    words = values << offsets
    packed = np.zeros((values.size * bits + 7) // 8, dtype=np.uint8)
    byte_count = (bits + 14) // 8
    for byte_offset in range(byte_count):
        indices = byte_indices + byte_offset
        valid = indices < packed.size
        np.bitwise_or.at(
            packed,
            indices[valid],
            ((words[valid] >> (8 * byte_offset)) & 0xFF).astype(np.uint8),
        )
    return packed


def _unpack_unsigned(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    """Inverse of _pack_unsigned."""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    starts = np.arange(count, dtype=np.uint64) * bits
    byte_indices = (starts // 8).astype(np.int64)
    offsets = starts % 8
    words = np.zeros(count, dtype=np.uint64)
    byte_count = (bits + 14) // 8
    for byte_offset in range(byte_count):
        indices = byte_indices + byte_offset
        valid = indices < packed.size
        words[valid] |= packed[indices[valid]].astype(np.uint64) << (
            8 * byte_offset
        )
    mask = (1 << bits) - 1
    return ((words >> offsets) & mask).astype(np.int32)


def _encode_subspace_ids(ids: np.ndarray, k: int) -> tuple[dict[str, np.ndarray], dict]:
    """Store only recovered IDs and use a one-bit mask for partial coverage."""
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    recovered = ids >= 0
    recovered_ids = ids[recovered]
    if recovered_ids.size and int(recovered_ids.max()) >= k:
        raise ValueError("Subspace ID is outside [0, K)")
    bits = max(1, int(np.ceil(np.log2(k))))
    arrays = {
        "nearest_subspace_ids.npy": _pack_unsigned(recovered_ids, bits),
    }
    mask_stored = not bool(np.all(recovered))
    if mask_stored:
        arrays["recovered_tile_mask.npy"] = np.packbits(
            recovered, bitorder="little"
        )
    metadata = {
        "subspace_id_encoding": "packed_fixed_width_v1",
        "subspace_id_bits": bits,
        "subspace_id_count": int(recovered_ids.size),
        "recovered_mask_stored": mask_stored,
    }
    return arrays, metadata


def _load_subspace_ids(artifact_dir: Path, metadata: dict) -> np.ndarray:
    """Load packed-v1 IDs, with backward compatibility for legacy artifacts."""
    stored = np.load(artifact_dir / "nearest_subspace_ids.npy")
    if metadata.get("subspace_id_encoding") != "packed_fixed_width_v1":
        return stored.astype(np.int32)

    n_tiles = int(metadata["n_tiles"])
    count = int(metadata["subspace_id_count"])
    bits = int(metadata["subspace_id_bits"])
    recovered_ids = _unpack_unsigned(stored, count, bits)
    if bool(metadata.get("recovered_mask_stored", False)):
        mask_bytes = np.load(artifact_dir / "recovered_tile_mask.npy")
        recovered = np.unpackbits(mask_bytes, bitorder="little")[:n_tiles].astype(bool)
    else:
        recovered = np.ones(n_tiles, dtype=bool)
    if int(recovered.sum()) != count:
        raise ValueError("Recovered mask and packed ID count disagree")
    ids = np.full(n_tiles, -1, dtype=np.int32)
    ids[recovered] = recovered_ids
    return ids


def _quantize_coefficients(
    coefficients: np.ndarray,
    bits: int,
    *,
    encoding: str = "scalar",
    codebook_size: int = 256,
    sample_size: int = 50000,
    stages: int = 1,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict]:
    """Encode coefficients with scalar quantization or vector K-means."""
    if encoding == "vq":
        if stages < 1:
            raise ValueError("coefficient_vq_stages must be at least 1")
        if bits != int(np.ceil(np.log2(codebook_size))):
            raise ValueError(
                "For coefficient VQ, coefficient_bits must equal "
                "ceil(log2(coefficient_codebook_size))"
            )
        if codebook_size < 2 or codebook_size > len(coefficients):
            raise ValueError(
                f"coefficient_codebook_size={codebook_size} requires at least "
                f"that many coefficient vectors; found {len(coefficients)}"
            )
        rng = np.random.default_rng(seed)
        if sample_size > 0 and len(coefficients) > sample_size:
            train = coefficients[
                rng.choice(len(coefficients), sample_size, replace=False)
            ]
        else:
            train = coefficients
        residual_train = train.astype(np.float32, copy=True)
        residual_all = coefficients.astype(np.float32, copy=True)
        codebooks = []
        all_code_ids = np.empty((len(coefficients), stages), dtype=np.int32)
        reconstructed = np.zeros_like(coefficients, dtype=np.float32)
        prediction_batch = 65536
        stage_mse = []
        for stage in range(stages):
            # K-means++ becomes the dominant cost for very large codebooks
            # such as 2^16. Initialize those codebooks by sampling coefficient
            # vectors directly, then refine them with MiniBatchKMeans.
            if codebook_size > 4096:
                init_indices = rng.choice(
                    len(residual_train), size=codebook_size, replace=False
                )
                init = residual_train[init_indices].astype(
                    np.float32, copy=True
                )
                n_init = 1
                max_iter = 30
                minibatch_size = min(
                    len(residual_train), max(4096, codebook_size)
                )
            else:
                init = "k-means++"
                n_init = 3
                max_iter = 100
                minibatch_size = min(4096, len(residual_train))
            model = MiniBatchKMeans(
                n_clusters=codebook_size,
                init=init,
                n_init=n_init,
                batch_size=minibatch_size,
                max_iter=max_iter,
                reassignment_ratio=0.0,
                random_state=seed + stage,
            )
            model.fit(residual_train)
            codebook = model.cluster_centers_.astype(np.float32)
            codebooks.append(codebook)
            train_ids = model.predict(residual_train)
            residual_train -= codebook[train_ids]
            for start in range(0, len(coefficients), prediction_batch):
                batch = residual_all[start : start + prediction_batch]
                ids = model.predict(batch)
                all_code_ids[start : start + len(batch), stage] = ids
                selected = codebook[ids]
                reconstructed[start : start + len(batch)] += selected
                residual_all[start : start + len(batch)] -= selected
            stage_mse.append(float(np.mean(residual_all ** 2)))
        vq_mse = float(np.mean((coefficients - reconstructed) ** 2))
        return {
            "tile_coefficients.npy": _pack_unsigned(all_code_ids, bits),
            "coefficient_codebook.npy": np.stack(codebooks).astype(np.float16),
        }, {
            "coefficient_shape": list(coefficients.shape),
            "coefficient_quantization": "vector_kmeans_v1",
            "coefficient_codebook_size": codebook_size,
            "coefficient_code_bits": bits,
            "coefficient_code_count": int(all_code_ids.size),
            "coefficient_vector_count": int(len(coefficients)),
            "coefficient_vq_stages": stages,
            "coefficient_vq_training_sample": int(len(train)),
            "coefficient_vq_mse": vq_mse,
            "coefficient_vq_stage_mse": stage_mse,
        }
    if encoding != "scalar":
        raise ValueError("coefficient_encoding must be 'scalar' or 'vq'")
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
    if metadata.get("coefficient_quantization") == "vector_kmeans_v1":
        count = int(metadata["coefficient_code_count"])
        code_bits = int(metadata["coefficient_code_bits"])
        stages = int(metadata.get("coefficient_vq_stages", 1))
        vector_count = int(metadata.get("coefficient_vector_count", count))
        code_ids = _unpack_unsigned(stored, count, code_bits).reshape(
            vector_count, stages
        )
        codebook = np.load(
            artifact_dir / "coefficient_codebook.npy"
        ).astype(np.float32)
        if codebook.ndim == 2:
            codebook = codebook[None, ...]
        if len(codebook) != stages:
            raise ValueError("Coefficient codebook stage count mismatch")
        if code_ids.size and int(code_ids.max()) >= codebook.shape[1]:
            raise ValueError("Coefficient code ID is outside the codebook")
        coefficients = np.zeros(
            (vector_count, codebook.shape[2]), dtype=np.float32
        )
        for stage in range(stages):
            coefficients += codebook[stage, code_ids[:, stage]]
        return coefficients
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
    tile_width: int = 1,
    sample_size: int = 0,
    iterations: int = 15,
    batch_size: int = 2048,
    seed: int = 42,
    selection_pool: int = 100,
    selection_seed: int = 42,
    coefficient_bits: int = 16,
    coefficient_encoding: str = "scalar",
    coefficient_codebook_size: int = 256,
    coefficient_vq_sample_size: int = 50000,
    coefficient_vq_stages: int = 1,
) -> dict:
    started = time.perf_counter()
    W = original_weight_out_in.T.astype(np.float32, copy=False)
    rows, columns = W.shape
    if C <= 0:
        raise ValueError("C must be positive")
    if tile_width <= 0:
        raise ValueError("tile_width must be positive")
    if rows % C:
        raise ValueError(f"Input dimension {rows} must be divisible by C={C}")
    if columns % tile_width:
        raise ValueError(
            f"Output dimension {columns} must be divisible by "
            f"tile_width={tile_width}"
        )

    n_tiles, all_tiles, cm_idx = _weight_tiles_flat(
        W, C, tile_width, np.float32
    )
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
        saved_coefficients,
        coefficient_bits,
        encoding=coefficient_encoding,
        codebook_size=coefficient_codebook_size,
        sample_size=coefficient_vq_sample_size,
        stages=coefficient_vq_stages,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    id_arrays, id_metadata = _encode_subspace_ids(ids_ordered, k)
    arrays = {
        "structured_pruning.npy": compact,
        "subspace_bases.npy": bases.astype(np.float16),
        **id_arrays,
        **coefficient_arrays,
    }
    for name, array in arrays.items():
        np.save(output_dir / name, array)

    total_energy = float(np.sum(tiles.astype(np.float64) ** 2))
    metadata = {
        "format": "seedlm_k_subspaces_v3",
        "rows": rows,
        "columns": columns,
        "C": C,
        "tile_width": tile_width,
        "tile_shape": [C, tile_width],
        "tile_dimension": C * tile_width,
        "sparsity": sparsity,
        "k": k,
        "rank": rank,
        "coefficient_bits": coefficient_bits,
        "coefficient_encoding": coefficient_encoding,
        **id_metadata,
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
    artifact_bytes = int(sum(file_bytes.values()))
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
        "artifact_bytes": artifact_bytes,
        "artifact_bits_per_original_parameter": (
            8.0 * artifact_bytes / (rows * columns)
        ),
        "artifact_files": file_bytes,
    }


def load_weight(artifact_dir: Path) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    arrays_started = time.perf_counter()
    compact = np.load(artifact_dir / "structured_pruning.npy")
    bases = np.load(artifact_dir / "subspace_bases.npy").astype(np.float32)
    ids = _load_subspace_ids(artifact_dir, metadata)
    coefficients = _load_coefficients(artifact_dir, metadata)
    load_seconds = time.perf_counter() - arrays_started

    rows = int(metadata["rows"])
    columns = int(metadata["columns"])
    C = int(metadata["C"])
    tile_width = int(metadata.get("tile_width", 1))
    if rows % C:
        raise ValueError(f"Stored rows={rows} are not divisible by C={C}")
    if columns % tile_width:
        raise ValueError(
            f"Stored columns={columns} are not divisible by "
            f"tile_width={tile_width}"
        )
    row_blocks = rows // C
    column_blocks = columns // tile_width
    tile_dimension = C * tile_width
    n_tiles = row_blocks * column_blocks
    if len(ids) != n_tiles:
        raise ValueError(f"ID count {len(ids)} does not match expected {n_tiles}")

    reconstruction_started = time.perf_counter()
    tiles = np.empty((n_tiles, tile_dimension), dtype=np.float16)
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
        tiles.reshape(row_blocks, column_blocks, C, tile_width)
        .transpose(0, 2, 1, 3)
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
#
# COEFFICIENT_BITS=8 \
# STAGE_POLICY=uniform \
# DEFAULT_STAGES=2 \
# ./SeedLM/k-space/final/run_final_vq_opt125m.sh