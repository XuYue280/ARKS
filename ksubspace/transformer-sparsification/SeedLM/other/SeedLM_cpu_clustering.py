from pathlib import Path
import time

import numpy as np
import pandas as pd
import os
import random
from functools import lru_cache
from scipy import stats
from typing import Tuple, Dict, Optional, Union, List
from numba import njit, prange
import torch
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus

'''
Example usage for memory storage cost analysis:
python -c "
from SeedLM.SeedLM_cpu import calculate_memory_storage_cost
calculate_memory_storage_cost(M=4, C=8, P=3, K_bits=16)
"
'''


# Terminology:
# - tile: C x selection_pool block in W (structured pruning pool)
# - col_matrix: C x 1 column slice within a tile
# Support functions
def _num_col_matrices(matrix: np.ndarray, C: int) -> int:
    """Count Cx1 col_matrices in this weight matrix."""
    rows, cols = matrix.shape
    return (rows // C) * cols


def _pruned_col_matrix_count(
        num_col_matrices: int,
        sparsity: float,
        selection_pool: int = 100,
        selection_seed: int = 42,
) -> int:
    return int(
        _pruned_col_matrix_indices_from_sparsity(
            num_col_matrices, sparsity, selection_pool, selection_seed
        ).size
    )


def _knn_k_values_pow2(
        num_col_matrices: int,
        sparsity: float,
        selection_pool: int = 100,
        selection_seed: int = 42,
        min_k: int = 1,
) -> List[int]:
    max_k = _pruned_col_matrix_count(num_col_matrices, sparsity, selection_pool, selection_seed)
    if max_k == 0:
        return [0]
    if max_k < min_k:
        raise ValueError(
            f"[_knn_k_values_pow2] cannot find valid K values: pruned_col_matrix_count={max_k} < min_k={min_k}")
    vals: List[int] = []
    k = min_k
    while k <= max_k:
        vals.append(k)
        k *= 2
    return vals


# Support functions END


def magnitude_pruning(w: np.ndarray, sparsity: float) -> np.ndarray:
    """Magnitude pruning: return the pruned matrix."""
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("sparsity must be in [0, 1].")

    w_pruned = np.array(w, copy=True)
    if sparsity == 0.0:
        return w_pruned

    num_elements = w_pruned.size
    num_prune = int(round(num_elements * sparsity))
    if num_prune <= 0:
        return w_pruned
    if num_prune >= num_elements:
        w_pruned[:] = 0
        return w_pruned

    flat = w_pruned.ravel()
    prune_idx = np.argpartition(np.abs(flat), num_prune - 1)[:num_prune]
    flat[prune_idx] = 0
    return w_pruned


def _pruned_col_matrix_indices_from_sparsity(
        num_col_matrices: int,
        sparsity: float,
        selection_pool: int = 100,
        selection_seed: int = 42,
) -> np.ndarray:
    """Structured pruning: return flat col_matrix indices to prune within each tile."""
    if selection_pool <= 0:
        raise ValueError("selection_pool must be positive.")
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("sparsity must be in [0, 1].")

    rng = np.random.default_rng(selection_seed)
    num_tiles = num_col_matrices // selection_pool
    remainder = num_col_matrices % selection_pool
    k_full = min(selection_pool, max(0, int(round(selection_pool * sparsity))))

    pruned_ids: list[np.ndarray] = []
    for tile_idx in range(num_tiles):
        if k_full == 0:
            continue
        base = tile_idx * selection_pool
        pruned_ids.append(base + rng.choice(selection_pool, size=k_full, replace=False))

    if remainder > 0:
        k_rem = min(remainder, max(0, int(round(remainder * sparsity))))
        if k_rem > 0:
            base = num_tiles * selection_pool
            pruned_ids.append(base + rng.choice(remainder, size=k_rem, replace=False))

    if not pruned_ids:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(pruned_ids).astype(np.int64)


# Same as Table 6 in the paper, each row = [K, indices j]
_LFSR_ALPHA_TABLE: dict[int, tuple[int, ...]] = {
    2: (0, 1),
    3: (0, 1),
    4: (0, 1),
    5: (0, 2),
    6: (0, 1),
    7: (0, 1),
    8: (0, 2, 3, 4),
    9: (0, 4),
    10: (0, 3),
    11: (0, 2),
    12: (0, 1, 2, 8),
    13: (0, 1, 2, 5),
    14: (0, 1, 2, 12),
    15: (0, 1),
    16: (0, 1, 3, 12),
    17: (0, 3),
    18: (0, 7),
    19: (0, 1, 2, 5),
    20: (0, 3),
    21: (0, 2),
    22: (0, 1),
    23: (0, 5),
    24: (0, 1, 2, 7),
}


def XOR_LFSR(row_bin: np.ndarray, alpha: tuple[int, ...], K: int) -> np.ndarray:
    """
    Single-step XOR-LFSR update (LSB -> MSB bit order).

    `alpha` is interpreted as tap indices. The new bit is XOR(parity) of all tapped
    positions from `row_bin`, then the register is left-shifted by one.
    """
    tap_mask = np.zeros(K, dtype=bool)
    for j in alpha:
        tap_mask[j] = True

    # XOR the new bit
    new_bit = bool(np.bitwise_xor.reduce(row_bin[tap_mask])) if np.any(tap_mask) else False

    row_result_bin = np.empty_like(row_bin, dtype=bool)
    # shift 1-K to 0-(K-1)
    row_result_bin[:-1] = row_bin[1:K]
    # add the new bit to the end
    row_result_bin[-1] = new_bit
    return row_result_bin


# if K is same, don't compute U_dec again, use cached value
@lru_cache(maxsize=None)
def _cached_LFSR_U_dec(K: int) -> np.ndarray:
    """
    Build and cache full LFSR integer codebook U_dec for a given K.
    Returned array is int32 and has length (2**K - 1).
    generate the circular-like array U_dec with shape (2^K-1, K) with bool.
    """
    if K not in _LFSR_ALPHA_TABLE:
        raise ValueError(
            f"LFSR_get_Us: K={K} not in alpha_table (supported K: {sorted(_LFSR_ALPHA_TABLE.keys())})"
        )

    taps = _LFSR_ALPHA_TABLE[K]
    n_rows = 2 ** K - 1

    # 1) Build all binary LFSR states (row). U_bin is a loop-like array with shape (2^K-1, K) with bool.
    U_bin = np.zeros((n_rows, K), dtype=bool)
    for tap_idx in taps:
        U_bin[0, tap_idx] = True
    for row_idx in range(1, n_rows):
        U_bin[row_idx] = XOR_LFSR(U_bin[row_idx - 1], taps, K)

    # 2) Convert each binary row to decimal.
    # Bit order in this file: index 0 is LSB, index -1 is MSB.
    U_dec = np.zeros(n_rows, dtype=np.uint16)
    for row_idx in range(n_rows):
        value = 0
        for bit_idx in range(K):
            if U_bin[row_idx, bit_idx]:
                value += 1 << bit_idx
        # U_dec cannot have 0
        if value == 0:
            raise ValueError(f"LFSR_get_Us: value is 0 for row {row_idx}")
        elif value > 2 ** 16 - 1:
            raise ValueError(f"LFSR_get_Us: value is greater than 2^16-1 for row {row_idx}, value: {value}")
        U_dec[row_idx] = np.uint16(value)

    return U_dec


def LFSR_get_Us(C: int, P: int, K: int, seed: int) -> np.ndarray:
    """
    Build LFSR-based ``U`` (SeedLM paper, Table 6 tap positions), then map entries to ``[-1, 1]``.

    dtypes:
        - LFSR state ``U_bin``: bool
        - Integer codebook ``U_dec``: int32 (values ``1 .. 2^K-1``)
        - Filled grid ``Vs``: float32 (must not cast integer codebook to float16 first:
          for large ``K``, values can exceed float16 max ~65504 and become ``inf``.)
        - Return ``Us``: float16

    Normalization (same as before): ``Us = (Vs - 2^(K-1)) / (2^(K-1) - 1)``.
    """

    rng = np.random.default_rng(seed)
    n_rows = 2 ** K - 1

    # if K is same, don't compute U_dec again, use cached value
    U_dec = _cached_LFSR_U_dec(K)

    # random start index in U_dec
    start_idx = int(rng.integers(0, n_rows))
    idx = (start_idx + np.arange(C * P, dtype=np.uint16).reshape(C, P)) % n_rows  # 2D array with shape (C, P)
    Vs = U_dec[idx].astype(np.float32)  # use float 32 to store uint16 range

    coeff = np.float32(2 ** (K - 1))
    Us = ((Vs - coeff) / (coeff - 1.0)).astype(np.float16)
    return Vs, Us


# def random_Us(C: int, P: int, K: int, seed: int) -> np.ndarray:
#     """
#     Same ``Vs`` construction and normalization as :func:`LFSR_get_Us`, but ``U_dec`` is
#     filled by **random permutation** of ``1 .. 2^K - 1`` (all non-zero K-bit pattern
#     decimals) instead of XOR-LFSR iterated bit rows.

#     ``start_idx`` into ``U_dec`` is also drawn at random (analogous to ``seed`` in
#     :func:`LFSR_get_Us`).
#     """
#     if C < 1 or P < 1 or K < 1:
#         raise ValueError(f"C, P, K must be >= 1, got C={C}, P={P}, K={K}")
#     rng = np.random.default_rng(seed)
#     U_dec = rng.permutation(np.arange(2**K-1, dtype=np.int32))

#     start_idx = int(rng.integers(0, 2**K-1))
#     idx = (start_idx + np.arange(C * P, dtype=np.int32).reshape(C, P)) % (2**K-1)
#     Vs = U_dec[idx].astype(np.float16)
#     coeff = np.float16(2 ** (K - 1))

#     Us = ((Vs - coeff) / (coeff - 1.0)).astype(np.float16)
#     return Vs, Us


# Section 2: Quantization
def get_uniform_quantization_level(range_min: float, range_max: float, bits: int) -> np.ndarray:
    '''
    get the quantization level for uniform quantization

    input:
    range_min: minimum value of the range
    range_max: maximum value of the range
    bits: number of bits

    output:
    quantization_level(LUT) with uniform step size
    '''
    return np.linspace(range_min, range_max, 2 ** bits).astype(np.float16)


def get_value_and_shared_exponent(t: np.ndarray, value_bits: int):
    '''
    SeedLM paper quantization method: value_bits of 2's complement integer + shared exponent
    a shared exponent for each tile
    shared exponent = max_i (log2|t_i|), loop through all elements in t
    '''
    for i in range(t.shape[0]):
        shared_exponent = np.max(np.log2(np.abs(t[i])))

    return None


def quantize(weight: np.ndarray, quantization_level: np.ndarray, bits: int) -> np.ndarray:
    '''
    quantize the weight to the given level

    input:
    weight can be 1D or 2D
    quantization_level: quantized value [x1, x2, ..., xn]

    output:
    quantization_index, same dimension as input weight, each elements (int 4) is the index of the quantization level
    can use this index + quantization_level[index] to get the quantized weight
    '''
    n_levels = 2 ** bits
    if len(quantization_level) != n_levels:
        raise ValueError(f"quantization_level size {len(quantization_level)} != 2**bits ({n_levels})")

    w = weight.copy().astype(np.float16)

    indices = np.argmin(np.abs(w[..., None] - quantization_level), axis=-1)

    if bits <= 8:
        return indices.astype(np.uint8)
    if bits <= 16:
        return indices.astype(np.uint16)
    return indices.astype(np.int32)


def reconstruct_quantized_weight(
        row: int,
        col: int,
        quantization_index: np.ndarray,
        quantization_level: np.ndarray,
        bits: int,
) -> np.ndarray:
    """Recover the quantized weight matrix from index + LUT."""
    n_levels = 2 ** bits
    if len(quantization_level) != n_levels:
        raise ValueError(
            f"quantization_level size {len(quantization_level)} != 2**bits ({n_levels})"
        )

    idx = quantization_index.reshape(row, col)
    return quantization_level[idx].astype(np.float16)


def quantization_compress(
        quantization_upper_bound: float = 1.0,
        quantization_lower_bound: float = -1.0,
        *,
        bits: int,
        load_path: Path | str,
        load_name: str,
        storage_path: Path | str,
        storage_LUT_name: str,
        storage_quantization_index_name: str,
        quantization_level: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, int]]:  # compress_timings, stored_bytes
    '''
    compress the weight to the given quantization level
    '''
    compress_timings: dict[str, float] = {
        "load_from_local": 0.0,
        "get_quantization_level": 0.0,
        "quantization": 0.0,
        "save_to_local": 0.0,
    }
    stored_bytes: dict[str, int] = {
        "LUT": 0,
        "quantization_index": 0,
    }

    t0 = time.perf_counter()
    weight = np.load(load_path / load_name)
    compress_timings["load_from_local"] = time.perf_counter() - t0

    if quantization_level is None:
        t0 = time.perf_counter()
        quantization_level = get_uniform_quantization_level(
            quantization_upper_bound, quantization_lower_bound, bits
        )
        compress_timings["get_quantization_level"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    quantization_index = quantize(weight, quantization_level, bits)
    compress_timings["quantization"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.save(storage_path / storage_LUT_name, quantization_level)
    stored_bytes["LUT"] = quantization_level.nbytes
    np.save(storage_path / storage_quantization_index_name, quantization_index)
    stored_bytes["quantization_index"] = quantization_index.nbytes
    compress_timings["save_to_local"] = time.perf_counter() - t0

    return compress_timings, stored_bytes


def quantization_load(
        row: int,
        col: int,
        bits: int,
        *,
        load_path: Path | str,
        load_lut_name: str,
        load_quantization_index_name: str,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:
    """Load LUT and quantization index from disk and reconstruct [row, col] weight.

    Returns (quantized_weight, load_timings, load_bytes).
    """
    load_timings: dict[str, float] = {
        "load_from_local": 0.0,
        "reconstruct_quantized_weight": 0.0,
    }
    load_bytes: dict[str, int] = {
        "LUT": 0,
        "quantization_index": 0,
    }

    load_path = Path(load_path)

    t0 = time.perf_counter()
    lut = np.load(load_path / load_lut_name)
    load_bytes["LUT"] = lut.nbytes
    quantization_index = np.load(load_path / load_quantization_index_name)
    load_bytes["quantization_index"] = quantization_index.nbytes
    load_timings["load_from_local"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    quantized_weight = reconstruct_quantized_weight(
        row, col, quantization_index, lut, bits
    )
    load_timings["reconstruct_quantized_weight"] = time.perf_counter() - t0

    return quantized_weight, load_timings, load_bytes


# def GPTQ_ts(M: int, ts: np.ndarray):
#     """
#     Quantize ``ts`` with IST-DASLab GPTQ ``Quantizer`` + ``quantize`` (``quant.py``).

#     Returns a **dequantized** coefficient tensor ``t_rec`` stored as ``float16``:
#     values lie on the M-bit affine quantization lattice from GPTQ (what you called
#     "float4 range" when ``M == 4``), but the container dtype is fp16.

#     Return tuple keeps three slots for older call sites:
#         ``(t_rec_float16, reserved_int8, reserved_empty_int8)``
#     The last two are unused placeholders; do **not** apply ``* 2**e`` to ``t_rec``.
#     """
#     if not (1 <= M <= 16):
#         raise ValueError(f"M (GPTQ bits) must be in [1, 16], got {M}")

#     t = np.where(
#         np.isfinite(t := np.asarray(ts, dtype=np.float16)),
#         t,
#         0.0,
#     )
#     orig_shape = t.shape
#     if t.size == 0:
#         z = np.zeros(0, dtype=np.float16)
#         return z, np.int8(0), np.array([], dtype=np.int8)

#     import sys

#     gptq_repo_dir = os.path.join(
#     os.path.dirname(os.path.dirname(__file__)),
#     "external_repos",
#     "gptq",)
#     if gptq_repo_dir not in sys.path:
#         sys.path.insert(0, gptq_repo_dir)

#     try:
#         from quant import Quantizer, quantize as gptq_quantize
#     except Exception as exc:
#         raise ImportError(
#             "GPTQ quant module not found. Clone IST-DASLab/gptq under external_repos/gptq."
#         ) from exc

#     x_torch = torch.from_numpy(t.reshape(-1, 1).astype(np.float32))
#     qzr = Quantizer()
#     qzr.configure(bits=M, perchannel=False, sym=False, mse=False)
#     qzr.find_params(x_torch, weight=False)
#     t_rec = gptq_quantize(x_torch, qzr.scale, qzr.zero, qzr.maxq).numpy().reshape(orig_shape)
#     t_rec = np.asarray(t_rec, dtype=np.float16)

#     return t_rec, np.int8(0), np.array([], dtype=np.int8)


def _precompute_one_seed_entry(
        si: int, C: int, P: int, K: int, init_seed: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    """One candidate seed: LFSR Us and pinv_U (independent per si, safe for multiprocessing)."""
    seed = init_seed + si  # fix seed, to make it reproducible
    _, Us = LFSR_get_Us(C, P, K, seed)
    Us = Us.astype(np.float32)
    pinv_U = np.linalg.pinv(Us).astype(np.float32)
    return seed, Us, pinv_U


def _build_precomputed(
        C: int, P: int, K: int, init_seed: int, n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute all 2^K (seed, Us, pinv_U); return stacked float32 arrays for Numba/BLAS."""
    n_seeds = 2 ** K
    if n_jobs == 1:
        entries = [_precompute_one_seed_entry(si, C, P, K, init_seed) for si in range(n_seeds)]
    else:
        from joblib import Parallel, delayed

        entries = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_precompute_one_seed_entry)(si, C, P, K, init_seed) for si in range(n_seeds)
        )

    seeds = np.empty(n_seeds, dtype=np.int64)
    us_stack = np.empty((n_seeds, C, P), dtype=np.float32)
    pinv_stack = np.empty((n_seeds, P, C), dtype=np.float32)
    for si, (seed, us, pinv_u) in enumerate(entries):
        seeds[si] = seed
        us_stack[si] = us
        pinv_stack[si] = pinv_u
    return seeds, us_stack, pinv_stack


# K clustering support functions
@njit(parallel=True)
def nearest_cluster_numba(col_matrices_flat, pruned_col_matrix_indices, centers):
    """Numba nopython does not support float16; pass float32 arrays."""
    num_col_matrices = col_matrices_flat.shape[0]
    num_pruned = pruned_col_matrix_indices.shape[0]
    K_cluster = centers.shape[0]
    C = centers.shape[1]

    nearest_ids = np.full(num_col_matrices, -1)

    for ii in prange(num_pruned):
        col_matrix_idx = pruned_col_matrix_indices[ii]

        best_k = -1
        best_dist = np.inf

        for k in range(K_cluster):
            dist = 0.0
            for c in range(C):
                diff = col_matrices_flat[col_matrix_idx, c] - centers[k, c]
                dist += diff * diff

            if dist < best_dist:
                best_dist = dist
                best_k = k

        nearest_ids[col_matrix_idx] = best_k

    return nearest_ids


@njit
def select_best_seed_numba(
        centers: np.ndarray,
        us_stack: np.ndarray,
        pinv_stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each cluster center, pick the seed with minimum L2 reconstruction error.
    Pass float32 centers / Us / pinv_U (Numba does not support float16).

    Returns (best_si, best_t, best_w_rec):
        best_si[j]   — seed index for cluster center j
        best_t[j, :] — (P,) coefficients t = pinv_U @ center
        best_w_rec[j, :] — (C,) reconstructed col_matrix Us @ t
    """
    n_centers, c = centers.shape
    s = us_stack.shape[0]
    p = us_stack.shape[2]
    min_err = np.full(n_centers, np.inf)
    best_si = np.full(n_centers, -1, dtype=np.int32)
    best_t = np.zeros((n_centers, p), dtype=np.float32)
    best_w_rec = np.zeros((n_centers, c), dtype=np.float32)

    for si in range(s):
        t = pinv_stack[si] @ centers.T  # (P, n_centers)
        w_rec = us_stack[si] @ t  # (C, n_centers)
        for j in range(n_centers):
            err = 0.0
            for cc in range(c):
                d = centers[j, cc] - w_rec[cc, j]
                err += d * d
            err = np.sqrt(err)
            if err < min_err[j]:
                min_err[j] = err
                best_si[j] = si
                for pp in range(p):
                    best_t[j, pp] = t[pp, j]
                for cc in range(c):
                    best_w_rec[j, cc] = w_rec[cc, j]

    return best_si, best_t, best_w_rec


# @njit(parallel=True)
# def fill_pruned_tiles_numba(
#     w_seedlm: np.ndarray,
#     pruned_tile_indices: np.ndarray,
#     nearest_cluster_center_ids: np.ndarray,
#     best_w_rec: np.ndarray,
#     best_si: np.ndarray,
#     tiles_per_col: int,
#     c: int,
# ) -> None:
#     """
#     Write precomputed cluster reconstructions into pruned tile columns.
#     Pass float32 w_seedlm / best_w_rec (Numba does not support float16).
#     Flat tile id: col * tiles_per_col + row_tile.
#     """
#     n_pruned = pruned_tile_indices.shape[0]
#     for k in prange(n_pruned):
#         tile_idx = pruned_tile_indices[k]
#         center_id = nearest_cluster_center_ids[tile_idx]
#         if center_id < 0:
#             continue
#         if best_si[center_id] < 0:
#             continue
#         row_tile = tile_idx % tiles_per_col
#         col_idx = tile_idx // tiles_per_col
#         row_start = row_tile * c
#         for cc in range(c):
#             w_seedlm[row_start + cc, col_idx] = best_w_rec[center_id, cc]


def _weight_col_matrices_flat(
        W: np.ndarray,
        C: int,
        dtype: np.dtype = np.float16,
) -> tuple[int, np.ndarray, np.ndarray]:
    Wrows, Wcols = W.shape
    col_matrices_per_col = Wrows // C
    num_col_matrices = col_matrices_per_col * Wcols
    col_matrices_flat = (
        W.reshape(col_matrices_per_col, C, Wcols)
        .transpose(2, 0, 1)
        .reshape(num_col_matrices, C)
        .astype(dtype, copy=False)
    )
    cm_idx = (
            (np.arange(num_col_matrices) % Wcols) * col_matrices_per_col
            + (np.arange(num_col_matrices) // Wcols)
    )
    return num_col_matrices, col_matrices_flat, cm_idx


def _structured_pruning_compact_from_weight(
        W: np.ndarray,
        C: int,
        sparsity: float,
        selection_pool: int,
        selection_seed: int,
        dtype: np.dtype = np.float16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_col_matrices, col_matrices_flat, cm_idx = _weight_col_matrices_flat(W, C, dtype)
    pruned_col_matrix_ids = _pruned_col_matrix_indices_from_sparsity(
        num_col_matrices, sparsity, selection_pool, selection_seed
    )
    compact = col_matrices_flat[cm_idx[~np.isin(cm_idx, pruned_col_matrix_ids)]]
    return compact, pruned_col_matrix_ids, col_matrices_flat, cm_idx


def find_K_cluster_centers(
        original_matrix: np.ndarray,
        C: int,
        K: int,
        init_seed: int,
        sparsity: float,
        selection_pool: int = 100,
        selection_seed: int = 42,
        *,
        pruned_col_matrix_indices: np.ndarray | None = None,
):
    '''
    time: O(K)
    space: O(KC)

    RANDOMLY pick K distinct cluster centers from pruned col_matrices (sparsity-defined).
    If pruned col_matrix count < K, use all pruned col_matrices as centers.

    return: list of cluster centers (Cx1 col_matrix each, with actual values)
    '''
    rng = np.random.default_rng(init_seed)
    W = original_matrix
    Wrows, Wcols = W.shape
    if Wrows % C != 0:
        raise ValueError(f"[find_K_cluster_centers] Wrows={Wrows} must be a multiple of C={C}.")

    col_matrices_per_col = Wrows // C
    num_col_matrices = col_matrices_per_col * Wcols
    if pruned_col_matrix_indices is None:  # no structured pruning, pick k center from all col_matrices
        pruned_col_matrix_indices = _pruned_col_matrix_indices_from_sparsity(
            num_col_matrices, sparsity, selection_pool, selection_seed
        )

    if pruned_col_matrix_indices.size == 0:
        raise ValueError(
            "[find_K_cluster_centers] Empty pruned col_matrices list; cannot pick cluster centers."
        )

    K_eff = min(K, pruned_col_matrix_indices.size)
    col_matrix_indices = rng.choice(pruned_col_matrix_indices, size=K_eff, replace=False)

    cluster_centers = []
    for col_matrix_idx in col_matrix_indices:
        row_block = int(col_matrix_idx % col_matrices_per_col)
        col_idx = int(col_matrix_idx // col_matrices_per_col)
        cluster_centers.append(W[row_block * C: (row_block + 1) * C, col_idx])

    return cluster_centers


def find_cluster_centers(
        col_matrices_flat: np.ndarray,
        pruned_col_matrix_indices: np.ndarray,
        K: int,
        init_seed: int,
        method: str = "random",
        sample_size: int = 50000,
        medoid_sample_size: int = 5000,
        batch_size: int = 4096,
) -> np.ndarray:
    """Select/learn K centers from pruned C-dimensional column matrices."""
    x_all = col_matrices_flat[pruned_col_matrix_indices].astype(np.float32, copy=False)
    if x_all.shape[0] == 0:
        raise ValueError("No pruned col_matrices available for clustering.")
    k = min(int(K), x_all.shape[0])
    rng = np.random.default_rng(init_seed)

    def sampled(x: np.ndarray, limit: int) -> np.ndarray:
        if x.shape[0] <= limit:
            return x
        return x[rng.choice(x.shape[0], size=limit, replace=False)]

    if method == "random":
        ids = rng.choice(x_all.shape[0], size=k, replace=False)
        return x_all[ids].copy()

    if method == "kmeans++":
        x = sampled(x_all, max(sample_size, k))
        centers, _ = kmeans_plusplus(x, n_clusters=k, random_state=init_seed)
        return centers.astype(np.float32, copy=False)

    if method == "minibatch":
        x = sampled(x_all, max(sample_size, k))
        model = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=3,
            batch_size=min(batch_size, x.shape[0]),
            max_iter=50,
            reassignment_ratio=0.01,
            random_state=init_seed,
        )
        model.fit(x)
        return model.cluster_centers_.astype(np.float32, copy=False)

    if method == "kmedoids":
        try:
            from sklearn_extra.cluster import KMedoids
        except ImportError as exc:
            raise RuntimeError(
                "kmedoids requires: pip install scikit-learn-extra"
            ) from exc
        x = sampled(x_all, medoid_sample_size)
        if k > x.shape[0]:
            raise ValueError(f"K={k} exceeds K-medoids sample size={x.shape[0]}")
        model = KMedoids(
            n_clusters=k,
            metric="euclidean",
            method="pam",
            init="k-medoids++",
            random_state=init_seed,
        )
        model.fit(x)
        return model.cluster_centers_.astype(np.float32, copy=False)

    raise ValueError(f"Unknown clustering method: {method}")


def nearest_cluster_center_search(
        original_matrix: np.ndarray,
        C: int,
        cluster_centers: List[np.ndarray],
        sparsity: float,
        selection_pool: int = 100,
        selection_seed: int = 42,
        *,
        pruned_col_matrix_indices: np.ndarray | None = None,
        col_matrices_flat: np.ndarray | None = None,
):
    '''
    time: O(num_pruned_col_matrices * K * C)
    space: O(num_col_matrices)

    Assign each pruned col_matrix to its nearest cluster center.
    Returns nearest cluster id per col_matrix (length num_col_matrices, -1 if not pruned).
    '''
    Wrows, Wcols = original_matrix.shape
    col_matrices_per_col = Wrows // C
    num_col_matrices = col_matrices_per_col * Wcols

    if pruned_col_matrix_indices is None:  # no structured pruning, assign all col_matrices to nearest cluster center
        pruned_col_matrix_indices = _pruned_col_matrix_indices_from_sparsity(
            num_col_matrices, sparsity, selection_pool, selection_seed
        )
    nearest_cluster_center_ids = np.full(num_col_matrices, -1, dtype=np.int32)

    if pruned_col_matrix_indices.size == 0 or len(cluster_centers) == 0:
        return nearest_cluster_center_ids

    if col_matrices_flat is None:
        col_matrices_flat = (
            original_matrix.reshape(col_matrices_per_col, C, Wcols)
            .transpose(2, 0, 1)
            .reshape(num_col_matrices, C)
            .astype(np.float32, copy=False)
        )
    else:
        col_matrices_flat = col_matrices_flat.astype(np.float32, copy=False)
    # equivalent to:
    # col_matrices = []
    # for col in range(Wcols):
    #     for rb in range(col_matrices_per_col):
    #         start = rb * C
    #         end = start + C
    #         col_matrix = original_matrix[start:end, col]
    #         col_matrices.append(col_matrix)
    # col_matrices_flat = np.array(col_matrices, dtype=np.float32)

    centers = np.stack(cluster_centers, axis=0).astype(np.float32, copy=False)
    nearest_cluster_center_ids = nearest_cluster_numba(
        col_matrices_flat,
        pruned_col_matrix_indices,
        centers,
    ).astype(np.int32)

    return nearest_cluster_center_ids


def structured_pruning_compress(
        sparsity: float,
        selection_pool: int,
        selection_seed: int,
        *,
        C: int = 8,
        load_path: Path | str,
        load_name: str,
        storage_path: Path | str,
        storage_name: str,
        storage_dtype: np.dtype = np.float16,
) -> tuple[dict[str, float], dict[str, int]]:  # compress_timings, stored_bytes
    """
    load original model from disk and compress it.
    Extract non-pruned col_matrices into compact form and save .npy.
    Returns (compress_timings, storage_bytes).
    """

    compress_timings: dict[str, float] = {
        "load_from_local": 0.0,
        "structured_pruning": 0.0,
        "save_to_local": 0.0,
    }
    stored_bytes: dict[str, int] = {
        "compact": 0,
    }

    t0 = time.perf_counter()
    W = np.load(load_path / load_name)
    W = W.T
    compress_timings["load_from_local"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    compact, _, _, _ = _structured_pruning_compact_from_weight(
        W, C, sparsity, selection_pool, selection_seed, storage_dtype
    )
    compress_timings["structured_pruning"] = time.perf_counter() - t0
    stored_bytes["compact"] = compact.nbytes

    t0 = time.perf_counter()
    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    np.save(storage_path / storage_name, compact)
    compress_timings["save_to_local"] = time.perf_counter() - t0

    return compress_timings, stored_bytes


def structured_pruning_load(
        row: int,
        col: int,
        sparsity: float,
        selection_pool: int,
        selection_seed: int,
        *,
        C: int = 8,
        load_path: Path | str,
        load_compact_name: str,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:
    """Load compact structured-pruning artifact and expand to dense matrix; pruned col_matrices are zero.

    Returns (W_out, load_timings, load_bytes).
    """
    load_timings: dict[str, float] = {
        "load_from_local": 0.0,
        "reconstruct_matrix": 0.0,
    }
    load_bytes: dict[str, int] = {
        "compact": 0,
    }

    load_path = Path(load_path)

    t0 = time.perf_counter()
    compact = np.load(load_path / load_compact_name)
    load_bytes["compact"] = compact.nbytes
    load_timings["load_from_local"] = time.perf_counter() - t0

    if row % C != 0:
        raise ValueError(f"row={row} must be divisible by C={C}.")

    t0 = time.perf_counter()
    col_matrices_per_col = row // C
    num_col_matrices = col_matrices_per_col * col
    pruned_col_matrix_ids = set(
        _pruned_col_matrix_indices_from_sparsity(
            num_col_matrices, sparsity, selection_pool, selection_seed
        )
    )

    out = np.zeros((row, col), dtype=np.float16)
    compact_i = 0
    for col_matrix_idx in range(num_col_matrices):
        row_block = col_matrix_idx // col
        col_idx = col_matrix_idx % col
        cm_idx = col_idx * col_matrices_per_col + row_block
        if cm_idx not in pruned_col_matrix_ids:
            out[row_block * C: (row_block + 1) * C, col_idx] = compact[compact_i].astype(
                np.float16, copy=False
            )
            compact_i += 1
    load_timings["reconstruct_matrix"] = time.perf_counter() - t0

    return out, load_timings, load_bytes


# compress and store
# TODO: ADD detailed timing for stack, _build_precomputed, select_best_seed_numba
def SeedLM_compress(
        sparsity: float,
        K_cluster: int,
        init_seed: int,
        selection_pool: int,
        selection_seed: int,
        *,
        C: int = 8,
        P: int = 3,
        K_bits: int = 16,
        clustering_method: str = "random",
        clustering_sample_size: int = 50000,
        medoid_sample_size: int = 5000,
        clustering_batch_size: int = 4096,
        pruned_col_matrix_storage_dtype: np.dtype = np.float16,
        ts_storage_dtype: np.dtype = np.float16,
        nearest_cluster_center_ids_storage_dtype: np.dtype = np.int32,
        seeds_storage_dtype: np.dtype = np.int32,
        load_path: Path | str,
        load_name: str,
        storage_path: Path | str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Load weight from disk, compress K-cluster SeedLM artifacts, and save .npy files.

    Returns (compress_timings, storage_bytes).
    """
    compress_timings: dict[str, float] = {
        "load_from_local": 0.0,
        "structured_pruning": 0.0,
        "find_k_cluster_centers": 0.0,
        "seedlm_fit": 0.0,
        "nearest_cluster_center_search": 0.0,
        "save_to_local": 0.0,
    }
    storage_bytes: dict[str, int] = {
        "structured_pruning_compact_format": 0,
        "seeds": 0,
        "ts": 0,
        "nearest_cluster_center_ids": 0,
    }

    t0 = time.perf_counter()
    W = np.load(Path(load_path) / load_name).T
    compress_timings["load_from_local"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    non_pruned_compact, pruned_col_matrix_ids, col_matrices_flat, cm_idx = (
        _structured_pruning_compact_from_weight(
            W, C, sparsity, selection_pool, selection_seed, pruned_col_matrix_storage_dtype
        )
    )
    compress_timings["structured_pruning"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cluster_centers = find_cluster_centers(
        col_matrices_flat,
        pruned_col_matrix_ids,
        K_cluster,
        init_seed,
        method=clustering_method,
        sample_size=clustering_sample_size,
        medoid_sample_size=medoid_sample_size,
        batch_size=clustering_batch_size,
    )
    compress_timings["find_k_cluster_centers"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    centers = np.asarray(cluster_centers, dtype=np.float32)
    seeds, us_stack, pinv_stack = _build_precomputed(C, P, K_bits, init_seed)
    best_seeds, best_t, _best_w_rec = select_best_seed_numba(
        centers, us_stack, pinv_stack
    )
    compress_timings["seedlm_fit"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nearest_ids_cm = nearest_cluster_center_search(
        W,
        C,
        list(cluster_centers),
        sparsity,
        selection_pool,
        selection_seed,
        pruned_col_matrix_indices=pruned_col_matrix_ids,
        col_matrices_flat=col_matrices_flat,
    )
    compress_timings["nearest_cluster_center_search"] = time.perf_counter() - t0

    # Params live in parent folder path; keep fixed artifact names inside this layer dir.
    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    structured_arr = non_pruned_compact.astype(pruned_col_matrix_storage_dtype)
    seeds_arr = seeds[best_seeds].astype(seeds_storage_dtype)
    ts_arr = best_t.astype(ts_storage_dtype)
    nearest_arr = nearest_ids_cm[cm_idx].astype(nearest_cluster_center_ids_storage_dtype)

    t0 = time.perf_counter()
    np.save(storage_path / "structured_pruning.npy", structured_arr)
    storage_bytes["structured_pruning_compact_format"] = structured_arr.nbytes
    np.save(storage_path / "seeds.npy", seeds_arr)
    storage_bytes["seeds"] = seeds_arr.nbytes
    np.save(storage_path / "ts.npy", ts_arr)
    storage_bytes["ts"] = ts_arr.nbytes
    np.save(storage_path / "nearest_cluster_center_ids.npy", nearest_arr)
    storage_bytes["nearest_cluster_center_ids"] = nearest_arr.nbytes
    compress_timings["save_to_local"] = time.perf_counter() - t0

    return compress_timings, storage_bytes


def SeedLM_load(
        row: int,
        col: int,
        sparsity: float,
        K_cluster: int,
        *,
        load_path: Path | str,
        C: int = 8,
        P: int = 3,
        K_bits: int = 16,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:  # W_out, load_timings, load_bytes
    """
    Load artifacts from disk
    reconstruct the weight matrix by SeedLM with K clustering method

    Returns (W_out, load_timings, load_bytes).
    """
    load_timings: dict[str, float] = {
        "load_from_local": 0.0,
        "reconstruct_k_cluster_centers": 0.0,
        "reconstruct_matrix": 0.0,
    }
    load_bytes: dict[str, int] = {
        "structured_pruning_compact_format": 0,
        "seeds": 0,
        "ts": 0,
        "nearest_cluster_center_ids": 0,
    }

    load_path = Path(load_path)

    t0 = time.perf_counter()
    structured_arr = np.load(load_path / "structured_pruning.npy")
    load_bytes["structured_pruning_compact_format"] = structured_arr.nbytes
    seeds_arr = np.load(load_path / "seeds.npy")
    load_bytes["seeds"] = seeds_arr.nbytes
    ts_arr = np.load(load_path / "ts.npy")
    load_bytes["ts"] = ts_arr.nbytes
    nearest_arr = np.load(load_path / "nearest_cluster_center_ids.npy")
    load_bytes["nearest_cluster_center_ids"] = nearest_arr.nbytes
    load_timings["load_from_local"] = time.perf_counter() - t0

    seeds_len = seeds_arr.shape[0]
    if seeds_len != ts_arr.shape[0]:
        raise ValueError(f"seeds length {seeds_len} != ts length {ts_arr.shape[0]}")
    if row % C != 0:
        raise ValueError(f"row={row} must be divisible by C={C}.")

    col_matrices_per_col = row // C
    num_col_matrices = col_matrices_per_col * col
    if nearest_arr.shape[0] != num_col_matrices:
        raise ValueError(
            f"nearest_cluster_center_ids length {nearest_arr.shape[0]} != num_col_matrices {num_col_matrices}"
        )

    t0 = time.perf_counter()
    seedlm_col_matrix_reconstruction = np.zeros((seeds_len, C), dtype=np.float32)
    for i in range(seeds_len):
        _, us = LFSR_get_Us(C, P, K_bits, seeds_arr[i])
        seedlm_col_matrix_reconstruction[i] = us @ ts_arr[i]
    load_timings["reconstruct_k_cluster_centers"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = np.zeros((row, col), dtype=np.float16)
    compact_i = 0
    for col_matrix_idx in range(num_col_matrices):
        row_block = col_matrix_idx // col
        col_idx = col_matrix_idx % col
        center_id = int(nearest_arr[col_matrix_idx])
        if center_id == -1:
            col_matrix = structured_arr[compact_i]
            compact_i += 1
        else:
            col_matrix = seedlm_col_matrix_reconstruction[center_id]
        out[row_block * C: (row_block + 1) * C, col_idx] = col_matrix.astype(
            np.float16, copy=False
        )
    load_timings["reconstruct_matrix"] = time.perf_counter() - t0

    return out, load_timings, load_bytes

