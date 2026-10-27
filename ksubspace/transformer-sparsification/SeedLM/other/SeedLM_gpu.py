from typing import Tuple, Union
import numpy as np
import torch

def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _cols_to_drop(cols: int, sparsity: float, device: torch.device) -> torch.Tensor:
    """Dropped column indices (same stride rule as SeedLM_cpu.prune_matrix_cols_by_percent)."""
    from SeedLM.other.SeedLM_cpu import _structured_col_prune_params

    _, _, cols_np = _structured_col_prune_params(cols, sparsity)
    return torch.from_numpy(cols_np).to(device)


def _tile_sparsity(num_row_tiles: int, cols: int, n_drop_cols: int) -> float:
    total_tiles = num_row_tiles * cols
    if total_tiles == 0:
        return 0.0
    return (n_drop_cols * num_row_tiles) / total_tiles


def prune_matrix_cols_by_percent_gpu(
    A: torch.Tensor,
    sparsity: float,
    C: int,
    device: str | None = None,
    return_indices: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    GPU version of SeedLM_cpu.prune_matrix_cols_by_percent.

    Structured column pruning (each dropped column removes Cx1 tiles in that column).
    ``sparsity`` in [0, 1] is the target fraction of columns to prune (rounded).

    Returns:
        S: pruned full-size matrix (float16), zeros at dropped columns
        dense: kept columns concatenated (float16), shape (rows, cols - k)
        cols_to_drop: dropped column indices, for restore
    """
    if A.ndim != 2:
        raise ValueError("prune_matrix_cols_by_percent_gpu expects a 2D matrix.")

    if device is None:
        device = str(A.device) if A.device.type != "cpu" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    dev = torch.device(device)
    A = A.to(device=dev, dtype=torch.float16)

    rows, cols = A.shape
    if rows % C != 0:
        raise ValueError(f"rows={rows} must be a multiple of tile height C={C}.")

    num_row_tiles = rows // C

    print(
        f"[prune_gpu input] shape=({rows}, {cols})  "
        f"target_sparsity={sparsity:.4f}  tiles={num_row_tiles * cols} (C={C})  "
        f"size={_tensor_nbytes(A)} bytes (float16)"
    )

    cols_to_drop = _cols_to_drop(cols, sparsity, dev)
    k = int(cols_to_drop.numel())
    keep_mask = torch.ones(cols, dtype=torch.bool, device=dev)
    if k > 0:
        keep_mask[cols_to_drop] = False

    S = A.clone()
    if k > 0:
        S[:, cols_to_drop] = 0
    dense = A[:, keep_mask]

    output_tile_sparsity = _tile_sparsity(num_row_tiles, cols, k)
    print(
        f"[prune_gpu output] S.shape=({rows}, {cols})  dense.shape={tuple(dense.shape)}  "
        f"tile_sparsity={output_tile_sparsity:.4f}  "
        f"S_size={_tensor_nbytes(S)} bytes  dense_size={_tensor_nbytes(dense)} bytes (float16)"
    )

    if return_indices:
        dropped_indices = (
            torch.arange(rows, device=dev, dtype=torch.int64)[:, None] * cols
            + cols_to_drop[None, :]
        ).reshape(-1)
        dropped_values = A.reshape(-1)[dropped_indices].clone()
        return S, dense, cols_to_drop, dropped_values, dropped_indices
    return S, dense, cols_to_drop


def restore_pruned_matrix_cols_gpu(
    dense: torch.Tensor,
    sparsity: float,
    rows: int,
    cols: int,
    cols_to_drop: torch.Tensor | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """
    input is dense matrix,
    this function is to add (restore) tiles with all zeros to pruned position
    
    Restore full pruned matrix from packed dense columns + drop pattern."""
    if device is None:
        device = str(dense.device)
    dev = torch.device(device)
    dense = dense.to(device=dev, dtype=torch.float16)

    if cols_to_drop is None:
        cols_to_drop = _cols_to_drop(cols, sparsity, dev)
    else:
        cols_to_drop = cols_to_drop.to(device=dev, dtype=torch.int64)

    keep_mask = torch.ones(cols, dtype=torch.bool, device=dev)
    if cols_to_drop.numel() > 0:
        keep_mask[cols_to_drop] = False

    S = torch.zeros(rows, cols, dtype=torch.float16, device=dev)
    S[:, keep_mask] = dense
    return S










# LFSR initial "1" position, refers to SeedLM paper Table 6.
_LFSR_ALPHA_TABLE: dict[int, Tuple[int, ...]] = {
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


def LFSR_get_Us(
    C: int,
    P: int,
    K: int,
    seed: int,
    device: str | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU version of LFSR_get_Us.

    Returns:
        Vs: float32, shape (C, P) — integer codebook values before normalization
        Us: float16, shape (C, P) — normalized to [-1, 1]
    """
    if K not in _LFSR_ALPHA_TABLE:
        raise ValueError(
            f"LFSR_get_Us: K={K} not in alpha_table "
            f"(supported K: {sorted(_LFSR_ALPHA_TABLE.keys())})"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    taps = _LFSR_ALPHA_TABLE[K]
    n_rows = 2**K - 1
    
    # build all binary LFSR states (row). U_bin is a loop-like array with shape (2^K-1, K) with bool.
    U_bin = torch.zeros((n_rows, K), dtype=torch.uint8, device=dev)
    
    # initialize the first row of U_bin with the taps (_LFSR_ALPHA_TABLE[K]).
    for tap_idx in taps:
        U_bin[0, tap_idx] = 1
    
    # iterate over the rest of the rows to build the rest of the U_bin.
    for row_idx in range(1, n_rows):
        prev = U_bin[row_idx - 1]
        # compute the new bit using XOR operation, see forluma in 3.1 in SeedLM paper.
        new_bit = (prev[list(taps)].sum() % 2).to(dtype=torch.uint8)
        # shift the previous row to the left by one and add the new bit to the end.
        U_bin[row_idx, :-1] = prev[1:K]
        U_bin[row_idx, -1] = new_bit

    bit_weights = torch.arange(K, device=dev, dtype=torch.int32)
    U_dec = (U_bin.to(torch.int32) << bit_weights).sum(dim=1)

    # Match SeedLM_cpu.py RNG (numpy PCG64); torch.Generator uses a different sequence.
    start_idx = int(np.random.default_rng(seed).integers(0, n_rows))
    idx = (start_idx + torch.arange(C * P, device=dev, dtype=torch.int64).reshape(C, P)) % n_rows
    Vs = U_dec[idx].to(torch.float32)

    coeff = torch.tensor(2 ** (K - 1), dtype=torch.float32, device=dev)
    Us = ((Vs - coeff) / (coeff - 1.0)).to(output_dtype)
    return Vs, Us
