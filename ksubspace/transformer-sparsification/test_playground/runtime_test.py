import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from numba import njit

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

'''
Example usage (run from benchmark/ root with .venv activated):

pytest test_playground/runtime_test.py -s
pytest test_playground/runtime_test.py -s -k test_select_best_seed_runtime
'''

N_CENTERS = 1000
C = 8
P = 3
N_SEEDS = 2**16
RNG_SEED = 42
N_WARMUP = 1
N_REPEAT = 3


@njit
def select_best_seed_numba_with_square_error(
    centers: np.ndarray,
    us_stack: np.ndarray,
    pinv_stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each cluster center, pick the seed with minimum squared L2 reconstruction error.
    Pass float32 centers / Us / pinv_U (Numba does not support float16).

    Returns (best_si, best_t, best_w_rec):
        best_si[j]   — seed index for cluster center j
        best_t[j, :] — (P,) coefficients t = pinv_U @ center
        best_w_rec[j, :] — (C,) reconstructed tile Us @ t
    """
    n_centers, c = centers.shape
    s = us_stack.shape[0]
    p = us_stack.shape[2]
    min_err = np.full(n_centers, np.inf)
    best_si = np.full(n_centers, -1, dtype=np.int32)
    best_t = np.zeros((n_centers, p), dtype=np.float32)
    best_w_rec = np.zeros((n_centers, c), dtype=np.float32)

    for si in range(s):
        t = pinv_stack[si] @ centers.T
        w_rec = us_stack[si] @ t
        for j in range(n_centers):
            err = 0.0
            for cc in range(c):
                d = centers[j, cc] - w_rec[cc, j]
                err += d * d
            if err < min_err[j]:
                min_err[j] = err
                best_si[j] = si
                for pp in range(p):
                    best_t[j, pp] = t[pp, j]
                for cc in range(c):
                    best_w_rec[j, cc] = w_rec[cc, j]

    return best_si, best_t, best_w_rec


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
        best_w_rec[j, :] — (C,) reconstructed tile Us @ t
    """
    n_centers, c = centers.shape
    s = us_stack.shape[0]
    p = us_stack.shape[2]
    min_err = np.full(n_centers, np.inf)
    best_si = np.full(n_centers, -1, dtype=np.int32)
    best_t = np.zeros((n_centers, p), dtype=np.float32)
    best_w_rec = np.zeros((n_centers, c), dtype=np.float32)

    for si in range(s):
        t = pinv_stack[si] @ centers.T
        w_rec = us_stack[si] @ t
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


def _format_runtime(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600}h {(total % 3600) // 60}m {total % 60}s"


def _generate_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)

    # 1000 centers, each (8, 1) column vector -> (1000, 8) for Numba.
    centers = rng.uniform(-1.0, 1.0, size=(N_CENTERS, C, 1)).astype(np.float16)
    centers_f32 = centers.reshape(N_CENTERS, C).astype(np.float32)

    us_stack = np.empty((N_SEEDS, C, P), dtype=np.float32)
    pinv_stack = np.empty((N_SEEDS, P, C), dtype=np.float32)
    for si in range(N_SEEDS):
        us = rng.uniform(-1.0, 1.0, size=(C, P)).astype(np.float16)
        us_f32 = us.astype(np.float32)
        us_stack[si] = us_f32
        pinv_stack[si] = np.linalg.pinv(us_f32).astype(np.float32)

    return centers_f32, us_stack, pinv_stack


def _bench(
    fn,
    centers: np.ndarray,
    us_stack: np.ndarray,
    pinv_stack: np.ndarray,
    *,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
) -> list[float]:
    for _ in range(n_warmup):
        fn(centers, us_stack, pinv_stack)
    times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn(centers, us_stack, pinv_stack)
        times.append(time.perf_counter() - t0)
    return times


def test_select_best_seed_runtime() -> None:
    print(
        f"\n[runtime_test] generating inputs: "
        f"centers=({N_CENTERS}, {C}), seeds={N_SEEDS}, Us=({C}, {P})"
    )
    t0 = time.perf_counter()
    centers, us_stack, pinv_stack = _generate_inputs()
    setup_time = time.perf_counter() - t0
    print(f"[runtime_test] data generation time = {_format_runtime(setup_time)}")
    print(
        f"[runtime_test] centers dtype={centers.dtype}, "
        f"us_stack shape={us_stack.shape}, pinv_stack shape={pinv_stack.shape}"
    )

    best_sq = select_best_seed_numba_with_square_error(centers, us_stack, pinv_stack)
    best_rt = select_best_seed_numba(centers, us_stack, pinv_stack)
    assert np.array_equal(best_sq[0], best_rt[0]), "sqrt vs square error should pick same seeds"

    times_sq = _bench(select_best_seed_numba_with_square_error, centers, us_stack, pinv_stack)
    times_rt = _bench(select_best_seed_numba, centers, us_stack, pinv_stack)

    print("[runtime_test] select_best_seed_numba_with_square_error:")
    for i, t in enumerate(times_sq, start=1):
        print(f"  run {i}: {_format_runtime(t)} ({t:.3f}s)")
    print(f"  best: {_format_runtime(min(times_sq))} ({min(times_sq):.3f}s)")
    print(f"  avg:  {_format_runtime(float(np.mean(times_sq)))} ({float(np.mean(times_sq)):.3f}s)")

    print("[runtime_test] select_best_seed_numba (with sqrt):")
    for i, t in enumerate(times_rt, start=1):
        print(f"  run {i}: {_format_runtime(t)} ({t:.3f}s)")
    print(f"  best: {_format_runtime(min(times_rt))} ({min(times_rt):.3f}s)")
    print(f"  avg:  {_format_runtime(float(np.mean(times_rt)))} ({float(np.mean(times_rt)):.3f}s)")

    speedup = float(np.mean(times_rt)) / float(np.mean(times_sq))
    print(f"[runtime_test] avg speedup (sqrt / square): {speedup:.3f}x\n")
