import time
from typing import Dict, Tuple

import torch

M, K, N = 768, 3072, 768


def _gflops(m: int, n: int, k: int, run_time_s: float) -> float:
    return 2.0 * m * n * k / run_time_s / 1e9


def _csr_csr_spmm_gflops(a_csr: torch.Tensor, b_csr: torch.Tensor, run_time_s: float) -> float:
    """GFLOPS for CSR @ CSR SPMM: 2 * sum_{(i,k) in A} nnz(B[k, :])."""
    a_col = a_csr.col_indices()
    b_crow = b_csr.crow_indices()
    row_nnz_b = b_crow[1:] - b_crow[:-1]
    flops = 2 * row_nnz_b[a_col].sum()
    return flops.item() / run_time_s / 1e9

def _csr_dense_spmm_gflops(a_csr, b_dense, run_time_s):
    nnz_a = a_csr._nnz()
    n = b_dense.shape[1]
    flops = 2 * nnz_a * n
    return flops / run_time_s / 1e9


def gemm_cpu(seed: int = 42, warmup: int = 3, repeats: int = 10, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, Dict[str, float]]:
    """A (M x K) @ B (K x N) on CPU."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    a = torch.randn(M, K, generator=gen, dtype=dtype).contiguous()
    b = torch.randn(K, N, generator=gen, dtype=dtype).contiguous()

    for _ in range(warmup):
        _ = a @ b

    start = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    run_time = (time.perf_counter() - start) / repeats

    return c, {"run_time_s": run_time, "gflops": _gflops(M, N, K, run_time)}


def gemm_gpu(seed: int = 42, warmup: int = 3, repeats: int = 10, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, Dict[str, float]]:
    """A (M x K) @ B (K x N) on GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run gemm_gpu.")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    a = torch.randn(M, K, generator=gen, dtype=dtype).to(device).contiguous()
    b = torch.randn(K, N, generator=gen, dtype=dtype).to(device).contiguous()

    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    torch.cuda.synchronize()
    run_time = (time.perf_counter() - start) / repeats

    return c, {"run_time_s": run_time, "gflops": _gflops(M, N, K, run_time)}


def compare_gemm(seed: int = 42, warmup: int = 3, repeats: int = 10) -> None:
    _, cpu_stats = gemm_cpu(seed=seed, warmup=warmup, repeats=repeats)
    print(f"GEMM shape: A({M}x{K}) @ B({K}x{N}) -> C({M}x{N})")
    print(
        f"CPU  run_time={cpu_stats['run_time_s']*1e3:.3f} ms  "
        f"GFLOPS={cpu_stats['gflops']:.2f}"
    )

    if torch.cuda.is_available():
        _, gpu_stats = gemm_gpu(seed=seed, warmup=warmup, repeats=repeats)
        speedup = cpu_stats["run_time_s"] / gpu_stats["run_time_s"]
        print(
            f"GPU  run_time={gpu_stats['run_time_s']*1e3:.3f} ms  "
            f"GFLOPS={gpu_stats['gflops']:.2f}  speedup={speedup:.2f}x"
        )
    else:
        print("GPU  skipped (CUDA not available)")


def csr_dense_spmm_cpu(
    seed: int = 42,
    warmup: int = 3,
    repeats: int = 10,
    sparsity: float = 0.9,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CSR sparse A (M x K) @ CSR sparse B (K x N) on CPU."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    a_dense = torch.randn(M, K, generator=gen, dtype=dtype)
    a_dense *= torch.rand(M, K, generator=gen) > sparsity
    a = a_dense.contiguous().to_sparse_csr()
    b_dense = torch.randn(K, N, generator=gen, dtype=dtype)
    # b_dense *= torch.rand(K, N, generator=gen) > sparsity
    # b = b_dense.contiguous().to_sparse_csr()
    b = b_dense.contiguous()

    for _ in range(warmup):
        _ = torch.sparse.mm(a, b)

    start = time.perf_counter()
    for _ in range(repeats):
        c = torch.sparse.mm(a, b)
    run_time = (time.perf_counter() - start) / repeats

    return c, {"run_time_s": run_time, "gflops": _csr_dense_spmm_gflops(a, b, run_time)}


def csr_dense_spmm_gpu(
    seed: int = 42,
    warmup: int = 3,
    repeats: int = 10,
    sparsity: float = 0.9,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CSR sparse A (M x K) @ CSR sparse B (K x N) on GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run spmm_gpu.")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    a_dense = torch.randn(M, K, generator=gen, dtype=dtype)
    a_dense *= torch.rand(M, K, generator=gen) > sparsity
    a = a_dense.contiguous().to_sparse_csr().to(device)
    b_dense = torch.randn(K, N, generator=gen, dtype=dtype)
    # b_dense *= torch.rand(K, N, generator=gen) > sparsity
    # b = b_dense.contiguous().to_sparse_csr().to(device)
    b = b_dense.to(device).contiguous()

    for _ in range(warmup):
        _ = torch.sparse.mm(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        c = torch.sparse.mm(a, b)
    torch.cuda.synchronize()
    run_time = (time.perf_counter() - start) / repeats

    return c, {"run_time_s": run_time, "gflops": _csr_dense_spmm_gflops(a, b, run_time)}



def compare_csr_dense_spmm(
    seed: int = 42,
    warmup: int = 3,
    repeats: int = 10,
    sparsity: float = 0.9,
) -> None:
    _, cpu_stats = csr_dense_spmm_cpu(seed=seed, warmup=warmup, repeats=repeats, sparsity=sparsity)
    print(f"SPMM shape: CSR A({M}x{K}) @ CSR B({K}x{N}) -> CSR C({M}x{N})  sparsity={sparsity:.0%}")
    print(
        f"CPU  run_time={cpu_stats['run_time_s']*1e3:.3f} ms  "
        f"GFLOPS={cpu_stats['gflops']:.2f}"
    )

    if torch.cuda.is_available():
        _, gpu_stats = csr_dense_spmm_gpu(seed=seed, warmup=warmup, repeats=repeats, sparsity=sparsity)
        speedup = cpu_stats["run_time_s"] / gpu_stats["run_time_s"]
        print(
            f"GPU  run_time={gpu_stats['run_time_s']*1e3:.3f} ms  "
            f"GFLOPS={gpu_stats['gflops']:.2f}  speedup={speedup:.2f}x"
        )
    else:
        print("GPU  skipped (CUDA not available)")


if __name__ == "__main__":
    compare_gemm()
    print()
    compare_csr_dense_spmm()
