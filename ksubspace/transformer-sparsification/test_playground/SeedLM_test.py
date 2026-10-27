import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
TEST_SEEDLM_DIR = _ROOT / "data" / "TEST" / "SeedLM_compressed"


from SeedLM.SeedLM_cpu import (
    XOR_LFSR,
    LFSR_get_Us,
    magnitude_pruning,
    find_K_cluster_centers,
    _pruned_tile_indices_from_sparsity,
    _build_precomputed,
    select_best_seed_numba,
    nearest_cluster_center_search,
    SeedLM_load,
    SeedLM_compress,
    structured_pruning_compress,
    structured_pruning_load,
    get_uniform_quantization_level,
    quantize,
    reconstruct_quantized_weight,
)
from SeedLM.SeedLM_gpu import (
    LFSR_get_Us as LFSR_get_Us_gpu,
    prune_matrix_cols_by_percent_gpu,
    restore_pruned_matrix_cols_gpu,
)

'''
Example usage (run from benchmark/ root with .venv activated):

# run all tests in this file (show print output)
pytest test_playground/SeedLM_test.py -s

# run a single test
pytest test_playground/SeedLM_test.py -s -k test_magnitude_pruning
'''

# test CPU
def test_XOR_LFSR():
    row_bin = np.array([1,1,0], dtype=bool)
    alpha = (1, 1, 0)
    K = 3
    for i in range(10):
        row_bin = XOR_LFSR(row_bin, alpha, K)
        print(f"[XOR_LFSR] row_bin: {row_bin}, LSB -> MSB")

# test CPU
def test_SeedLM():
    C = 5
    P = 3
    K = 3
    Vs, Us = LFSR_get_Us(C, P, K, 3569)
    print(f"\n[LFSR_get_Us] Vs shape: {Vs.shape}")
    print(f"[LFSR_get_Us] Vs: {Vs}")
    print(f"[LFSR_get_Us] Us shape: {Us.shape}")
    print(f"[LFSR_get_Us] Us: {Us}\n\n")

# Test quantization
def test_quantization():
    rows, cols = 3072, 768
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, size=(rows, cols)).astype(np.float16)

    for bits in [4, 5, 6, 7, 8]:
        quantization_level = get_uniform_quantization_level(-1.0, 1.0, bits)
        print(
            f"[quantization bits={bits}] quantization_level shape: {quantization_level.shape}, "
            f"dtype: {quantization_level.dtype}"
        )
        assert len(quantization_level) == 2**bits
        quantization_index = quantize(A, quantization_level, bits)
        print(
            f"[quantization bits={bits}] quantization_index shape: {quantization_index.shape}, "
            f"dtype: {quantization_index.dtype}"
        )
        assert quantization_index.shape == A.shape
        quantized_weight = reconstruct_quantized_weight(
            rows, cols, quantization_index, quantization_level, bits
        )
        print(
            f"[quantization bits={bits}] quantized_weight shape: {quantized_weight.shape}, "
            f"dtype: {quantized_weight.dtype}"
        )

        mse = np.mean((A.astype(np.float32) - quantized_weight.astype(np.float32)) ** 2)
        l2_norm = np.linalg.norm(A.astype(np.float32) - quantized_weight.astype(np.float32))
        standardized_l2_norm = l2_norm / np.linalg.norm(A.astype(np.float32))
        print(f"[quantization bits={bits}] MSE: {mse:.4f}")
        print(f"[quantization bits={bits}] L2 norm: {l2_norm:.4f}")
        print(f"[quantization bits={bits}] Standardized L2 norm: {standardized_l2_norm:.4f}\n")

def test_magnitude_pruning():
    sparsity = 0.38
    rng = np.random.default_rng(42)
    w = rng.uniform(-1.0, 1.0, size=(16, 128)).astype(np.float32)

    pruned = magnitude_pruning(w, sparsity)

    num_elements = w.size
    num_nonzero = int(np.count_nonzero(pruned))
    num_pruned_expected = int(round(num_elements * sparsity))
    num_nonzero_expected = num_elements - num_pruned_expected

    print(
        f"[magnitude_pruning] shape={w.shape}, sparsity={sparsity}\n"
        f"  num_elements={num_elements}\n"
        f"  num_nonzero after prune={num_nonzero}\n"
        f"  expected pruned (num_elements * sparsity)={num_pruned_expected}\n"
        f"  expected nonzero (num_elements - pruned)={num_nonzero_expected}"
    )

    assert pruned.shape == w.shape
    assert num_nonzero == num_nonzero_expected
    assert np.array_equal(magnitude_pruning(w, 0.0), w)
    assert np.all(magnitude_pruning(w, 1.0) == 0)

def test_SeedLM_quantization():
    value_bits, exponent_bits = 4, 4
    C, P, K = 8, 3, 16
    init_seed = 25565

    rng = np.random.default_rng(422)
    tile = rng.uniform(-1, 1, size=C).astype(np.float16)

    _, us_stack, pinv_stack = _build_precomputed(C, P, K, init_seed)
    _, best_t, _ = select_best_seed_numba(
        tile.reshape(1, C).astype(np.float32),
        us_stack,
        pinv_stack,
    )

    t = best_t[0].astype(np.float16)

    vmax = 2 ** (value_bits - 1) - 1    # 7 for 4-bit
    vmin = -2 ** (value_bits - 1)      # -8 for 4-bit

    print(f"[SeedLM_quantization] vmax: {vmax}, vmin: {vmin}")

    nz = t != 0
    if np.any(nz):
        max_abs = np.max(np.abs(t[nz])).astype(np.float32)

        # choose exponent so max_abs / 2^e fits into vmax
        shared_exponent = np.round(np.log2(max_abs / vmax)).astype(np.int8)
    else:
        shared_exponent = np.int8(0)

    scale = 2.0 ** shared_exponent

    int_value = np.clip(
        np.round(t.astype(np.float32) / scale),
        vmin,
        vmax,
    ).astype(np.int8)

    t_rec = (int_value.astype(np.float32) * scale).astype(np.float16)

    print(f"[SeedLM_quantization] shared_exponent: {shared_exponent} ({exponent_bits}-bit)")
    print(f"[SeedLM_quantization] int_value: {int_value} ({value_bits}-bit)")
    print(f"[SeedLM_quantization] t: {t}")
    print(f"[SeedLM_quantization] t_rec: {t_rec}")
    print(f"[SeedLM_quantization] diff: {t - t_rec}")
    print(f"[SeedLM_quantization] pass\n\n")



def test_prune_only():
    # Structured column-tile pruning: compress -> load -> inference (zeros pruned tiles).
    C = 8
    P = 3
    K_BITS = 4
    INIT_SEED = 25565
    rows, cols = 16, 100
    sparsity = 0.38
    K_cluster = 2
    selection_seed = 4256
    selection_pool = 100

    rng = np.random.default_rng(42)
    A = rng.standard_normal((rows, cols)).astype(np.float16)

    artifact_name = f"test_prune_only_{sparsity:.2f}.npy"
    structured_pruning_compress(
        A, sparsity, selection_pool, selection_seed,
        C=C, store_path=TEST_SEEDLM_DIR, artifact_name=artifact_name,
    )
    S = structured_pruning_load(
        rows,
        cols,
        sparsity,
        selection_pool,
        selection_seed,
        C=C,
        load_path=TEST_SEEDLM_DIR,
        load_compact_name=artifact_name,
    )
    assert S.shape == A.shape
    tiles_per_col = rows // C
    pruned_tile_idx = [
        col * tiles_per_col + tr
        for col in range(cols)
        for tr in range(tiles_per_col)
        if np.all(S[tr * C:(tr + 1) * C, col] == 0)
    ]
    print(f"[prune_only] pruned_tile_idx ({len(pruned_tile_idx)}): {pruned_tile_idx}")
    num_tiles = tiles_per_col * cols
    pruned_pct = 100.0 * len(pruned_tile_idx) / num_tiles
    print(f"[prune_only] tiles: {len(pruned_tile_idx)}/{num_tiles}, calculated pruned = {pruned_pct:.1f}%")
    print(f"[prune_only] pass in sparsity={sparsity}\n\n")


def test_prune_and_k_cluster():
    pass
    # Structured pruning + SeedLM K-cluster: compress then load.
    C=8
    P=3
    K_BITS=4
    M=4
    INIT_SEED=25565
    rows, cols = 16, 100
    sparsity = 0.38
    K_cluster = 2
    selection_seed = 4256
    selection_pool = 100

    rng = np.random.default_rng(42)
    A = rng.standard_normal((rows, cols)).astype(np.float16)

    TEST_SEEDLM_DIR.mkdir(parents=True, exist_ok=True)
    weight_name = "test_weight.npy"
    np.save(TEST_SEEDLM_DIR / weight_name, A.T.astype(np.float16))

    SeedLM_compress(
        sparsity,
        K_cluster,
        INIT_SEED,
        selection_pool,
        selection_seed,
        C=C,
        P=P,
        K_bits=K_BITS,
        load_path=TEST_SEEDLM_DIR,
        load_name=weight_name,
        storage_path=TEST_SEEDLM_DIR,
    )
    S_K_cluster, _, _ = SeedLM_load(
        rows,
        cols,
        sparsity,
        K_cluster,
        load_path=TEST_SEEDLM_DIR,
        C=C,
        P=P,
        K_bits=K_BITS,
    )
    artifact_name = f"test_prune_and_k_cluster_{sparsity:.2f}.npy"
    structured_pruning_compress(
        A, sparsity, selection_pool, selection_seed,
        C=C, store_path=TEST_SEEDLM_DIR, artifact_name=artifact_name,
    )
    S_with_zeros = structured_pruning_load(
        rows,
        cols,
        sparsity,
        selection_pool,
        selection_seed,
        C=C,
        load_path=TEST_SEEDLM_DIR,
        load_compact_name=artifact_name,
    )

    print(f"[prune_and_k_cluster] S_with_zeros.shape: {S_with_zeros.shape}")
    print(f"[prune_and_k_cluster] S_K_cluster.shape: {S_K_cluster.shape}")

    print(f"[prune_and_k_cluster] first 5 rows and columns of S_with_zeros: {S_with_zeros[:5, :5]}")
    print(f"[prune_and_k_cluster] first 5 rows and columns of S_K_cluster: {S_K_cluster[:5, :5]}")

    assert S_K_cluster.shape == A.shape
    tiles_per_col = rows // C
    recovered_tiles = 0
    for col in range(cols):
        for tr in range(tiles_per_col):
            tile_pruned = S_with_zeros[tr * C:(tr + 1) * C, col]
            tile_rec = S_K_cluster[tr * C:(tr + 1) * C, col]
            if np.all(tile_pruned == 0) and not np.all(tile_rec == 0):
                recovered_tiles += 1
    num_tiles = tiles_per_col * cols
    print(f"[prune_and_k_cluster] k_cluster={K_cluster}, recovered_tiles={recovered_tiles}/{num_tiles}, calculated pruned = {100.0 * recovered_tiles / num_tiles:.1f}%")
    print(f"[prune_and_k_cluster] pass in sparsity={sparsity}\n\n")
    assert recovered_tiles > 0, "Expected at least one pruned tile to be recovered"


SELECTION_SEEDS = [25565]
SPARSITY=0.2
ROWS=1024
COLS=64
C = 8
P_VALUES = [1,2,3,4,5,6,7,8]
K_bits = 16 
K_CLUSTER=int(ROWS/C*COLS*0.04)
'''
Run test:

pytest test_playground/SeedLM_test.py -s -k test_store

pytest test_playground/SeedLM_test.py -s -k test_load_and_inference
'''
def test_store():

    init_seed=25565
    selection_pool=100

    rng = np.random.default_rng(42)
    A = rng.standard_normal((ROWS, COLS)).astype(np.float16)

    TEST_SEEDLM_DIR.mkdir(parents=True, exist_ok=True)
    weight_name = "test_store_weight.npy"
    np.save(TEST_SEEDLM_DIR / weight_name, A.T.astype(np.float16))
    np.save(TEST_SEEDLM_DIR / "A.npy", A)

    for selection_seed in SELECTION_SEEDS:
        for P in P_VALUES:
            storage_path = (
                TEST_SEEDLM_DIR / f"selection_seed_{selection_seed}_P_{P}"
            )
            compress_timings, storage_bytes = SeedLM_compress(
                SPARSITY,
                K_CLUSTER,
                init_seed,
                selection_pool,
                selection_seed,
                C=C,
                P=P,
                K_bits=K_bits,
                load_path=TEST_SEEDLM_DIR,
                load_name=weight_name,
                storage_path=storage_path,
            )
            # print(
            #     f"[store_precomputation] selection_seed={selection_seed}, P={P}, "
            #     f"storage_path={storage_path}"
            # )
            # print(f"[store_precomputation] compress_timings: {compress_timings}")
            # print(f"[store_precomputation] storage_bytes: {storage_bytes}")

    print(f"\n\n")


def test_load_and_inference():
    # Requires test_store() to have run first (artifacts under TEST_SEEDLM_DIR).


    tiles_per_col = ROWS // C
    num_tiles = tiles_per_col * COLS
    A = np.load(TEST_SEEDLM_DIR / "A.npy")
    assert A.shape == (ROWS, COLS)
    A_f32 = A.astype(np.float32)
    A_l2_norm = np.linalg.norm(A_f32)

    for selection_seed in SELECTION_SEEDS:
        for P in P_VALUES:
            load_path = TEST_SEEDLM_DIR / f"selection_seed_{selection_seed}_P_{P}"
            pruned_tiles_loaded = np.load(load_path / "structured_pruning.npy")
            seeds_loaded = np.load(load_path / "seeds.npy")
            ts_loaded = np.load(load_path / "ts.npy")
            nearest_ids_loaded = np.load(
                load_path / "nearest_cluster_center_ids.npy"
            )

            assert pruned_tiles_loaded.shape == (
                int(round(num_tiles * (1 - SPARSITY))),
                C,
            )
            assert seeds_loaded.ndim == 1
            assert ts_loaded.ndim == 2
            assert nearest_ids_loaded.ndim == 1
            assert nearest_ids_loaded.shape[0] == num_tiles
            assert nearest_ids_loaded.dtype == np.int32

            reconstructed_matrix, load_timings, stored_bytes = SeedLM_load(
                ROWS,
                COLS,
                SPARSITY,
                K_CLUSTER,
                load_path=load_path,
                C=C,
                P=P,
                K_bits=K_bits,
            )
            assert reconstructed_matrix.shape == A.shape

            rec_f32 = reconstructed_matrix.astype(np.float32)
            mse = np.mean((A_f32 - rec_f32) ** 2)
            l2_norm = np.linalg.norm(A_f32 - rec_f32)
            standardized_l2_norm = l2_norm / A_l2_norm

            print(
                "[load_and_inference] parameters: "
                f"[selection_seed={selection_seed}, sparsity={SPARSITY}, "
                f"C={C}, P={P}, K_CLUSTER={K_CLUSTER}]"
            )
            print(f"[load_and_inference] MSE: {mse:.6f}")
            print(
                f"[load_and_inference] Standardized L2 norm: "
                f"{standardized_l2_norm:.6f}\n"
            )

    print(f"[load_and_inference] pass in sparsity={SPARSITY}\n")





#TODO: test GPU correctness later
# # test CPU
# def test_Random_Us():
#     C = 5
#     P = 3
#     K = 3
#     Vs, Us = random_Us(C, P, K, 3569)
#     print(f"\n[random_Us] Vs shape: {Vs.shape}")
#     print(f"[random_Us] Vs: {Vs}")
#     print(f"\n[random_Us] Us shape: {Us.shape}")
#     print(f"[random_Us] Us: {Us}")


# test GPU
# def test_LFSR_get_Us_gpu():
#     C = 5
#     P = 3
#     K = 3
#     Vs_gpu, Us_gpu = LFSR_get_Us_gpu(C, P, K, 3569)
#     Vs_cpu, Us_cpu = LFSR_get_Us(C, P, K, 3569)
#     Vs_gpu_np = Vs_gpu.detach().cpu().numpy()
#     Us_gpu_np = Us_gpu.detach().cpu().numpy()
#     assert (Vs_gpu_np == Vs_cpu).all(), "Vs_gpu != Vs_cpu"
#     assert (Us_gpu_np == Us_cpu).all(), "Us_gpu != Us_cpu"
#     print(f"\n[LFSR_get_Us_gpu] Vs shape: {Vs_gpu.shape}")
#     print(f"[LFSR_get_Us_gpu] Vs: {Vs_gpu}")
#     print(f"[LFSR_get_Us_gpu] Us shape: {Us_gpu.shape}")
#     print(f"[LFSR_get_Us_gpu] Us: {Us_gpu}")
#     print(f"\n[LFSR_get_Us_cpu] Vs shape: {Vs_cpu.shape}")
#     print(f"[LFSR_get_Us_cpu] Vs: {Vs_cpu}")
#     print(f"[LFSR_get_Us_cpu] Us shape: {Us_cpu.shape}")
#     print(f"[LFSR_get_Us_cpu] Us: {Us_cpu}\n\n")

# # test CPU
# def test_prune_matrix_cols_by_percent_cpu():
#     C = 8
#     rows, cols = 16, 2048
#     sparsity = 0.8964
#     rng = np.random.default_rng(42)
#     A = rng.standard_normal((rows, cols)).astype(np.float16)

#     S, dropped_values = prune_matrix_cols_by_percent(A, sparsity * 100)
#     _, stride, cols_drop = _structured_col_prune_params(cols, sparsity)
#     n_drop_cols = cols_drop.size

#     print(f"\n[prune_cpu] A.shape={A.shape}  sparsity={sparsity}  stride={stride}  dropped_cols={n_drop_cols}")
#     print(f"[prune_cpu] cols_to_drop={cols_drop}")
#     print(f"[prune_cpu] S.shape={S.shape}  dropped_values.len={dropped_values.size}")

#     assert S.shape == A.shape
#     assert S.dtype == np.float16
#     assert dropped_values.size == rows * n_drop_cols
#     assert (S == 0).sum() == rows * n_drop_cols
#     if n_drop_cols > 0:
#         assert np.array_equal(dropped_values, A[:, cols_drop].ravel())


# # test GPU (+ restore), compare S with CPU
# def test_prune_matrix_cols_by_percent_gpu():
#     C = 8
#     rows, cols = 16, 2048
#     sparsity = 0.8964
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     rng = np.random.default_rng(42)
#     A_np = rng.standard_normal((rows, cols)).astype(np.float16)

#     S_cpu, _ = prune_matrix_cols_by_percent(A_np, sparsity * 100)
#     S_gpu, dense, cols_to_drop = prune_matrix_cols_by_percent_gpu(
#         torch.from_numpy(A_np), sparsity, C, device=device
#     )
#     S_restored = restore_pruned_matrix_cols_gpu(
#         dense, sparsity, rows, cols, cols_to_drop, device=device
#     )

#     S_gpu_np = S_gpu.detach().cpu().numpy()
#     S_restored_np = S_restored.detach().cpu().numpy()
#     _, _, cols_drop = _structured_col_prune_params(cols, sparsity)
#     n_drop_cols = cols_drop.size

#     assert (S_gpu_np == S_cpu).all(), "S_gpu != S_cpu"
#     assert (S_restored_np == S_cpu).all(), "restored S != S_cpu"
#     assert dense.shape == (rows, cols - n_drop_cols)
#     assert dense.dtype == torch.float16

#     print(f"\n[prune_gpu test] device={device}  dense.shape={tuple(dense.shape)}")
#     print(f"[prune_gpu test] cols_to_drop={cols_to_drop.detach().cpu().numpy()}")
#     print(f"[prune_gpu test] S_gpu == S_cpu, restore OK")