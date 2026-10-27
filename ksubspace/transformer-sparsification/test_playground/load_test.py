from pathlib import Path

import numpy as np

'''
pytest test_playground/load_test.py -s
'''


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "SeedLM_compressed"


def test_load_ts():
    for path in sorted(DATA_DIR.glob("ts_*.npy")):
        ts = np.load(path)
        print(
            f"{path.name}: shape={ts.shape}, len={len(ts)}, "
            f"min={ts.min():.6g}, max={ts.max():.6g}, dtype={ts.dtype}"
        )
