# Adaptive Rank Allocation: How to Run

The Bash script automatically runs four steps:

1. Create the maximum-rank compression profile.
2. Select sensitive matrices using calibration.
3. Create the final adaptive-rank compressed model.
4. Calculate perplexity on WikiText-2.

## Run the OPT-125M script

Place the Bash file in the root of the `transformer-sparsification` repository:

```text
transformer-sparsification/
├── run_first_order_adaptive_opt125m_k512_sp100.sh
└── SeedLM/
```

Run:

```bash
cd /mnt/c/Users/jamalids/Documents/sparsity/transformer-sparsification
source .venv/bin/activate
chmod +x run_first_order_adaptive_opt125m_k512_sp100.sh
./run_first_order_adaptive_opt125m_k512_sp100.sh
```

Run in the background:

```bash
nohup ./run_first_order_adaptive_opt125m_k512_sp100.sh \
  > first_order_opt125m.out 2>&1 &

tail -f first_order_opt125m.out
```

## Use the script for another model

Make a copy:

```bash
cp run_first_order_adaptive_opt125m_k512_sp100.sh run_my_model.sh
```

Open `run_my_model.sh` and change the variables near the beginning.

### 1. Model name and model directory

OPT-125M:

```bash
MODEL="facebook/opt-125m"
INPUT_DIR="SeedLM/model/opt-125m"
```

Llama-3.2-1B:

```bash
MODEL="meta-llama/Llama-3.2-1B"
INPUT_DIR="SeedLM/model/Llama-3.2-1B"
```

### 2. Compression settings

Change these values as required:

```bash
K=512
BASE_RANK=4
MAX_RANK=6
SPARSITY=1.0
EXTRA_BUDGET_MB=1.88
```

`SPARSITY=1.0` uses all selected tiles. `SPARSITY=0.5` uses 50% tile coverage.

The script automatically selects which matrices receive `MAX_RANK`. The other matrices use `BASE_RANK`.

### 3. Expected matrix count

```bash
EXPECTED_MATRICES=72
```

Use:

```text
OPT-125M:       72
Llama-3.2-1B: 112
```

### 4. Output paths

Change the model name, K, and sparsity in all output paths:

```bash
PROFILE_ROOT="SeedLM/data/SeedLM_compressed/opt-125m/first_order_profile_K512_sp100"
PROFILE_DIR="${PROFILE_ROOT}/sp_1.00_K512_R6_B8_C8_W1"

ALLOCATION_DIR="results/opt-125m/first_order_full_K512_sp100"
OVERRIDE_FILE="${ALLOCATION_DIR}/rank_override_arguments.txt"

ADAPTIVE_ROOT="SeedLM/data/SeedLM_compressed/opt-125m/first_order_adaptive_K512_sp100"
INFERENCE_DIR="results/opt-125m/inference_first_order_K512_sp100"
```

The `PROFILE_DIR` name must match the settings exactly:

```text
sp_1.00_K512_R6_B8_C8_W1
```

For example, Llama-3.2-1B with K=256 and full tile coverage uses:

```bash
K=256
BASE_RANK=4
MAX_RANK=6
SPARSITY=1.0
EXTRA_BUDGET_MB=64.13
EXPECTED_MATRICES=112

PROFILE_ROOT="SeedLM/data/SeedLM_compressed/Llama-3.2-1B/first_order_profile_K256_sp100"
PROFILE_DIR="${PROFILE_ROOT}/sp_1.00_K256_R6_B8_C8_W1"
ALLOCATION_DIR="results/Llama-3.2-1B/first_order_full_K256_sp100"
OVERRIDE_FILE="${ALLOCATION_DIR}/rank_override_arguments.txt"
ADAPTIVE_ROOT="SeedLM/data/SeedLM_compressed/Llama-3.2-1B/first_order_adaptive_K256_sp100"
INFERENCE_DIR="results/Llama-3.2-1B/inference_first_order_K256_sp100"
```

After editing the file, run:

```bash
chmod +x run_my_model.sh
./run_my_model.sh
```

The final PPL CSV is saved inside the directory specified by `INFERENCE_DIR`.
