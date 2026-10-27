# benchmark

create venv and install requirements:
```bash
cd benchmark/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Memory cost analysis:
```bash
python -c "
from SeedLM.SeedLM_cpu import calculate_memory_storage_cost
calculate_memory_storage_cost(M=4, C=8, P=3, K_bits=16)
"
```
