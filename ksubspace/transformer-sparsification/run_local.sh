#!/bin/bash
# =============================================================================
# SeedLM OPT benchmark — local single-process full pipeline
#
# Uses globals from SeedLM/bench_compress_and_store.py
# (OPT_MODELS, DEFAULT_SPARSITIES, K_CLUSTER_TILE_FRACTION, QUANTIZATION_BITS).
#
# Runs:
#   1) SeedLM/bench_compress_and_store.py  (all sections)
#   2) SeedLM/bench_load_and_inference.py  (all sections)
#
# JSON is written under results/{model}/CPU_{compress,inference}_result/
# (not server_temp). For cluster job-split runs use ./run.sh.
#
# Example usage:
#   ./run_local.sh setup
#       Create .venv and install requirements.txt
#
#   ./run_local.sh
#       Compress all models in OPT_MODELS, then load/inference
#
#   ./run_local.sh --device cuda --max-ppl-samples 8
#       Forward extra flags to both benches
#
#   ./run_local.sh compress
#       Only compress & store
#
#   ./run_local.sh load --device cuda
#       Only load & inference (requires compress artifacts)
#
#   ./run_local.sh --model facebook/opt-125m
#       Note: --model must also be in OPT_MODELS choices
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
COMPRESS_PY="$ROOT/SeedLM/bench_compress_and_store.py"
LOAD_PY="$ROOT/SeedLM/bench_load_and_inference.py"

setup_venv() {
  echo "[setup] creating venv at $VENV"
  python3 -m venv "$VENV"
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r "$ROOT/requirements.txt"
  echo "[setup] done — activate with: source $VENV/bin/activate"
}

ensure_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN"
    echo "Run first: $ROOT/run_local.sh setup"
    exit 1
  fi
}

print_globals() {
  echo "[local] reading globals from SeedLM/bench_compress_and_store.py"
  echo -n "  OPT_MODELS: "
  "$PYTHON_BIN" -c "from SeedLM.bench_compress_and_store import OPT_MODELS; print(OPT_MODELS)"
  echo -n "  DEFAULT_SPARSITIES: "
  "$PYTHON_BIN" -c "from SeedLM.bench_compress_and_store import DEFAULT_SPARSITIES; print(DEFAULT_SPARSITIES)"
  echo -n "  K_CLUSTER_TILE_FRACTION: "
  "$PYTHON_BIN" -c "from SeedLM.bench_compress_and_store import K_CLUSTER_TILE_FRACTION; print(K_CLUSTER_TILE_FRACTION)"
  echo -n "  QUANTIZATION_BITS: "
  "$PYTHON_BIN" -c "from SeedLM.bench_compress_and_store import QUANTIZATION_BITS; print(QUANTIZATION_BITS)"
}

run_compress() {
  echo "[local] compress: $COMPRESS_PY $*"
  "$PYTHON_BIN" "$COMPRESS_PY" "$@"
}

run_load() {
  echo "[local] load: $LOAD_PY $*"
  "$PYTHON_BIN" "$LOAD_PY" "$@"
}

usage() {
  sed -n '1,40p' "$0" | sed 's/^# \\?//'
}

if [[ "${1:-}" == "setup" ]]; then
  setup_venv
  exit 0
fi

if [[ "${1:-}" == "help" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ensure_python
print_globals

mode="all"
extra=()
if [[ "${1:-}" == "compress" || "${1:-}" == "load" ]]; then
  mode="$1"
  shift
fi
extra=("$@")

case "$mode" in
  all)
    run_compress "${extra[@]}"
    run_load "${extra[@]}"
    ;;
  compress)
    run_compress "${extra[@]}"
    ;;
  load)
    run_load "${extra[@]}"
    ;;
esac

echo "[local] done"
