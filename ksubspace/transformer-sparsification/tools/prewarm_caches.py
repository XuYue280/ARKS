#!/usr/bin/env python3
"""Populate every cache a network-less compute node will need, on the login node.

Trillium's GPU compute nodes (trig*) have no outbound network, and nothing in
final1/ passes local_files_only=. So anything that would touch the Hub or the
datasets server at run time has to be materialised here first:

  1. the model snapshot (HF hub cache), and
  2. the *tokenised* corpora (ARKS_TOKEN_CACHE) -- for both the evaluation
     corpora and the allocator's calibration split.

(2) matters more than it looks: with the token cache warm, the compute node
never calls load_dataset() at all, so the `datasets` library never needs the
network even for C4's pinned shard or PTB's raw githubusercontent URL.

The token-cache key includes the tokenizer signature and the transformers
version, so this must run in the same venv the jobs will use.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "final1"))

# eval corpus -> (split, recipe). Mirrors run.sh's EVAL_SPECS.
EVAL_CORPORA = {
    "wikitext2": ("test", "wikitext2"),
    "ptb": ("test", "ptb"),
    "c4": ("validation", "c4"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="wikitext2,ptb,c4")
    ap.add_argument("--calibration-dataset", default="wikitext2")
    ap.add_argument("--calibration-split", default="train")
    ap.add_argument("--skip-weights", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("ARKS_TOKEN_CACHE"):
        print("ARKS_TOKEN_CACHE is not set -- token caches would not persist", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer, AutoConfig
    from arks_common import cached_token_ids, dataset_text_column, DATASET_PRESETS
    from bench_inference_minibatch_vq import test_loader, resolve_dataset

    ok = True

    # ---- 1. model snapshot -------------------------------------------------
    if not args.skip_weights:
        from huggingface_hub import snapshot_download, list_repo_files
        try:
            files = list_repo_files(args.model)
            has_st = any(f.endswith(".safetensors") for f in files)
            patterns = ["*.json", "*.txt", "*.model", "*.tiktoken", "tokenizer*", "*.py"]
            patterns.append("*.safetensors" if has_st else "*.bin")
            patterns.append("*.safetensors.index.json" if has_st else "*.bin.index.json")
            # `or None`, never `or True`: True *requires* a stored token and
            # fails on public repos when HF_HOME has none. None = anonymous for
            # public repos, and HF_TOKEN still covers the gated Llama ones.
            path = snapshot_download(
                args.model, allow_patterns=patterns,
                token=os.environ.get("HF_TOKEN") or None,
            )
            nw = len([f for f in os.listdir(path) if f.endswith((".safetensors", ".bin"))])
            print(f"[prewarm] weights OK  {args.model} -> {path} ({nw} file(s))", flush=True)
        except Exception as exc:
            ok = False
            print(f"[prewarm] weights FAIL {args.model}: {exc}", file=sys.stderr, flush=True)
            return 1

    # Config + tokenizer must load offline later; touch them now.
    AutoConfig.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- 2. evaluation corpora (tokenised) ---------------------------------
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        split, recipe = EVAL_CORPORA.get(name, ("test", "default"))
        ns = argparse.Namespace(
            dataset=name, dataset_config=None, dataset_split=split,
            dataset_text_column="auto", dataset_data_files=None, dataset_recipe=recipe,
        )
        try:
            # test_loader() -> cached_token_ids() writes the cache entry the
            # compute node will hit. maximum=1 keeps this cheap; the cache
            # stores the full token stream regardless of the chunk cap.
            test_loader(tok, 2048, 1, 1, args.model, **resolve_dataset(ns))
            print(f"[prewarm] eval  OK  {args.model} {name} split={split} recipe={recipe}", flush=True)
        except Exception as exc:
            ok = False
            print(f"[prewarm] eval  FAIL {args.model} {name}: {exc}", file=sys.stderr, flush=True)

    # ---- 3. allocator calibration split ------------------------------------
    preset = DATASET_PRESETS.get(args.calibration_dataset)
    cal_name = preset["name"] if preset else args.calibration_dataset
    cal_cfg = preset["config"] if preset else None
    cal_files = (preset or {}).get("data_files") or None
    if isinstance(cal_files, dict):
        cal_files = cal_files.get(args.calibration_split)
    try:
        # Same call shape as first_order_rank_allocator.calibration_tensor().
        cached_token_ids(
            tok, model_id=args.model, dataset_name=cal_name, dataset_config=cal_cfg,
            split=args.calibration_split, variant="join_all",
            build_text=lambda ds: "\n\n".join(ds[dataset_text_column(ds, "auto")]),
            data_files=cal_files,
        )
        print(f"[prewarm] calib OK  {args.model} {args.calibration_dataset} split={args.calibration_split}", flush=True)
    except Exception as exc:
        ok = False
        print(f"[prewarm] calib FAIL {args.model}: {exc}", file=sys.stderr, flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
