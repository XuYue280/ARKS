"""Prove that layer-wise inference reproduces the all-on-GPU perplexity exactly.

Runs the same corpus twice on one model -- once with the whole model resident
on the GPU, once streaming a single decoder layer at a time -- and compares
the perplexities at full float precision, plus the peak VRAM and wall time of
each. Anything short of an exact match is a bug, not a tolerance question:
with --batch-size 1 both paths run identical modules, in identical dtypes, on
the same device, in the same order.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "final1"))

from transformers import AutoTokenizer  # noqa: E402
from bench_inference_minibatch_vq import (  # noqa: E402
    load_eval_model,
    perplexity,
    perplexity_layerwise,
    resolve_dataset,
    test_loader,
)

CORPORA = {"wikitext2": ("test", 0), "ptb": ("test", 0), "c4": ("validation", 256)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--corpora", default="wikitext2,ptb,c4")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--loss-chunk-tokens", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = []
    ok = True
    for name in [c.strip() for c in args.corpora.split(",") if c.strip()]:
        split, cap = CORPORA[name]
        ns = argparse.Namespace(
            dataset=name, dataset_config=None, dataset_split=split,
            dataset_text_column="auto", dataset_data_files=None, dataset_recipe=name,
        )
        loader = test_loader(tok, args.seqlen, 1, cap, args.model, **resolve_dataset(ns))

        out = {}
        for mode, fn in (("full", perplexity), ("layerwise", perplexity_layerwise)):
            model = load_eval_model(args.model, device, layerwise=(mode == "layerwise"))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            ppl, ntok = fn(model, loader, device, tok.pad_token_id, args.loss_chunk_tokens)
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            out[mode] = (ppl, ntok, elapsed, peak)
            del model
            torch.cuda.empty_cache()
            print(f"  [{name}/{mode}] ppl={ppl!r} tokens={ntok} {elapsed:.1f}s peak={peak:.2f}GiB",
                  flush=True)

        (p1, n1, t1, m1), (p2, n2, t2, m2) = out["full"], out["layerwise"]
        exact = (p1 == p2) and (n1 == n2)
        ok &= exact
        rows.append((name, p1, p2, t1, t2, m1, m2, exact))

    print("\n" + "=" * 96)
    print(f"{'corpus':<11}{'full PPL':>20}{'layerwise PPL':>20}{'full s':>9}{'lw s':>9}"
          f"{'full GiB':>10}{'lw GiB':>9}{'':>6}")
    print("=" * 96)
    for name, p1, p2, t1, t2, m1, m2, exact in rows:
        print(f"{name:<11}{p1:>20.12f}{p2:>20.12f}{t1:>9.1f}{t2:>9.1f}"
              f"{m1:>10.2f}{m2:>9.2f}{'  EXACT' if exact else '  DIFF':>6}")
    print("=" * 96)
    print("\nRESULT:", "所有语料逐位一致" if ok else "存在差异")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
