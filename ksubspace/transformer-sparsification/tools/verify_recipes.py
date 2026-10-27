"""Verify the patched eval path reproduces the reference dense perplexities.

Calls the repo's own test_loader()/perplexity() exactly as arks_pipeline.sh does.
Targets (facebook/opt-125m, seqlen 2048): wikitext2 27.65, ptb 38.99, c4 26.56
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "final1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from bench_inference_minibatch_vq import test_loader, perplexity, resolve_dataset
from arks_common import DATASET_PRESETS

MODEL = os.environ.get("PROBE_MODEL", "facebook/opt-125m")
TARGETS = {"wikitext2": 27.65, "ptb": 38.99, "c4": 26.56}
# split + chunk cap, mirroring what run.sh passes per corpus
SPEC = {
    "wikitext2": ("test", 0),
    "ptb":       ("test", 0),
    "c4":        ("validation", 256),
}

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()

ok = True
for name, (split, cap) in SPEC.items():
    a = argparse.Namespace(
        dataset=name, dataset_config=None, dataset_split=split,
        dataset_text_column="auto", dataset_data_files=None, dataset_recipe=name,
    )
    kw = resolve_dataset(a)
    print(f"\n[{name}] resolve_dataset -> {kw}", flush=True)
    loader = test_loader(tok, 2048, 1, cap, MODEL, **kw)
    ppl, ntok = perplexity(model, loader, "cuda", tok.pad_token_id)
    tgt = TARGETS[name]
    hit = abs(ppl - tgt) < 0.05
    ok &= hit
    print(f"[{name}] ppl={ppl:.4f}  target={tgt}  tokens={ntok}  {'MATCH' if hit else 'MISMATCH'}", flush=True)

print("\nRESULT:", "ALL MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
