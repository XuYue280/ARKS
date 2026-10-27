"""Measure dense opt-125m PPL under several text-assembly recipes.

Reference targets (GPTQ/SparseGPT convention):
    wikitext2 27.65   ptb 38.99   c4 26.56
"""
import math, os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = os.environ.get("PROBE_MODEL", "facebook/opt-125m")
SEQLEN = 2048
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
pad = tok.pad_token_id


@torch.no_grad()
def ppl_from_ids(ids, max_chunks=None):
    """Same math as final1/bench_inference_minibatch_vq.py:perplexity()."""
    n = ids.numel() // SEQLEN
    x = ids[: n * SEQLEN].reshape(n, SEQLEN)
    if max_chunks:
        x = x[:max_chunks]
    lf = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad if pad is not None else -100)
    tot, cnt = 0.0, 0
    for i in range(x.shape[0]):
        b = x[i : i + 1].to(DEV)
        logits = model(input_ids=b, attention_mask=torch.ones_like(b)).logits[:, :-1, :]
        lab = b[:, 1:]
        nll = lf(logits.permute(0, 2, 1), lab)
        m = (lab != pad) if pad is not None else torch.ones_like(lab, dtype=torch.bool)
        tot += float((nll * m).sum()); cnt += int(m.sum())
    return math.exp(tot / cnt), cnt, x.shape[0]


def enc(text):
    return tok(text, return_tensors="pt").input_ids[0]


results = {}

def run(tag, ids, max_chunks=None):
    p, ntok, nch = ppl_from_ids(ids, max_chunks)
    results[tag] = p
    print(f"{tag:<46} ppl={p:8.4f}  tokens={ntok:>9}  chunks={nch}", flush=True)


# ---------------- WikiText-2 ----------------
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
run("wikitext2  CURRENT/GPTQ  '\\n\\n'.join  full", enc("\n\n".join(wt["text"])))

# ---------------- PTB ----------------
ptb_src = {}
try:
    d = load_dataset("ptb_text_only", "penn_treebank", split="test")
    ptb_src["ptb_text_only.sentence"] = list(d["sentence"])
except Exception as e:
    print("  [ptb_text_only unavailable]", str(e)[:120])
for repo, col in [("ilovefpga/ptb_text", None), ("FALcon6/ptb_text_only", None)]:
    try:
        d = load_dataset(repo, split="test")
        col = col or ("sentence" if "sentence" in d.column_names else d.column_names[0])
        ptb_src[f"{repo}.{col}"] = list(d[col])
    except Exception as e:
        print(f"  [{repo} unavailable]", str(e)[:120])
raw = "/scratch/xuy280/hf_cache/ptb_text_only_raw/ptb.test.txt"
if os.path.exists(raw):
    lines = open(raw).read().split("\n")
    ptb_src["raw ptb.test.txt lines"] = lines
    ptb_src["raw ptb.test.txt lines.strip()"] = [l.strip() for l in lines]

for name, rows in ptb_src.items():
    run(f"ptb  [{name}]  ' '.join", enc(" ".join(rows)))
    run(f"ptb  [{name}]  '\\n\\n'.join (CURRENT)", enc("\n\n".join(rows)))

# ---------------- C4 ----------------
c4 = load_dataset(
    "allenai/c4", "en",
    data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
    split="validation",
)
run("c4  GPTQ  val[:1100] ' '.join  256x2048", enc(" ".join(c4[:1100]["text"])), 256)
run("c4  CURRENT-ish  full '\\n\\n'.join  256x2048", enc("\n\n".join(c4["text"][:20000])), 256)

print("\nJSON:", json.dumps(results))
