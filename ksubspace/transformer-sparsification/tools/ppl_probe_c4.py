import math, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = os.environ.get("PROBE_MODEL", "facebook/opt-125m"); SEQLEN = 2048; DEV = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
pad = tok.pad_token_id

@torch.no_grad()
def ppl(ids, max_chunks=None):
    n = ids.numel() // SEQLEN
    x = ids[: n * SEQLEN].reshape(n, SEQLEN)
    if max_chunks: x = x[:max_chunks]
    lf = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad if pad is not None else -100)
    tot = cnt = 0
    for i in range(x.shape[0]):
        b = x[i:i+1].to(DEV)
        lg = model(input_ids=b, attention_mask=torch.ones_like(b)).logits[:, :-1, :]
        lab = b[:, 1:]
        nll = lf(lg.permute(0, 2, 1), lab)
        m = (lab != pad) if pad is not None else torch.ones_like(lab, dtype=torch.bool)
        tot += float((nll*m).sum()); cnt += int(m.sum())
    return math.exp(tot/cnt), cnt, x.shape[0]

c4 = load_dataset("allenai/c4", "en",
                  data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                  split="validation", verification_mode="no_checks")
print("c4 validation rows:", len(c4), flush=True)
for tag, text, mc in [
    ("c4 GPTQ  val[:1100] ' '.join  -> first 256x2048", " ".join(c4[:1100]["text"]), 256),
    ("c4       val[:1100] '\\n\\n'.join -> first 256x2048", "\n\n".join(c4[:1100]["text"]), 256),
]:
    ids = tok(text, return_tensors="pt").input_ids[0]
    p, ntok, nch = ppl(ids, mc)
    print(f"{tag:<52} ppl={p:8.4f} tokens={ntok} chunks={nch}", flush=True)
