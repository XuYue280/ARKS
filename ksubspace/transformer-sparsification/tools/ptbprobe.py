import math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
SEQ=2048
tok=AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
model=AutoModelForCausalLM.from_pretrained("facebook/opt-125m", dtype=torch.float16).to("cuda").eval()
pad=tok.pad_token_id
@torch.no_grad()
def ppl(ids):
    n=ids.numel()//SEQ; x=ids[:n*SEQ].reshape(n,SEQ)
    lf=torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad if pad is not None else -100)
    tot=cnt=0
    for i in range(n):
        b=x[i:i+1].to("cuda")
        lg=model(input_ids=b, attention_mask=torch.ones_like(b)).logits[:,:-1,:]
        lab=b[:,1:]; nll=lf(lg.permute(0,2,1),lab)
        m=(lab!=pad) if pad is not None else torch.ones_like(lab,dtype=torch.bool)
        tot+=float((nll*m).sum()); cnt+=int(m.sum())
    return math.exp(tot/cnt), cnt
# baseline 用的源: ptb_text_only 的 sentence 列
for repo in ["ilovefpga/ptb_text","ptb_text_only"]:
    try:
        d=load_dataset(repo, split="test") if repo!="ptb_text_only" else load_dataset(repo,"penn_treebank",split="test")
        col="sentence" if "sentence" in d.column_names else d.column_names[0]
        rows=list(d[col])
        for sep,lab in [("\n\n",'"\\n\\n".join  <- baseline 原生做法'),(" ",'" ".join      <- GPTQ/ARKS 做法')]:
            ids=tok(sep.join(rows), return_tensors="pt").input_ids[0]
            p,c=ppl(ids); print(f"  {repo:<22}{col:<10}{lab:<38} ppl={p:8.4f} tokens={c}")
    except Exception as e:
        print(f"  {repo}: {str(e)[:90]}")
