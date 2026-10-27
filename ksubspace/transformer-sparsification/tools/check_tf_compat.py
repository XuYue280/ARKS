"""Check whether each baseline's transformers imports still resolve in the installed version."""
import ast, importlib, os, sys, traceback

REPOS = {
    "ASVD4LLM": "/scratch/xuy280/ASVD4LLM",
    "Basis_Sharing": "/scratch/xuy280/Basis_Sharing",
    "SVD-LLM": "/scratch/xuy280/SVD-LLM",
}

def tf_imports(root):
    """(module, symbol, file, line) for every `from transformers... import X`."""
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(p, encoding="utf-8", errors="ignore").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("transformers"):
                    for a in node.names:
                        out.append((node.module, a.name, os.path.relpath(p, root), node.lineno))
    return out

import transformers
print(f"installed transformers = {transformers.__version__}\n")
overall = {}
for name, root in REPOS.items():
    imps = tf_imports(root)
    missing = []
    for mod, sym, f, ln in sorted(set(imps)):
        if sym == "*":
            continue
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            missing.append((mod, sym, f, ln, f"module import failed: {type(e).__name__}"))
            continue
        if not hasattr(m, sym):
            missing.append((mod, sym, f, ln, "symbol not found"))
    overall[name] = (len(set(imps)), missing)
    print(f"=== {name}: {len(set(imps))} 个 transformers 导入, {len(missing)} 个失效 ===")
    for mod, sym, f, ln, why in missing:
        print(f"   ✗ {f}:{ln}  from {mod} import {sym}   ({why})")
    if not missing:
        print("   ✓ 全部可解析")
    print()
print("小结:")
for n, (tot, miss) in overall.items():
    print(f"  {n:<16} {tot - len(miss)}/{tot} 可用  ->  {'兼容' if not miss else '不兼容'}")
