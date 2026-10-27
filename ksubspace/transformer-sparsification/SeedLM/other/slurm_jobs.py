"""Generate SLURM job manifests and merge server_temp JSON shards.

Reads global experiment knobs exclusively from SeedLM.bench_compress_and_store.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from SeedLM.other.bench_compress_and_store import (
    DEFAULT_SPARSITIES,
    K_CLUSTER_TILE_FRACTION,
    OPT_MODELS,
    QUANTIZATION_BITS,
    _model_tag,
)


def _sp_tag(sp: float) -> str:
    return f"{sp:.2f}"


def _kfrac_tag(frac: float) -> str:
    return f"{frac:.4g}"


def active_sparsities() -> list[float]:
    return [sp for sp in DEFAULT_SPARSITIES if sp != 0.0]


def build_store_jobs(models: list[str] | None = None) -> list[dict]:
    jobs: list[dict] = []
    for model in models or list(OPT_MODELS):
        jobs.append(
            {
                "phase": "store",
                "model": model,
                "section": "store",
                "job_tag": "store",
            }
        )
    return jobs


def build_compress_mid_jobs(models: list[str] | None = None) -> list[dict]:
    """s2 / s3 / s4 / s5a — may run after store (s5b needs s2 first)."""
    jobs: list[dict] = []
    sparsities = active_sparsities()
    for model in models or list(OPT_MODELS):
        for bits in QUANTIZATION_BITS:
            jobs.append(
                {
                    "phase": "compress_mid",
                    "model": model,
                    "section": "s2",
                    "quantization_bits": bits,
                    "job_tag": f"s2_bits{bits}",
                }
            )
        for sp in sparsities:
            jobs.append(
                {
                    "phase": "compress_mid",
                    "model": model,
                    "section": "s3",
                    "sparsity": sp,
                    "job_tag": f"s3_sp{_sp_tag(sp)}",
                }
            )
            jobs.append(
                {
                    "phase": "compress_mid",
                    "model": model,
                    "section": "s4",
                    "sparsity": sp,
                    "job_tag": f"s4_sp{_sp_tag(sp)}",
                }
            )
        for sp in sparsities:
            for kfrac in K_CLUSTER_TILE_FRACTION:
                jobs.append(
                    {
                        "phase": "compress_mid",
                        "model": model,
                        "section": "s5a",
                        "sparsity": sp,
                        "k_cluster_fraction": kfrac,
                        "job_tag": f"s5a_sp{_sp_tag(sp)}_kfrac{_kfrac_tag(kfrac)}",
                    }
                )
    return jobs


def build_s5b_prep_jobs(models: list[str] | None = None) -> list[dict]:
    jobs: list[dict] = []
    for model in models or list(OPT_MODELS):
        for bits in QUANTIZATION_BITS:
            jobs.append(
                {
                    "phase": "s5b_prep",
                    "model": model,
                    "section": "s5b_prep",
                    "quantization_bits": bits,
                    "job_tag": f"s5b_prep_bits{bits}",
                }
            )
    return jobs


def build_s5b_jobs(models: list[str] | None = None) -> list[dict]:
    jobs: list[dict] = []
    sparsities = active_sparsities()
    for model in models or list(OPT_MODELS):
        for bits in QUANTIZATION_BITS:
            for sp in sparsities:
                for kfrac in K_CLUSTER_TILE_FRACTION:
                    jobs.append(
                        {
                            "phase": "s5b",
                            "model": model,
                            "section": "s5b",
                            "quantization_bits": bits,
                            "sparsity": sp,
                            "k_cluster_fraction": kfrac,
                            "job_tag": (
                                f"s5b_bits{bits}_sp{_sp_tag(sp)}_kfrac{_kfrac_tag(kfrac)}"
                            ),
                        }
                    )
    return jobs


def build_load_jobs(models: list[str] | None = None) -> list[dict]:
    jobs: list[dict] = []
    sparsities = active_sparsities()
    for model in models or list(OPT_MODELS):
        jobs.append(
            {
                "phase": "load",
                "model": model,
                "section": "s1",
                "job_tag": "s1_baseline",
            }
        )
        for bits in QUANTIZATION_BITS:
            jobs.append(
                {
                    "phase": "load",
                    "model": model,
                    "section": "s2",
                    "quantization_bits": bits,
                    "job_tag": f"s2_bits{bits}",
                }
            )
        for sp in sparsities:
            jobs.append(
                {
                    "phase": "load",
                    "model": model,
                    "section": "s3",
                    "sparsity": sp,
                    "job_tag": f"s3_sp{_sp_tag(sp)}",
                }
            )
            jobs.append(
                {
                    "phase": "load",
                    "model": model,
                    "section": "s4",
                    "sparsity": sp,
                    "job_tag": f"s4_sp{_sp_tag(sp)}",
                }
            )
        for sp in sparsities:
            for kfrac in K_CLUSTER_TILE_FRACTION:
                jobs.append(
                    {
                        "phase": "load",
                        "model": model,
                        "section": "s5a",
                        "sparsity": sp,
                        "k_cluster_fraction": kfrac,
                        "job_tag": f"s5a_sp{_sp_tag(sp)}_kfrac{_kfrac_tag(kfrac)}",
                    }
                )
        for bits in QUANTIZATION_BITS:
            for sp in sparsities:
                for kfrac in K_CLUSTER_TILE_FRACTION:
                    jobs.append(
                        {
                            "phase": "load",
                            "model": model,
                            "section": "s5b",
                            "quantization_bits": bits,
                            "sparsity": sp,
                            "k_cluster_fraction": kfrac,
                            "job_tag": (
                                f"s5b_bits{bits}_sp{_sp_tag(sp)}_kfrac{_kfrac_tag(kfrac)}"
                            ),
                        }
                    )
    return jobs


# Merge order must match full-pipeline logging order.
_COMPRESS_SORT_KEY = {
    "store": 0,
    "s2": 1,
    "s3": 2,
    "s4": 3,
    "s5a": 4,
    "s5b_prep": 5,
    "s5b": 6,
}

_LOAD_SORT_KEY = {
    "s1_baseline": 0,
    "s1": 0,
}


def _shard_sort_key_compress(path: Path) -> tuple:
    stem = path.stem
    if stem == "store":
        return (0, "", 0.0, 0.0, 0)
    if stem.startswith("s2_bits"):
        return (1, "", 0.0, 0.0, int(stem.replace("s2_bits", "")))
    if stem.startswith("s3_sp"):
        return (2, "", float(stem.replace("s3_sp", "")), 0.0, 0)
    if stem.startswith("s4_sp"):
        return (3, "", float(stem.replace("s4_sp", "")), 0.0, 0)
    if stem.startswith("s5a_sp"):
        # s5a_sp0.50_kfrac0.01
        rest = stem[len("s5a_sp") :]
        sp_s, _, k_s = rest.partition("_kfrac")
        return (4, "", float(sp_s), float(k_s), 0)
    if stem.startswith("s5b_prep_bits"):
        return (5, "", 0.0, 0.0, int(stem.replace("s5b_prep_bits", "")))
    if stem.startswith("s5b_bits"):
        # s5b_bits3_sp0.50_kfrac0.01
        rest = stem[len("s5b_bits") :]
        bits_s, _, rem = rest.partition("_sp")
        sp_s, _, k_s = rem.partition("_kfrac")
        return (6, "", float(sp_s), float(k_s), int(bits_s))
    return (99, stem, 0.0, 0.0, 0)


def _shard_sort_key_load(path: Path) -> tuple:
    stem = path.stem
    if stem in ("s1_baseline", "s1"):
        return (0, "", 0.0, 0.0, 0)
    if stem.startswith("s2_bits"):
        return (1, "", 0.0, 0.0, int(stem.replace("s2_bits", "")))
    if stem.startswith("s3_sp"):
        return (2, "", float(stem.replace("s3_sp", "")), 0.0, 0)
    if stem.startswith("s4_sp"):
        return (3, "", float(stem.replace("s4_sp", "")), 0.0, 0)
    if stem.startswith("s5a_sp"):
        rest = stem[len("s5a_sp") :]
        sp_s, _, k_s = rest.partition("_kfrac")
        return (4, "", float(sp_s), float(k_s), 0)
    if stem.startswith("s5b_bits"):
        rest = stem[len("s5b_bits") :]
        bits_s, _, rem = rest.partition("_sp")
        sp_s, _, k_s = rem.partition("_kfrac")
        return (5, "", float(sp_s), float(k_s), int(bits_s))
    return (99, stem, 0.0, 0.0, 0)


def merge_server_temp(
    *,
    kind: str,
    model_id: str | None = None,
) -> list[Path]:
    """Merge server_temp shards into results/{model}/CPU_*_result/{ts}.JSON."""
    if kind == "compress":
        temp_root = _ROOT / "results" / "server_temp" / "CPU_compress_result"
        sort_key = _shard_sort_key_compress
        result_subdir = "CPU_compress_result"
    elif kind == "inference":
        temp_root = _ROOT / "results" / "server_temp" / "CPU_inference_result"
        sort_key = _shard_sort_key_load
        result_subdir = "CPU_inference_result"
    else:
        raise ValueError(f"kind must be compress|inference, got {kind}")

    if not temp_root.exists():
        raise FileNotFoundError(f"No server_temp dir: {temp_root}")

    model_dirs = (
        [temp_root / _model_tag(model_id)]
        if model_id
        else sorted(p for p in temp_root.iterdir() if p.is_dir())
    )
    written: list[Path] = []
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        shards = sorted(model_dir.glob("*.JSON"), key=sort_key)
        if not shards:
            print(f"[merge] skip empty {model_dir}")
            continue
        merged: list[dict] = []
        for shard in shards:
            data = json.loads(shard.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {shard}")
            merged.extend(data)
        out_dir = _ROOT / "results" / model_dir.name / result_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ts}.JSON"
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[merge] {len(shards)} shards -> {out_path} ({len(merged)} records)")
        written.append(out_path)
    return written


def write_job_list(phase: str, out_path: Path, models: list[str] | None = None) -> int:
    builders = {
        "store": build_store_jobs,
        "compress_mid": build_compress_mid_jobs,
        "s5b_prep": build_s5b_prep_jobs,
        "s5b": build_s5b_jobs,
        "load": build_load_jobs,
    }
    if phase not in builders:
        raise ValueError(f"Unknown phase {phase}")
    jobs = builders[phase](models)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"[jobs] wrote {len(jobs)} jobs -> {out_path}")
    return len(jobs)


def cli_args_for_job(job: dict, *, stage: str) -> list[str]:
    """Build argv flags for compress or load bench from a job dict."""
    args = ["--model", job["model"], "--section", job["section"], "--server-temp", "--job-tag", job["job_tag"]]
    if "sparsity" in job:
        args += ["--sparsity", str(job["sparsity"])]
    if "k_cluster_fraction" in job:
        args += ["--k-cluster-fraction", str(job["k_cluster_fraction"])]
    if "quantization_bits" in job:
        args += ["--quantization-bits", str(job["quantization_bits"])]
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="SLURM job list / merge helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen", help="Write a job-list JSON for one phase.")
    p_gen.add_argument("phase", choices=["store", "compress_mid", "s5b_prep", "s5b", "load"])
    p_gen.add_argument("--out", type=Path, required=True)
    p_gen.add_argument("--model", action="append", default=None)

    p_merge = sub.add_parser("merge", help="Merge server_temp shards into a single JSON.")
    p_merge.add_argument("kind", choices=["compress", "inference"])
    p_merge.add_argument("--model", default=None, help="HF model id or omit for all.")

    p_print = sub.add_parser("print-globals", help="Print shared globals for shell scripts.")
    p_print.add_argument(
        "--what",
        choices=["models", "sparsity", "kfrac", "bits"],
        required=True,
    )

    p_runargs = sub.add_parser("job-argv", help="Print bench argv for jobs[i] in a list file.")
    p_runargs.add_argument("--jobs", type=Path, required=True)
    p_runargs.add_argument("--index", type=int, required=True)
    p_runargs.add_argument("--stage", choices=["compress", "load"], required=True)

    args = parser.parse_args()
    if args.cmd == "gen":
        write_job_list(args.phase, args.out, args.model)
    elif args.cmd == "merge":
        merge_server_temp(kind=args.kind, model_id=args.model)
    elif args.cmd == "print-globals":
        if args.what == "models":
            print("\n".join(OPT_MODELS))
        elif args.what == "sparsity":
            print(" ".join(str(x) for x in DEFAULT_SPARSITIES))
        elif args.what == "kfrac":
            print(" ".join(str(x) for x in K_CLUSTER_TILE_FRACTION))
        elif args.what == "bits":
            print(" ".join(str(x) for x in QUANTIZATION_BITS))
    elif args.cmd == "job-argv":
        jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
        job = jobs[args.index]
        print(" ".join(cli_args_for_job(job, stage=args.stage)))


if __name__ == "__main__":
    main()
