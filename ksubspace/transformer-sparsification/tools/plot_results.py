#!/usr/bin/env python3
"""Render the ARKS perplexity results as a three-panel figure.

Reads every inference_*.csv under arks_runs/ and draws:
  (a) PPL degradation by model and eval corpus, fp16 coefficients
  (b) the same for int8 coefficients
  (c) the compression-quality trade-off: rho against mean degradation

Colour carries ONE meaning throughout -- the eval corpus -- so panel (c),
whose series are the two coefficient precisions, encodes them with marker
fill instead of a second colour vocabulary.
"""

import csv
import glob
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model display order, with parameter counts for the trade-off panel.
MODELS = [
    ("opt-125m", 0.125e9),
    ("opt-6.7b", 6.7e9),
    ("opt-13b", 13.0e9),
    ("opt-30b", 30.0e9),
    ("opt-66b", 66.0e9),
    ("Llama-3.2-1B", 1.24e9),
    ("Llama-3.1-8B", 8.0e9),
    ("Llama-3.1-70B", 70.0e9),
]
CORPORA = ["wikitext2", "ptb", "c4"]
PRECISIONS = ["fp16", "int8"]

# Reference palette, light mode. Slots 1-3 validate all-pairs in both modes.
SERIES = {"wikitext2": "#2a78d6", "ptb": "#eb6834", "c4": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e1"

# Panel (c) direct-label offsets in points; default (9, 4).
LABEL_NUDGE = {"opt-66b": (9, -14), "opt-13b": (9, 6), "opt-30b": (9, 5)}


def load():
    """rows[(model, precision, corpus)] -> dict of the numbers we plot."""
    rows = {}
    pattern = os.path.join(ROOT, "arks_runs", "*", "seed_0", "ppl", "*", "*", "inference_*.csv")
    for path in glob.glob(pattern):
        # .../arks_runs/<model>/seed_0/ppl/<corpus>/<experiment>/inference_*.csv
        parts = path.split(os.sep)
        model, corpus, experiment = parts[-6], parts[-3], parts[-2]
        with open(path) as fh:
            rec = next(csv.DictReader(fh))
        precision = "int8" if "int8" in experiment else "fp16"
        dense, arks = float(rec["dense_ppl"]), float(rec["ppl"])
        rows[(model, precision, corpus)] = dict(
            dense=dense,
            arks=arks,
            pct=float(rec["ppl_increase_percent"]),
            dlog=math.log(arks) - math.log(dense),
            rho=float(rec["total_model_bits_per_parameter"]) / 16.0,
        )
    return rows


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def bar_panel(ax, rows, precision, title, ymax):
    labels = [m for m, _ in MODELS]
    x = np.arange(len(labels), dtype=float)
    # 2px surface gap between adjacent bars: width chosen so the group leaves a gap.
    width = 0.26
    offsets = {"wikitext2": -width, "ptb": 0.0, "c4": width}

    missing = []
    for corpus in CORPORA:
        vals, pos = [], []
        for i, (model, _) in enumerate(MODELS):
            rec = rows.get((model, precision, corpus))
            if rec is None:
                if model not in missing:
                    missing.append(model)
                continue
            vals.append(rec["pct"])
            pos.append(x[i] + offsets[corpus])
        bars = ax.bar(pos, vals, width - 0.02, label=corpus,
                      color=SERIES[corpus], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        # Relief rule: aqua is under 3:1 on this surface, so every bar is labelled.
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, val + ymax * 0.015,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7, color=INK_2, zorder=4)

    for model in missing:
        i = labels.index(model)
        ax.text(x[i], ymax * 0.5, "queued", ha="center", va="center",
                fontsize=9, color=INK_MUTED, style="italic", rotation=90)
        ax.axvspan(x[i] - 0.45, x[i] + 0.45, color="#f2f1ed", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", color=INK_2, fontsize=9)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("PPL degradation (%)", color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=10, fontweight="bold")
    style_axes(ax)


def tradeoff_panel(ax, rows):
    """rho against mean degradation; a thin line joins each model's two points."""
    for model, _ in MODELS:
        pts = []
        for precision in PRECISIONS:
            recs = [rows[(model, precision, c)] for c in CORPORA if (model, precision, c) in rows]
            if not recs:
                continue
            pts.append((precision,
                        float(np.mean([r["rho"] for r in recs])),
                        float(np.mean([r["pct"] for r in recs]))))
        if not pts:
            continue
        if len(pts) == 2:
            ax.plot([pts[0][1], pts[1][1]], [pts[0][2], pts[1][2]],
                    color=INK_MUTED, linewidth=1.2, zorder=2, alpha=0.7)
        for precision, rho, pct in pts:
            filled = precision == "int8"
            ax.plot(rho, pct, marker="o", markersize=9,
                    markerfacecolor=INK if filled else SURFACE,
                    markeredgecolor=INK, markeredgewidth=1.8, zorder=3,
                    linestyle="none")
        # Direct label at the fp16 (right-hand) end; a few need nudging off
        # a neighbouring connector line.
        anchor = max(pts, key=lambda p: p[1])
        dx, dy = LABEL_NUDGE.get(model, (9, 4))
        ax.annotate(model, (anchor[1], anchor[2]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.5, color=INK_2)

    ax.set_xlabel(r"Parameter retention ratio  $\rho$  =  compressed / original   (lower = more compression)", color=INK_2, fontsize=10)
    ax.set_ylabel("Mean PPL degradation over 3 corpora (%)", color=INK_2, fontsize=10)
    ax.set_title("(c) Compression-quality trade-off   (hollow = fp16, filled = int8)", color=INK,
                 fontsize=11.5, loc="left", pad=10, fontweight="bold")
    ax.set_xlim(0.28, 0.86)
    ax.margins(y=0.14)
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.text(0.995, 0.03,
            "opt-66b / Llama-3.1-70B: int8 still queued, so only their fp16 point is plotted",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=INK_MUTED, style="italic")


def main():
    rows = load()
    done = len(rows)

    fig = plt.figure(figsize=(13.5, 11.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          top=0.838, bottom=0.06, left=0.07, right=0.985,
                          hspace=0.42, wspace=0.16)

    ymax = 47.0
    ax_a = fig.add_subplot(gs[0, 0])
    bar_panel(ax_a, rows, "fp16", "(a) fp16 coefficients (16 bit) - 3 eval corpora", ymax)
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    bar_panel(ax_b, rows, "int8", "(b) int8 coefficients (8 bit) - 3 eval corpora", ymax)
    ax_b.set_ylabel("")

    handles, labels = ax_a.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False,
                     bbox_to_anchor=(0.07, 0.936), fontsize=10,
                     title="PPL eval corpus", alignment="left",
                     handlelength=1.1, handleheight=1.1, columnspacing=1.8)
    leg.get_title().set_color(INK)
    leg.get_title().set_fontsize(10)
    leg.get_title().set_fontweight("bold")
    for text in leg.get_texts():
        text.set_color(INK_2)

    ax_c = fig.add_subplot(gs[1, :])
    tradeoff_panel(ax_c, rows)

    fig.suptitle("ARKS K-subspace compression: perplexity cost vs. compression",
                 color=INK, fontsize=15, fontweight="bold", x=0.07, ha="left", y=0.985)
    fig.text(0.07, 0.958,
             f"Evaluated on 3 corpora (wikitext2, PTB, C4); calibrated on wikitext2 throughout"
             f"  ·  K=512  ·  extra budget: OPT 100 MB, Llama 200 MB  ·  seed 0"
             f"  ·  {done}/42 results complete",
             color=INK_2, fontsize=9.5, ha="left")

    for ext in ("png", "pdf"):
        out = os.path.join(ROOT, f"arks_results.{ext}")
        fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
