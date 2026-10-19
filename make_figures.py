#### Draws all result figures of the paper from stats.json and cosine.npz (written by compute_stats.py).
#### Runs on a laptop (numpy + matplotlib only). Figures are drawn at their printed size,
#### so the font sizes below are the sizes on the page.
####
####   main text:  <out>/{Llama,Qwen}_logprob_MMLU.pdf     (Fig. 4)
####               <out>/{Llama,Qwen}_cosine_sim_MMLU.pdf  (Fig. 5)
####   appendix:   <out>/LogitDistribution/<variant>.pdf   (Figs. 6-7)
####               <out>/CosineSimilarity/<variant>.pdf    (Figs. 8-9)
####
#### Usage: python3 make_figures.py --stats stats/ --out ../../Figures/Results/v2

#### Libraries
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from registry import BENCHMARKS, MODELS, MODEL_ORDER, PANEL_TITLES

#### Style
FONT = "Times New Roman" if any("Times New Roman" in f.name for f in font_manager.fontManager.ttflist) else "STIXGeneral"
plt.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
})

BASE_LINE, BASE_FILL = "#1D548F", "#4C78A8"   # dark blue
VAR_LINE, VAR_FILL = "#089099", "#7CCBA2"     # teal
MEAN = "#DC267F"                              # pink mean lines and std bands
STAT_FONT = 6.5

MAIN_HEADROOM = 1.6       # y-range above the highest curve, for the statistics
APPENDIX_HEADROOM = 1.6
COSINE_HEADROOM = 1.45
MAIN_WIDTH = 6.05   # 0.84 \textwidth in inches
HALF_WIDTH = 3.55   # 0.495 \textwidth in inches


def num(x, digits=2):
    # Number with a real minus sign and no negative zero
    s = f"{x:.{digits}f}"
    if s == "-0." + "0" * digits:
        s = s[1:]
    return s.replace("-", "−")


def signed(x, digits=2):
    s = num(x, digits)
    return s if s.startswith("−") or float(s) == 0 else "+" + s


def stats_box(ax, lines, size=STAT_FONT):
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=size,
            color="#222222", linespacing=1.15)


def density(counts, width):
    counts = np.asarray(counts, dtype=float)
    return counts / (counts.sum() * width) if counts.sum() > 0 else counts


#### Panels
def logprob_panel(ax, s, bins, title, lines, headroom):
    # The statistics sit in the top part of the panel; mean lines and bands stop below them
    width = bins[1] - bins[0]
    centers = 0.5 * (bins[:-1] + bins[1:])
    b, v = s["base_full"], s["variant_full"]
    db, dv = density(b["hist"], width), density(v["hist"], width)

    ax.fill_between(centers, db, 0, color=BASE_FILL, alpha=0.35, linewidth=0)
    ax.fill_between(centers, dv, 0, color=VAR_FILL, alpha=0.35, linewidth=0)
    ax.plot(centers, db, color=BASE_LINE, linewidth=1.0)
    ax.plot(centers, dv, color=VAR_LINE, linewidth=1.0)
    for st in (b, v):
        ax.axvline(st["mean"], ymax=1 / headroom, color=MEAN, linewidth=0.7, alpha=0.9)
        ax.axvspan(st["mean"] - st["std"], st["mean"] + st["std"], ymax=1 / headroom, color=MEAN, alpha=0.08, linewidth=0)

    ax.set_title(title, pad=3)
    ax.set_xlim(bins[0], bins[-1])
    ax.xaxis.set_major_locator(MultipleLocator(10.0))
    ax.xaxis.set_minor_locator(MultipleLocator(5.0))
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    stats_box(ax, lines)
    return max(db.max(), dv.max())


def cosine_panel(ax, sims, s, title):
    counts, _, _ = ax.hist(sims, bins=60, range=(-1.0, 1.0), color=BASE_FILL, alpha=0.7, edgecolor=BASE_LINE, linewidth=0.4)
    ax.set_ylim(0, counts.max() * COSINE_HEADROOM)   # room for the statistics above the bars
    mu, sd = s["cosine"]["mean"], s["cosine"]["std"]
    ax.axvline(mu, ymax=1 / COSINE_HEADROOM, color=MEAN, linewidth=0.7, alpha=0.9)
    ax.axvspan(mu - sd, mu + sd, ymax=1 / COSINE_HEADROOM, color=MEAN, alpha=0.1, linewidth=0)
    ax.set_title(title, pad=3)
    ax.set_xlim(-1.05, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    stats_box(ax, [rf"$\mu$ = {num(mu, 3)}", rf"$\sigma$ = {num(sd, 3)}"])


def logprob_lines(s):
    v = s["variant_full"]
    delta = v["mean"] - s["base_full"]["mean"]
    return [rf"$\mu$ = {num(v['mean'])}", rf"$\sigma$ = {num(v['std'])}",
            rf"$\Delta\mu$ = {signed(delta)}", f"KLD = {num(s['kl']['mean'], 3)}"]


def finish(fig, path, xlabel, ylabel, legend=None):
    fig.supxlabel(xlabel, fontsize=8, y=0.075 if legend else 0.01)
    fig.supylabel(ylabel, fontsize=8, x=0.005)
    if legend:
        fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False,
                   bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)
    print(f"Wrote {path}")


#### Figures
def main_figures(results, cos, out, bench="MMLU"):
    bins = np.array(results["bins"])
    for family in ("Llama", "Qwen"):
        variants = [m for m in MODEL_ORDER if MODELS[m]["family"] == family and MODELS[m]["base"]]

        # Fig. 4: full-vocabulary log-probabilities
        fig, axes = plt.subplots(2, 3, figsize=(MAIN_WIDTH, 3.55), sharex=True, sharey=True)
        ymax = 0.0
        for ax, m in zip(axes.flat, variants):
            s = results["pairs"][m][bench]
            ymax = max(ymax, logprob_panel(ax, s, bins, PANEL_TITLES[m], logprob_lines(s), MAIN_HEADROOM))
        for ax in axes.flat:
            ax.set_ylim(0, ymax * MAIN_HEADROOM)
        fig.tight_layout(rect=[0.02, 0.10, 1.0, 1.0], h_pad=0.6, w_pad=0.8)
        legend = [Line2D([0], [0], color=BASE_LINE, linewidth=1.5, label="Base"),
                  Line2D([0], [0], color=VAR_LINE, linewidth=1.5, label="Variant")]
        finish(fig, os.path.join(out, f"{family}_logprob_{bench}.pdf"), "Next-token log-probability", "Density", legend)

        # Fig. 5: cosine similarity
        fig, axes = plt.subplots(2, 3, figsize=(MAIN_WIDTH, 3.3), sharex=True)
        for ax, m in zip(axes.flat, variants):
            cosine_panel(ax, cos[f"{m}|{bench}"], results["pairs"][m][bench], PANEL_TITLES[m])
        fig.tight_layout(rect=[0.02, 0.03, 1.0, 1.0], h_pad=0.6, w_pad=0.8)
        finish(fig, os.path.join(out, f"{family}_cosine_sim_{bench}.pdf"), "Cosine similarity of logits", "Count")


def appendix_figures(results, cos, out):
    bins = np.array(results["bins"])
    os.makedirs(os.path.join(out, "LogitDistribution"), exist_ok=True)
    os.makedirs(os.path.join(out, "CosineSimilarity"), exist_ok=True)

    for m, per_bench in results["pairs"].items():
        base = MODELS[m]["base"]

        # Figs. 6-7: one panel per benchmark
        fig, axes = plt.subplots(2, 2, figsize=(HALF_WIDTH, 2.95), sharex=True, sharey=True)
        ymax = 0.0
        for ax, bench in zip(axes.flat, BENCHMARKS):
            s = per_bench[bench]
            b, v = s["base_full"], s["variant_full"]
            lines = [rf"Base: $\mu$ = {num(b['mean'])}, $\sigma$ = {num(b['std'])}",
                     rf"Variant: $\mu$ = {num(v['mean'])}, $\sigma$ = {num(v['std'])}",
                     f"KLD = {num(s['kl']['mean'], 3)}"]
            ymax = max(ymax, logprob_panel(ax, s, bins, bench, lines, APPENDIX_HEADROOM))
        for ax in axes.flat:
            ax.set_ylim(0, ymax * APPENDIX_HEADROOM)
        fig.tight_layout(rect=[0.03, 0.12, 1.0, 1.0], h_pad=0.5, w_pad=0.6)
        legend = [Line2D([0], [0], color=BASE_LINE, linewidth=1.5, label=base),
                  Line2D([0], [0], color=VAR_LINE, linewidth=1.5, label=m)]
        finish(fig, os.path.join(out, "LogitDistribution", f"{m}.pdf"), "Next-token log-probability", "Density", legend)

        # Figs. 8-9
        fig, axes = plt.subplots(2, 2, figsize=(HALF_WIDTH, 2.75), sharex=True)
        for ax, bench in zip(axes.flat, BENCHMARKS):
            cosine_panel(ax, cos[f"{m}|{bench}"], per_bench[bench], bench)
        fig.tight_layout(rect=[0.03, 0.04, 1.0, 1.0], h_pad=0.5, w_pad=0.6)
        finish(fig, os.path.join(out, "CosineSimilarity", f"{m}.pdf"), "Cosine similarity of logits", "Count")


#### Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=str, default="stats")
    parser.add_argument("--out", type=str, default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = json.load(open(os.path.join(args.stats, "stats.json")))
    cos = dict(np.load(os.path.join(args.stats, "cosine.npz")))
    main_figures(results, cos, args.out)
    appendix_figures(results, cos, args.out)
