#### Computes every number and distribution used in the paper from the saved logits.
#### Needs only numpy. Reads <logits>/<model>/<bench>/{logits.npy,meta.json} written by extract_logits.py
#### and writes a small results folder that can be copied to a laptop for plotting:
####   <out>/stats.json   accuracy, top-1/top-50/full-vocabulary statistics, KLD, cosine summaries, histograms
####   <out>/cosine.npz   per-question cosine similarity for each variant and benchmark
####   <out>/report.md    the paper's tables recomputed and compared with the values in main.tex
####
#### Definitions (Section VI-C of the paper):
####   log-probabilities  log-softmax of the next-token logits over the full vocabulary
####   confidence         top-1 log-probability, mean and std over questions
####   top-50             the 50 largest log-probabilities, pooled over questions
####   shape              mean and std of all log-probabilities, pooled over questions
####   KLD                D_KL(p_base || p_variant), averaged over questions
####   cosine             cosine similarity of the raw logit vectors, per question
#### The LoRA model is compared on the vocabulary it shares with its base. Questions where a
#### quantized model returns NaN or infinite logits are dropped from its comparison.
####
#### Usage: python3 compute_stats.py --logits /output/slm-logits --out /output/slm-logits/stats

#### Libraries
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from registry import BENCHMARKS, MODELS, MODEL_ORDER

HIST_BINS = np.linspace(-40.0, 5.0, 160)   # same bins as the original figures
TOP_K = 50
CHUNK = 64


#### Helpers
def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def histogram(x):
    lo, hi = HIST_BINS[0], HIST_BINS[-1]
    width = HIST_BINS[1] - HIST_BINS[0]
    x = x[(x >= lo) & (x <= hi)]
    idx = np.minimum(((x - lo) / width).astype(np.int64), len(HIST_BINS) - 2)
    return np.bincount(idx, minlength=len(HIST_BINS) - 1)


class Pooled:
    """Running mean, std, and histogram of a pooled set of values."""
    def __init__(self, hist=True):
        self.n, self.s, self.ss = 0, 0.0, 0.0
        self.counts = np.zeros(len(HIST_BINS) - 1, dtype=np.int64) if hist else None

    def add(self, x):
        x = x[np.isfinite(x)]
        self.n += x.size
        self.s += float(x.sum())
        self.ss += float((x * x).sum())
        if self.counts is not None:
            self.counts += histogram(x.ravel())

    def summary(self):
        mu = self.s / self.n
        out = dict(n=self.n, mean=mu, std=float(np.sqrt(max(self.ss / self.n - mu * mu, 0.0))))
        if self.counts is not None:
            out["hist"] = self.counts.tolist()
        return out


def mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std()))


def load(logits_dir, model, bench):
    d = os.path.join(logits_dir, model, bench)
    meta = json.load(open(os.path.join(d, "meta.json")))
    if not meta.get("done"):
        raise RuntimeError(f"{model}/{bench} is not finished ({meta['completed']}/{meta['num_questions']})")
    return np.load(os.path.join(d, "logits.npy"), mmap_mode="r"), meta


def shared_vocab(logits_dir, base, variant, n_base, n_var):
    # Indices of the tokens that both tokenizers share, sorted by token string
    vb = json.load(open(os.path.join(logits_dir, base, "vocab.json")))
    vv = json.load(open(os.path.join(logits_dir, variant, "vocab.json")))
    shared = sorted(t for t in set(vb) & set(vv) if vb[t] < n_base and vv[t] < n_var)
    print(f"  shared vocabulary {len(shared)} (base {len(vb)}, variant {len(vv)})", flush=True)
    return np.array([vb[t] for t in shared]), np.array([vv[t] for t in shared])


#### Single-model statistics: accuracy, confidence, top-50, full-vocabulary shape
def model_stats(logits_dir, model, bench):
    Z, meta = load(logits_dir, model, bench)
    letters = meta["choice_letters"]
    token_ids = meta["choice_token_ids"]

    correct = scored = invalid = 0
    top1, top50, full = [], Pooled(hist=False), Pooled()
    for a in range(0, Z.shape[0], CHUNK):
        z = np.asarray(Z[a:a + CHUNK], dtype=np.float64)
        ok = np.isfinite(z).all(axis=1)
        invalid += int((~ok).sum())
        with np.errstate(invalid="ignore", over="ignore"):
            lp = log_softmax(z)

        for r in range(z.shape[0]):
            gold = meta["gold"][a + r]
            if gold not in letters:   # ARC answer keys that are numbers or "E"
                continue
            scores = np.array([lp[r, ids].max() for ids in token_ids])
            # torch.argmax returns the first NaN if there is one; keep that behavior
            pred = int(np.argmax(np.isnan(scores))) if np.isnan(scores).any() else int(np.argmax(scores))
            correct += int(letters[pred] == gold)
            scored += 1

        lp = lp[ok]
        if lp.shape[0] == 0:
            continue
        top = -np.partition(-lp, TOP_K - 1, axis=1)[:, :TOP_K]
        top1.append(top.max(axis=1))
        top50.add(top)
        full.add(lp)

    return dict(accuracy=100.0 * correct / scored, scored=scored, questions=int(Z.shape[0]),
                invalid=invalid, vocab_size=int(Z.shape[1]),
                top1=mean_std(np.concatenate(top1)), top50=top50.summary(), full=full.summary())


#### Pair statistics: variant against its base
def pair_stats(logits_dir, variant, bench):
    base = MODELS[variant]["base"]
    Zb, _ = load(logits_dir, base, bench)
    Zv, _ = load(logits_dir, variant, bench)
    assert Zb.shape[0] == Zv.shape[0], (base, variant, bench)

    ib = iv = None
    if MODELS[variant]["loader"] == "lora":
        ib, iv = shared_vocab(logits_dir, base, variant, Zb.shape[1], Zv.shape[1])
    else:
        assert Zb.shape[1] == Zv.shape[1], (base, variant, bench, Zb.shape, Zv.shape)

    kl, cos, skipped = [], [], 0
    fb, fv = Pooled(), Pooled()
    for a in range(0, Zb.shape[0], CHUNK):
        zb = np.asarray(Zb[a:a + CHUNK], dtype=np.float64)
        zv = np.asarray(Zv[a:a + CHUNK], dtype=np.float64)
        if ib is not None:
            zb, zv = zb[:, ib], zv[:, iv]
        ok = np.isfinite(zb).all(axis=1) & np.isfinite(zv).all(axis=1)
        skipped += int((~ok).sum())
        zb, zv = zb[ok], zv[ok]
        if zb.shape[0] == 0:
            continue

        cos.append((zb * zv).sum(axis=1) / (np.linalg.norm(zb, axis=1) * np.linalg.norm(zv, axis=1)))
        lb, lv = log_softmax(zb), log_softmax(zv)
        kl.append((np.exp(lb) * (lb - lv)).sum(axis=1))
        fb.add(lb)
        fv.add(lv)

    kl, cos = np.concatenate(kl), np.concatenate(cos)
    return dict(base=base, used=int(kl.size), skipped=skipped,
                kl=mean_std(kl), cosine=mean_std(cos),
                base_full=fb.summary(), variant_full=fv.summary()), cos


#### Jobs
def run_job(job):
    kind, logits_dir, name, bench = job
    print(f"{kind} {name} {bench}", flush=True)
    if kind == "model":
        return job, model_stats(logits_dir, name, bench), None
    stats, cos = pair_stats(logits_dir, name, bench)
    return job, stats, cos


#### Report: recompute the paper's tables and compare with main.tex
PAPER_ACCURACY = {  # Table II
    "Llama-3.1-8B": [62.67, 24.92, 32.60, 88.38],
    "Llama-3.1-8B-Instruct": [67.12, 29.51, 30.78, 92.91],
    "DeepSeek-R1-Distill-Llama-8B": [51.53, 23.82, 34.72, 83.05],
    "Llama-3.1-8B-Pruned-4-Layers": [60.63, 24.25, 30.25, 81.20],
    "Code-Llama-3.1-8B-Text-to-SQL-LoRA": [60.89, 26.60, 32.22, 88.07],
    "Llama-3.1-8B-bnb-4bit": [60.69, 25.03, 30.55, 86.97],
    "Llama-3.1-8B-Instruct-FP8-dynamic": [66.79, 29.61, 30.48, 93.13],
    "Qwen-2.5-7B": [71.44, 39.33, 44.35, 95.95],
    "Qwen-2.5-7B-Instruct": [72.00, 38.39, 45.34, 95.95],
    "Qwen-2.5-Math-7B": [57.42, 39.30, 46.78, 88.33],
    "DeepSeek-R1-Distill-Qwen-7B": [51.82, 31.76, 45.19, 84.32],
    "Qwen-2.5-Math-7B-Instruct": [52.90, 36.72, 48.60, 85.60],
    "Qwen-2.5-7B-bnb-4bit": [70.49, 39.36, 44.50, 95.42],
    "Qwen-2.5-7B-FP8-dynamic": [71.44, 39.43, 43.29, 95.68],
}
PAPER_KL = {  # Table III
    "Llama-3.1-8B-Instruct": [0.27, 0.57, 2.26, 1.83],
    "DeepSeek-R1-Distill-Llama-8B": [0.41, 0.62, 1.31, 0.75],
    "Llama-3.1-8B-Pruned-4-Layers": [0.18, 0.70, 0.99, 0.48],
    "Code-Llama-3.1-8B-Text-to-SQL-LoRA": [0.15, 0.34, 0.45, 0.51],
    "Llama-3.1-8B-bnb-4bit": [0.05, 0.08, 0.13, 0.08],
    "Llama-3.1-8B-Instruct-FP8-dynamic": [0.00, 0.02, 0.00, 0.00],
    "Qwen-2.5-7B-Instruct": [0.78, 1.32, 2.04, 1.31],
    "Qwen-2.5-Math-7B": [0.69, 0.80, 0.94, 0.60],
    "DeepSeek-R1-Distill-Qwen-7B": [0.58, 1.35, 1.45, 0.62],
    "Qwen-2.5-Math-7B-Instruct": [1.05, 1.95, 1.72, 0.98],
    "Qwen-2.5-7B-bnb-4bit": [0.03, 0.06, 0.15, 0.04],
    "Qwen-2.5-7B-FP8-dynamic": [0.00, 0.00, 0.02, 0.01],
}


def write_report(results, path):
    lines = ["# Recomputed tables", "",
             "Each cell is `new (paper)`; cells marked ** differ from the paper after rounding.", ""]

    def cell(new, old, fmt):
        s = fmt.format(new)
        return f"{s} ({fmt.format(old)})" + ("" if s == fmt.format(old) else " **")

    lines += ["## Table II: accuracy (%)", "", "| Model | " + " | ".join(BENCHMARKS) + " |",
              "|---|" + "---|" * len(BENCHMARKS)]
    for m in MODEL_ORDER:
        row = [cell(results["models"][m][b]["accuracy"], PAPER_ACCURACY[m][i], "{:.2f}") for i, b in enumerate(BENCHMARKS)]
        lines.append(f"| {m} | " + " | ".join(row) + " |")

    lines += ["", "## Table III: KLD from the base model", "", "| Model | " + " | ".join(BENCHMARKS) + " |",
              "|---|" + "---|" * len(BENCHMARKS)]
    for m in results["pairs"]:
        row = [cell(results["pairs"][m][b]["kl"]["mean"], PAPER_KL[m][i], "{:.2f}") for i, b in enumerate(BENCHMARKS)]
        lines.append(f"| {m} | " + " | ".join(row) + " |")

    lines += ["", "## Table IV: top-1 and top-50 log-probabilities, mean (std)", "",
              "| Model | " + " | ".join(f"{b} top-1 | {b} top-50" for b in BENCHMARKS) + " |",
              "|---|" + "---|---|" * len(BENCHMARKS)]
    for m in MODEL_ORDER:
        row = []
        for b in BENCHMARKS:
            s = results["models"][m][b]
            row += [f"{s['top1']['mean']:.2f} ({s['top1']['std']:.2f})", f"{s['top50']['mean']:.2f} ({s['top50']['std']:.2f})"]
        lines.append(f"| {m} | " + " | ".join(row) + " |")

    lines += ["", "## Cosine similarity and full-vocabulary shape (pair statistics)", "",
              "| Variant | Bench | cos mean | cos std | base mean (std) | variant mean (std) | delta mean | used | skipped |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m, per_bench in results["pairs"].items():
        for b in BENCHMARKS:
            s = per_bench[b]
            bf, vf = s["base_full"], s["variant_full"]
            lines.append(f"| {m} | {b} | {s['cosine']['mean']:.3f} | {s['cosine']['std']:.3f} | "
                         f"{bf['mean']:.2f} ({bf['std']:.2f}) | {vf['mean']:.2f} ({vf['std']:.2f}) | "
                         f"{vf['mean'] - bf['mean']:+.2f} | {s['used']} | {s['skipped']} |")

    lines += ["", "## Questions", "", "| Model | " + " | ".join(f"{b} scored/total (invalid)" for b in BENCHMARKS) + " |",
              "|---|" + "---|" * len(BENCHMARKS)]
    for m in MODEL_ORDER:
        row = [f"{results['models'][m][b]['scored']}/{results['models'][m][b]['questions']} ({results['models'][m][b]['invalid']})"
               for b in BENCHMARKS]
        lines.append(f"| {m} | " + " | ".join(row) + " |")

    open(path, "w").write("\n".join(lines) + "\n")


#### Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=str, default="/output/slm-logits")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = args.out or os.path.join(args.logits, "stats")
    os.makedirs(out, exist_ok=True)

    jobs = [("model", args.logits, m, b) for m in MODEL_ORDER for b in BENCHMARKS]
    jobs += [("pair", args.logits, m, b) for m in MODEL_ORDER if MODELS[m]["base"] for b in BENCHMARKS]

    results = dict(bins=HIST_BINS.tolist(), benchmarks=BENCHMARKS, models={}, pairs={})
    cosines = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (kind, _, name, bench), stats, cos in pool.map(run_job, jobs):
            if kind == "model":
                results["models"].setdefault(name, {})[bench] = stats
            else:
                results["pairs"].setdefault(name, {})[bench] = stats
                cosines[f"{name}|{bench}"] = cos

    # Keep the registry order
    results["models"] = {m: results["models"][m] for m in MODEL_ORDER}
    results["pairs"] = {m: results["pairs"][m] for m in MODEL_ORDER if m in results["pairs"]}

    json.dump(results, open(os.path.join(out, "stats.json"), "w"))
    np.savez_compressed(os.path.join(out, "cosine.npz"), **cosines)
    write_report(results, os.path.join(out, "report.md"))
    print(f"\nWrote {out}/stats.json, cosine.npz, report.md")
