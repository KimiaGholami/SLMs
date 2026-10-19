#### Statistical analysis of the distribution metrics against accuracy (numpy only, runs on a laptop).
#### Reads stats/stats.json (compute_stats.py) and stats/per_question.npz (per_question.py) and writes
#### stats/analysis.md and stats/analysis.json:
####   A  accuracy with 95% Wilson intervals, with the paper's scoring and with uppercase letters only
####   B  accuracy change of each variant against its base: paired 95% interval and exact McNemar test
####   C  per variant and benchmark: change in confidence, entropy, and shape; KLD, JSD, cosine, centered cosine
####   D  across the 48 variant-benchmark pairs: Spearman correlation of each metric with the accuracy
####      change, with a 95% interval from a bootstrap that resamples variants (each has 4 benchmarks)
####   E  per question: AUROC of each metric for detecting (i) questions whose answer changes and
####      (ii) questions the base answers correctly and the variant answers wrongly
####   F  calibration over the answer options (expected calibration error, 10 bins)
####
#### Usage: python3 analyze.py --stats stats

#### Libraries
import argparse
import json
import math
import os

import numpy as np

from registry import BENCHMARKS as REGISTRY_BENCHMARKS, MODELS

N_BOOT = 10000
RNG = np.random.default_rng(0)


#### Statistics helpers
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(b, c):
    # Exact two-sided test on the discordant pairs
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0)


def ranks(x):
    # Average ranks with ties
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / counts)[inv]


def spearman(x, y):
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx * rx).sum() * (ry * ry).sum()))


def auroc(score, label):
    # Probability that a positive question scores higher than a negative one
    label = np.asarray(label, dtype=bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = ranks(np.asarray(score))
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(conf, correct, bins=10):
    # conf: probability of the predicted option, renormalized over the options
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return 100 * total / len(conf)


#### Analysis
def main(stats_dir):
    st = json.load(open(os.path.join(stats_dir, "stats.json")))
    pq = np.load(os.path.join(stats_dir, "per_question.npz"))
    get = lambda m, b, f: pq[f"{m}|{b}|{f}"]
    getp = lambda v, b, f: pq[f"pair|{v}|{b}|{f}"]
    BENCHMARKS = st.get("benchmarks", REGISTRY_BENCHMARKS)
    model_list = list(st["models"])
    variants = list(st["pairs"])
    has_upper = f"{model_list[0]}|{BENCHMARKS[0]}|pred_upper" in pq.files
    out = dict(A={}, B={}, C={}, D={}, E={}, F={}, benchmarks=BENCHMARKS, models=model_list, variants=variants)

    # A: accuracy
    for m in model_list:
        for b in BENCHMARKS:
            gold = get(m, b, "gold")
            s = gold >= 0
            k, n = int((get(m, b, "pred")[s] == gold[s]).sum()), int(s.sum())
            out["A"][f"{m}|{b}"] = dict(n=n, acc=100 * k / n, ci=wilson(k, n))
            if has_upper:
                ku = int((get(m, b, "pred_upper")[s] == gold[s]).sum())
                out["A"][f"{m}|{b}"].update(acc_upper=100 * ku / n,
                                            changed=100 * float((get(m, b, "pred")[s] != get(m, b, "pred_upper")[s]).mean()))

    # B, C, E, F per variant and benchmark
    for v in variants:
        base = MODELS[v]["base"]
        for b in BENCHMARKS:
            gold = get(v, b, "gold")
            s = gold >= 0
            cb = get(base, b, "pred")[s] == gold[s]
            cv = get(v, b, "pred")[s] == gold[s]
            d = cv.astype(float) - cb.astype(float)
            boot = RNG.integers(0, d.size, size=(2000, d.size))
            means = d[boot].mean(axis=1)
            nb, nc = int((cb & ~cv).sum()), int((~cb & cv).sum())
            out["B"][f"{v}|{b}"] = dict(delta=100 * d.mean(), ci=(100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)),
                                        lost=nb, gained=nc, p=mcnemar(nb, nc))

            shape = st["pairs"][v][b]
            out["C"][f"{v}|{b}"] = dict(
                d_acc=100 * d.mean(),
                d_conf=float(get(v, b, "top1").mean() - get(base, b, "top1").mean()),
                d_entropy=float(get(v, b, "entropy").mean() - get(base, b, "entropy").mean()),
                d_mu=shape["variant_full"]["mean"] - shape["base_full"]["mean"],
                d_sigma=shape["variant_full"]["std"] - shape["base_full"]["std"],
                kl=float(getp(v, b, "kl").mean()), js=float(getp(v, b, "js").mean()),
                cos=float(getp(v, b, "cos").mean()), ccos=float(getp(v, b, "ccos").mean()))

            # E: per question, all questions for flips, base-correct questions for breaks
            pb, pv = get(base, b, "pred"), get(v, b, "pred")
            flip = pb != pv
            per_q = {
                "|d_conf|": np.abs(get(v, b, "top1") - get(base, b, "top1")),
                "|d_entropy|": np.abs(get(v, b, "entropy") - get(base, b, "entropy")),
                "kl": getp(v, b, "kl"), "js": getp(v, b, "js"),
                "-cos": -getp(v, b, "cos"), "-ccos": -getp(v, b, "ccos")}
            breaks_mask = s & (pb == gold)
            breaks = (pv != gold)[breaks_mask]
            out["E"][f"{v}|{b}"] = dict(
                flip_rate=100 * float(flip.mean()), break_rate=100 * float(breaks.mean()),
                flip={k: auroc(x, flip) for k, x in per_q.items()},
                brk={k: auroc(x[breaks_mask], breaks) for k, x in per_q.items()})

    for m in model_list:
        for b in BENCHMARKS:
            gold = get(m, b, "gold")
            s = gold >= 0
            conf = get(m, b, "conf") if f"{m}|{b}|conf" in pq.files else get(m, b, "opt_prob").max(axis=1)
            out["F"][f"{m}|{b}"] = ece(conf[s], get(m, b, "pred")[s] == gold[s])

    # D: correlation with the accuracy change across variant-benchmark pairs
    metrics = ["d_conf", "d_entropy", "d_mu", "d_sigma", "kl", "js", "cos", "ccos"]
    groups = {"All": variants}
    for fam in ("Llama", "Qwen"):
        groups[fam] = [v for v in variants if MODELS[v]["family"] == fam]
        sweep = [v for v in groups[fam] if "transform" in MODELS[v]]
        if len(sweep) >= 4:
            groups[f"Sweep-{fam}"] = sweep
    for group, vs in groups.items():
        if len(vs) < 3:
            continue
        res = {}
        for metric in metrics:
            x = np.array([[out["C"][f"{v}|{b}"][metric] for b in BENCHMARKS] for v in vs])
            y = np.array([[out["C"][f"{v}|{b}"]["d_acc"] for b in BENCHMARKS] for v in vs])
            rho = spearman(x.ravel(), y.ravel())
            boots = []
            for _ in range(N_BOOT // 5):
                i = RNG.integers(0, len(vs), len(vs))
                xb, yb = x[i].ravel(), y[i].ravel()
                if np.unique(xb).size > 2 and np.unique(yb).size > 2:
                    boots.append(spearman(xb, yb))
            res[metric] = dict(rho=rho, ci=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))))
        out["D"][group] = res

    json.dump(out, open(os.path.join(stats_dir, "analysis.json"), "w"), indent=1, default=float)
    write_md(out, variants, metrics, BENCHMARKS, os.path.join(stats_dir, "analysis.md"))
    print(f"Wrote {stats_dir}/analysis.md and analysis.json")


#### Report
def write_md(out, variants, metrics, BENCHMARKS, path):
    L = ["# Analysis of distribution metrics against accuracy", ""]

    L += ["## A. Accuracy (%) with 95% Wilson interval", "",
          "| Model | Bench | n | Acc. | 95% CI | Acc. (uppercase only) | Predictions changed (%) |", "|---|---|---|---|---|---|---|"]
    for k, a in out["A"].items():
        m, b = k.split("|")
        extra = f"{a['acc_upper']:.2f} | {a['changed']:.2f}" if "acc_upper" in a else "- | -"
        L.append(f"| {m} | {b} | {a['n']} | {a['acc']:.2f} | [{a['ci'][0]:.1f}, {a['ci'][1]:.1f}] | {extra} |")

    L += ["", "## B. Accuracy change against the base (points), paired 95% CI, McNemar", "",
          "| Variant | Bench | Delta | 95% CI | Lost | Gained | p |", "|---|---|---|---|---|---|---|"]
    for k, r in out["B"].items():
        v, b = k.split("|")
        L.append(f"| {v} | {b} | {r['delta']:+.2f} | [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['lost']} | {r['gained']} | {r['p']:.2g} |")

    L += ["", "## C. Metrics per variant and benchmark", "",
          "| Variant | Bench | dAcc | dConf | dEntropy | dMu | dSigma | KLD | JSD | cos | centered cos |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in out["C"].items():
        v, b = k.split("|")
        L.append(f"| {v} | {b} | {c['d_acc']:+.2f} | {c['d_conf']:+.3f} | {c['d_entropy']:+.3f} | {c['d_mu']:+.2f} | {c['d_sigma']:+.2f} | "
                 f"{c['kl']:.3f} | {c['js']:.3f} | {c['cos']:.3f} | {c['ccos']:.3f} |")

    L += ["", "## D. Spearman correlation with the accuracy change (95% CI, bootstrap over variants)", "",
          "| Metric | " + " | ".join(out["D"].keys()) + " |", "|---|" + "---|" * len(out["D"])]
    for metric in metrics:
        L.append(f"| {metric} | " + " | ".join(f"{g[metric]['rho']:+.2f} [{g[metric]['ci'][0]:+.2f}, {g[metric]['ci'][1]:+.2f}]"
                                            for g in out["D"].values()) + " |")

    keys = list(next(iter(out["E"].values()))["flip"].keys())
    for task, title in (("flip", "questions whose answer changes"), ("brk", "questions the base gets right and the variant gets wrong")):
        L += ["", f"## E. Per-question AUROC for detecting {title}", "",
              "| Variant | Bench | rate (%) | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
        for k, e in out["E"].items():
            v, b = k.split("|")
            rate = e["flip_rate"] if task == "flip" else e["break_rate"]
            L.append(f"| {v} | {b} | {rate:.1f} | " + " | ".join(f"{e[task][m]:.2f}" for m in keys) + " |")
        mean = {m: np.nanmean([e[task][m] for e in out["E"].values()]) for m in keys}
        L.append("| **Mean** | | | " + " | ".join(f"**{mean[m]:.2f}**" for m in keys) + " |")

    L += ["", "## F. Expected calibration error over the options (%)", "", "| Model | " + " | ".join(BENCHMARKS) + " |",
          "|---|" + "---|" * len(BENCHMARKS)]
    for m in out["models"]:
        L.append(f"| {m} | " + " | ".join(f"{out['F'][f'{m}|{b}']:.1f}" for b in BENCHMARKS) + " |")

    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=str, default="stats")
    main(parser.parse_args().stats)
