#### Statistics from run_harness.py outputs (numpy only). For one setting (e.g. zeroshot), reads
#### <harness>/<setting>/<model>/{samples/<bench>.jsonl.gz, <bench>/logits.npy, vocab.json} for every model
#### found, and writes to <out>:
####   stats.json         per model and benchmark: accuracy and confidence/shape statistics; per variant and
####                      benchmark: KLD, JSD, cosine, centered cosine, and full-vocabulary histograms
####                      (same layout as compute_stats.py, so make_figures.py reads it unchanged)
####   cosine.npz         per-question cosine similarity, keyed "<variant>|<bench>"
####   per_question.npz   per-question data for analyze.py, keyed "<model>|<bench>|<field>" and
####                      "pair|<variant>|<bench>|<field>"
#### Accuracy follows the harness: acc for lettered tasks (MMLU, GSM8K-MC) and length-normalized acc_norm for
#### cloze tasks (ARC, MathQA), as on the Open LLM Leaderboard. All distribution statistics use each model's
#### real vocabulary (rows of the output layer without a token are removed); pairs use the token ids both share.
####
#### Usage: python3 harness_stats.py --harness /output/harness --setting zeroshot --out /output/harness/stats-zeroshot

#### Libraries
import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from compute_stats import CHUNK, HIST_BINS, TOP_K, Pooled, log_softmax, mean_std
from registry import MODELS

NORM_BENCHES = {"ARC-Easy", "ARC-Challenge", "MathQA"}


#### Loading
def load_samples(root, model, bench):
    return [json.loads(l) for l in gzip.open(os.path.join(root, model, "samples", f"{bench}.jsonl.gz"), "rt")]


def load_logits(root, model, bench):
    meta = json.load(open(os.path.join(root, model, bench, "meta.json")))
    assert meta.get("done"), (model, bench)
    return np.load(os.path.join(root, model, bench, "logits.npy"), mmap_mode="r")


def real_ids(root, model, n_rows):
    vocab = json.load(open(os.path.join(root, model, "vocab.json")))
    return np.array(sorted(i for i in vocab.values() if i < n_rows))


#### Single model
def model_job(root, model, bench):
    rows = load_samples(root, model, bench)
    gold = np.array([r["target"] for r in rows])
    ll = [np.array(r["loglik"]) for r in rows]
    if bench in NORM_BENCHES:
        # Harness acc_norm: log-likelihood divided by the length of the answer text in characters
        # (the logged continuation starts with the space that separates it from the prompt)
        length = lambda c: len(c[1:]) if c.startswith(" ") else len(c)
        scores = [l / np.array([length(c) for c in r["choices"]], dtype=float) for l, r in zip(ll, rows)]
    else:
        scores = ll
    pred = np.array([int(np.argmax(s)) for s in scores])
    opt_prob = [np.exp(s - s.max()) / np.exp(s - s.max()).sum() for s in scores]
    conf = np.array([p.max() for p in opt_prob])
    harness_metric = "acc_norm" if bench in NORM_BENCHES else "acc"
    harness_correct = np.array([r[harness_metric] for r in rows], dtype=float)
    mismatch = int((harness_correct != (pred == gold)).sum())
    if mismatch:
        print(f"WARNING {model} {bench}: {mismatch} questions differ from the harness metric", flush=True)

    Z = load_logits(root, model, bench)
    ids = real_ids(root, model, Z.shape[1])
    top1, ent, top50, full = [], [], Pooled(hist=False), Pooled()
    for a in range(0, Z.shape[0], CHUNK):
        lp = log_softmax(np.asarray(Z[a:a + CHUNK], dtype=np.float64)[:, ids])
        top = -np.partition(-lp, TOP_K - 1, axis=1)[:, :TOP_K]
        top1.append(top.max(axis=1))
        ent.append(-(np.exp(lp) * lp).sum(axis=1))
        top50.add(top)
        full.add(lp)
    top1, ent = np.concatenate(top1), np.concatenate(ent)

    stats = dict(accuracy=100 * float((pred == gold).mean()), harness_accuracy=100 * float(harness_correct.mean()),
                 metric=harness_metric, questions=len(rows), mismatch=mismatch,
                 top1=mean_std(top1), entropy=mean_std(ent), top50=top50.summary(), full=full.summary())
    per_q = dict(gold=gold, pred=pred, conf=conf, top1=top1, entropy=ent)
    return stats, per_q


#### Variant against its base
def pair_job(root, variant, bench):
    base = MODELS[variant]["base"]
    Zb, Zv = load_logits(root, base, bench), load_logits(root, variant, bench)
    assert Zb.shape[0] == Zv.shape[0], (variant, bench)
    ids = np.intersect1d(real_ids(root, base, Zb.shape[1]), real_ids(root, variant, Zv.shape[1]))

    out = {k: [] for k in ("kl", "js", "cos", "ccos")}
    fb, fv = Pooled(), Pooled()
    for a in range(0, Zb.shape[0], CHUNK):
        zb = np.asarray(Zb[a:a + CHUNK], dtype=np.float64)[:, ids]
        zv = np.asarray(Zv[a:a + CHUNK], dtype=np.float64)[:, ids]
        lb, lv = log_softmax(zb), log_softmax(zv)
        pb, pv = np.exp(lb), np.exp(lv)
        lm = np.log(0.5 * (pb + pv))
        out["kl"].append((pb * (lb - lv)).sum(axis=1))
        out["js"].append(0.5 * (pb * (lb - lm)).sum(axis=1) + 0.5 * (pv * (lv - lm)).sum(axis=1))
        out["cos"].append((zb * zv).sum(axis=1) / (np.linalg.norm(zb, axis=1) * np.linalg.norm(zv, axis=1)))
        cb, cv = zb - zb.mean(axis=1, keepdims=True), zv - zv.mean(axis=1, keepdims=True)
        out["ccos"].append((cb * cv).sum(axis=1) / (np.linalg.norm(cb, axis=1) * np.linalg.norm(cv, axis=1)))
        fb.add(lb)
        fv.add(lv)
    per_q = {k: np.concatenate(v) for k, v in out.items()}
    stats = dict(base=base, used=int(per_q["kl"].size), skipped=0, vocab=int(ids.size),
                 kl=mean_std(per_q["kl"]), js=mean_std(per_q["js"]), cosine=mean_std(per_q["cos"]),
                 ccos=mean_std(per_q["ccos"]), base_full=fb.summary(), variant_full=fv.summary())
    return stats, per_q


def run_job(job):
    kind, root, name, bench = job
    print(f"{kind} {name} {bench}", flush=True)
    return job, (model_job if kind == "model" else pair_job)(root, name, bench)


#### Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=str, default="/output/harness")
    parser.add_argument("--setting", type=str, default="zeroshot")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = os.path.join(args.harness, args.setting)
    out = args.out or os.path.join(args.harness, f"stats-{args.setting}")
    os.makedirs(out, exist_ok=True)

    # Models with finished results and logits, in registry order
    done = {m for m in os.listdir(root) if os.path.exists(os.path.join(root, m, "results.json"))}
    models = [m for m in MODELS if m in done]
    benches = list(json.load(open(os.path.join(root, models[0], "results.json")))["n_samples"])
    has_logits = lambda m: all(os.path.exists(os.path.join(root, m, b, "meta.json")) for b in benches)
    models = [m for m in models if has_logits(m)]
    pairs = [m for m in models if MODELS[m]["base"] in models]
    print(f"{len(models)} models, {len(pairs)} pairs, benchmarks {benches}")

    jobs = [("model", root, m, b) for m in models for b in benches] + [("pair", root, v, b) for v in pairs for b in benches]
    results = dict(bins=HIST_BINS.tolist(), benchmarks=benches, setting=args.setting, models={}, pairs={})
    arrays, cosines = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (kind, _, name, bench), (stats, per_q) in pool.map(run_job, jobs):
            if kind == "model":
                results["models"].setdefault(name, {})[bench] = stats
                arrays.update({f"{name}|{bench}|{k}": v for k, v in per_q.items()})
            else:
                results["pairs"].setdefault(name, {})[bench] = stats
                arrays.update({f"pair|{name}|{bench}|{k}": v for k, v in per_q.items()})
                cosines[f"{name}|{bench}"] = per_q["cos"]

    results["models"] = {m: results["models"][m] for m in models}
    results["pairs"] = {m: results["pairs"][m] for m in pairs}
    json.dump(results, open(os.path.join(out, "stats.json"), "w"))
    np.savez_compressed(os.path.join(out, "cosine.npz"), **cosines)
    np.savez_compressed(os.path.join(out, "per_question.npz"), **arrays)
    print(f"\nWrote {out}/stats.json, cosine.npz, per_question.npz")
