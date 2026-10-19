#### Per-question statistics for the analysis in analyze.py. Needs only numpy.
#### Reads the logits written by extract_logits.py and writes <out>/per_question.npz with, for every
#### model and benchmark (keys "<model>|<bench>|<field>"):
####   gold         index of the correct option (-1 if the answer key is not an option letter)
####   pred         predicted option, scoring each option by its best letter token (as in the paper)
####   pred_upper   predicted option, using only the uppercase letter with or without a leading space
####   opt_prob     probabilities of the options, renormalized over the options (paper scoring)
####   top1         top-1 log-probability over the vocabulary
####   entropy      entropy of the next-token distribution (nats)
#### and for every variant and benchmark (keys "pair|<variant>|<bench>|<field>"):
####   kl           D_KL(p_base || p_variant)
####   js           Jensen-Shannon divergence (nats)
####   cos          cosine similarity of the raw logits
####   ccos         cosine similarity of the centered logits (Pearson correlation)
#### All quantities use each model's real vocabulary: rows of the output layer without a token
#### (Qwen pads 151,665 tokens to 152,064 rows) are removed before the softmax.
####
#### Usage: python3 per_question.py --logits /output/slm-logits --out /output/slm-logits/stats

#### Libraries
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from compute_stats import CHUNK, load, log_softmax
from registry import BENCHMARKS, MODELS, MODEL_ORDER


#### Helpers
def real_ids(logits_dir, model, n_rows):
    vocab = json.load(open(os.path.join(logits_dir, model, "vocab.json")))
    ids = np.array(sorted(i for i in vocab.values() if i < n_rows))
    return ids, vocab


def upper_ids(vocab, letters):
    # "A" and " A" (byte-level BPE writes the space as "Ġ")
    return [[vocab[t] for t in (L, "Ġ" + L) if t in vocab] for L in letters]


def entropy(lp):
    return -(np.exp(lp) * lp).sum(axis=1)


#### Single model
def model_rows(logits_dir, model, bench):
    Z, meta = load(logits_dir, model, bench)
    ids, vocab = real_ids(logits_dir, model, Z.shape[1])
    # Token ids index the full logit vector; map them to positions in the real vocabulary
    pos = np.full(Z.shape[1], -1)
    pos[ids] = np.arange(ids.size)
    letters = meta["choice_letters"]
    paper_sets = [pos[np.array(t)] for t in meta["choice_token_ids"]]
    upper_sets = [pos[np.array(t)] for t in upper_ids(vocab, letters)]
    assert all((s >= 0).all() for s in paper_sets + upper_sets)

    out = {k: [] for k in ("pred", "pred_upper", "opt_prob", "top1", "entropy")}
    for a in range(0, Z.shape[0], CHUNK):
        z = np.asarray(Z[a:a + CHUNK], dtype=np.float64)[:, ids]
        with np.errstate(invalid="ignore", over="ignore"):
            lp = log_softmax(z)
        scores = np.stack([lp[:, s].max(axis=1) for s in paper_sets], axis=1)
        upper = np.stack([lp[:, s].max(axis=1) for s in upper_sets], axis=1)
        out["pred"].append(scores.argmax(axis=1))
        out["pred_upper"].append(upper.argmax(axis=1))
        out["opt_prob"].append(np.exp(log_softmax(scores)))
        out["top1"].append(lp.max(axis=1))
        out["entropy"].append(entropy(lp))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["gold"] = np.array([letters.index(g) if g in letters else -1 for g in meta["gold"]])
    return out


#### Variant against its base
def pair_rows(logits_dir, variant, bench):
    base = MODELS[variant]["base"]
    Zb, _ = load(logits_dir, base, bench)
    Zv, _ = load(logits_dir, variant, bench)
    ib, vb = real_ids(logits_dir, base, Zb.shape[1])
    iv, vv = real_ids(logits_dir, variant, Zv.shape[1])
    if MODELS[variant]["loader"] == "lora":
        # Compare on the tokens both tokenizers share
        inv_b = {i: t for t, i in vb.items()}
        shared = sorted(t for t in (inv_b[i] for i in ib) if t in vv and vv[t] < Zv.shape[1])
        ib, iv = np.array([vb[t] for t in shared]), np.array([vv[t] for t in shared])
    else:
        # Same token ids; some derived tokenizers rename special tokens, so match by id
        ib = iv = np.intersect1d(ib, iv)

    out = {k: [] for k in ("kl", "js", "cos", "ccos")}
    for a in range(0, Zb.shape[0], CHUNK):
        zb = np.asarray(Zb[a:a + CHUNK], dtype=np.float64)[:, ib]
        zv = np.asarray(Zv[a:a + CHUNK], dtype=np.float64)[:, iv]
        lb, lv = log_softmax(zb), log_softmax(zv)
        pb, pv = np.exp(lb), np.exp(lv)
        lm = np.log(0.5 * (pb + pv))
        out["kl"].append((pb * (lb - lv)).sum(axis=1))
        out["js"].append(0.5 * (pb * (lb - lm)).sum(axis=1) + 0.5 * (pv * (lv - lm)).sum(axis=1))
        out["cos"].append((zb * zv).sum(axis=1) / (np.linalg.norm(zb, axis=1) * np.linalg.norm(zv, axis=1)))
        cb, cv = zb - zb.mean(axis=1, keepdims=True), zv - zv.mean(axis=1, keepdims=True)
        out["ccos"].append((cb * cv).sum(axis=1) / (np.linalg.norm(cb, axis=1) * np.linalg.norm(cv, axis=1)))
    return {k: np.concatenate(v) for k, v in out.items()}


def run_job(job):
    kind, logits_dir, name, bench = job
    print(f"{kind} {name} {bench}", flush=True)
    rows = model_rows(logits_dir, name, bench) if kind == "model" else pair_rows(logits_dir, name, bench)
    return job, rows


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

    arrays = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (kind, _, name, bench), rows in pool.map(run_job, jobs):
            prefix = f"{name}|{bench}" if kind == "model" else f"pair|{name}|{bench}"
            for k, v in rows.items():
                arrays[f"{prefix}|{k}"] = v

    np.savez_compressed(os.path.join(out, "per_question.npz"), **arrays)
    print(f"\nWrote {out}/per_question.npz")
