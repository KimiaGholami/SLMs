#### Runs lm-evaluation-harness on one model and saves the next-token logits for the harness prompts.
#### Output in <out>/<setting>/<model>/:
####   results.json                    harness metrics, task versions, and settings
####   samples/<bench>.jsonl.gz        per question: doc id, target, prompt, log-likelihood of each option, metrics
####   <bench>/logits.npy              next-token logits after each prompt [questions, vocab] (with --logits)
####   <bench>/meta.json               question order and the token ids of the prompt length
#### The setting name records the evaluation setup (e.g. zeroshot, 5shot, chat). The prompts are encoded
#### exactly as the harness encodes them (its tok_encode, BOS token added, trailing spaces removed).
####
#### Usage:
####   python3 run_harness.py --model Llama-3.1-8B --setting zeroshot --logits
####   python3 run_harness.py --group main --index 0 --setting zeroshot --logits   (index from JOB_COMPLETION_INDEX)

#### Libraries
import argparse
import gzip
import importlib.metadata
import json
import os
import time

import numpy as np
import torch

from models import load_model
from registry import MODELS, group_models

HERE = os.path.dirname(os.path.abspath(__file__))

# Benchmark name in the paper -> harness tasks (MMLU is a group of 57 subject tasks)
BENCH_TASKS = {
    "MMLU": "mmlu",
    "ARC-Easy": "arc_easy",
    "ARC-Challenge": "arc_challenge",
    "MathQA": "mathqa_jet",
    "GSM8K-MC": "gsm8k_mc",
}


#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--group", type=str, default="main")
parser.add_argument("--index", type=int, default=None)
parser.add_argument("--setting", type=str, default="zeroshot")
parser.add_argument("--benchmarks", nargs="+", default=list(BENCH_TASKS))
parser.add_argument("--num-fewshot", type=int, default=0)
parser.add_argument("--chat", action="store_true", help="apply the model's chat template")
parser.add_argument("--logits", action="store_true", help="also save next-token logits for every prompt")
parser.add_argument("--limit", type=int, default=None, help="questions per task (for testing)")
parser.add_argument("--out", type=str, default="/output/harness")
parser.add_argument("--tasks-dir", type=str, default=os.path.join(HERE, "tasks"))
args = parser.parse_args()

if args.model is None:
    index = args.index if args.index is not None else int(os.environ["JOB_COMPLETION_INDEX"])
    args.model = group_models(args.group)[index]
spec = MODELS[args.model]
out_dir = os.path.join(args.out, args.setting, args.model)
os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
print(f"Model {args.model} ({spec['hf']}), setting {args.setting}, few-shot {args.num_fewshot}, chat {args.chat}")


#### Device
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}, {free / 2**30:.1f}/{total / 2**30:.1f} GiB free")
    if free < 0.9 * total:
        raise RuntimeError(f"GPU {i} is already in use by another process")
gpu_gib = torch.cuda.get_device_properties(0).total_memory / 2**30


#### Model
tokenizer, model = load_model(spec)
print(f"Loaded on {model.device}, dtype {next(model.parameters()).dtype}")
json.dump(tokenizer.get_vocab(), open(os.path.join(out_dir, "vocab.json"), "w"))

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

batch_size = 4 if gpu_gib < 30 else 16
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, add_bos_token=True)


#### Harness
results_path = os.path.join(out_dir, "results.json")
if os.path.exists(results_path):
    print("Harness results exist, skipping evaluation")
    samples_by_bench = {b: [json.loads(l) for l in gzip.open(os.path.join(out_dir, "samples", f"{b}.jsonl.gz"), "rt")]
                        for b in args.benchmarks}
else:
    t0 = time.time()
    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=[BENCH_TASKS[b] for b in args.benchmarks],
        num_fewshot=args.num_fewshot,
        log_samples=True,
        apply_chat_template=args.chat,
        fewshot_as_multiturn=args.chat and args.num_fewshot > 0,
        task_manager=TaskManager(include_path=args.tasks_dir),
        limit=args.limit,
    )
    print(f"Harness finished in {(time.time() - t0) / 60:.1f} min")

    # Samples per benchmark; MMLU subjects are concatenated in alphabetical order
    samples_by_bench = {}
    for b in args.benchmarks:
        task = BENCH_TASKS[b]
        names = sorted(k for k in res["samples"] if k == task or (task == "mmlu" and k.startswith("mmlu_")))
        rows = []
        for name in names:
            for s in sorted(res["samples"][name], key=lambda s: s["doc_id"]):
                rows.append(dict(task=name, doc_id=s["doc_id"], target=s["target"],
                                 context=s["arguments"][0][0], choices=[a[1] for a in s["arguments"]],
                                 loglik=[float(r[0]) for r in s["filtered_resps"]],
                                 **{m: s[m] for m in ("acc", "acc_norm") if m in s}))
        samples_by_bench[b] = rows
        with gzip.open(os.path.join(out_dir, "samples", f"{b}.jsonl.gz"), "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    summary = dict(model=args.model, hf=spec["hf"], setting=args.setting, num_fewshot=args.num_fewshot,
                   chat=args.chat, lm_eval_version=importlib.metadata.version("lm_eval"), batch_size=batch_size,
                   add_bos_token=True, dtype=str(next(model.parameters()).dtype),
                   results=res["results"], groups=res.get("groups", {}), versions=res.get("versions", {}),
                   n_samples={b: len(r) for b, r in samples_by_bench.items()})
    json.dump(summary, open(results_path, "w"), indent=1, default=str)
    for b, r in samples_by_bench.items():
        acc = np.mean([x["acc"] for x in r]) * 100
        norm = np.mean([x["acc_norm"] for x in r]) * 100 if "acc_norm" in r[0] else float("nan")
        print(f"{b}: acc {acc:.2f}, acc_norm {norm:.2f}, n {len(r)}")


#### Next-token logits after each harness prompt
if args.logits:
    for b, rows in samples_by_bench.items():
        bench_dir = os.path.join(out_dir, b)
        os.makedirs(bench_dir, exist_ok=True)
        meta_path, logits_path = os.path.join(bench_dir, "meta.json"), os.path.join(bench_dir, "logits.npy")
        if os.path.exists(meta_path) and json.load(open(meta_path)).get("done"):
            print(f"{b}: logits exist, skipping")
            continue

        with torch.no_grad():
            first = torch.tensor([lm.tok_encode(rows[0]["context"].rstrip())], device=model.device)
            vocab_size = model(first).logits.shape[-1]
        logits_mm = np.lib.format.open_memmap(logits_path, mode="w+", dtype="float32", shape=(len(rows), vocab_size))
        lengths = []
        t0 = time.time()
        for q, r in enumerate(rows):
            ids = lm.tok_encode(r["context"].rstrip())
            lengths.append(len(ids))
            with torch.no_grad():
                out = model(torch.tensor([ids], device=model.device))
            logits_mm[q] = out.logits[0, -1].float().cpu().numpy()
            if (q + 1) % 2000 == 0:
                print(f"{b} logits: {q + 1}/{len(rows)} ({(q + 1) / (time.time() - t0):.1f}/s)", flush=True)
        logits_mm.flush()
        del logits_mm
        json.dump(dict(model=args.model, benchmark=b, num_questions=len(rows), vocab_size=vocab_size,
                       prompt_tokens=lengths, done=True), open(meta_path, "w"))
        print(f"{b}: saved logits {len(rows)} x {vocab_size}")

print(f"\nDone with {args.model}")
