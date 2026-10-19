#### Saves the next-token logits after each benchmark prompt, for one model.
#### Output (per model and benchmark):
####   <out>/<model>/<bench>/logits.npy   [num_questions, vocab_size]
####   <out>/<model>/<bench>/meta.json    gold answers, answer-letter token ids, progress
####   <out>/<model>/vocab.json           tokenizer vocabulary (used to align the LoRA vocab)
#### The run can be restarted: finished benchmarks are skipped and unfinished ones resume.
####
#### Usage:
####   python3 extract_logits.py --model Llama-3.1-8B
####   python3 extract_logits.py --index 0           # index into registry.MODEL_ORDER
####   python3 extract_logits.py --group full --index 0   # index into the float32 models only
####   (in an Indexed Job, the index is read from JOB_COMPLETION_INDEX)

#### Libraries
import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from registry import BENCHMARKS, CHOICE_LETTERS, MODELS, MODEL_ORDER, format_example, load_benchmark
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate.utils.modeling")

SAVE_INTERVAL = 500  # flush logits and progress every N questions


#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--index", type=int, default=None)
parser.add_argument("--group", type=str, default="all", choices=["all", "full", "small"],
                    help="models the index refers to: all, float32 models (full), or quantized and LoRA models (small)")
parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
parser.add_argument("--out", type=str, default="/output/slm-logits")
parser.add_argument("--store-dtype", type=str, default="float32", choices=["float32", "float16"])
parser.add_argument("--limit", type=int, default=None, help="only the first N questions (for testing)")
args = parser.parse_args()

if args.model is None:
    index = args.index if args.index is not None else int(os.environ["JOB_COMPLETION_INDEX"])
    group = [m for m in MODEL_ORDER
             if args.group == "all" or (MODELS[m]["loader"] == "full") == (args.group == "full")]
    args.model = group[index]

spec = MODELS[args.model]
model_dir = os.path.join(args.out, args.model)
os.makedirs(model_dir, exist_ok=True)
print(f"\nModel {args.model} ({spec['hf']}), loader={spec['loader']}, dtype={spec['dtype']}")


#### Device
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}, {free / 2**30:.1f}/{total / 2**30:.1f} GiB free")
        # Some nodes hand out GPUs that another process still occupies
        if free < 0.9 * total:
            raise RuntimeError(f"GPU {i} is already in use by another process")


#### Load Model (same settings as the original per-model scripts)
def load_model(spec):
    hf = spec["hf"]
    # Float32 models (about 30 GiB) fill the GPUs first; on a 24 GB GPU the remaining layers are kept in
    # CPU memory and moved to the GPU for each forward pass (slower, same results). The other models
    # fit on one GPU; bitsandbytes refuses a device map that places any module off the GPU, and PEFT
    # fails when a LoRA model is split over devices.
    if spec["loader"] == "full":
        device_map = "auto"
        # When layers are offloaded, the GPU also needs room for the largest of them when it is moved
        # there (Qwen's float32 output layer is 2 GiB), so smaller GPUs keep more memory free
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total = int(torch.cuda.get_device_properties(i).total_memory / 2**30)
            max_memory[i] = f"{total - (2 if total >= 30 else 5)}GiB"
        max_memory["cpu"] = "24GiB"
        print(f"max_memory {max_memory}")
    else:
        device_map, max_memory = {"": 0}, None
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)

    if spec["loader"] == "full":
        config = AutoConfig.from_pretrained(hf, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            print("Removing unsupported quantization_config:", config.quantization_config)
            del config.quantization_config
        model = AutoModelForCausalLM.from_pretrained(hf, config=config, torch_dtype=getattr(torch, spec["dtype"]),
                                                     device_map=device_map, max_memory=max_memory, trust_remote_code=True, low_cpu_mem_usage=True)

    elif spec["loader"] == "bnb4":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(hf, quantization_config=bnb_config,
                                                     device_map=device_map, max_memory=max_memory, trust_remote_code=True, low_cpu_mem_usage=True)
        if spec["family"] == "Qwen":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

    elif spec["loader"] == "fp8":
        model = AutoModelForCausalLM.from_pretrained(hf, device_map=device_map, max_memory=max_memory, trust_remote_code=True, low_cpu_mem_usage=True)

    elif spec["loader"] == "lora":
        # Same steps as AutoPeftModelForCausalLM: load the base model, resize its embeddings to the
        # adapter's tokenizer, and attach the adapter (its LoRA weights are upcast to float32, as in
        # the original run).
        from peft import PeftConfig, PeftModel
        base_name = PeftConfig.from_pretrained(hf).base_model_name_or_path
        print(f"LoRA base model {base_name}")
        model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, device_map=device_map,
                                                     trust_remote_code=True, low_cpu_mem_usage=True)
        # The adapter stores full embedding and output layers, so the new rows are overwritten when it
        # is attached; mean_resizing=False skips a covariance fit that fails on some GPUs
        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model = PeftModel.from_pretrained(model, hf, torch_device="cpu")  # read the adapter file into CPU memory first
    else:
        raise ValueError(spec["loader"])

    model.eval()
    devices = set(getattr(model, "hf_device_map", {"": 0}).values())
    print(f"devices {devices}")
    if "disk" in devices or ("cpu" in devices and spec["loader"] != "full"):
        raise RuntimeError(f"Part of the model was offloaded to {devices}; use a larger GPU")
    if "cpu" in devices:
        print("Some layers are kept in CPU memory; this run will be slower")
    return tokenizer, model


def letter_token_ids(tokenizer, letter):
    # Last token of the letter, with or without a leading space or newline, in upper or lower case
    ids_set = set()
    for prefix in ["", " ", "\n", "\n "]:
        for L in (letter, letter.lower()):
            ids = tokenizer(prefix + L, add_special_tokens=False)["input_ids"]
            if ids:
                ids_set.add(ids[-1])
    return sorted(ids_set)


print(f"\nLoading Model {spec['hf']}")
tokenizer, model = load_model(spec)

with open(os.path.join(model_dir, "vocab.json"), "w") as f:
    json.dump(tokenizer.get_vocab(), f)


#### Benchmarks
for bench in args.benchmarks:
    bench_dir = os.path.join(model_dir, bench)
    os.makedirs(bench_dir, exist_ok=True)
    meta_path = os.path.join(bench_dir, "meta.json")
    logits_path = os.path.join(bench_dir, "logits.npy")

    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None

    print(f"\nLoading {bench} Benchmark")
    ds = load_benchmark(bench)

    # Prompts in the original order, without the skipped questions
    items = []
    for idx, example in enumerate(ds):
        prompt, gold, skip = format_example(bench, example)
        if skip:
            print(f"Skipping question {idx} (fewer than 4 options)")
            continue
        items.append((idx, prompt, gold))
    if bench == "MathQA":
        # The original format splits options on commas, so an option that contains a comma is cut apart
        n_bad = sum(len(ex["options"].split(",")) != 5 for ex in ds)
        print(f"MathQA questions whose options do not split into exactly 5: {n_bad}/{len(ds)}")
    if args.limit is not None:
        items = items[:args.limit]
    num_questions = len(items)

    if meta is not None and meta.get("done") and meta["num_questions"] == num_questions:
        print(f"{bench}: already done, skipping")
        continue

    # Logits of one prompt, to get the vocabulary size
    with torch.no_grad():
        test = tokenizer(items[0][1], return_tensors="pt", truncation=True).to(model.device)
        vocab_size = model(**test, return_dict=True).logits.shape[-1]

    resume = (meta is not None and os.path.exists(logits_path)
              and meta["num_questions"] == num_questions and meta["vocab_size"] == vocab_size)
    if resume:
        logits_mm = np.load(logits_path, mmap_mode="r+")
        start = meta["completed"]
        print(f"Resuming {bench} at question {start}/{num_questions}")
    else:
        logits_mm = np.lib.format.open_memmap(logits_path, mode="w+", dtype=args.store_dtype,
                                              shape=(num_questions, vocab_size))
        start = 0
        meta = dict(model=args.model, hf=spec["hf"], benchmark=bench,
                    num_questions=num_questions, vocab_size=vocab_size, store_dtype=args.store_dtype,
                    dataset_index=[it[0] for it in items],
                    gold=[it[2] for it in items],
                    choice_letters=CHOICE_LETTERS[bench],
                    choice_token_ids=[letter_token_ids(tokenizer, L) for L in CHOICE_LETTERS[bench]],
                    completed=0, done=False)

    print(f"Questions: {num_questions}, vocab size: {vocab_size}")
    for L, ids in zip(meta["choice_letters"], meta["choice_token_ids"]):
        print(f"  {L}: {ids} {[tokenizer.decode([i]) for i in ids]}")

    t0 = time.time()
    for q in range(start, num_questions):
        inputs = tokenizer(items[q][1], return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        logits_mm[q] = outputs.logits[0, -1, :].float().cpu().numpy()

        if (q + 1) % SAVE_INTERVAL == 0 or q + 1 == num_questions:
            logits_mm.flush()
            meta["completed"] = q + 1
            meta["done"] = (q + 1 == num_questions)
            json.dump(meta, open(meta_path, "w"))
            rate = (q + 1 - start) / (time.time() - t0)
            print(f"{bench}: {q+1}/{num_questions} ({rate:.1f} questions/s)", flush=True)

    del logits_mm
    print(f"Done with {bench}")

print(f"\nDone with {args.model}")
