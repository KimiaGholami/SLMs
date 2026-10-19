#### Models, benchmarks, and prompt formats used in the paper.
#### Prompt formats and dataset order are copied from the original per-model scripts
#### (e.g. Llama/Llama-3.1-8B/<BENCH>/INIT.py) so that the saved logits match the paper.

#### Models
# key: short name used for files and figures
#   hf:     Hugging Face model id
#   loader: how to load it (full, bnb4, fp8, lora)
#   dtype:  dtype of the original runs (full-precision models were loaded in float32,
#           the transformers default at the time)
#   base:   model it was derived from (None for base models)
#   family: Llama or Qwen
#   label:  name shown in the paper
MODELS = {
    # Llama-3.1 family
    "Llama-3.1-8B":                       dict(hf="meta-llama/Llama-3.1-8B",                         loader="full", dtype="float32",  base=None,                    family="Llama"),
    "Llama-3.1-8B-Instruct":              dict(hf="meta-llama/Llama-3.1-8B-Instruct",                loader="full", dtype="float32",  base="Llama-3.1-8B",          family="Llama"),
    "DeepSeek-R1-Distill-Llama-8B":       dict(hf="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",        loader="full", dtype="float32",  base="Llama-3.1-8B",          family="Llama"),
    "Llama-3.1-8B-Pruned-4-Layers":       dict(hf="Na0s/Llama-3.1-8B-Pruned-4-Layers",               loader="full", dtype="float32",  base="Llama-3.1-8B",          family="Llama"),
    "Code-Llama-3.1-8B-Text-to-SQL-LoRA": dict(hf="philschmid/code-llama-3-1-8b-text-to-sql-lora",   loader="lora", dtype="bfloat16", base="Llama-3.1-8B",          family="Llama"),
    "Llama-3.1-8B-bnb-4bit":              dict(hf="unsloth/Meta-Llama-3.1-8B-bnb-4bit",              loader="bnb4", dtype="bfloat16", base="Llama-3.1-8B",          family="Llama"),
    "Llama-3.1-8B-Instruct-FP8-dynamic":  dict(hf="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic", loader="fp8",  dtype="auto",     base="Llama-3.1-8B-Instruct", family="Llama"),
    # Qwen-2.5 family
    "Qwen-2.5-7B":                        dict(hf="Qwen/Qwen2.5-7B",                                 loader="full", dtype="float32",  base=None,                    family="Qwen"),
    "Qwen-2.5-7B-Instruct":               dict(hf="Qwen/Qwen2.5-7B-Instruct",                        loader="full", dtype="float32",  base="Qwen-2.5-7B",           family="Qwen"),
    "Qwen-2.5-Math-7B":                   dict(hf="Qwen/Qwen2.5-Math-7B",                            loader="full", dtype="float32",  base="Qwen-2.5-7B",           family="Qwen"),
    "DeepSeek-R1-Distill-Qwen-7B":        dict(hf="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",         loader="full", dtype="float32",  base="Qwen-2.5-Math-7B",      family="Qwen"),
    "Qwen-2.5-Math-7B-Instruct":          dict(hf="Qwen/Qwen2.5-Math-7B-Instruct",                   loader="full", dtype="float32",  base="Qwen-2.5-Math-7B",      family="Qwen"),
    "Qwen-2.5-7B-bnb-4bit":               dict(hf="unsloth/Qwen2.5-7B-bnb-4bit",                     loader="bnb4", dtype="bfloat16", base="Qwen-2.5-7B",           family="Qwen"),
    "Qwen-2.5-7B-FP8-dynamic":            dict(hf="RedHatAI/Qwen2.5-7B-FP8-dynamic",                 loader="fp8",  dtype="auto",     base="Qwen-2.5-7B",           family="Qwen"),
}

# Order of the models in the paper (Table I), also the index of the extraction job
MODEL_ORDER = list(MODELS.keys())

# Small language models built from larger ones or trained at small scale
SLM_MODELS = {
    "Llama-3.1-Minitron-4B-Width":       dict(hf="nvidia/Llama-3.1-Minitron-4B-Width-Base", base="Llama-3.1-8B"),
    "Llama-3.1-Minitron-4B-Depth":       dict(hf="nvidia/Llama-3.1-Minitron-4B-Depth-Base", base="Llama-3.1-8B"),
    "Llama-3.2-3B":                      dict(hf="meta-llama/Llama-3.2-3B", base="Llama-3.1-8B"),
    "Llama-3.2-3B-Instruct":             dict(hf="meta-llama/Llama-3.2-3B-Instruct", base="Llama-3.2-3B"),
    "Llama-3.2-1B":                      dict(hf="meta-llama/Llama-3.2-1B", base="Llama-3.1-8B"),
    "Llama-3.2-1B-Instruct":             dict(hf="meta-llama/Llama-3.2-1B-Instruct", base="Llama-3.2-1B"),
    "Qwen-2.5-3B":                       dict(hf="Qwen/Qwen2.5-3B", base=None),
    "Qwen-2.5-3B-Instruct":              dict(hf="Qwen/Qwen2.5-3B-Instruct", base="Qwen-2.5-3B"),
    "Qwen-2.5-1.5B":                     dict(hf="Qwen/Qwen2.5-1.5B", base=None),
    "Qwen-2.5-1.5B-Instruct":            dict(hf="Qwen/Qwen2.5-1.5B-Instruct", base="Qwen-2.5-1.5B"),
    "Qwen-2.5-0.5B":                     dict(hf="Qwen/Qwen2.5-0.5B", base=None),
    "Qwen-2.5-0.5B-Instruct":            dict(hf="Qwen/Qwen2.5-0.5B-Instruct", base="Qwen-2.5-0.5B"),
    "Qwen-2.5-Math-1.5B":                dict(hf="Qwen/Qwen2.5-Math-1.5B", base="Qwen-2.5-1.5B"),
    "Qwen-2.5-Math-1.5B-Instruct":       dict(hf="Qwen/Qwen2.5-Math-1.5B-Instruct", base="Qwen-2.5-Math-1.5B"),
    "DeepSeek-R1-Distill-Qwen-1.5B":     dict(hf="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", base="Qwen-2.5-Math-1.5B"),
}
for name, spec in SLM_MODELS.items():
    spec.update(loader="full", dtype="bfloat16", family="Llama" if "Llama" in name else "Qwen")

# Controlled sweep: one method at increasing strength, applied to each base model
SWEEP_MODELS = {}
for base in ("Llama-3.1-8B", "Qwen-2.5-7B"):
    steps = [(f"HQQ-{b}bit", {"hqq_bits": b}) for b in (8, 4, 3, 2)]
    steps += [(f"Magnitude-{int(100 * p)}", {"prune": p}) for p in (0.1, 0.3, 0.5)]
    steps += [("Magnitude-2-4", {"prune_2_4": True})]
    steps += [(f"Drop-{n}-Layers", {"drop_layers": n}) for n in (2, 4, 8)]
    for suffix, transform in steps:
        SWEEP_MODELS[f"{base}-{suffix}"] = dict(hf=MODELS[base]["hf"], loader="full", dtype="bfloat16", base=base,
                                                family=MODELS[base]["family"], transform=transform)

MODELS.update(SLM_MODELS)
MODELS.update(SWEEP_MODELS)

# Short panel titles for the main-text figures (family prefix removed)
PANEL_TITLES = {
    "Llama-3.1-8B-Instruct":              "Instruct",
    "DeepSeek-R1-Distill-Llama-8B":       "DeepSeek-R1-Distill-Llama-8B",
    "Llama-3.1-8B-Pruned-4-Layers":       "Pruned-4-Layers",
    "Code-Llama-3.1-8B-Text-to-SQL-LoRA": "Text-to-SQL-LoRA",
    "Llama-3.1-8B-bnb-4bit":              "bnb-4bit",
    "Llama-3.1-8B-Instruct-FP8-dynamic":  "Instruct-FP8-dynamic",
    "Qwen-2.5-7B-Instruct":               "Instruct",
    "Qwen-2.5-Math-7B":                   "Math",
    "DeepSeek-R1-Distill-Qwen-7B":        "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen-2.5-Math-7B-Instruct":          "Math-Instruct",
    "Qwen-2.5-7B-bnb-4bit":               "bnb-4bit",
    "Qwen-2.5-7B-FP8-dynamic":            "FP8-dynamic",
}

QUANTIZED = {k for k, v in MODELS.items() if v["loader"] in ("bnb4", "fp8")}

# Models evaluated by run_harness.py, by group (the job index selects a model within its group)
# Llama-3.2 is a gated repository; its group runs once the Hugging Face token has access
GATED = [m for m in SLM_MODELS if m.startswith("Llama-3.2")]
GROUPS = {"main": MODEL_ORDER, "slm": [m for m in SLM_MODELS if m not in GATED], "slm-gated": GATED,
          "sweep": list(SWEEP_MODELS)}


def group_models(group):
    return GROUPS[group]


#### Benchmarks
BENCHMARKS = ["MMLU", "MathQA", "GSM8K-MC", "ARC"]

# Number of answer options scored for each benchmark
CHOICE_LETTERS = {
    "MMLU":     ["A", "B", "C", "D"],
    "MathQA":   ["A", "B", "C", "D", "E"],
    "GSM8K-MC": ["A", "B", "C", "D"],
    "ARC":      ["A", "B", "C", "D"],
}


def load_benchmark(bench):
    from datasets import load_dataset
    if bench == "MMLU":
        ds = load_dataset("cais/mmlu", "all")
    elif bench == "MathQA":
        ds = load_dataset("jet-ai/math_qa")
    elif bench == "GSM8K-MC":
        ds = load_dataset("XiangPan/gsm8k-mc")
    elif bench == "ARC":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    else:
        raise ValueError(bench)
    return ds["test"].shuffle(seed=42)


def format_example(bench, example):
    """Returns (prompt, gold_letter, skip). skip=True drops the question (ARC with < 4 options)."""
    if bench == "MMLU":
        choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(example["choices"])])
        prompt = f"{example['question']}\n{choices}\nAnswer:"
        return prompt, chr(65 + int(example["answer"])), False

    if bench == "MathQA":
        options = [opt[4:] for opt in example["options"].split(",")]
        choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(options)])
        prompt = f"{example['Problem']}\n{choices}\nAnswer:"
        return prompt, example["correct"].strip().upper(), False

    if bench == "GSM8K-MC":
        prompt = example["question"]
        for L in "ABCD":
            prompt += f"\n{L}) {example[L]}"
        return f"{prompt}\nanswer:", example["Answer"], False

    if bench == "ARC":
        texts = example["choices"]["text"]
        if len(texts) < 4:
            return None, None, True
        prompt = example["question"]
        for i, L in enumerate("ABCD"):
            prompt += f"\n{L}) {texts[i]}"
        return f"{prompt}\nanswer:", example["answerKey"], False

    raise ValueError(bench)
