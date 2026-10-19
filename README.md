# Building Efficient Small Language Models: How SLM-Building Methods Change the Next-Token Distribution

Code for the paper *Building Efficient Small Language Models (SLMs)*.

Small language models (SLMs) are usually built from larger models by fine-tuning, distillation, reinforcement learning, pruning, or quantization. Each of these methods changes the model's next-token distribution, but these changes are rarely measured directly. This repository evaluates public models built with these methods, saves their next-token logits on standard benchmarks, and compares each model's distribution with that of the model it was derived from.

## What the paper studies

The paper organizes SLM-building methods into three families, by the source of the change they introduce:

| Family | Training signal | Methods |
|---|---|---|
| Knowledge-based | A fixed target to imitate | Supervised fine-tuning (SFT), continued pre-training, logit and feature knowledge distillation (KD) |
| Feedback-based | Feedback on generated responses | On-policy KD, reinforcement learning (RLHF, RLAIF, GRPO), preference optimization (DPO) |
| Parameter-based | A direct change to the weights | Parameter-efficient fine-tuning (LoRA), pruning, quantization |

The experiments ask three questions:

1. **How far does each method move the next-token distribution away from the base model?**
2. **Can label-free divergence from the base model locate the questions on which a variant gets worse?**
3. **What does repeated (stacked) specialization do to the distribution and to accuracy?**

## Models

All models are public Hugging Face checkpoints. Each variant is compared with the model it was derived from (its *base*). The full list, with loaders and base models, is in [`registry.py`](registry.py).

**Main models** (registry group `main`, 14 models): two families with variants built by the methods above.

| Model | Base model | Method |
|---|---|---|
| Llama-3.1-8B | – | Pre-training |
| Llama-3.1-8B-Instruct | Llama-3.1-8B | SFT + rejection sampling + DPO |
| DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B | SFT on DeepSeek-R1 outputs |
| Llama-3.1-8B-Pruned-4-Layers | Llama-3.1-8B | Depth pruning |
| Code-Llama-3.1-8B-Text-to-SQL-LoRA | Llama-3.1-8B | SFT with LoRA |
| Llama-3.1-8B-bnb-4bit | Llama-3.1-8B | 4-bit NF4 quantization |
| Llama-3.1-8B-Instruct-FP8-dynamic | Llama-3.1-8B-Instruct | FP8 quantization |
| Qwen-2.5-7B | – | Pre-training |
| Qwen-2.5-7B-Instruct | Qwen-2.5-7B | SFT + DPO + GRPO |
| Qwen-2.5-Math-7B | Qwen-2.5-7B | Continued pre-training on math |
| DeepSeek-R1-Distill-Qwen-7B | Qwen-2.5-Math-7B | SFT on DeepSeek-R1 outputs |
| Qwen-2.5-Math-7B-Instruct | Qwen-2.5-Math-7B | Math SFT + reward model + GRPO |
| Qwen-2.5-7B-bnb-4bit | Qwen-2.5-7B | 4-bit NF4 quantization |
| Qwen-2.5-7B-FP8-dynamic | Qwen-2.5-7B | FP8 quantization |

**Small models** (groups `slm` and `slm-gated`, 15 models): Llama-3.1-Minitron-4B Width and Depth (pruned and distilled from Llama-3.1-8B), Llama-3.2-1B and 3B with their Instruct versions (gated on Hugging Face), Qwen-2.5-0.5B, 1.5B, and 3B with their Instruct versions, and the Qwen-2.5-Math-1.5B line (Math, Math-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), which repeats the 7B specialization chain at 1.5B.

**Controlled sweep** (group `sweep`, 22 models): one method at increasing strength, applied to Llama-3.1-8B and to Qwen-2.5-7B, with no other change. The transforms are implemented in [`models.py`](models.py):

| Method | Settings |
|---|---|
| HQQ quantization (calibration-free) | 8, 4, 3, 2 bits, group size 64 |
| Unstructured magnitude pruning of the linear layers in the Transformer blocks | 10%, 30%, 50% |
| Semi-structured 2:4 magnitude pruning | 2 of every 4 weights kept |
| Depth pruning (remove the blocks before the last one) | 2, 4, 8 blocks |

## Evaluation protocol

Accuracy uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.5 with its default prompts, zero-shot, without a chat template, in bfloat16, with the BOS token added (`add_bos_token=True`).

| Benchmark | Harness task | Format | Metric | Questions |
|---|---|---|---|---|
| MMLU | `mmlu` | Lettered options, letter log-likelihood | acc | 14,042 |
| ARC-Easy | `arc_easy` | Cloze (answer text log-likelihood) | acc_norm | 2,376 |
| ARC-Challenge | `arc_challenge` | Cloze | acc_norm | 1,172 |
| MathQA | `mathqa_jet` ([`tasks/`](tasks)) | Cloze, harness built-in task on the `jet-ai/math_qa` copy | acc_norm | 2,985 |
| GSM8K-MC | `gsm8k_mc` ([`tasks/`](tasks)) | Lettered options, in the MMLU format | acc | 1,319 |

`acc_norm` divides each option's log-likelihood by its length in characters, as on the Open LLM Leaderboard.

For every question, the next-token logits after the harness prompt are saved, using the harness's own tokenization of the prompt. The distribution analysis is therefore done on exactly the inputs that the accuracy is measured on.

## Distribution metrics

For each question, the next-token logits are converted to log-probabilities with a log-softmax over the model's real vocabulary. Rows of the output layer without a token (Qwen pads 151,665 tokens to 152,064 or 151,936 rows) are removed. A variant and its base are compared on the token ids they share.

| Quantity | Definition |
|---|---|
| Confidence | Top-1 log-probability over the vocabulary, mean and standard deviation over questions |
| Entropy | Entropy of the next-token distribution (nats) |
| Shape | Mean and standard deviation of all log-probabilities, pooled over the questions of a benchmark; histogram on 159 bins from −40 to 5 |
| Top-50 | Mean and standard deviation of the 50 largest log-probabilities, pooled over questions |
| KLD | $D_{\mathrm{KL}}(p_{\text{base}} \,\|\, p_{\text{variant}})$, averaged over questions |
| JSD | Jensen–Shannon divergence between the two distributions (nats) |
| Cosine | Cosine similarity of the raw logit vectors |
| Centered cosine | Cosine similarity after subtracting each vector's mean (the Pearson correlation of the logits); unlike the raw cosine, it does not change when a constant is added to all logits |

## Statistics

[`analyze.py`](analyze.py) reports:

- **Accuracy** with 95% Wilson intervals.
- **Accuracy change** of each variant against its base: paired 95% bootstrap interval and exact McNemar test on the questions only one of the two models answers correctly.
- **Across models:** Spearman correlation between each distribution metric and the accuracy change over all variant–benchmark pairs, with a 95% interval from a bootstrap that resamples variants (each variant contributes one point per benchmark). Computed for all variants, per family, and for the controlled sweep alone.
- **Per question:** AUROC of each metric for detecting (i) questions whose predicted answer changes and (ii) questions the base answers correctly and the variant answers wrongly.
- **Calibration:** expected calibration error over the answer options (10 bins).

## Repository layout

| Path | Purpose |
|---|---|
| [`registry.py`](registry.py) | Models, base models, loaders, model groups, benchmarks, and the original prompt formats |
| [`models.py`](models.py) | Model loading for the harness runs (bfloat16, one GPU) and the sweep transforms |
| [`run_harness.py`](run_harness.py) | GPU: lm-evaluation-harness on one model, plus next-token logits for every harness prompt |
| [`harness_stats.py`](harness_stats.py) | CPU: accuracy, confidence, shape, KLD, JSD, cosine, and per-question data from the harness outputs |
| [`analyze.py`](analyze.py) | Laptop: statistical analysis (Section "Statistics") |
| [`make_figures.py`](make_figures.py) | Laptop: all result figures (log-probability distributions and cosine similarities, main text and appendix) |
| [`extract_logits.py`](extract_logits.py), [`compute_stats.py`](compute_stats.py), [`per_question.py`](per_question.py) | Original pipeline (see below) |
| [`tasks/`](tasks) | Custom harness tasks (`mathqa_jet`, `gsm8k_mc`) |
| [`jobs/`](jobs) | Kubernetes job templates for every step |

## Running the harness pipeline

Each step can be rerun on its own. The GPU step needs one GPU with at least 24 GB for the 7B and 8B models.

```bash
pip install -r requirements-gpu.txt

# 1. Evaluate one model and save its logits (GPU)
python3 run_harness.py --model Llama-3.1-8B --setting zeroshot --logits --out outputs/harness
#    or one model of a group: --group main --index 0   (the index is read from JOB_COMPLETION_INDEX if omitted)
#    quick test: --limit 10

# 2. Statistics for every finished model (CPU)
python3 harness_stats.py --harness outputs/harness --setting zeroshot --out stats-zeroshot

# 3. Analysis and figures (laptop)
pip install -r requirements.txt
python3 analyze.py --stats stats-zeroshot          # writes stats-zeroshot/analysis.md and analysis.json
python3 make_figures.py --stats stats-zeroshot --out figures
```

Outputs of `run_harness.py`, in `<out>/<setting>/<model>/`:

- `results.json`: harness metrics, task versions, and settings.
- `samples/<bench>.jsonl.gz`: per question, the prompt, the log-likelihood of each option, and the metrics.
- `<bench>/logits.npy`: next-token logits after each prompt, shape `[questions, vocabulary]`, float32.
- `<bench>/meta.json` and `vocab.json`: question order, prompt lengths, and the tokenizer vocabulary.

Storage: about 11 to 13 GB of logits per model for the five benchmarks (21,894 questions).

Other settings use the same script: `--setting 5shot --num-fewshot 5` for few-shot evaluation, and `--setting chat --chat` to apply the model's chat template.

## Running on a Kubernetes cluster

The experiments in the paper ran on the [National Research Platform (Nautilus)](https://nationalresearchplatform.org/). The templates in [`jobs/`](jobs) run each step as a Kubernetes Job:

- `harness-main.yaml`, `harness-slm.yaml`, `harness-slm-gated.yaml`, and `harness-sweep.yaml` are indexed jobs with one index per model in a registry group. Each pod uses one GPU.
- `harness-stats.yaml` runs the statistics on CPU.

To adapt the templates to another cluster, change:

- the namespace;
- the persistent volume claim mounted at `/output`;
- the secret `hf-token-secret`, which holds a Hugging Face token (the Llama models require accepting their licenses);
- the GPU types and node exclusions in `nodeAffinity`.

The code reaches the pods through ConfigMaps:

```bash
kubectl create configmap slm-pipeline --from-file=registry.py --from-file=models.py \
  --from-file=run_harness.py --from-file=harness_stats.py --from-file=compute_stats.py \
  --from-file=extract_logits.py --from-file=per_question.py --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap slm-pipeline-tasks --from-file=tasks/mathqa_jet.yaml \
  --from-file=tasks/gsm8k_mc.yaml --from-file=tasks/utils.py --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f jobs/harness-main.yaml
```

## Original pipeline

The first version of the paper used custom zero-shot prompts and letter scoring. Its pipeline is kept so that those figures can be reproduced:

1. `extract_logits.py` (GPU, float32; `jobs/extract-full.yaml`, `jobs/extract-small.yaml`) saves the next-token logits after each prompt.
2. `compute_stats.py` (CPU; `jobs/compute-stats.yaml`) computes accuracy, confidence, shape, KLD, and cosine similarity.
3. `per_question.py` (CPU; `jobs/per-question.yaml`) saves per-question data.
4. `analyze.py` and `make_figures.py` run as above.

The prompt formats are in `registry.py`. Its MathQA formatter splits options on commas, which breaks the 71 of 2,985 questions whose options contain a comma. The harness pipeline uses the harness's own option parser instead.

## Software versions

PyTorch 2.4.1 with CUDA 12.1 (`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`), lm-evaluation-harness 0.4.5, transformers 4.46.3, accelerate 1.1.1, datasets 3.1.0, peft 0.13.2, bitsandbytes 0.44.1, compressed-tensors 0.10.2, hqq 0.2.2, NumPy < 2, pyarrow 17.0.0. See [`requirements-gpu.txt`](requirements-gpu.txt).

## Citation

The citation will be added when the paper is published.
