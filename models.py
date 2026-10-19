#### Model loading for run_harness.py (bfloat16 on one GPU, the precision used by standard evaluations).
#### Registry entries may add a "transform" that is applied after loading (used for the sweep models):
####   {"prune": fraction}        unstructured magnitude pruning of every linear layer in the blocks
####   {"prune_2_4": True}        2:4 semi-structured magnitude pruning (2 of every 4 weights kept)
####   {"drop_layers": n}         remove the n blocks before the last one (Gromov et al., 2024)
####   {"hqq_bits": b}            calibration-free HQQ quantization of the linear layers to b bits

#### Libraries
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model(spec):
    hf = spec["hf"]
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    kwargs = dict(device_map={"": 0}, trust_remote_code=True, low_cpu_mem_usage=True)
    transform = spec.get("transform", {})

    if spec["loader"] == "full":
        config = AutoConfig.from_pretrained(hf, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        if "hqq_bits" in transform:
            from transformers import HqqConfig
            kwargs["quantization_config"] = HqqConfig(nbits=transform["hqq_bits"], group_size=64)
        model = AutoModelForCausalLM.from_pretrained(hf, config=config, torch_dtype=torch.bfloat16, **kwargs)

    elif spec["loader"] == "bnb4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(hf, quantization_config=bnb, **kwargs)

    elif spec["loader"] == "fp8":
        model = AutoModelForCausalLM.from_pretrained(hf, **kwargs)

    elif spec["loader"] == "lora":
        # Attach the adapter and merge it into the weights
        from peft import PeftConfig, PeftModel
        base_name = PeftConfig.from_pretrained(hf).base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, **kwargs)
        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        # Read the adapter file into CPU memory first; it carries full copies of the embedding and output layers
        model = PeftModel.from_pretrained(model, hf, torch_device="cpu").merge_and_unload()
    else:
        raise ValueError(spec["loader"])

    if "prune" in transform or "prune_2_4" in transform:
        magnitude_prune(model, transform.get("prune"), transform.get("prune_2_4", False))
    if "drop_layers" in transform:
        drop_layers(model, transform["drop_layers"])

    model.eval()
    return tokenizer, model


#### Pruning
def block_linears(model):
    for layer in model.model.layers:
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                yield module


@torch.no_grad()
def magnitude_prune(model, fraction, two_four):
    # Zero the smallest-magnitude weights of each linear layer in the Transformer blocks
    for lin in block_linears(model):
        w = lin.weight
        if two_four:
            groups = w.abs().reshape(-1, 4)
            idx = groups.topk(2, dim=1, largest=False).indices
            mask = torch.ones_like(groups, dtype=torch.bool).scatter_(1, idx, False)
            w.mul_(mask.reshape(w.shape))
        else:
            k = int(fraction * w.numel())
            if k > 0:
                threshold = w.abs().flatten().float().kthvalue(k).values.to(w.dtype)
                w.mul_(w.abs() > threshold)


def drop_layers(model, n):
    # Keep the first blocks and the last block; remove the n blocks before the last one
    layers = model.model.layers
    keep = list(range(len(layers) - 1 - n)) + [len(layers) - 1]
    model.model.layers = torch.nn.ModuleList([layers[i] for i in keep])
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i
    model.config.num_hidden_layers = len(model.model.layers)
