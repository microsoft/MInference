# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from minference import MInference


def run_target_length(m: int, model, attn_type: str):
    # wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
    prompt_complex = open("./prompt_hardest.txt").read()
    input_ids = tokenizer(prompt_complex)["input_ids"]
    n = len(input_ids)
    b = m // n + 1

    new_input_ids = (input_ids * b)[:m]
    prompt = tokenizer.decode(new_input_ids)
    data = tokenizer(prompt, return_tensors="pt")
    input_ids = data["input_ids"].cuda()
    attention_mask = data["attention_mask"].cuda()
    s = 0
    T = 10
    for _ in range(T):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            if attn_type != "inf_llm":
                model(
                    input_ids,
                    attention_mask,
                    use_cache=False,
                    num_logits_to_keep=1,
                )
            else:
                model.generate(
                    input_ids, generation_config=GenerationConfig(max_new_tokens=1)
                )
        torch.cuda.synchronize()
        s += time.time() - start
    print(attn_type, m, s / T)
    return s / T


def run_benchmark(model_name: str):
    TARGET_LENS = [l * 1000 for l in [10, 50, 100, 200, 300, 500, 1000]]
    ATTN_TYPES = ["dense", "a_shape", "minference"]
    ATTN_TYPES2NAME = {
        "dense": "FlashAttention-2",
        "a_shape": "A-Shape",
        "inf_llm": "InfLLM",
        "minference": "MInference",
    }
    latency = defaultdict(list)

    for attn_type in ATTN_TYPES:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            _attn_implementation="flash_attention_2",
        )
        attn_kwargs = {} if args.attn_type != "inf_llm" else {"dense_decoding": False}
        if attn_type != "hf":
            minference_patch = MInference(
                attn_type, model_name, attn_kwargs=attn_kwargs
            )
            model = minference_patch(model)
        for l in TARGET_LENS:
            if l >= 700000 and attn_type not in ["inf_llm", "hf"]:
                minference_patch = MInference(attn_type, model_name, kv_cache_cpu=True)
                model = minference_patch(model)

            t = run_target_length(l, model, attn_type)
            latency[ATTN_TYPES2NAME[attn_type]].append([l, f"{t:.5f}"])
            print(attn_type, t, l)
            torch.cuda.empty_cache()

    res = [[""] + [ATTN_TYPES2NAME[attn_type] for attn_type in ATTN_TYPES]]
    for idx in range(len(TARGET_LENS)):
        l = TARGET_LENS[idx]
        res.append(
            [f"{l//1000}K"]
            + [latency[ATTN_TYPES2NAME[attn_type]][idx][-1] for attn_type in ATTN_TYPES]
        )
    print("\n".join(["\t".join(ii) for ii in res]))
    with open("res.csv", "w") as f:
        f.write("\n".join(["\t".join(ii) for ii in res]))
    return res


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_name",
        type=str,
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        # default="Qwen/Qwen2.5-7B-Instruct",
    )
    args.add_argument(
        "--attn_type",
        type=str,
        choices=[
            "hf",
            "a_shape",
            "minference",
            "dense",
            "inf_llm",
        ],
    )
    args.add_argument("--context_window", type=int, default=100_000)
    args.add_argument("--run_benchmark", action="store_true")
    args.add_argument("--kv_cache_cpu", action="store_true")
    args.add_argument("--trust_remote_code", action="store_true")
    args = args.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=args.trust_remote_code
    )
    if args.run_benchmark:
        run_benchmark(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            _attn_implementation="flash_attention_2",
        )
        attn_kwargs = {} if args.attn_type != "inf_llm" else {"dense_decoding": False}
        if args.attn_type != "hf":
            minference_patch = MInference(
                args.attn_type,
                model_name,
                kv_cache_cpu=args.kv_cache_cpu,
                attn_kwargs=attn_kwargs,
            )
            model = minference_patch.patch_model(model)

        run_target_length(args.context_window, model, args.attn_type)
