import argparse
import sys
import time

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)

from minference import MInference


def run_target_length(m: int, model):
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
            model(input_ids, attention_mask, use_cache=False)
            # model.generate(input_ids, generation_config=GenerationConfig(max_new_tokens=1))
        torch.cuda.synchronize()
        s += time.time() - start
    print(m, s / T)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_name",
        type=str,
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    )
    args.add_argument(
        "--attn_type",
        type=str,
        required=True,
        choices=[
            "hf",
            "streaming",
            "minference",
            "minference_wo_cache",
            "inf_llm",
        ],
    )
    args.add_argument("--context_window", type=int, default=100000)
    args = args.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    minference_patch = MInference(args.attn_type, model_name, None, starting_layer=0)

    model = minference_patch.patch_model(model)
    run_target_length(args.context_window, model)
