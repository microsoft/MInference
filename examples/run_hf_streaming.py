import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import os
import re
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from minference import MInference


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    torch.cuda.synchronize()
    st = time.time()
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    torch.cuda.synchronize()
    print(f"\033[1;31m TTFT: {time.time() - st:.2f}s\033[0m\n")

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        if seq_len > 5000:
            print(
                "\n" + "USER: " + prompt[:2500] + prompt[-2500:] + "\n\nASSISTANT: ",
                end="",
            )
        else:
            print("\n" + prompt, end="")
        print(seq_len)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )


def main(args):
    model_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="cuda",
    )

    # Patch MInference Module
    minference_patch = MInference(args.attn_type, model_name_or_path, kv_cache_cpu=True)
    model = minference_patch(model)

    prompts = [
        open("data/pg2600.txt").read()
        + "\nCould you help me summarization the previous book."
    ]

    kv_cache = None

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument(
        "--attn_type",
        type=str,
        required=True,
        choices=[
            "vllm",
            "hf",
            "streaming",
            "minference",
            "inf_llm",
        ],
    )
    args = parser.parse_args()

    main(args)
