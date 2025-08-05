# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import copy
import gc
import random
import time
import uuid

import fire
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from minference import MInference

kv_retrieval_prompt_template = (
    """
<s> Extract the value corresponding to the specified key in the data below.

Data:
{formatted_kv_records}

Extract the value corresponding to this key:
key: {key}
corresponding value:
""".strip()
    + " "
)

kv_retrieval_prompt_template_llama2_chat = (
    """
<s> [INST] Extract the value corresponding to the specified key in the data below.

Data:
{formatted_kv_records}

Extract the value corresponding to this key:
key: {key}

Please directly output the corresponding value without outputing anything else. [/INST]  Sure! The value corresponding to the key "{key}" is:
""".strip()
    + "\n\nvalue: "
)


kv_retrieval_prompt_template_llama3_instruct = (
    """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Extract the value corresponding to the specified key in the data below.

Data:
{formatted_kv_records}

Extract the value corresponding to this key:
key: {key}

Please directly output the corresponding value without outputing anything else.
value:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip()
    + "\n\n"
)

kv_retrieval_prompt_template_qwen_instruct = (
    """
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Extract the value corresponding to the specified key in the data below.

Data:
{formatted_kv_records}

Extract the value corresponding to this key:
key: {key}

Please directly output the corresponding value without outputing anything else.<|im_end|>
<|im_start|>nuser
""".strip()
    + "\n"
)


def get_kv_retrieval_prompt(
    data,
    key,
    model_name,
):
    # Format the KV data into a string
    formatted_kv_records = ""
    for index, record in enumerate(data):
        data_string = f"key: {record[0]} value: {record[1]}\n"
        formatted_kv_records += data_string

    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        prompt_template = kv_retrieval_prompt_template_llama2_chat
    elif (
        model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
        or model_name == "gradientai/Llama-3-8B-Instruct-262k"
        or model_name == "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
        or model_name == "meta-llama/Llama-3.1-8B-Instruct"
    ):
        prompt_template = kv_retrieval_prompt_template_llama3_instruct
    elif (
        model_name == "meta-llama/Llama-2-7b-hf"
        or model_name == "meta-llama/Meta-Llama-3-8B"
    ):
        prompt_template = kv_retrieval_prompt_template
    elif model_name == "Qwen/Qwen2.5-7B-Instruct":
        prompt_template = kv_retrieval_prompt_template_qwen_instruct

    return prompt_template.format(formatted_kv_records=formatted_kv_records, key=key)


def quick_get_random_kv_samples(
    model_name, tokenizer, gold_index, n_kv_num=10, n_sample=100
):
    samples = []
    sample_counter = 0
    # create n_kv_num key-value pairs
    for idx in range(n_sample):
        ordered_kv_records = [
            [str(uuid.uuid4()), str(uuid.uuid4())] for _ in range(n_kv_num)
        ]
        key = str(uuid.uuid4())
        value = str(uuid.uuid4())
        ordered_kv_records.insert(gold_index, [key, value])
        kv_prompt = get_kv_retrieval_prompt(
            data=ordered_kv_records,
            key=key,
            model_name=model_name,
        )
        input_ids = tokenizer(kv_prompt, add_special_tokens=False)["input_ids"]
        samples.append(
            {
                "input_ids": input_ids,
                "key": key,
                "value": value,
            }
        )
    return samples


def main(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    method="dense",
    starting_layer=None,
    gamma=None,
):
    seq_len_list = [
        32000,
        48000,
        64000,
        80000,
        96000,
        112000,
        128000,
    ]
    n_times = 3
    if starting_layer is None:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":
            starting_layer = 16
        elif model_name == "gradientai/Llama-3-8B-Instruct-262k":
            starting_layer = 16
        elif model_name == "Qwen/Qwen2.5-7B-Instruct":
            starting_layer = 20
        else:
            raise NotImplementedError
    if gamma is None:
        gamma = 0.95
    if method == "dense":
        kwargs = dict(attn_type="dense")
    elif method == "tri_mix":
        kwargs = dict(
            attn_type="tri_mix",
            attn_kwargs={
                "last_n": 128,
                "starting_layer": starting_layer,
                "n_local": 512,
                "n_init": 8,
            },
        )
    elif method == "flexprefill":
        kwargs = dict(
            attn_type="flexprefill",
            attn_kwargs={"gamma": gamma},
        )
    elif method == "minference":
        kwargs = dict(
            attn_type="minference",
        )
    elif method == "tri_mix_minference":
        kwargs = dict(
            attn_type="tri_mix_minference",
            attn_kwargs={
                "last_n": 128,
                "starting_layer": starting_layer,
                "n_local": 512,
                "n_init": 8,
            },
        )
    else:
        raise NotImplementedError

    model_name_to_saving_name = {
        "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
        "gradientai/Llama-3-8B-Instruct-262k": "Llama-3-8B-Instruct-262k",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    minference_patch = MInference(
        model_name=model_name,
        config_path=None,
        starting_layer=-1,
        kv_type="dense",
        is_search=False,
        kv_cache_cpu=False,
        kv_cache_cpu_device="cpu",
        **kwargs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    model = minference_patch(model)
    samples = quick_get_random_kv_samples(
        model_name, tokenizer, 3000, n_kv_num=6000, n_sample=n_times
    )

    # warmup
    for seq_len in seq_len_list:
        input_ids = samples[0]["input_ids"][:seq_len]
        input_ids = torch.tensor([input_ids], device=model.device)
        with torch.no_grad():
            model(input_ids, use_cache=False)
        torch.cuda.empty_cache()

    # start test
    ret_list = []
    print("---------------------------")
    for seq_len in seq_len_list:
        dur_list = []
        for i in range(n_times):
            sample = samples[i]
            input_ids = sample["input_ids"]
            # print(len(input_ids))
            assert len(input_ids) >= seq_len
            input_ids = input_ids[:seq_len]
            input_ids = torch.tensor([input_ids], device=model.device)

            with torch.no_grad():
                torch.cuda.synchronize(device=model.device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                model(input_ids, use_cache=False)
                torch.cuda.synchronize(device=model.device)
                end_event.record()
                torch.cuda.synchronize(device=model.device)
                elapsed_time_ms = start_event.elapsed_time(end_event)
            torch.cuda.empty_cache()
            dur_list.append((elapsed_time_ms) / 1000.0)
        print("seq_len: {:<20} time: {:.2f}s".format(seq_len, np.mean(dur_list)))
        print("---------------------------")
        ret_list.append(
            {
                "model": model_name_to_saving_name[model_name],
                "method": method,
                "seq_len": seq_len,
                "time": np.mean(dur_list),
            }
        )

    if gamma != 0.95:
        method = "{}_{:.2f}".format(method, gamma)
    pd.DataFrame(ret_list).to_csv(
        f"speed_test_{model_name_to_saving_name[model_name]}_result_{method}.csv",
        index=False,
    )


if __name__ == "__main__":
    fire.Fire(main)
