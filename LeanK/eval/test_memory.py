# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minference import MInference


def main(model_name):
    n_times = 6
    n_warmup = 1

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test HF Baseline
    for bs in [52]:
        for t in range(n_times):
            seq_len = 4096
            dur_list = []
            mem_list = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            input_ids = torch.randint(
                0,
                model.vocab_size,
                (
                    bs,
                    seq_len,
                ),
            ).to(model.device)
            attention_mask = torch.randint(0, 2, (bs, seq_len)).to(model.device)
            start_time = time.time()

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=127,
                    do_sample=False,
                    return_legacy_cache=True,
                )
                torch.cuda.synchronize()

            end_time = time.time()
            print("run", t, "takes", (end_time - start_time), "seconds")

            peak_memory = torch.cuda.max_memory_allocated()
            if t > n_warmup:
                dur_list.append((end_time - start_time))
                mem_list.append(peak_memory / 1024 / 1024 / 1024)
        print(
            "------------------------------------------------------------------------------------------------------------"
        )
        print(
            "HF       seq_len: {:<20} batch_size: {:<20} time: {:<10.2f} memory: {:<10.2f} ".format(
                seq_len, bs, np.mean(dur_list), np.mean(mem_list)
            )
        )
        print(
            "------------------------------------------------------------------------------------------------------------"
        )

    minference_patch = MInference(
        attn_type="dense",
        model_name=model_name,
        kv_type="leank",
        attn_kwargs={"recent_size": 250, "sink_size": 6, "accumu_size": 128},
    )
    model = minference_patch(model)

    # test LeanK
    for bs in [52, 64]:
        for t in range(n_times):
            seq_len = 4096
            dur_list = []
            mem_list = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            input_ids = torch.randint(
                0,
                model.vocab_size,
                (
                    bs,
                    seq_len,
                ),
            ).to(model.device)
            attention_mask = torch.randint(0, 2, (bs, seq_len)).to(model.device)

            start_time = time.time()

            with torch.no_grad():
                model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=127,
                    do_sample=False,
                    temperature=1.0,
                )
                torch.cuda.synchronize()

            end_time = time.time()
            print("run", t, "takes", (end_time - start_time), "seconds")

            peak_memory = torch.cuda.max_memory_allocated()
            if t > n_warmup:
                dur_list.append((end_time - start_time))
                mem_list.append(peak_memory / 1024 / 1024 / 1024)
            torch.cuda.empty_cache()
        print(
            "------------------------------------------------------------------------------------------------------------"
        )
        print(
            "LeanK    seq_len: {:<20} batch_size: {:<20} time: {:<10.2f} memory: {:<10.2f} ".format(
                seq_len, bs, np.mean(dur_list), np.mean(mem_list)
            )
        )
        print(
            "------------------------------------------------------------------------------------------------------------"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="model path",
    )
    args = parser.parse_args()
    main(args.model_name)
