# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import importlib
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from tqdm import tqdm

SERVER_TYPES = (
    "trtllm",
    "vllm",
    "openai",
    "gemini",
    "hf",
    "minference",
    "streaming",
    "InfLLM",
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument(
    "--data_dir", type=Path, required=True, help="path to load the dataset jsonl files"
)
parser.add_argument(
    "--save_dir",
    type=Path,
    required=True,
    help="path to save the prediction jsonl files",
)
parser.add_argument(
    "--benchmark", type=str, default="synthetic", help="Options: [synthetic]"
)
parser.add_argument(
    "--task", type=str, required=True, help="Options: tasks in benchmark"
)
parser.add_argument(
    "--subset", type=str, default="validation", help="Options: validation or test"
)
parser.add_argument(
    "--chunk_idx", type=int, default=0, help="index of current split chunk"
)
parser.add_argument("--chunk_amount", type=int, default=1, help="size of split chunk")

# Server
parser.add_argument(
    "--server_type", default="nemo", action=ServerAction, choices=SERVER_TYPES
)
parser.add_argument("--server_host", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=str, default="5000")
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="gpt-3.5-turbo",
    help="supported models from OpenAI or HF (provide a key or a local path to the checkpoint)",
)

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default="")
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)

# MInference
parser.add_argument("--config_path", type=str)
parser.add_argument("--starting_layer", type=int, default=-1)
parser.add_argument("--kv_cache_cpu", action="store_true")
parser.add_argument("--kv_cache_cpu_device", type=str, default="cpu")
parser.add_argument("--use_snapkv", action="store_true")
parser.add_argument("--trust_remote_code", action="store_true")

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(",")))
# if args.server_type == 'hf' or args.server_type == 'gemini' or args.server_type == 'minference':
if args.server_type in [
    "hf",
    "gemini",
    "minference",
    "streaming",
    "InfLLM",
]:
    args.threads = 1


def get_llm(tokens_to_generate):
    if args.server_type == "trtllm":
        from client_wrappers import TRTLLMClient

        llm = TRTLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
            max_attention_window_size=args.sliding_window_size,
        )

    elif args.server_type == "vllm":
        from client_wrappers import VLLMClient

        llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == "openai":
        from client_wrappers import OpenAIClient

        llm = OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == "gemini":
        from client_wrappers import GeminiClient

        llm = GeminiClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == "hf":
        from model_wrappers import HuggingFaceModel

        llm = HuggingFaceModel(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )

    elif args.server_type == "minference":
        from model_wrappers import MInferenceModel

        llm = MInferenceModel(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
            config_path=args.config_path,
            kv_cache_cpu=args.kv_cache_cpu,
            kv_cache_cpu_device=args.kv_cache_cpu_device,
            use_snapkv=args.use_snapkv,
            trust_remote_code=args.trust_remote_code,
            starting_layer=args.starting_layer,
        )

    elif args.server_type == "InfLLM":
        from model_wrappers import InfLLM

        llm = InfLLM(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )

    elif args.server_type == "streaming":
        from model_wrappers import Streaming

        llm = Streaming(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
            config_path=args.config_path,
            kv_cache_cpu=args.kv_cache_cpu,
            kv_cache_cpu_device=args.kv_cache_cpu_device,
            use_snapkv=args.use_snapkv,
            trust_remote_code=args.trust_remote_code,
            starting_layer=args.starting_layer,
        )

    else:
        raise RuntimeError(f"Unsupported server type {args.server_type}")

    return llm


def main():
    start_time = time.time()

    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config["task"]])

    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"

    if args.chunk_amount > 1:
        pred_file = args.save_dir / f"{args.task}-{args.chunk_idx}.jsonl"
    else:
        pred_file = args.save_dir / f"{args.task}.jsonl"

    print(f"Predict {args.task} \nfrom {task_file}\nto {pred_file}")
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample["index"] for sample in read_manifest(pred_file)]
        data = [
            sample
            for sample in read_manifest(task_file)
            if sample["index"] not in pred_index
        ]
    else:
        data = read_manifest(task_file)

    # Load api
    llm = get_llm(config["tokens_to_generate"])

    def get_output(idx, index, input, outputs, others, truncation, length):
        while True:
            try:
                pred = llm(prompt=input)
                break
            except Exception as e:
                traceback.print_exc()

        if len(pred["text"]) > 0:
            outputs_parallel[idx] = {
                "index": index,
                "pred": pred["text"][0],
                "input": input,
                "outputs": outputs,
                "others": others,
                "truncation": truncation,
                "length": length,
            }

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]
    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, "at", encoding="utf-8", buffering=1) as fout:
        for idx, data_point in tqdm(enumerate(data), total=len(data)):
            get_output(
                idx,
                data_point["index"],
                data_point["input"],
                data_point["outputs"],
                data_point.get("others", {}),
                data_point.get("truncation", -1),
                data_point.get("length", -1),
            )

            fout.write(json.dumps(outputs_parallel[idx]) + "\n")

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    main()
