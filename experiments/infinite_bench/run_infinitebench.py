# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from args import parse_args
from compute_scores import compute_scores
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from vllm import LLM, SamplingParams

from minference import MInference


# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_input_length: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
    attn_type: str = "vllm",
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens = truncate_by_tokens(input_text, tok, max_input_length)
    if verbose:
        print("# tokens:", len(input_tokens))
        print("=============== Input ===============")
        print(tok.decode(input_tokens[:200]))
        print("...")
        print(tok.decode(input_tokens[-200:]))
        print("=====================================")
    if "vllm" in attn_type:
        if len(input_tokens) != 1:
            input_tokens = [input_tokens]
        outputs = model.generate(
            prompt_token_ids=input_tokens,
            sampling_params=generation_config,
        )
        output = outputs[0].outputs[0].text
        output = output.strip()
    else:
        input_tensors = {
            "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
        }
        # cache = SinkCache(window_length=200000, num_sink_tokens=10000)
        # if attn_type == "minference_kv_cache_cpu":
        #     input_tensors["use_cache"] = False
        outputs = model.generate(**input_tensors, generation_config=generation_config)
        # outputs = model.generate(**input_tensors, generation_config=generation_config, past_key_values=cache)

        output = outputs[0, len(input_tokens) :]
        output = tok.decode(output, skip_special_tokens=True)
        output = output.strip()
    # print(input_text[:5000], input_text[-5000:])
    print("Chunked generation:", output)
    return output


def load_model(
    model_name: str,
    topk: int = -1,
    starting_layer: int = -1,
    topk_dims_file_path: str = "",
    use_sparq: bool = False,
    attn_type: str = "vllm",
    max_seq_length: int = None,
    is_search: bool = False,
    kv_type: str = "",
    trust_remote_code: bool = False,
    kv_cache_cpu: bool = False,
    kv_cache_cpu_device: str = "cpu",
    tensor_parallel_size: int = 1,
):
    tok = AutoTokenizer.from_pretrained(
        model_name, resume_download=None, trust_remote_code=trust_remote_code
    )
    tok.pad_token = tok.eos_token

    minference_patch = MInference(
        attn_type,
        model_name,
        config_path=topk_dims_file_path,
        starting_layer=starting_layer,
        kv_type=kv_type,
        is_search=is_search,
        kv_cache_cpu=kv_cache_cpu,
        kv_cache_cpu_device=kv_cache_cpu_device,
    )

    if "vllm" in attn_type:
        llm = LLM(
            model=model_name,
            max_model_len=max_seq_length,
            enable_chunked_prefill=False,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            swap_space=64,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_name, resume_download=None, trust_remote_code=trust_remote_code
        )
        if "LWM" in model_name:
            c = {
                "theta": 10000000,
                "max_sequence_length": 131072,
                "scan_attention": True,
                "scan_query_chunk_size": 1024,
                "scan_key_chunk_size": 1024,
                "scan_mlp": True,
                "scan_mlp_chunk_size": 1024,
                "scan_layers": True,
            }
            config.update(c)

        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype="auto",
            device_map="auto",
            resume_download=None,
            trust_remote_code=trust_remote_code,
            _attn_implementation="flash_attention_2",
        )

    if attn_type not in ["vllm", "hf"]:
        llm = minference_patch(llm)

    print("Model and tokenizer loaded.")
    return llm, tok


if __name__ == "__main__":
    args = parse_args()

    check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    # Model
    model, tok = load_model(
        model_name,
        args.topk,
        args.starting_layer,
        args.topk_dims_file_path,
        args.use_sparq,
        attn_type=args.attn_type,
        max_seq_length=max_seq_length,
        is_search=args.is_search,
        kv_type=args.kv_type,
        trust_remote_code=args.trust_remote_code,
        kv_cache_cpu=args.kv_cache_cpu,
        kv_cache_cpu_device=args.kv_cache_cpu_device,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    results = {}

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        if "vllm" in args.attn_type:
            generation_config = SamplingParams(
                temperature=0,
                max_tokens=max_new_tokens,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                # temperature=0,
                # top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )

        # Data
        result_dir = Path(args.output_dir, f"{real_model_name}_{args.attn_type}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}.jsonl"
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")

        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            compute_scores(output_path, data_name, real_model_name, max_seq_length)
            with open(output_path) as f:
                preds = [json.loads(ii) for ii in f.readlines()]

        for i, eg in tqdm(enumerate(examples)):
            if i < args.start_example_id or i < len(preds):
                continue
            input_text = create_prompt(eg, data_name, real_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)
            # print(input_text.index(ground_truth), len(input_text), input_text.index(ground_truth) / len(input_text))
            # print(f"====== Example {i} ======")

            msgs = [dict(role="system", content=input_text)]
            input_text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            pred = get_pred(
                model,
                tok,
                input_text,
                max_input_length=max_seq_length - max_new_tokens,
                verbose=args.verbose,
                generation_config=generation_config,
                attn_type=args.attn_type,
            )
            print("Ground Truth", get_answer(eg, data_name))
            if args.verbose:
                print(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()

        result_file_path = f"{real_model_name}_{args.attn_type}"
        score = compute_scores(output_path, data_name, result_file_path)
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))
