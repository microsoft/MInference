# Copyright (c) 2024 Microsoft
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
from datasets import load_dataset
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    GreedySearch,
    GreedySearch_InfLLM,
    GreedySearch_Mamba2,
    GreedySearch_RetrAttn,
    GreedySearch_RetrAttn_Legacy,
    GreedySearch_vLLM,
    check_benchmark_availability,
    create_multiturn_prompt,
    create_scdq_prompt,
    dump_jsonl,
    get_compressed_examples,
    get_ground_truth,
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
    MambaForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.import_utils import _is_package_available

if _is_package_available("vllm"):
    from vllm import LLM, SamplingParams
if _is_package_available("lmcache_vllm"):
    from lmcache_vllm.vllm import LLM as LMCacheLLM
    import lmcache_vllm

import random

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
    eg,
    data_name,
    max_new_tokens,
    max_input_length: int,
    attn_type: str = "vllm",
    tok=None,
    use_chat_template=False,
    scdq_mode=False,
    disable_golden_context=False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    if scdq_mode:
        encoded_eg = create_scdq_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=("vllm" in attn_type),
        )
    else:
        # multi-turn mode
        encoded_eg = create_multiturn_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=("vllm" in attn_type),
            disable_golden_context=disable_golden_context,
        )
    context = truncate_by_tokens(
        encoded_eg["prompts"][0], model.tokenizer, max_input_length
    )
    encoded_eg["prompts"][0] = context
    if scdq_mode:
        # scdq mode has no action for disable_golden_context
        outputs = model.test_scdq(encoded_eg, max_length=max_new_tokens)
    else:
        # multi-turn mode test
        outputs = model.test(
            encoded_eg,
            max_length=max_new_tokens,
            disable_golden_context=disable_golden_context,
        )

    print("Chunked generation:", json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


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
    hyper_param: dict = None,
):
    if model_name == "THUDM/glm-4-9b-chat-1m":
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, revision="refs/pr/19"
        )
    else:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    # tok.pad_token = tok.eos_token

    if attn_type == "vllm_blend":
        llm = LMCacheLLM(
            model=model_name,
            enable_prefix_caching=True,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=0.5,
            swap_space=64,
        )
        llm = GreedySearch_vLLM(llm, tok)
    elif attn_type == "vllm_kv":
        llm = LLM(
            model=model_name,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=True,
            swap_space=64,
            enforce_eager=True,
            enable_kvcompress=True,
            block_size=16,
            kv_head_bias_path=None,
            kv_head_bias_weight=0,
            disable_log_stats=True,
            prefill_metric_collection_window_size=32,
            prefill_metric_collection_block_size=4096,
            max_kv_per_compression=50_000_000,
            metric_aggregation="L2-sum",
            maxpool_metrics=True,
        )
        llm = GreedySearch_vLLM(
            llm,
            tok,
            is_kv_compress=True,
        )
    elif "vllm" in attn_type:
        # num_gpus
        llm = LLM(
            model=model_name,
            enable_prefix_caching="Jamba" not in model_name,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=trust_remote_code,
            swap_space=64,
        )
        if attn_type != "vllm":
            minference_patch = MInference(
                attn_type,
                model_name,
                config_path=topk_dims_file_path,
                starting_layer=starting_layer,
                attn_kwargs=hyper_param,
            )
            llm = minference_patch(llm)
        llm = GreedySearch_vLLM(llm, tok)
    else:
        minference_patch = MInference(
            attn_type.replace("_sink", ""),
            model_name,
            config_path=topk_dims_file_path,
            starting_layer=starting_layer,
            kv_type=kv_type,
            is_search=is_search,
            kv_cache_cpu=kv_cache_cpu,
            kv_cache_cpu_device=kv_cache_cpu_device,
            attn_kwargs=hyper_param,
        )
        if "mamba" in model_name.lower() or "recurrentgemma" in model_name.lower():
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                resume_download=None,
                trust_remote_code=trust_remote_code,
            )
            llm = GreedySearch_Mamba2(llm, tok)

            return llm, tok
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=trust_remote_code,
                attn_implementation="flash_attention_2",
            )
            llm = minference_patch(llm)

        if attn_type == "inf_llm":
            llm = GreedySearch_InfLLM(llm.model, tok)
            return llm, tok
        elif kv_type in ["retr_attn", "kivi"]:
            llm = GreedySearch_RetrAttn(
                llm,
                tok,
            )
            return llm, tok

        llm = GreedySearch(
            llm,
            tok,
        )

    print("Model and tokenizer loaded.")
    return llm, tok


if __name__ == "__main__":
    args = parse_args()

    # check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task
    scdq_mode = args.same_context_different_query

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    if max_seq_length == -1:
        max_seq_length = 160_000

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
        hyper_param=args.hyper_param.copy(),
    )

    disable_golden_context = (
        "_disable_golden_context" if args.disable_golden_context else ""
    )
    verbalize_hyper_param = (
        f"_{'-'.join([f'{k}={v}' for k, v in args.hyper_param.items() if k != 'best_pattern'])}"
        if args.hyper_param
        else ""
    )
    result_dir = Path(
        args.output_dir,
        f"{real_model_name}_{args.attn_type}{disable_golden_context}_{args.kv_type}{verbalize_hyper_param}",
    )
    result_dir.mkdir(exist_ok=True, parents=True)
    use_scdq = "_scdq" if scdq_mode else "_multi_turn"
    use_llmlingua = "_lingua" if args.use_llmlingua else ""
    real_model_name = f"{real_model_name}_{args.attn_type}{use_scdq}{disable_golden_context}_{args.kv_type}{verbalize_hyper_param}"  # add all the args to the real_model_name, for easy identification

    results = {}
    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if isinstance(max_new_tokens, dict):
            assert (
                max(max_new_tokens.values()) <= max_seq_length
            ), "max_new_tokens must be less than max_seq_length"
        elif max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        # Data
        output_path = (
            result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.jsonl"
        )
        examples = load_dataset("microsoft/SCBench", data_name, split="test")

        if args.use_llmlingua:
            # do prompt compression here
            compression_ratio = (
                args.hyper_param.get("llmlingua_ratio", 3) if args.hyper_param else 3
            )
            examples = get_compressed_examples(
                examples, data_name, args.data_dir, rate=1 / compression_ratio
            )
        max_turn_size = len(examples[0]["multi_turns"])
        if args.max_turns > 0 and args.max_turns < max_turn_size:
            examples = [
                {**eg, "multi_turns": eg["multi_turns"][: args.max_turns]}
                for eg in examples
            ]
            max_turn_size = args.max_turns

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print(f"==== Evaluation {data_name}====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Num of turns: {max_turn_size}")

        done = set()
        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    tmp = json.loads(line)
                    done.add(int(tmp["id"]))
                    preds.append(tmp)
            # examples = examples[len(preds):]
            compute_scores(
                output_path, data_name, real_model_name, max_seq_length, scdq_mode
            )

        for i, eg in tqdm(enumerate(examples)):
            if i < args.start_example_id or i in done:
                continue
            if data_name in [
                "scbench_summary_with_needles",
                "scbench_repoqa_and_kv",
            ]:
                max_input_length = max_seq_length - (
                    sum(list(max_new_tokens.values())) * max_turn_size // 2
                )
            else:
                max_input_length = max_seq_length - max_new_tokens * max_turn_size
            if scdq_mode:
                max_input_length -= 1000

            pred = get_pred(
                model,
                eg,
                data_name,
                max_new_tokens,
                max_input_length=max_input_length,
                attn_type=args.attn_type,
                tok=tok,
                use_chat_template=args.use_chat_template,
                scdq_mode=scdq_mode,
                disable_golden_context=args.disable_golden_context,
            )
            # a list of ground truth answers for each turn
            gts = get_ground_truth(eg, data_name)
            for turn_idx, (ans, gt, turn) in enumerate(
                zip(pred["answers"], gts, eg["multi_turns"])
            ):
                case = {
                    "id": i,
                    "turn_idx": turn_idx,
                    "prediction": ans,
                    "ground_truth": gt,
                }
                if "task" in pred:
                    case["task"] = pred["task"][turn_idx]
                if data_name == "scbench_repoqa":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    case["func_name"] = turn["name"]
                if data_name == "scbench_repoqa_and_kv":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    if turn["task"] == "scbench_repoqa":
                        case["func_name"] = turn["name"]
                if data_name == "scbench_kv_compressible":
                    case["task"] = eg["task"]
                preds.append(case)
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()
            done.add(i)

        score = compute_scores(
            output_path,
            data_name,
            real_model_name,
            max_seq_length=max_seq_length,
            scdq_mode=scdq_mode,
        )
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))
    try:
        lmcache_vllm.close_lmcache_engine()
    except:
        pass
