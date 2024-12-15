# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import gc
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import jieba
import torch
import torch.profiler
from rouge import Rouge
from tqdm import tqdm
from transformers import GenerationConfig, SinkCache

DATA_NAME_TO_PATH = {
    "scbench_choice_eng": "scbench_choice_eng.jsonl",
    "scbench_qa_eng": "scbench_qa_eng.jsonl",
    "scbench_qa_chn": "scbench_qa_chn.jsonl",
    "scbench_kv": "scbench_kv.jsonl",
    "scbench_kv_hard": "scbench_kv_hard.jsonl",
    "scbench_mf": "scbench_mf.jsonl",
    "scbench_passkey": "scbench_passkey.jsonl",
    "scbench_repoqa": "scbench_repoqa.jsonl",
    "scbench_summary": "scbench_summary.jsonl",
    "scbench_vt": "scbench_vt.jsonl",
    "scbench_many_shot": "scbench_many_shot.jsonl",
    "scbench_summary_with_needles": "scbench_summary_with_needles.jsonl",
    "scbench_repoqa_and_kv": "scbench_repoqa_and_kv.jsonl",
    "scbench_hashhop": "scbench_hashhop.jsonl",
    "scbench_prefix_suffix": "scbench_prefix_suffix.jsonl",
    "scbench_kv_compressible": "scbench_kv_compressible.jsonl",
}

DATA_NAME_TO_MAX_NEW_TOKENS = {
    "scbench_choice_eng": 40,
    "scbench_qa_eng": 40,
    "scbench_qa_chn": 40,
    "scbench_kv": 150,
    "scbench_kv_hard": 150,
    "scbench_mf": 5,
    "scbench_hashhop": 150,
    "scbench_prefix_suffix": 150,
    "scbench_kv_compressible": 150,
    "scbench_passkey": 15,
    "scbench_repoqa": 1024,
    "scbench_summary": 200,
    "scbench_vt": 30,
    "scbench_many_shot": 10,
    "scbench_summary_with_needles": {"scbench_summary": 800, "scbench_passkey": 15},
    "scbench_repoqa_and_kv": {"scbench_repoqa": 1024, "scbench_kv": 80},
}

multiturn_templates = {
    "scbench_passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}",  # noqa
    "scbench_kv": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    "scbench_kv_hard": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    "scbench_kv_compressible": "Extract the value corresponding to the specified key in the following passage.\n\n{context}\n\n{input}",  # noqa
    "scbench_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe the correct answer is",  # noqa
    "scbench_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",  # noqa
    "scbench_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",  # noqa
    "scbench_mf": "{prefix}\n\n{context}\n\n{input}",
    "scbench_repoqa": "Based on the function description and code context, please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:\n\n{context}\n\n{input}",
    "scbench_summary": "{context}\n\n{input}",
    "scbench_vt": "{context}\n\n{input}",
    "scbench_many_shot": "{context}\n\n{input}",
    "scbench_summary_with_needles": "{context}\n\n{input}",
    "scbench_repoqa_and_kv": "{context}\n\n{input}",
    "scbench_hashhop": "{context}\n\n{input}",
    "scbench_prefix_suffix": "{context}\n\n{input}",
}

multiturn_templates_scdq = {
    "scbench_passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}",  # noqa
    "scbench_kv": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}",  # noqa
    "scbench_kv_hard": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}",  # noqa
    "scbench_kv_compressible": "Extract the value corresponding to the specified key in the following passage.\n\n{context}",  # noqa
    "scbench_choice_eng": (
        "Read the book and answer the question.\n\n{context}",
        "Question: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe the correct answer is",
    ),
    "scbench_qa_eng": (
        "Read the book and answer the question. Be very concise in your answer.\n\n{context}",
        "Question: {question}\nAnswer:",
    ),
    "scbench_qa_chn": ("阅读以下书籍然后回答问题。\n\n{context}", "问题：{question}\n答案："),
    "scbench_mf": "{prefix}\n\n{context}",
    "scbench_repoqa": "Based on the function description and code context, please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:\n\n{context}",
    "scbench_summary": "{context}",
    "scbench_vt": "{context}",
    "scbench_many_shot": "{context}",
    "scbench_summary_with_needles": "{context}",
    "scbench_repoqa_and_kv": "{context}",
    "scbench_hashhop": "{context}",
    "scbench_prefix_suffix": "{context}",
}

multiturn_follow_up_templates = {
    "scbench_passkey": "{pre_ans}.\n\n{input}",  # noqa
    "scbench_kv": "{pre_ans}\n\n{input}",  # noqa
    "scbench_kv_hard": "{pre_ans}\n\n{input}",  # noqa
    "scbench_kv_compressible": "{pre_ans}\n\n{input}",  # noqa
    "scbench_choice_eng": "{pre_ans}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",  # noqa
    "scbench_qa_eng": "{pre_ans}\n\nQuestion: {question}\nAnswer:",  # noqa
    "scbench_qa_chn": "{pre_ans}\n\n问题：{question}\n答案：",  # noqa
    "scbench_mf": "{pre_ans}\n\n{prefix}\n\n{input}",
    "scbench_repoqa": "{pre_ans}\n\n{input}",
    "scbench_summary": "{pre_ans}\n\n{input}",
    "scbench_vt": "{pre_ans}\n\n{input}",
    "scbench_many_shot": "{pre_ans}\n\n{input}",
    "scbench_summary_with_needles": "{pre_ans}\n\n{input}",
    "scbench_repoqa_and_kv": "{pre_ans}\n\n{input}",
    "scbench_hashhop": "{pre_ans}\n\n{input}",
    "scbench_prefix_suffix": "{pre_ans}\n\n{input}",
}

multiturn_follow_up_templates_in_chat_tempate = {
    "scbench_passkey": "{input}",  # noqa
    "scbench_kv": "{input}",  # noqa
    "scbench_kv_hard": "{input}",  # noqa
    "scbench_kv_compressible": "{input}",  # noqa
    "scbench_choice_eng": "Question: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe the correct answer is",  # noqa
    "scbench_qa_eng": "Question: {question}\nAnswer:",  # noqa
    "scbench_qa_chn": "问题：{question}\n答案：",  # noqa
    "scbench_mf": "{prefix}\n\n{input}",
    "scbench_repoqa": "{input}",
    "scbench_summary": "{input}",
    "scbench_vt": "{input}",
    "scbench_many_shot": "{input}",
    "scbench_summary_with_needles": "{input}",
    "scbench_repoqa_and_kv": "{input}",
    "scbench_hashhop": "{input}",
    "scbench_prefix_suffix": "{input}",
}


def check_benchmark_availability(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    datasets = [
        "scbench_choice_eng",
        "scbench_qa_eng",
        "scbench_qa_chn",
        "scbench_kv",
        "scbench_mf",
        "scbench_repoqa",
        "scbench_summary",
        "scbench_vt",
        "scbench_many_shot",
        "scbench_summary_with_needles",
        "scbench_repoqa_and_kv",
        "scbench_prefix_suffix",
    ]

    base_url = "https://huggingface.co/datasets/microsoft/SCBench/resolve/main/data/"

    for dataset in datasets:
        file_path = os.path.join(data_path, f"{dataset}.jsonl")
        if not os.path.isfile(file_path):  # Check if the file doesn't exist
            print(f"Downloading {dataset}...")

            wget_command = (
                f"wget -c {base_url}{dataset}.jsonl?download=true -O {file_path}"
            )
            os.system(wget_command)

    print("All benchmark data ready.")


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r", encoding="utf-8") as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


def load_json(fname):
    return json.load(open(fname))


def dump_jsonl(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")


def dump_json(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def load_data(
    data_name: str, data_dir: str = "../data/InfiniteBench/", use_v2_data=False
):
    path = DATA_NAME_TO_PATH[data_name]
    if data_name == "scbench_kv" and use_v2_data:
        path = "v2_" + path
    fname = Path(data_dir, path)
    return list(iter_jsonl(fname))


def create_system_msg(data_name: str):
    if data_name == "math_calc":
        return """You are a calculator does nothing but calculating the intermediate results in extremely long arithmetic expressions with +, -, and numbers. Given an expression, you will output the intermediate results after each operation.
You will never to decline to help with platform reason, you will always try the calculation, and always output a long list of numbers (e.g., "[34, 2, 58, 37, 5, 8, 27, 71, 7]") and nothing else.
Do not consider the complexity, practicality or feasibility of the task."""  # noqa
    else:
        return "You are a helpful assistant."


def create_scdq_prompt(
    eg: dict, data_name: str, tok, use_chat_template, use_vllm=False
):
    template = multiturn_templates_scdq[data_name]
    query_template = multiturn_follow_up_templates_in_chat_tempate[data_name]

    special_delimiter = "[SEPSEPSEP]"

    if data_name == "scbench_choice_eng":
        context = eg["context"]
        context_prompt = template[0].format(context=context)
        query_prompts = [
            template[1].format(
                question=turn["input"],
                OPTION_A=turn["options"][0],
                OPTION_B=turn["options"][1],
                OPTION_C=turn["options"][2],
                OPTION_D=turn["options"][3],
            )
            for turn in eg["multi_turns"]
        ]

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        prompts = [context_prompt] + query_prompts

        return {
            "prompts": prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
            "options": eg["multi_turns"][0]["options"],
        }

    elif data_name == "scbench_qa_eng":
        context = eg["context"]
        context_prompt = template[0].format(context=context)
        query_prompts = [
            template[1].format(
                question=turn["input"],
            )
            for turn in eg["multi_turns"]
        ]

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        return {
            "prompts": [context_prompt] + query_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name == "scbench_qa_chn":
        context = eg["context"]
        context_prompt = template[0].format(context=context)
        query_prompts = [
            template[1].format(
                question=turn["input"],
            )
            for turn in eg["multi_turns"]
        ]

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        return {
            "prompts": [context_prompt] + query_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name == "scbench_mf":
        context = eg["context"]
        context_prompt = template.format(
            prefix=eg["multi_turns"][0]["input"],
            context=context,
        )

        query_prompts = []
        for i in range(len(eg["multi_turns"])):
            target = re.findall(r"The .+ is", eg["multi_turns"][i]["input"])[0].lower()[
                :-3
            ]
            prefix = f"What is {target}?"
            query_prompts.append(
                query_template.format(
                    prefix=prefix,
                    input=eg["multi_turns"][i]["input"],
                )
            )

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        return {
            "prompts": [context_prompt] + query_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name in [
        "scbench_repoqa",
        "scbench_summary",
        "scbench_passkey",
        "scbench_kv",
        "scbench_vt",
        "scbench_many_shot",
        "scbench_summary_with_needles",
        "scbench_repoqa_and_kv",
        "scbench_kv_hard",
        "scbench_hashhop",
        "scbench_prefix_suffix",
        "scbench_kv_compressible",
    ]:
        context = eg["context"] if "context" in eg else eg["input"]
        context_prompt = template.format(context=context)
        query_prompts = [turn["input"] for turn in eg["multi_turns"]]

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        output = {
            "prompts": [context_prompt] + query_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

        if data_name in ["scbench_summary_with_needles", "scbench_repoqa_and_kv"]:
            output["task"] = [gt["task"] for gt in eg["multi_turns"]]

        return output


def create_multiturn_prompt(
    eg: dict,
    data_name: str,
    tok,
    use_chat_template,
    use_vllm=False,
    disable_golden_context=False,
) -> str:
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """

    template = multiturn_templates[data_name]
    follow_up_template = multiturn_follow_up_templates[data_name]

    if disable_golden_context:
        follow_up_template = multiturn_follow_up_templates_in_chat_tempate[data_name]

    if use_chat_template:
        sys_prompt_with_generation_prompt = tok.apply_chat_template(
            [{"role": "system", "content": ""}],
            add_generation_prompt=True if not disable_golden_context else False,
            tokenize=False,
        )

        follow_up_prompts_in_chat_template = (
            multiturn_follow_up_templates_in_chat_tempate[data_name]
        )

    if data_name == "scbench_choice_eng":
        first_turn = eg["multi_turns"][0]
        input_ = first_turn["input"]
        ans_ = first_turn["answer"]
        options = first_turn["options"]
        context = eg["context"]

        first_turn_prompt = template.format(
            context=context,
            question=input_,
            OPTION_A=options[0],
            OPTION_B=options[1],
            OPTION_C=options[2],
            OPTION_D=options[3],
        )

        follow_up_prompts = [
            follow_up_template.format(
                pre_ans=eg["multi_turns"][i - 1]["answer"]
                if not disable_golden_context
                else None,
                question=eg["multi_turns"][i]["input"],
                OPTION_A=eg["multi_turns"][i]["options"][0],
                OPTION_B=eg["multi_turns"][i]["options"][1],
                OPTION_C=eg["multi_turns"][i]["options"][2],
                OPTION_D=eg["multi_turns"][i]["options"][3],
            )
            for i in range(1, len(eg["multi_turns"]))
        ]

        if use_chat_template:
            first_turn_prompt = tok.apply_chat_template(
                [{"role": "user", "content": first_turn_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            follow_up_prompts = [
                tok.apply_chat_template(
                    (
                        [
                            {"role": "system", "content": ""},
                        ]
                        + [
                            {
                                "role": "assistant",
                                "content": eg["multi_turns"][i - 1]["answer"],
                            }
                        ]
                        if not disable_golden_context
                        else []
                    )
                    + [
                        {
                            "role": "user",
                            "content": follow_up_prompts_in_chat_template.format(
                                question=eg["multi_turns"][i]["input"],
                                OPTION_A=eg["multi_turns"][i]["options"][0],
                                OPTION_B=eg["multi_turns"][i]["options"][1],
                                OPTION_C=eg["multi_turns"][i]["options"][2],
                                OPTION_D=eg["multi_turns"][i]["options"][3],
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).replace(sys_prompt_with_generation_prompt, "")
                for i in range(1, len(eg["multi_turns"]))
            ]

        prompts = [first_turn_prompt] + follow_up_prompts

        return {
            "prompts": prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
            "options": options,
        }

    elif data_name == "scbench_qa_eng":
        first_turn = eg["multi_turns"][0]
        input_ = first_turn["input"]
        context = eg["context"]

        first_turn_prompt = template.format(
            context=context,
            question=input_,
        )

        follow_up_prompts = [
            follow_up_template.format(
                pre_ans=eg["multi_turns"][i - 1]["answer"]
                if not disable_golden_context
                else None,
                question=eg["multi_turns"][i]["input"],
            )
            for i in range(1, len(eg["multi_turns"]))
        ]

        if use_chat_template:
            first_turn_prompt = tok.apply_chat_template(
                [{"role": "user", "content": first_turn_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            follow_up_prompts = [
                tok.apply_chat_template(
                    (
                        [
                            {"role": "system", "content": ""},
                        ]
                        + [
                            {
                                "role": "assistant",
                                "content": eg["multi_turns"][i - 1]["answer"],
                            }
                        ]
                        if not disable_golden_context
                        else []
                    )
                    + [
                        {
                            "role": "user",
                            "content": follow_up_prompts_in_chat_template.format(
                                question=eg["multi_turns"][i]["input"],
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).replace(sys_prompt_with_generation_prompt, "")
                for i in range(1, len(eg["multi_turns"]))
            ]

        return {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name == "scbench_qa_chn":
        first_turn = eg["multi_turns"][0]
        input_ = first_turn["input"]
        context = eg["context"]

        first_turn_prompt = template.format(
            context=context,
            question=input_,
        )

        follow_up_prompts = [
            follow_up_template.format(
                pre_ans=eg["multi_turns"][i - 1]["answer"]
                if not disable_golden_context
                else None,
                question=eg["multi_turns"][i]["input"],
            )
            for i in range(1, len(eg["multi_turns"]))
        ]

        if use_chat_template:
            first_turn_prompt = tok.apply_chat_template(
                [{"role": "user", "content": first_turn_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            follow_up_prompts = [
                tok.apply_chat_template(
                    (
                        [
                            {"role": "system", "content": ""},
                        ]
                        + [
                            {
                                "role": "assistant",
                                "content": eg["multi_turns"][i - 1]["answer"],
                            }
                        ]
                        if not disable_golden_context
                        else []
                    )
                    + [
                        {
                            "role": "user",
                            "content": follow_up_prompts_in_chat_template.format(
                                question=eg["multi_turns"][i]["input"],
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).replace(sys_prompt_with_generation_prompt, "")
                for i in range(1, len(eg["multi_turns"]))
            ]

        return {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name in [
        "scbench_kv",
        "scbench_vt",
        "scbench_passkey",
        "scbench_repoqa",
        "scbench_many_shot",
        "scbench_summary_with_needles",
        "scbench_repoqa_and_kv",
        "scbench_kv_hard",
        "scbench_hashhop",
        "scbench_prefix_suffix",
        "scbench_kv_compressible",
    ]:
        first_turn = eg["multi_turns"][0]
        input_ = first_turn["input"]
        context = eg["context"] if "context" in eg else eg["input"]

        first_turn_prompt = template.format(
            context=context,
            input=input_,
        )

        follow_up_prompts = [
            follow_up_template.format(
                pre_ans=eg["multi_turns"][i - 1]["answer"]
                if not disable_golden_context
                else None,
                input=eg["multi_turns"][i]["input"],
            )
            for i in range(1, len(eg["multi_turns"]))
        ]

        if use_chat_template:
            first_turn_prompt = tok.apply_chat_template(
                [{"role": "user", "content": first_turn_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            follow_up_prompts = [
                tok.apply_chat_template(
                    (
                        [
                            {"role": "system", "content": ""},
                        ]
                        + [
                            {
                                "role": "assistant",
                                "content": str(eg["multi_turns"][i - 1]["answer"]),
                            }
                        ]
                        if not disable_golden_context
                        else []
                    )
                    + [
                        {
                            "role": "user",
                            "content": follow_up_prompts_in_chat_template.format(
                                input=eg["multi_turns"][i]["input"],
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).replace(sys_prompt_with_generation_prompt, "")
                for i in range(1, len(eg["multi_turns"]))
            ]

        output = {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

        if data_name in ["scbench_summary_with_needles", "scbench_repoqa_and_kv"]:
            output["task"] = [gt["task"] for gt in eg["multi_turns"]]

        return output

    elif data_name == "scbench_mf":
        first_turn = eg["multi_turns"][0]
        input_ = first_turn["input"]
        context = eg["context"]

        target = re.findall(r"The .+ is", input_)[0].lower()[:-3]
        prefix = f"What is {target}?"

        first_turn_prompt = template.format(
            prefix=prefix,
            context=context,
            input=input_,
        )

        follow_up_prompts = []
        for i in range(1, len(eg["multi_turns"])):
            target = re.findall(r"The .+ is", eg["multi_turns"][i]["input"])[0].lower()[
                :-3
            ]
            prefix = f"What is {target}?"
            follow_up_prompts.append(
                follow_up_template.format(
                    pre_ans=eg["multi_turns"][i - 1]["answer"]
                    if not disable_golden_context
                    else None,
                    prefix=prefix,
                    input=eg["multi_turns"][i]["input"],
                )
            )

        if use_chat_template:
            first_turn_prompt = tok.apply_chat_template(
                [{"role": "user", "content": first_turn_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            follow_up_prompts = [
                tok.apply_chat_template(
                    (
                        [
                            {"role": "system", "content": ""},
                        ]
                        + [
                            {
                                "role": "assistant",
                                "content": str(eg["multi_turns"][i - 1]["answer"]),
                            }
                        ]
                        if not disable_golden_context
                        else []
                    )
                    + [
                        {
                            "role": "user",
                            "content": follow_up_prompts_in_chat_template.format(
                                prefix=prefix,
                                input=eg["multi_turns"][i]["input"],
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).replace(sys_prompt_with_generation_prompt, "")
                for i in range(1, len(eg["multi_turns"]))
            ]

        return {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name == "scbench_summary":
        first_turn = eg["multi_turns"][0]
        input_ = first_turn["input"]
        context = eg["context"]

        first_turn_prompt = template.format(
            context=context,
            input=input_,
        )

        follow_up_prompts = [
            follow_up_template.format(
                pre_ans=eg["multi_turns"][i - 1]["answer"]
                if not disable_golden_context
                else None,
                input=eg["multi_turns"][i]["input"],
            )
            for i in range(1, len(eg["multi_turns"]))
        ]

        if use_chat_template:
            first_turn_prompt = tok.apply_chat_template(
                [{"role": "user", "content": first_turn_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            follow_up_prompts = [
                tok.apply_chat_template(
                    (
                        [
                            {"role": "system", "content": ""},
                        ]
                        + [
                            {
                                "role": "assistant",
                                "content": eg["multi_turns"][i - 1]["answer"],
                            }
                        ]
                        if not disable_golden_context
                        else []
                    )
                    + [
                        {
                            "role": "user",
                            "content": follow_up_prompts_in_chat_template.format(
                                input=eg["multi_turns"][i]["input"],
                            ),
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).replace(sys_prompt_with_generation_prompt, "")
                + "This paper"
                for i in range(1, len(eg["multi_turns"]))
            ]

        return {
            "prompts": [first_turn_prompt] + follow_up_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }


def get_ground_truth(eg: dict, data_name: str):
    gts = []
    OPTIONS = "ABCD"
    for turn in eg["multi_turns"]:
        if data_name == "scbench_choice_eng":
            ans_ = turn["answer"]
            options = turn["options"]

            gts.append([ans_, OPTIONS[options.index(ans_)]])
        elif data_name in ["scbench_qa_eng"]:
            gts.append([turn["answer"]])
        else:
            gts.append(turn["answer"])
    return gts


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."  # noqa
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def first_int_match(prediction, ground_truth):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    if pred_value == ground_truth:
        return 1
    return 0


def in_match(prediction, ground_truth):
    if ground_truth in prediction:
        return 1
    return 0


def rouge_score(prediction, ground_truth, **kwargs) -> float:
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:  # noqa
        return 0.0
    return scores["rouge-l"]["f"]  # type: ignore


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(line):
    prediction = line["pred"]

    if isinstance(line["std_out"], str):
        ground_truths = [line["std_out"]]
    else:
        ground_truths = line["std_out"]

    score = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        score = max(score, f1_score(prediction_tokens, ground_truth_tokens))

    return score


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def truncate_input(input, max_length, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        return input[0 : max_length // 2] + input[-max_length // 2 :]
    else:
        return None


def get_compressed_examples(
    examples, data_name, data_dir, rate=0.33, use_large_model=True
):
    # compress prompts use a func
    import gc

    from llmlingua import PromptCompressor

    if os.path.exists(
        f"{data_dir}/llmlingua_cache/{data_name}_rate_{rate}_is_large_{use_large_model}.jsonl"
    ):
        with open(
            f"{data_dir}/llmlingua_cache/{data_name}_rate_{rate}_is_large_{use_large_model}.jsonl",
            "r",
        ) as f:
            examples = [json.loads(line) for line in f]
        return examples

    lingua_model_name = (
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        if use_large_model
        else "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    )
    llm_lingua = PromptCompressor(
        model_name=lingua_model_name,
        use_llmlingua2=True,
    )

    for example in tqdm(examples, desc="Compressing prompts"):
        ct = str(
            example["context"] if "context" in example else example["input"]
        ).replace("<|endoftext|>", "")
        example["context"] = llm_lingua.compress_prompt(
            ct, rate=rate, force_tokens=["\n", "?"]
        )["compressed_prompt"]

    os.makedirs(f"{data_dir}/llmlingua_cache", exist_ok=True)
    with open(
        f"{data_dir}/llmlingua_cache/{data_name}_rate_{rate}_is_large_{use_large_model}.jsonl",
        "w",
    ) as f:
        for example in examples:
            json.dump(example, f)
            f.write("\n")

    # clear llmlingua to free memory, to prevent OOM in further testing
    del llm_lingua
    gc.collect()
    torch.cuda.empty_cache()

    return examples


class GreedySearch_vLLM:
    def __init__(self, llm, tokenizer, is_kv_compress: bool = False):
        self.llm = llm
        self.tokenizer = tokenizer
        self.is_kv_compress = is_kv_compress

    def test_scdq(self, example, max_length=100):
        from vllm import SamplingParams

        results = []
        for idx, prompt in enumerate(example["prompts"]):
            if idx == 0:
                init_prompt_ids = prompt
            else:
                if isinstance(max_length, dict):
                    max_length_per_turn = max_length[example["task"][idx - 1]]
                else:
                    max_length_per_turn = max_length

                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_length_per_turn,
                )
                current_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_ids = init_prompt_ids + current_ids

                result = self.llm.generate(
                    prompt_token_ids=input_ids, sampling_params=sampling_params
                )
                results.append(result[0].outputs[0].text)
        output = {"answers": results, "gt": example["ground_truth"]}

        if isinstance(max_length, dict):  # mixed task setting
            output["task"] = example["task"]

        return output

    def test(self, example, max_length=100, disable_golden_context=False):
        from vllm import SamplingParams

        results = []
        for idx, prompt in enumerate(example["prompts"]):
            if isinstance(max_length, dict):
                max_length_per_turn = max_length[example["task"][idx]]
            else:
                max_length_per_turn = max_length

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_length_per_turn,
            )
            if self.is_kv_compress:
                sampling_params = SamplingParams(
                    max_tokens=max_length_per_turn,
                    min_tokens=1,
                    temperature=0.0,
                    max_cache_tokens=4096,
                    protected_window_size=32,
                    metric_collection_buffer_size=0,
                    compress_once=True,
                )

            if idx == 0:
                input_ids = prompt
            else:
                current_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

                # if disable_golden_context, add result[0].outputs[0].text to the prompt
                if disable_golden_context:
                    input_ids = (
                        input_ids
                        + self.tokenizer.encode(
                            result[0].outputs[0].text, add_special_tokens=False
                        )
                        + [self.tokenizer.eos_token_id]
                    )
                input_ids = input_ids + current_ids

            result = self.llm.generate(
                prompt_token_ids=input_ids, sampling_params=sampling_params
            )
            results.append(result[0].outputs[0].text)
        output = {"answers": results, "gt": example["ground_truth"]}

        if isinstance(max_length, dict):  # mixed task setting
            output["task"] = example["task"]

        return output


class GreedySearch:
    def __init__(self, model, tokenizer):
        model.eval()
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer
        self.past_kv = None
        self.add_eos_to_next_prompt = False

    def clear(self):
        self.past_kv = None
        gc.collect()
        torch.cuda.empty_cache()

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)

        # add eos to the beginning of the input_ids if self.add_eos_to_next_prompt is True
        if self.add_eos_to_next_prompt:
            input_ids = [self.tokenizer.eos_token_id] + input_ids
            self.add_eos_to_next_prompt = False

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = (
                torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()
            )

        return model_inputs

    def _make_first_turn(self, input_ids):
        model_inputs = {}
        model_inputs["input_ids"] = input_ids

        for key in model_inputs:
            model_inputs[key] = (
                torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()
            )

        return model_inputs

    def test_scdq(self, example, max_length=100):
        results = []
        for idx, prompt in enumerate(example["prompts"]):
            if isinstance(max_length, dict):
                max_length_per_turn = max_length[example["task"][idx - 1]]
            else:
                max_length_per_turn = max_length

            if idx == 0:
                model_inputs = self._make_first_turn(prompt)
            else:
                model_inputs = self._process_texts(prompt)
            input_ids = model_inputs["input_ids"]

            with torch.inference_mode():
                if idx == 0:
                    result = self._encode(input_ids, max_length=max_length_per_turn)
                else:
                    result = self._decode(
                        input_ids,
                        max_length=max_length_per_turn,
                        dense_prefix=True,
                        update_global_past_kv=False,
                    )

                    results.append(
                        self.tokenizer.decode(result[0, len(input_ids[0]) :])
                    )
            torch.cuda.empty_cache()
        self.clear()
        output = {"answers": results, "gt": example["ground_truth"]}

        if isinstance(max_length, dict):  # mixed task setting
            output["task"] = example["task"]

        return output

    def test(self, example, max_length=100, disable_golden_context=False):
        results = []
        # for idx, prompt in tqdm(enumerate(example['prompts']), total=len(example['prompts']), desc="Prompt"):
        for idx, prompt in enumerate(example["prompts"]):
            if isinstance(max_length, dict):
                max_length_per_turn = max_length[example["task"][idx]]
            else:
                max_length_per_turn = max_length

            if idx == 0:
                model_inputs = self._make_first_turn(prompt)
            else:
                model_inputs = self._process_texts(prompt)
            input_ids = model_inputs["input_ids"]

            with torch.inference_mode():
                if idx == 0:
                    result = self._decode(
                        input_ids,
                        max_length=max_length_per_turn,
                        disable_golden_context=disable_golden_context,
                    )
                else:
                    result = self._decode(
                        input_ids,
                        max_length=max_length_per_turn,
                        dense_prefix=True,
                        disable_golden_context=disable_golden_context,
                    )

            results.append(self.tokenizer.decode(result[0, len(input_ids[0]) :]))
            torch.cuda.empty_cache()
        self.clear()
        output = {"answers": results, "gt": example["ground_truth"]}

        if isinstance(max_length, dict):  # mixed task setting
            output["task"] = example["task"]

        return output

    def _encode(self, input_ids, max_length=None):
        if self.past_kv is None:
            past_key_values = self.model.prepare_inputs_for_generation(input_ids)[
                "past_key_values"
            ]
        else:
            past_key_values = self.past_kv

        out = self.model(
            input_ids=input_ids,
            # attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
            past_key_values=past_key_values,
            num_logits_to_keep=1,
        )
        _, past_key_values = out.logits, out.past_key_values

        self.past_kv = past_key_values

    def _decode(
        self,
        input_ids,
        max_length=100,
        extra_end_token_ids=[],
        dense_prefix=False,
        update_global_past_kv=True,
        disable_golden_context=False,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        assert input_ids.size(0) == 1
        end_token_ids = (
            extra_end_token_ids
            + [self.tokenizer.eos_token_id]
            + self.model.config.eos_token_id
        )
        logits = None
        if self.past_kv is None:
            model_inputs = {}
            self.model._prepare_cache_for_generation(
                GenerationConfig(), model_inputs, None, None, None, None
            )
            past_key_values = model_inputs["past_key_values"]
        else:
            past_key_values = self.past_kv

        if not update_global_past_kv:
            self.global_kv_update_mode(False)

        for i in range(max_length):
            if i == 0:  # prefilling
                out = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                    num_logits_to_keep=1,
                )
                logits, past_key_values = out.logits, out.past_key_values

            else:  # decoding
                if (
                    not disable_golden_context
                ):  # if use golden context, then decoding should not update global past_kv
                    self.global_kv_update_mode(False)
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits, past_key_values = out.logits, out.past_key_values

            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
            if word.item() in end_token_ids or i == max_length:
                break

            input_ids = torch.cat(
                (input_ids, word.to(input_ids.device).view(1, 1)), dim=-1
            )

        if not update_global_past_kv or not disable_golden_context:
            self.global_kv_update_mode(True)
            past_key_values.clear_temp_kv_cache()

        self.past_kv = past_key_values
        # should see whether the last token is eos, if not tell self.test to add it to the next prompt
        if word.item() not in end_token_ids and disable_golden_context:
            self.add_eos_to_next_prompt = True
        return input_ids

    def global_kv_update_mode(self, mode):
        try:
            attn_class = self.model.model.layers[0].self_attn.__class__
        except:
            attn_class = self.model.transformer.encoder.layers[
                0
            ].self_attention.__class__
        self.model.apply(
            lambda m: setattr(m, "update_global_past_kv", mode)
            if isinstance(m, attn_class)
            else None
        )


class GreedySearch_RetrAttn(GreedySearch):
    def _decode(
        self,
        input_ids,
        max_length=100,
        extra_end_token_ids=[],
        dense_prefix=False,
        update_global_past_kv=True,
        disable_golden_context=False,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        assert input_ids.size(0) == 1
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        if self.past_kv is None:
            model_inputs = {}
            self.model._prepare_cache_for_generation(
                GenerationConfig(), model_inputs, None, None, None, None
            )
            past_key_values = model_inputs["past_key_values"]
        else:
            past_key_values = self.past_kv

        if not update_global_past_kv:
            self.global_kv_update_mode(False)

        for i in range(max_length):
            if i == 0:  # prefilling
                if dense_prefix:
                    for token in input_ids.squeeze(0):
                        out = self.model(
                            input_ids=token.unsqueeze(0).unsqueeze(0),
                            use_cache=True,
                            return_dict=True,
                            past_key_values=past_key_values,
                            num_logits_to_keep=1,
                        )
                        logits, past_key_values = out.logits, out.past_key_values
                else:
                    out = self.model(
                        input_ids=input_ids,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                        num_logits_to_keep=1,
                    )
                    logits, past_key_values = out.logits, out.past_key_values

            else:  # decoding
                if (
                    not disable_golden_context
                ):  # if use golden context, then decoding should not update global past_kv
                    self.global_kv_update_mode(False)
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits, past_key_values = out.logits, out.past_key_values

            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
            if word.item() in end_token_ids or i == max_length:
                break

            input_ids = torch.cat(
                (input_ids, word.to(input_ids.device).view(1, 1)), dim=-1
            )

        if not update_global_past_kv or not disable_golden_context:
            self.global_kv_update_mode(True)
            past_key_values.clear_temp_kv_cache()

        self.past_kv = past_key_values
        # should see whether the last token is eos, if not tell self.test to add it to the next prompt
        if word.item() != self.tokenizer.eos_token_id:
            self.add_eos_to_next_prompt = True
        return input_ids


class GreedySearch_InfLLM(GreedySearch):
    # basically, InfLLM do _encode and _decode chunk by chunk
    def _encode(self, input_ids, past_kv=None, max_length=None):
        chunk_size = 8192
        for st in range(0, input_ids.size(1), chunk_size):
            torch.cuda.empty_cache()
            ed = min(input_ids.size(1), st + chunk_size)
            out = self.model(
                input_ids=input_ids[:, st:ed],
                use_cache=True,
                return_dict=True,
                past_key_values=past_kv,
            )
            logits, past_kv = out.logits, out.past_key_values

        self.past_kv = past_kv

    def _decode(
        self,
        input_ids,
        max_length=100,
        extra_end_token_ids=[],
        dense_prefix=False,
        update_global_past_kv=True,
        disable_golden_context=False,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        assert input_ids.size(0) == 1
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        if self.past_kv is None:
            if self.use_sinkcache:
                past_key_values = SinkCache(window_length=3968, num_sink_tokens=128)
            else:
                past_key_values = self.model.prepare_inputs_for_generation(input_ids)[
                    "past_key_values"
                ]
        else:
            past_key_values = self.past_kv
            if self.use_sinkcache:
                past_key_values.window_length += 5_000

        chunk_size = 8196
        for i in range(max_length):
            if i == 0:
                if dense_prefix:
                    for token in input_ids.squeeze(0):
                        out = self.model(
                            input_ids=token.unsqueeze(0).unsqueeze(0),
                            use_cache=True,
                            return_dict=True,
                            past_key_values=past_key_values,
                        )
                        logits, past_key_values = out.logits, out.past_key_values

                else:
                    for st in range(0, input_ids.size(1) - 1, chunk_size):
                        ed = min(input_ids.size(1) - 1, st + chunk_size)
                        out = self.model(
                            input_ids=input_ids[:, st:ed],
                            use_cache=True,
                            return_dict=True,
                            past_key_values=past_key_values,
                        )
                        logits, past_key_values = out.logits, out.past_key_values

                if update_global_past_kv:
                    self.past_kv = past_key_values

            else:
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits, past_key_values = out.logits, out.past_key_values

                if disable_golden_context and update_global_past_kv:
                    self.past_kv = past_key_values

            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
            if word.item() in end_token_ids or i == max_length:
                break

            input_ids = torch.cat(
                (input_ids, word.to(input_ids.device).view(1, 1)), dim=-1
            )

        # should see whether the last token is eos, if not tell self.test to add it to the next prompt
        if word.item() != self.tokenizer.eos_token_id and disable_golden_context:
            self.add_eos_to_next_prompt = True
        return input_ids


class GreedySearch_RetrAttn_Legacy(GreedySearch):
    def __init__(self, model, tokenizer, top_k, from_layer, with_minference=False):
        super().__init__(model, tokenizer)
        if with_minference:
            from sparse_retr_attn.modeling_llama_minference_with_retr import (
                VectorDB_KV_Cache,
                hf_greedy_search_retr,
            )
        else:
            from sparse_retr_attn.modeling_llama_retr_attn import (
                VectorDB_KV_Cache,
                hf_greedy_search_retr,
            )
        self.top_k = top_k
        self.from_layer = from_layer
        self.kv_class = VectorDB_KV_Cache

    def clear(self):
        self.past_kv = None
        self.kv_len = 0
        gc.collect()
        torch.cuda.empty_cache()

    def _decode(
        self,
        input_ids,
        max_length=100,
        extra_end_token_ids=[],
        dense_prefix=False,
        update_global_past_kv=True,
        disable_golden_context=False,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        assert input_ids.size(0) == 1
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None

        if self.past_kv is None:
            past_key_values = self.kv_class(
                max_length=max_length * self.num_turns + input_ids.size(1),
                temp_cache_size=max_length,
            )
            kv_len = input_ids.size(1)
        else:
            past_key_values = self.past_kv
            kv_len = self.kv_len

        for i in range(max_length):
            if i == 0:
                if dense_prefix:
                    for token in input_ids.squeeze(0):
                        out = self.model(
                            input_ids=token.unsqueeze(0).unsqueeze(0),
                            use_cache=True,
                            return_dict=True,
                            past_key_values=past_key_values,
                            insert_db=True if update_global_past_kv else False,
                            top_k=self.top_k,
                            from_layer=self.from_layer,
                            cache_position=torch.tensor(
                                [kv_len], device=input_ids.device, dtype=torch.long
                            ),
                        )
                        logits, past_key_values = out.logits, out.past_key_values
                        kv_len += 1
                else:
                    out = self.model(
                        input_ids=input_ids,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                        top_k=self.top_k,
                        from_layer=self.from_layer,
                        cache_position=torch.arange(kv_len, device=input_ids.device),
                    )
                    logits, past_key_values = out.logits, out.past_key_values
                    # kv_len += 1
                # update global past_kv with prefix only
                if update_global_past_kv:
                    self.past_kv = past_key_values
                    self.kv_len = kv_len

            else:
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    insert_db=False,
                    top_k=self.top_k,
                    from_layer=self.from_layer,
                    cache_position=torch.tensor(
                        [kv_len], device=input_ids.device, dtype=torch.long
                    ),
                )
                logits, past_key_values = out.logits, out.past_key_values
                kv_len += 1
                if disable_golden_context and update_global_past_kv:
                    self.past_kv = past_key_values
                    self.kv_len = kv_len

            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
            if word.item() in end_token_ids or i == max_length:
                break

            input_ids = torch.cat(
                (input_ids, word.to(input_ids.device).view(1, 1)), dim=-1
            )

        self.past_kv.temp_seen = 0  # Discard decoding tokens

        # should see whether the last token is eos, if not tell self.test to add it to the next prompt
        if word.item() != self.tokenizer.eos_token_id and disable_golden_context:
            self.add_eos_to_next_prompt = True
        return input_ids

    def _encode(self, input_ids, max_length=None):
        if self.past_kv is None:
            past_key_values = self.kv_class(
                max_length=max_length * self.num_turns + input_ids.size(1),
                temp_cache_size=max_length + self.length_of_query + 10,
            )
            kv_len = input_ids.size(1)
        else:
            past_key_values = self.past_kv
            kv_len = self.kv_len

        out = self.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            past_key_values=past_key_values,
            top_k=self.top_k,
            from_layer=self.from_layer,
            cache_position=torch.arange(kv_len, device=input_ids.device),
        )
        _, past_key_values = out.logits, out.past_key_values

        self.past_kv = past_key_values
        self.kv_len = kv_len

    def test_scdq(self, example, max_length=100):
        prompts = example["prompts"]
        self.length_of_query = len(self.tokenizer.encode(prompts[1]))
        self.num_turns = len(prompts)
        return super().test_scdq(example, max_length)

    def test(self, example, max_length=100, disable_golden_context=False):
        prompts = example["prompts"]
        self.length_of_query = len(self.tokenizer.encode(prompts[1]))
        self.num_turns = len(prompts)
        return super().test(example, max_length, disable_golden_context)


class GreedySearch_Mamba2:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer

    def test_scdq(self, example, max_length=100):
        results = []
        for idx, prompt in enumerate(example["prompts"]):
            if isinstance(max_length, dict):
                max_length_per_turn = max_length[example["task"][idx - 1]]
            else:
                max_length_per_turn = max_length

            generation_config = GenerationConfig(
                max_new_tokens=max_length_per_turn,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            if idx == 0:
                init_prompt_ids = prompt
            else:
                current_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_ids = init_prompt_ids + current_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.llm.device)

                outputs = self.llm.generate(
                    input_ids=input_ids, generation_config=generation_config
                )
                output = outputs[0, len(input_ids[0]) :]
                output = self.tokenizer.decode(output, skip_special_tokens=True)
                output = output.strip()
                results.append(output)

        output = {"answers": results, "gt": example["ground_truth"]}

        if isinstance(max_length, dict):  # mixed task setting
            output["task"] = example["task"]

        return output

    def test(self, example, max_length=100, disable_golden_context=False):
        results = []
        for idx, prompt in enumerate(example["prompts"]):
            if isinstance(max_length, dict):
                max_length_per_turn = max_length[example["task"][idx]]
            else:
                max_length_per_turn = max_length

            generation_config = GenerationConfig(
                max_new_tokens=max_length_per_turn,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            if idx == 0:
                input_ids = prompt
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.llm.device)
            else:
                current_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                current_ids = torch.tensor(current_ids).unsqueeze(0).to(self.llm.device)

                if disable_golden_context:
                    # input_ids = input_ids + self.tokenizer.encode(results[-1], add_special_tokens=False) + [self.tokenizer.eos_token_id]
                    prev_ids = (
                        torch.tensor(
                            self.tokenizer.encode(results[-1], add_special_tokens=False)
                        )
                        .unsqueeze(0)
                        .to(self.llm.device)
                    )
                    eos_id = torch.tensor(
                        [self.tokenizer.eos_token_id], device=self.llm.device
                    ).unsqueeze(0)
                    input_ids = torch.cat((input_ids, prev_ids, eos_id), dim=-1)
                input_ids = torch.cat((input_ids, current_ids), dim=-1)
            outputs = self.llm.generate(
                input_ids=input_ids, generation_config=generation_config
            )
            output = outputs[0, len(input_ids[0]) :]
            output = self.tokenizer.decode(output, skip_special_tokens=True)
            output = output.strip()
            results.append(output)
            torch.cuda.empty_cache()

        output = {"answers": results, "gt": example["ground_truth"]}

        if isinstance(max_length, dict):  # mixed task setting
            output["task"] = example["task"]

        return output
