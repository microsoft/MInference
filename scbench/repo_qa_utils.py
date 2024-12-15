# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Refer to the code in https://github.com/evalplus/repoqa

import itertools
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tempdir
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from transformers import AutoConfig
from tree_sitter_languages import get_language, get_parser

FUNCTION_QUERY = {
    "python": "(function_definition name: (_)) @fdef",
    "java": "(method_declaration name: (_)) @fdef",
    "typescript": "(function_declaration name: (_)) @fdef",
    "rust": "(function_item name: (_)) @fdef",
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier))) @fdef",
    "go": "(function_declaration name: (_)) @fdef",
}

COMMENT_QUERY = {
    "python": [
        "(block (expression_statement (string) @docstring))",
        "(comment) @comment",
    ],
    "java": ["(line_comment) @comment", "(block_comment) @comment"],
    "cpp": ["(comment) @comment"],
    "rust": ["(line_comment) @comment", "(block_comment) @comment"],
    "typescript": ["(comment) @comment"],
    "go": ["(comment) @comment"],
}


def progress(note: str = "processing"):
    return Progress(
        TextColumn(f"{note} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )


LANGUAGES = list(FUNCTION_QUERY.keys())
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

import re

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def compute_function_similarity(
    candidate_function: str, reference_function: str
) -> float:
    candidate_tokens = [item for item in re.split("\s+", candidate_function.strip())]

    reference_tokens = [item for item in re.split("\s+", reference_function.strip())]

    chencherry = SmoothingFunction()

    return sentence_bleu(
        [reference_tokens], candidate_tokens, smoothing_function=chencherry.method4
    )


class Result(Enum):
    BEST_MATCH = "best_match"
    FAIL_MATCH = "fail_match"


# unbiased estimator from https://github.com/openai/human-eval
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def remove_comments(source_code: str, lang: str) -> str:
    source_bytes = bytes(source_code, "utf8")
    parser = get_parser(lang)
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    # Remove comments from source code
    capture_list = []
    for query_str in COMMENT_QUERY[lang]:
        comment_query = get_language(lang).query(query_str)
        capture_list += comment_query.captures(root_node)

    capture_list.sort(key=lambda cap: cap[0].start_byte, reverse=True)

    for node, _ in capture_list:
        source_bytes = source_bytes[: node.start_byte] + source_bytes[node.end_byte :]

    return source_bytes.decode("utf-8")


def sanitize_output(model_output: str, lang: str) -> str:
    model_output = model_output.strip()
    search_pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    code_blocks = re.findall(search_pattern, model_output, re.DOTALL | re.MULTILINE)

    parser = get_parser(lang)
    fn_query = get_language(lang).query(FUNCTION_QUERY[lang])

    # If not code blocks found, simply return model output
    if not code_blocks:
        return model_output

    processed_blocks = []
    for block in code_blocks:
        processed_blocks.append(block)

        # Try to use tree-sitter to parse if possible
        try:
            block_bytes = bytes(block, "utf8")
            tree = parser.parse(block_bytes)
            for capture in fn_query.captures(tree.root_node):
                node, _ = capture
                function_content = block_bytes[node.start_byte : node.end_byte]
                return function_content.decode("utf8")
        except:
            pass

    # no valid functions found by tree-sitter approach return first block
    return processed_blocks[0]


def print_result_table(model_name, pass_results):
    # Printing scores in a table
    table = Table(title=f"Scores (%) of {model_name} at different thresholds")
    table.add_column("Threshold", justify="center", style="bold magenta")
    for threshold in THRESHOLDS:
        table.add_column(f"{threshold}", justify="center")

    # Prepare data to determine the maximum values for each threshold
    threshold_scores = {threshold: [] for threshold in THRESHOLDS}
    for lang_results in pass_results.values():
        for thresh, value in lang_results.items():
            threshold_scores[thresh].append(value["pass@1"])

    # Calculate the maximum score for each threshold
    max_scores = {
        threshold: max(scores) for threshold, scores in threshold_scores.items()
    }
    min_scores = {
        threshold: min(scores) for threshold, scores in threshold_scores.items()
    }

    # Fill the table rows
    for language, lang_results in pass_results.items():
        row = [("⭐" if language == "all" else "") + language]
        for threshold, value in lang_results.items():
            score = value["pass@1"]
            formatted_score = f"{100 * score:.1f}"
            if max_scores[threshold] - score < 0.01:
                formatted_score = f"[bold green]{formatted_score}[/]"
            elif score - min_scores[threshold] < 0.01:
                formatted_score = f"[bold red]{formatted_score}[/]"
            row.append(formatted_score)
        if language == "all":
            row = [f"[bold yellow]{r}[/]" for r in row]
        table.add_row(*row)

    Console().print(table)


def needle_evaluator(
    model_output: str,
    ground_truth: str,
    needles,
    lang: str,
    ignore_comments: bool,
) -> Tuple[Result, str, float]:
    best_target = None
    best_similarity = 0
    sanitized_output = sanitize_output(model_output, lang)
    if ignore_comments:
        sanitized_output = remove_comments(sanitized_output, lang)
    for needle in needles:
        current_name = needle["name"]
        current_func = needle["needle"]
        if ignore_comments:
            current_func = remove_comments(current_func, lang)

        current_similarity = compute_function_similarity(sanitized_output, current_func)
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_target = current_name

    if best_target == ground_truth["func_name"]:
        verdict = Result.BEST_MATCH
    else:
        verdict = Result.FAIL_MATCH
    return verdict, best_target, best_similarity


def _get_repo(lang_data: Dict, repo_name: str) -> Dict:
    for repo in lang_data:
        if repo["repo"] == repo_name:
            return repo


def compute_language_results(evaluation_result: Dict, all_results: Dict) -> None:
    for language, lang_results in evaluation_result.items():
        current_result = {}
        total = np.array([1 for _ in lang_results])

        for threshold in THRESHOLDS:
            correct_result = []
            for res in lang_results:
                bc = 0
                if res["is_best_similar"] and res["best_similar_score"] >= threshold:
                    bc = 1
                correct_result.append(bc)
            correct_result = np.array(correct_result)

            pass_at_k = {
                f"pass@{k}": estimate_pass_at_k(total, correct_result, k).mean()
                for k in [1, 10, 100]
                if total.min() >= k
            }
            current_result[threshold] = pass_at_k
        all_results[language] = current_result


def compute_score(
    # model_name: str, dataset: Dict, model_output: List[Dict], ignore_comments: bool
    model_name: str,
    preds: list,
    labels: list,
    needle_by_repo: dict,
    ignore_comments: bool = False,
) -> Dict:
    evaluation_result = defaultdict(list)
    score_name = (
        model_name.replace("'", "")
        .replace("{", "")
        .replace("}", "")
        .replace(": ", "_")
        .replace(", ", "_")
    )
    with progress(f"Scoring {score_name}") as pbar:
        for result in pbar.track(preds):
            lang = result["lang"]
            repo_name = result["repo"]
            model_output = result["prediction"]
            ground_truth = {
                "func_name": result["func_name"],
                "ground_truth": result["ground_truth"],
            }
            needles = needle_by_repo[repo_name]

            verdict, best_target, best_similarity = needle_evaluator(
                model_output, ground_truth, needles, lang, ignore_comments
            )

            is_best_similar = False
            if verdict == Result.BEST_MATCH:
                is_best_similar = True

            current_task = {
                "repo": repo_name,
                "name": ground_truth,
                # "needle_position": result["position_ratio"],
                "is_best_similar": is_best_similar,
                "best_similar_score": best_similarity,
                "best_target": best_target,
                # "position": {
                #     "token_start": result["needle_token_start"],
                #     "token_end": result["needle_token_end"],
                # },
            }
            evaluation_result[lang].append(current_task)

    # Calculate pass@k
    pass_results = {}

    all_langs = []
    for lang in evaluation_result:
        all_langs += evaluation_result[lang]
    total = np.array([1 for _ in all_langs])

    pass_results["all"] = {}
    for threshold in THRESHOLDS:
        correct_result = []
        for res in all_langs:
            bc = 0
            if res["is_best_similar"] and res["best_similar_score"] >= threshold:
                bc = 1
            correct_result.append(bc)
        correct_result = np.array(correct_result)
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, correct_result, k).mean()
            for k in [1, 10, 100]
            if total.min() >= k
        }
        pass_results["all"][threshold] = pass_at_k

    compute_language_results(evaluation_result, pass_results)
    print_result_table(model_name, pass_results)

    output_json = {}
    model_json = {}
    model_json["eval_date"] = str(datetime.now())
    model_json["scores"] = pass_results
    model_json["results"] = evaluation_result

    output_json[model_name] = model_json

    return output_json


def save_json(output_json, result_path) -> None:
    if os.path.isfile(result_path):
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(output_json, f)
