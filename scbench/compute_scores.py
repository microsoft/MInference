# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import evaluate
from args import parse_args
from repo_qa_utils import compute_score as compute_repoqa_score
from tqdm import tqdm

ROUGE_SCORER = evaluate.load("rouge")


def normalize_answer(s: str) -> str:
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


def normalize_zh_answer(s: str) -> str:
    """Chinese version. Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."  # noqa
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def string_match_all(pred, ref, model_name=""):
    score = sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
    return round(score, 2)


def f1_score(prediction, ground_truth) -> tuple[float, float, float]:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths) -> float:
    """Computes the F1, recall, and precision."""
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(pred)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def qa_f1_score_zh(pred: str, ground_truths: list[str]) -> float:
    """
    QA F1 score for chinese.
    """
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        norm_pred = normalize_zh_answer(pred)
        norm_label = normalize_zh_answer(ground_truth)

        # One character one token.
        pred_tokens = list(norm_pred)
        label_tokens = list(norm_label)
        scores = f1_score(pred_tokens, label_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def load_json(fname):
    return json.load(open(fname))


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r", encoding="utf8") as fin:
        for line in fin:
            if line.strip() == "":  # Skip empty lines
                continue
            if i == cnt:
                break
            if line.strip() == "":  # Skip empty lines
                continue
            yield json.loads(line)
            i += 1


def first_int_match(prediction):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def split_retrieval_answer(pred: str):
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    return words


def get_score_one_kv_retrieval(pred, label, model_name: str) -> bool:
    # for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}", "</s>", "The", "To"]:
    #     pred = pred.replace(c, " ")
    # words = pred.split()
    return label in pred


def get_score_one_passkey(pred, label, model_name: str) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_number_string(pred, label, model_name: str) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_code_run(pred, label, model_name: str) -> bool:
    """
    Returns the score of one example in Code.Run.
    """
    if isinstance(label, list):
        label = label[0]
    pred = pred.strip()
    for c in ["\n", ".", "`", "'", '"', ":"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    if len(words) == 0:
        return False
    try:
        pred = int(words[-1])
        return label == pred
    except Exception:
        return False


def get_score_one_code_debug(pred, label, model_name: str) -> bool:
    """
    Returns the score of one example in Code.Debug.
    """
    pred = pred.strip()
    label_c = label[1]
    fn_name = label[0]
    if pred[:2] in [f"{label_c}.", f"{label_c}:"]:
        return True

    ans_prefixes = [
        "answer is:",
        # "answer is",
        # "error is",
        "is:",
        "answer:",
    ]
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        pred = pred[idx + len(prefix) + 1 :]
        for s in [label_c, fn_name]:
            if pred.startswith(s):
                return True
        return False
    return False


def get_score_one_math_find(pred, label, model_name: str) -> bool:
    if isinstance(label, list):
        # In math_find, there is always only one label.
        label = label[0]
    if isinstance(label, int):
        # Find first int or float
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return False
        first_num = first_num.group(0).strip()
        return int(float(first_num)) == label
    elif isinstance(label, float):
        # Find first float or int
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return False
        first_float = first_float.group(0).strip()
        return float(first_float) == label
    else:
        raise TypeError(f"Expected int or float, got {type(label)}")


def get_score_one_longdialogue_qa_eng(pred, label, model_name: str) -> bool:
    pred = pred.strip()
    pred = pred.upper()
    for item in label:
        if item.upper() in pred:
            return 1
    return 0


def get_score_one_longbook_choice_eng(pred, label, model_name: str) -> bool:
    # Just use the first letter as the prediction
    pred = pred.strip()
    if pred == "":
        return False
    if pred[0] in "ABCD":
        return pred[0] in label
    if pred in label:
        return True
    # Find a answer prefix
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally, just find the first occurrence of A, B, C, or D.
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False


def get_score_one_longbook_qa_eng(pred, label, model_name: str) -> float:
    return qa_f1_score(pred, label)


def get_score_one_longbook_sum_eng(pred: str, label: str, model_name: str) -> float:
    score = ROUGE_SCORER.compute(
        predictions=[pred], references=[label], use_aggregator=False
    )
    return score["rougeLsum"][0]  # type: ignore


def get_score_one_longbook_qa_chn(pred, label, model_name: str) -> float:
    return qa_f1_score_zh(pred, label)


def get_score_one_math_calc(pred, label, model_name: str) -> float:
    assert isinstance(label, list), f"Expected list, got {type(label)}"
    # assert isinstance(pred, list), f"Expected list, got {type(pred)}"
    pred_nums = []
    pred_list = re.split("[^0-9]", pred)
    for item in pred_list:
        if item != "":
            pred_nums.append(int(item))

    # Our prompts makes GPT4 always output the first number as the first value
    # in the predicted answer.
    if model_name == "gpt4":
        pred_nums = pred_nums[1:]

    cnt = 0
    for i in range(len(label)):
        if i >= len(pred_nums):
            break
        if label[i] == pred_nums[i]:
            cnt += 1
        else:
            break
    return cnt / len(label)


def get_score_one(pred: str, label: str, task_name: str, model_name: str) -> float:
    """
    Computes the score for one prediction.
    Returns one float (zero and one for boolean values).
    """
    NAME_TO_SCORE_GETTER = {
        # Retrieve
        "kv_retrieval": get_score_one_kv_retrieval,
        "kv_retrieval_prefix": get_score_one_kv_retrieval,
        "kv_retrieval_both": get_score_one_kv_retrieval,
        "passkey": get_score_one_passkey,
        "number_string": get_score_one_number_string,
        # Code
        "code_run": get_score_one_code_run,
        "code_debug": get_score_one_code_debug,
        # Longbook
        "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
        "longbook_qa_eng": get_score_one_longbook_qa_eng,
        "longbook_sum_eng": get_score_one_longbook_sum_eng,
        "longbook_choice_eng": get_score_one_longbook_choice_eng,
        "longbook_qa_chn": get_score_one_longbook_qa_chn,
        # Math
        "math_find": get_score_one_math_find,
        "math_calc": get_score_one_math_calc,
        # multi-turn nativ
        "scbench_summary": get_score_one_longbook_sum_eng,
        "scbench_vt": string_match_all,
        "scbench_many_shot": get_score_one_longdialogue_qa_eng,
        "scbench_kv_compressible": get_score_one_kv_retrieval,
    }
    assert task_name in NAME_TO_SCORE_GETTER, f"Invalid task name: {task_name}"
    score = NAME_TO_SCORE_GETTER[task_name](pred, label, model_name)
    return float(score)


def get_labels(preds: list, task_name: str = None) -> list[str]:
    possible_label_keys = ["ground_truth", "label"]
    labels = []
    for pred in preds:
        if task_name is not None and pred["task"] != task_name:
            continue
        for label_key in possible_label_keys:
            if label_key in pred:
                labels.append(pred[label_key])
                break
    if not labels:
        raise ValueError(f"Cannot find label in {preds[0]}")
    return labels


def get_preds(preds: list, data_name: str) -> list[str]:
    pred_strings = []
    possible_pred_keys = ["prediction", "pred"]
    for pred in preds:
        if "task" in pred:
            if pred["task"] != data_name:
                continue
        this_pred = "NO PREDICTION"
        for pred_key in possible_pred_keys:
            if pred_key in pred:
                this_pred = pred[pred_key]
                break
        else:
            raise ValueError(f"Cannot find prediction in {pred}")
        pred_strings.append(this_pred)
    if len(pred_strings) == 0:
        raise ValueError(f"No prediction found for {data_name}")
    return pred_strings


def get_score(labels: list, preds: list, data_name: str, model_name: str) -> float:
    """
    Computes the average score for a task.
    """
    assert len(labels) == len(preds)
    scores = []
    for label, pred in tqdm(zip(labels, preds)):
        score = get_score_one(pred, label, data_name, model_name)
        scores.append(score)
    return sum(scores) / len(scores)


Multiturnbench_to_Infinitebench = {
    "scbench_choice_eng": "longbook_choice_eng",
    "scbench_qa_eng": "longdialogue_qa_eng",
    "scbench_qa_chn": "longbook_qa_chn",
    "scbench_kv": "kv_retrieval",
    "scbench_kv_hard": "kv_retrieval",
    "scbench_hashhop": "kv_retrieval",
    "scbench_prefix_suffix": "kv_retrieval",
    "scbench_mf": "math_find",
    "scbench_passkey": "passkey",
}


def compute_scores(
    preds_path, data_name: str, model_name: str, max_seq_length=-1, scdq_mode=False
):
    print("Loading prediction results from", preds_path)
    preds = list(iter_jsonl(preds_path))

    if data_name in Multiturnbench_to_Infinitebench:
        task_name = Multiturnbench_to_Infinitebench[data_name]
    else:
        task_name = data_name

    if task_name in ["scbench_repoqa", "scbench_repoqa_and_kv"]:
        # collect needle wrt repos
        needle_by_repo = {}
        for pred in preds:
            if task_name == "scbench_repoqa_and_kv":
                if pred["task"] != "scbench_repoqa":
                    continue
            repo = pred["repo"]
            if repo not in needle_by_repo:
                needle_by_repo[repo] = []
            needle_by_repo[repo].append(
                {"needle": pred["ground_truth"], "name": pred["func_name"]}
            )

    if scdq_mode:
        if task_name == "scbench_repoqa":
            labels = get_labels(preds)
            acc = compute_repoqa_score(model_name, preds, labels, needle_by_repo)
        elif task_name == "scbench_summary_with_needles":
            subtasks = ["scbench_summary", "scbench_passkey"]
            acc = {}
            for subtask in subtasks:
                try:
                    labels = get_labels(preds, subtask)
                    preds_ = get_preds(preds, subtask)
                except ValueError:
                    print(f"No prediction for {subtask}")
                    acc[subtask] = 0
                    continue
                if subtask in Multiturnbench_to_Infinitebench:
                    task_ = Multiturnbench_to_Infinitebench[subtask]
                else:
                    task_ = subtask
                acc_ = get_score(labels, preds_, task_, model_name)
                acc[subtask] = acc_
        elif task_name == "scbench_repoqa_and_kv":
            subtasks = ["scbench_repoqa", "scbench_kv"]
            acc = {}
            for subtask in subtasks:
                try:
                    labels = get_labels(preds, subtask)
                    preds_ = [pred for pred in preds if pred["task"] == subtask]
                except ValueError:
                    print(f"No prediction for {subtask}")
                    acc[subtask] = 0
                    continue

                if subtask == "scbench_repoqa":
                    acc_ = compute_repoqa_score(
                        model_name, preds_, labels, needle_by_repo
                    )
                    acc_ = acc_[model_name]["scores"]["all"][0.8]["pass@1"]
                elif subtask == "scbench_kv":
                    acc_ = get_score(labels, preds_, "kv_retrieval", model_name)
                else:
                    raise ValueError(f"Invalid subtask: {subtask}")
                acc[subtask] = acc_
        elif task_name in ["scbench_kv_compressible"]:
            subtasks = list(set(pred["task"] for pred in preds))
            acc = {}
            for subtask in subtasks:
                try:
                    labels = get_labels(preds)
                    preds_ = get_preds(preds, subtask)
                except ValueError:
                    print(f"No prediction for {subtask}")
                    acc[subtask] = 0
                    continue

                acc_ = get_score(labels, preds_, task_name, model_name)
                acc[subtask] = acc_
        else:
            labels = get_labels(preds)
            preds = get_preds(preds, data_name)
            acc = get_score(labels, preds, task_name, model_name)

        print(
            f"===== Accuracy of {model_name} on {data_name} task for same-context-different-query mode is: {acc} ====="
        )

        pred_dir = preds_path.parent
        save_file = pred_dir / f"{model_name}_summary.txt"

        print(f"Saving results to {save_file}")
        with open(save_file, "a") as f:
            if task_name == "scbench_repoqa":
                acc_str = f"{acc[model_name]['scores']['all'][0.8]['pass@1'] * 100:.3f}"
            elif task_name == "scbench_summary_with_needles":
                acc_str = f"{acc['scbench_summary'] * 100:.3f},{acc['scbench_passkey'] * 100:.3f}"
            elif task_name == "scbench_repoqa_and_kv":
                acc_str = (
                    f"{acc['scbench_repoqa'] * 100:.3f},{acc['scbench_kv'] * 100:.3f}"
                )
            elif task_name == "scbench_kv_compressible":
                acc_str = f"{acc}"
            else:
                acc_str = f"{acc * 100:.3f}"
            if max_seq_length != -1:
                f.write(
                    f"{model_name},{data_name},scdq_mode,{max_seq_length},{acc_str}\n"
                )
            else:
                f.write(f"{model_name},{data_name},scdq_mode,{acc_str}\n")
        return acc_str

    preds_by_turns = {}
    # need to group preds with the turn_idx
    for pred in preds:
        turn_idx = pred["turn_idx"]
        if turn_idx not in preds_by_turns:
            preds_by_turns[turn_idx] = []
        preds_by_turns[turn_idx].append(pred)

    acc_by_turns = {}
    for turn_idx, preds in preds_by_turns.items():
        if task_name == "scbench_repoqa":
            labels = get_labels(preds)
            acc = compute_repoqa_score(model_name, preds, labels, needle_by_repo)
        elif task_name == "scbench_summary_with_needles":
            # compute acc for each task
            subtasks = ["scbench_summary", "scbench_passkey"]
            acc = {}
            for subtask in subtasks:
                try:
                    labels = get_labels(preds, subtask)
                    preds_ = get_preds(preds, subtask)
                except ValueError:
                    print(f"No prediction for {subtask}")
                    acc[subtask] = 0
                    continue
                if subtask in Multiturnbench_to_Infinitebench:
                    task_ = Multiturnbench_to_Infinitebench[subtask]
                else:
                    task_ = subtask
                acc_ = get_score(labels, preds_, task_, model_name)
                acc[subtask] = acc_
        elif task_name == "scbench_repoqa_and_kv":
            subtasks = ["scbench_repoqa", "scbench_kv"]
            acc = {}
            for subtask in subtasks:
                try:
                    labels = get_labels(preds, subtask)
                except ValueError:
                    print(f"No prediction for {subtask}")
                    acc[subtask] = 0
                    continue

                if subtask == "scbench_repoqa":
                    preds_ = [pred for pred in preds if pred["task"] == subtask]
                    acc_ = compute_repoqa_score(
                        model_name, preds_, labels, needle_by_repo
                    )
                    acc_ = acc_[model_name]["scores"]["all"][0.8]["pass@1"]
                elif subtask == "scbench_kv":
                    preds_ = get_preds(preds, subtask)
                    acc_ = get_score(labels, preds_, "kv_retrieval", model_name)
                else:
                    raise ValueError(f"Invalid subtask: {subtask}")
                acc[subtask] = acc_
        elif task_name in ["scbench_kv_compressible"]:
            subtasks = list(set(pred["task"] for pred in preds))
            acc = {}
            for subtask in subtasks:
                try:
                    labels = get_labels(preds, subtask)
                    preds_ = get_preds(preds, subtask)
                    assert len(preds_) == len(
                        labels
                    ), f"Length of preds_: {len(preds_)}, length of labels: {len(labels)}, subtask: {subtask}"
                except ValueError:
                    print(f"No prediction for {subtask}")
                    acc[subtask] = 0
                    continue

                acc_ = get_score(labels, preds_, task_name, model_name)
                acc[subtask] = acc_
        else:
            labels = get_labels(preds)
            preds = get_preds(preds, data_name)
            acc = get_score(labels, preds, task_name, model_name)
            print(
                f"===== Accuracy of {model_name} on {data_name} task for turn {turn_idx} is: {acc} ====="
            )
        acc_by_turns[turn_idx] = acc

    pred_dir = preds_path.parent
    save_file = pred_dir / f"{model_name}_summary.txt"

    print(f"Saving results to {save_file}")
    with open(save_file, "a") as f:
        if task_name == "scbench_repoqa":
            acc_str = ",".join(
                [
                    f"{v[model_name]['scores']['all'][0.8]['pass@1'] * 100:.3f}"
                    for k, v in acc_by_turns.items()
                ]
            )
        elif task_name == "scbench_summary_with_needles":
            # summary/passkey
            acc_str = ",".join(
                [
                    f"{v['scbench_summary'] * 100:.3f},{v['scbench_passkey'] * 100:.3f}"
                    for k, v in acc_by_turns.items()
                ]
            )
            # acc_str = ",".join([f"summary: {v['scbench_summary'] * 100:.3f}, passkey: {v['scbench_passkey'] * 100:.3f}" for k, v in acc_by_turns.items()])
        elif task_name == "scbench_repoqa_and_kv":
            # repoqa/kv
            acc_str = ",".join(
                [
                    f"{v['scbench_repoqa'] * 100:.3f},{v['scbench_kv'] * 100:.3f}"
                    for k, v in acc_by_turns.items()
                ]
            )
            # acc_str = ",".join([f"repoqa: {v['scbench_repoqa'] * 100:.3f}, kv: {v['scbench_kv'] * 100:.3f}" for k, v in acc_by_turns.items()])
        elif task_name == "scbench_kv_compressible":
            acc_str = ",".join([f"Turn-{k}:{v}" for k, v in acc_by_turns.items()])
        else:
            acc_str = ",".join([f"{v * 100:.3f}" for k, v in acc_by_turns.items()])
        if max_seq_length != -1:
            f.write(f"{model_name},{data_name},{max_seq_length},{acc_str}\n")
        else:
            f.write(f"{model_name},{data_name},{acc_str}\n")
    return acc_str


ALL_TASKS = [
    "passkey",
    "number_string",
    "kv_retrieval",
    "longdialogue_qa_eng",
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "longbook_qa_chn",
    "math_find",
    "math_calc",
    "code_run",
    "code_debug",
]

if __name__ == "__main__":
    args = parse_args()
    if args.task == "all":
        tasks = ALL_TASKS
    else:
        tasks = [args.task]
    for task in tasks:
        result_dir = Path(args.output_dir, args.model_name)
        preds_path = result_dir / f"preds_{task}.jsonl"
        assert preds_path.exists(), f"Predictions not found in: {preds_path}"
        compute_scores(preds_path, task, args.model_name)
