# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import gc
import json
import os

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from minference import MInference


class LongPPL:
    def __init__(
        self,
        model_name,
        attn_type,
        min_context,
        max_context,
        intervals: int = 10,
        run_name: str = None,
        output_path: str = "results/long-ppl/",
        data_path: str = "liyucheng/pg19-4k",
        num_eval_examples: int = 100,
        **kwargs,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prepare_data(
            data_path,
            min_context,
            max_context,
            intervals,
            num_eval_examples,
        )
        self.load_model(model_name, attn_type, **kwargs)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = os.path.join(
            output_path,
            f'{model_name.replace("/", "-")}_{attn_type}_{run_name if run_name is not None else ""}.json',
        )
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                self.results = json.load(f)

    def load_model(
        self,
        model_name: str,
        attn_type: str = "vllm",
        **kwargs,
    ):
        if attn_type == "vllm":
            pass
        else:
            topk_dims_file_path = kwargs.get("topk_dims_file_path", None)
            topk_from_layer = kwargs.get("topk_from_layer", -1)
            minference_patch = MInference(
                attn_type,
                model_name,
                topk_dims_file_path,
                starting_layer=topk_from_layer,
            )
            self.model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self.model.config.is_ppl = True
            self.model = minference_patch(self.model)
        return

    def prepare_data(
        self,
        data_path: str,
        min_context: int,
        max_context: int,
        intervals: int,
        num_eval_examples: int,
    ):
        def tok(x):
            return self.tokenizer(x["text"])

        def truncate(x, length=None):
            return {
                "input_ids": x["input_ids"][:length],
                "attention_mask": x["attention_mask"][:length],
            }

        all_lengths = [
            min_context + (max_context - min_context) // intervals * i
            for i in range(intervals + 1)
        ]

        ds = datasets.load_dataset(data_path, split="train")
        ds1k = ds.select(range(num_eval_examples))

        ds1k = ds1k.map(
            tok,
            # num_proc=16,
            remove_columns=ds.column_names,
        )

        self.test_data = {
            length: ds1k.map(
                truncate,
                # num_proc=16,
                fn_kwargs={"length": length},
            )
            for length in all_lengths
        }
        return

    def save_results(self):
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        return

    def ppl(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        prob = F.log_softmax(shift_logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        shift_labels = shift_labels.view(-1)
        prob = prob.view(-1, prob.size(-1))

        loss = -prob[torch.arange(prob.size(0)), shift_labels].mean()
        return loss.exp().item()

    def chunk_ppl(self, logits, labels, chunk_size=10000):
        total_loss = 0
        num_chunks = 0
        for i in range(0, logits.size(1), chunk_size):
            chunk_logits = logits[:, i : i + chunk_size][..., :-1, :].contiguous()
            chunk_labels = labels[:, i : i + chunk_size][..., 1:].contiguous()

            chunk_prob = F.log_softmax(chunk_logits, dim=-1, dtype=torch.float32).to(
                logits.dtype
            )
            chunk_labels = chunk_labels.view(-1)
            chunk_prob = chunk_prob.view(-1, chunk_prob.size(-1))

            chunk_loss = -chunk_prob[
                torch.arange(chunk_prob.size(0)), chunk_labels
            ].sum()
            total_loss += chunk_loss
            num_chunks += 1

        total_loss /= logits.size(1) - num_chunks
        return total_loss.exp().item()

    def start_test(self):
        print("Starting test...")

        if not hasattr(self, "results"):
            self.results = {}
        for length, ds in self.test_data.items():
            ppls = []
            with torch.no_grad():
                for example in tqdm(ds, desc=f"Testing with context length: {length}"):
                    gc.collect()
                    torch.cuda.empty_cache()

                    example = {
                        k: torch.tensor([v], device=self.model.device)
                        for k, v in example.items()
                    }
                    self.model.config.topk = length // 10
                    output = self.model(
                        **example, use_cache=False, output_hidden_states=False
                    )

                    if length > 10_000:
                        ppl = self.chunk_ppl(
                            output.logits, example["input_ids"], chunk_size=10_000
                        )
                    else:
                        ppl = self.ppl(output.logits, example["input_ids"])

                    ppls.append(ppl)

            length = f"{length // 1_000}K"
            self.results[length] = np.mean(ppls)
            print(f"Average PPL for {length}: {self.results[length]}")
            self.save_results()

        print("Completed.")

    @staticmethod
    def viz_results(
        result_path: str = "results/long-ppl/",
        output_path: str = "results/long-ppl/long-ppl-viz.png",
    ):
        from glob import glob

        import matplotlib.pyplot as plt

        plt.figure(dpi=200, figsize=(10, 6))

        all_res = glob(os.path.join(result_path, "*.json"))
        for res in all_res:
            model_name = res.split("/")[-1].split(".")[0]
            with open(res, "r") as f:
                data = json.load(f)
            plt.plot(data.keys(), data.values(), label=model_name)

        plt.legend()
        plt.grid(True)
        plt.xlabel("Context Length")
        plt.ylabel("PPL")

        plt.savefig(output_path)
        plt.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, required=True)
    args.add_argument(
        "--attn_type",
        type=str,
        choices=[
            "hf",
            "streaming",
            "minference",
            "dilated1",
            "dilated2",
        ],
        default="hf",
    )
    args.add_argument("--do_plot", action="store_true")
    args.add_argument("--min_seq_length", type=int, default=1_000)
    args.add_argument("--max_seq_length", type=int, default=100_000)
    args.add_argument("--intervals", type=int, default=9)
    args.add_argument("--run_name", type=str, default=None)
    args.add_argument("--num_eval_examples", type=int, default=100)

    args.add_argument("--topk", type=int, default=-1)
    args.add_argument("--topk_from_layer", type=int, default=-1)
    args.add_argument("--topk_dims_file_path", type=str, default=None)
    args.add_argument("--output_path", type=str, default="results/long-ppl/")

    args = args.parse_args()

    test = LongPPL(
        model_name=args.model_name,
        attn_type=args.attn_type,
        min_context=args.min_seq_length,
        max_context=args.max_seq_length,
        intervals=args.intervals,
        run_name=args.run_name,
        num_eval_examples=args.num_eval_examples,
        output_path=args.output_path,
        topk=args.topk,
        topk_from_layer=args.topk_from_layer,
        topk_dims_file_path=args.topk_dims_file_path,
    )
    test.start_test()

    if args.do_plot:
        LongPPL.viz_results("results/long-ppl/")
