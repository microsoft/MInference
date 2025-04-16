# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

import numpy as np
import torch
from absl.app import run
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from minference import MInference


class LLMNeedleHaystackTester:
    OURS_TEMPLATE = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n{context}\n\nQuestion: {question} Don't give information outside the document or repeat your findings. Keep your response short and direct. Answer: "
    RANDOM_NEEDLE_CITIES = [
        "Chicago",
        "Yangon",
        "Antananarivo",
        "Colombo",
        "Almaty",
        "Sydney",
        "Chicago",
        "Mexico City",
        "Seattle",
        "Lagos",
        "Amsterdam",
        "Belgrade",
        "Cairo",
        "Baghdad",
        "Damascus",
        "Kigali",
        "Dakar",
        "Dakar",
        "Sofia",
        "Kigali",
        "Victoria",
        "Tashkent",
        "Mumbai",
        "Barcelona",
        "Almaty",
        "Amman",
        "Toronto",
        "Bratislava",
        "Johannesburg",
        "Thimphu",
        "Bangkok",
        "Santiago",
        "Cairo",
        "San Francisco",
        "Lagos",
        "Amsterdam",
        "Paris",
        "Rabat",
        "Santiago",
        "Copenhagen",
        "Madrid",
        "Kigali",
        "Ho Chi Minh City",
        "Sarajevo",
        "Delhi",
        "Istanbul",
        "Ho Chi Minh City",
        "Khartoum",
        "Helsinki",
        "Doha",
        "Istanbul",
        "Kuala Lumpur",
        "Budapest",
        "Shanghai",
        "Moscow",
        "Los Angeles",
        "Oslo",
        "Johannesburg",
        "Berlin",
        "Bangalore",
        "Tokyo",
        "Melbourne",
        "Barcelona",
        "Chicago",
        "Port Louis",
        "Lisbon",
        "Nairobi",
        "Kampala",
        "Lima",
        "Maputo",
        "Vancouver",
        "Dubai",
        "Khartoum",
        "Jakarta",
        "Madrid",
        "Yerevan",
        "Beirut",
        "Athens",
        "Chicago",
        "Paris",
        "Bucharest",
        "Copenhagen",
        "Brussels",
        "Damascus",
        "Seattle",
        "Los Angeles",
        "Yerevan",
        "Victoria",
        "Tunis",
        "Astana",
        "Seoul",
        "Buenos Aires",
        "Bangkok",
        "Colombo",
        "Brussels",
        "Khartoum",
        "Doha",
        "San Francisco",
        "Vienna",
        "Jakarta",
    ]

    def __init__(
        self,
        config,
        retrieval_question="What is the special magic {} number?",
        results_version=1,
        rnd_number_digits=7,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_interval_type="linear",
        save_results=False,
        final_context_length_buffer=200,
        print_ongoing_status=True,
        **kwargs,
    ):
        haystack_file = config.haystack_file
        context_lengths_min = config.context_lengths_min
        context_lengths_max = config.context_lengths_max
        context_lengths_num_intervals = config.n_context_length_intervals
        document_depth_percent_intervals = config.n_document_depth_intervals

        self.config = config
        self.needle = "\nThe special magic {city} number is: {rnd_number}\n"
        if not haystack_file or not retrieval_question:
            raise ValueError(
                "Needle, haystack, and retrieval_question must be provided."
            )

        self.rnd_number_digits = rnd_number_digits
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.haystack_file = haystack_file
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        self.context_lengths = np.round(
            np.linspace(
                context_lengths_min,
                context_lengths_max,
                num=context_lengths_num_intervals,
                endpoint=True,
            )
        ).astype(int)
        if document_depth_percent_interval_type == "linear":
            self.document_depth_percents = np.round(
                np.linspace(
                    document_depth_percent_min,
                    document_depth_percent_max,
                    num=document_depth_percent_intervals,
                    endpoint=True,
                )
            ).astype(int)
        elif document_depth_percent_interval_type == "sigmoid":
            self.document_depth_percents = [
                self.logistic(x)
                for x in np.linspace(
                    document_depth_percent_min,
                    document_depth_percent_max,
                    document_depth_percent_intervals,
                )
            ]
        else:
            raise ValueError(
                f"Unsupported document_depth_percent_interval_type: {document_depth_percent_interval_type}"
            )

        if self.config.jobs is not None:
            start, end = self.config.jobs.split("-")
            print(self.context_lengths)
            self.context_lengths = self.context_lengths[int(start) : int(end)]
            print(self.context_lengths)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        minference_patch = MInference(
            self.config.attn_type,
            self.config.model_name,
            self.config.pattern_path,
            starting_layer=0,
            kv_cache_cpu=self.config.kv_cache_cpu,
            kv_cache_cpu_device=self.config.kv_cache_cpu_device,
            attn_kwargs=(
                {} if self.config.attn_type != "inf_llm" else {"dense_decoding": False}
            ),
            kv_type=self.config.kv_type,
        )
        if "vllm" in self.config.attn_type:
            #### use vllm implementation
            self.model = LLM(
                model=self.config.model_name,
                max_num_seqs=1,
                max_model_len=context_lengths_max,
                **kwargs,
            )
            self.generation_config = SamplingParams(temperature=0, max_tokens=64)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype="auto",
                device_map="cuda",
                trust_remote_code=config.trust_remote_code,
                _attn_implementation="flash_attention_2",
                **kwargs,
            )
            self.model = minference_patch(self.model)
            self.generation_config = GenerationConfig(
                max_new_tokens=32,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

    def generate_random_number(self, num_digits):
        lower_bound = 10 ** (num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def logistic(self, x, L=100, x0=50, k=0.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def read_context_files(self, n):
        max_context_length = max(self.context_lengths)
        contexts = []
        f = open(self.haystack_file, "r")
        for _ in range(n):
            context = ""
            toks = 0
            while toks < max_context_length:
                text = json.loads(f.readline())["text"]
                context += text
                toks += len(self.tokenizer.encode(text))
            contexts.append(context)
        return contexts

    def create_contexts(
        self,
        needle_rnd_number,
        insert_needle,
        random_city,
        trim_context,
        context_length,
        depth_percent,
        seed,
    ):
        needle = self.needle.format(city=random_city, rnd_number=needle_rnd_number)
        question = self.retrieval_question.format(random_city)
        if not insert_needle:
            needle = " "  # replace needle with a space
        context = self.insert_needle(
            needle, trim_context, depth_percent, context_length
        )
        results = {
            "context": context,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "needle": needle,
            "question": question,
            "insert_needle": insert_needle,
            "needle_rnd_number": needle_rnd_number,
            "seed": seed,
        }
        return results

    def insert_needle(self, needle, context, depth_percent, context_length):
        tokens_needle = self.tokenizer.encode(needle)
        tokens_context = self.tokenizer.encode(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.tokenizer.encode(".", add_special_tokens=False)

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.tokenizer.decode(tokens_new_context)
        return new_context

    def run_test(self):
        contexts = []
        template = self.OURS_TEMPLATE

        def _key_from_result(result):
            return (result["context_length"], result["depth_percent"], result["seed"])

        results = []
        full_contexts = self.read_context_files(self.config.n_rounds)
        full_tokens = [
            self.tokenizer.encode(full_context) for full_context in tqdm(full_contexts)
        ]

        start = time.time()
        for context_length in self.context_lengths:
            torch.cuda.empty_cache()
            trim_contexts = [
                self.tokenizer.decode(full_token[:context_length])
                for full_token in tqdm(full_tokens)
            ]
            contexts = []
            for depth_percent in self.document_depth_percents:
                for i in range(self.config.n_rounds):
                    random_city = random.choice(
                        LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES
                    )
                    insert_needle = True
                    needle_rnd_number = str(
                        self.generate_random_number(self.rnd_number_digits)
                    )
                    print("context length: " + str(context_length))
                    print("depth_percent : " + str(depth_percent))
                    context = self.create_contexts(
                        needle_rnd_number,
                        insert_needle,
                        random_city,
                        trim_contexts[i],
                        context_length,
                        depth_percent,
                        i,
                    )
                    contexts.append(context)

            for context in tqdm(contexts):
                prompt = template.format(
                    context=context["context"], question=context["question"]
                )
                if self.config.attn_type == "vllm":
                    outs = self.model.generate(prompt, self.generation_config)
                    out = outs[0].outputs[0].text
                else:
                    input_tensor = self.tokenizer(
                        prompt, return_tensors="pt", return_attention_mask=False
                    ).to(self.model.device)
                    with torch.no_grad():
                        outs = self.model.generate(
                            **input_tensor,
                            generation_config=self.generation_config,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    new_tokens = outs[0, input_tensor["input_ids"].shape[-1] :]
                    out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(
                    {
                        "context_length": context["context_length"],
                        "depth_percent": context["depth_percent"],
                        "response": out,
                        "answer": context["needle_rnd_number"],
                        "correct": context["needle_rnd_number"] in out,
                        "seed": context["seed"],
                    }
                )
            with open(self.config.output_file, "w") as f:
                json.dump(results, f)
        print("elapsed", time.time() - start)
        print("done")
        print(f"Saved results to {self.config.output_file}")

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%"
        )
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()
