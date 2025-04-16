# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from minference import MInference

ATTN_TYPES = ["dense", "a_shape", "tri_shape", "minference", "flexprefill"]
KV_TYPES = [
    "dense",
    "snapkv",
    "pyramidkv",
    "quest",
    "streamingllm",
    "retr_attn",
    "kivi",
]


class MInferenceE2ETester(unittest.TestCase):
    """
    End2end Test for MInference
    """

    @classmethod
    def setUpClass(cls):
        # paramaters
        cls.model_name = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
        # cls.model_name = "Qwen/Qwen2.5-7B-Instruct"
        trust_remote_code = True

        # init model and tokenizer
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, trust_remote_code=trust_remote_code
        )

        cls.prompt_complex = open("./prompt_hardest.txt").read()

    def forward(self, attn_type: str, kv_type: str, attn_kwargs: dict):
        def load_type():
            minference_patch = MInference(
                attn_type=attn_type,
                model_name=self.model_name,
                kv_type=kv_type,
                attn_kwargs=attn_kwargs,
            )
            return minference_patch.patch_model(self.model)

        def test_different_context_windows(seq_len: int):
            input_ids = self.tokenizer(self.prompt_complex)["input_ids"]
            n = len(input_ids)
            b = seq_len // n + 1

            new_input_ids = (input_ids * b)[:seq_len]
            prompt = self.tokenizer.decode(new_input_ids)
            data = self.tokenizer(prompt, return_tensors="pt")
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()

            with torch.no_grad():
                if attn_type != "inf_llm":
                    model(
                        input_ids,
                        attention_mask,
                        use_cache=False,
                        num_logits_to_keep=1,
                    )
                else:
                    model.generate(
                        input_ids, generation_config=GenerationConfig(max_new_tokens=1)
                    )
            torch.cuda.empty_cache()

        model = load_type()
        test_different_context_windows(100_000)
        # test_different_context_windows(1000000)
        del model
        torch.cuda.empty_cache()

    def test_dense(self):
        self.forward("dense", "dense", {})

    def test_minference(self):
        attn_kwargs = {}
        for kv_type in KV_TYPES:
            with self.subTest(attn_type="minference", kv_type=kv_type):
                self.forward("minference", kv_type, attn_kwargs)

    def test_all_kv_types(self):
        attn_kwargs = {}
        for kv_type in KV_TYPES:
            with self.subTest(attn_type="dense", kv_type=kv_type):
                self.forward("dense", kv_type, attn_kwargs)

    def test_all_attn_types(self):
        attn_kwargs = {}
        for attn_type in ATTN_TYPES:
            for kv_type in ["dense"]:
                with self.subTest(attn_type=attn_type, kv_type=kv_type):
                    self.forward(attn_type, kv_type, attn_kwargs)
