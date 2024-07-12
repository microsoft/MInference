# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from minference import MInference


class MInferenceE2ETester(unittest.TestCase):
    """
    End2end Test for MInference
    """

    def __init__(self, *args, **kwargs):
        super(MInferenceE2ETester, self).__init__(*args, **kwargs)

        # paramaters
        model_name = "gradientai/Llama-3-8B-Instruct-262k"
        trust_remote_code = False
        attn_type = "minference"
        kv_cache_cpu = True
        self.attn_type = attn_type

        # init model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        attn_kwargs = {}
        minference_patch = MInference(
            attn_type,
            model_name,
            kv_cache_cpu=kv_cache_cpu,
            attn_kwargs=attn_kwargs,
        )
        self.model = minference_patch.patch_model(model)

        self.prompt_complex = open("./prompt_hardest.txt").read()

    def test_general_minference(self):
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
                if self.attn_type != "inf_llm":
                    self.model(input_ids, attention_mask, use_cache=False)
                else:
                    self.model.generate(
                        input_ids, generation_config=GenerationConfig(max_new_tokens=1)
                    )

        test_different_context_windows(100000)
        test_different_context_windows(1000000)
