# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import logging
from typing import Dict, List, Optional

import requests
import torch


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )

        if "Yarn-Llama" in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
            )

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, **self.generation_kwargs)
            generated_text = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
        else:
            output = self.pipeline(
                text_inputs=prompt,
                **self.generation_kwargs,
            )
            assert len(output) == 1
            generated_text = output[0]["generated_text"]

        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}


class MInferenceModel:
    def __init__(
        self,
        name_or_path: str,
        config_path: str,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        temperature: float = 0.0,
        top_k: int = 32,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        kv_cache_cpu_device: str = None,
        kv_type: str = "",
        trust_remote_code: bool = False,
        attn_type: str = "minference",
    ) -> None:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
        )

        from minference import MInference

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            trust_remote_code=trust_remote_code,
            resume_download=None,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="cuda",
            resume_download=None,
            trust_remote_code=trust_remote_code,
            _attn_implementation="flash_attention_2",
        )
        minference_patch = MInference(
            attn_type,
            name_or_path,
            config_path=config_path,
            starting_layer=starting_layer,
            kv_type=kv_type,
            kv_cache_cpu=kv_cache_cpu,
            kv_cache_cpu_device=kv_cache_cpu_device,
            is_search=False,
        )
        self.model = minference_patch(model)

        self.pipeline = None
        generation_config = GenerationConfig(
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        if do_sample:
            generation_config.top_k = top_k
            generation_config.top_p = top_p
            generation_config.temperature = temperature

        self.generation_config = generation_config

        self.stop = stop

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        torch.cuda.empty_cache()
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=False
        ).to(self.model.device)
        output = self.model.generate(**inputs, generation_config=self.generation_config)
        generated_text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}


class InfLLM(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from minference import MInference

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
            resume_download=None,
            trust_remote_code=True,
        )
        minference_patch = MInference("inf_llm", name_or_path, None, starting_layer=0)
        self.model = minference_patch.patch_model(self.model)
        self.pipeline = None
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=False
        ).to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_kwargs["max_new_tokens"],
        )
        generated_text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}


class Streaming(MInferenceModel):
    def __init__(
        self,
        name_or_path: str,
        config_path: str,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        temperature: float = 0.0,
        top_k: int = 32,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        kv_cache_cpu_device: str = None,
        kv_type: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
        )

        from minference import MInference

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            trust_remote_code=trust_remote_code,
            resume_download=None,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="cuda",
            resume_download=None,
            trust_remote_code=trust_remote_code,
            _attn_implementation="flash_attention_2",
        )
        minference_patch = MInference(
            "a_shape",
            name_or_path,
            config_path=config_path,
            starting_layer=starting_layer,
            kv_type=kv_type,
            kv_cache_cpu=kv_cache_cpu,
            kv_cache_cpu_device=kv_cache_cpu_device,
            is_search=False,
        )
        self.model = minference_patch(model)

        self.pipeline = None
        generation_config = GenerationConfig(
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        if do_sample:
            generation_config.top_k = top_k
            generation_config.top_p = top_p
            generation_config.temperature = temperature

        self.generation_config = generation_config

        self.stop = stop


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(
            name_or_path, device=self.device, dtype=torch.bfloat16
        )
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")
        self.max_genlen = self.generation_kwargs.pop("max_new_tokens")
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {"text": [self.tokenizer.decode(out.sequences[0][input_ids.shape[1] :])]}
