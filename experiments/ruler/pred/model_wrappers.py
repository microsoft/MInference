# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Dict, List, Optional

import requests
import torch

from minference import MInference


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
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        minfence_patch = MInference("minference", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)

        self.pipeline = None
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        torch.cuda.empty_cache()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, **self.generation_kwargs)
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


class Dilated1(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        minfence_patch = MInference("dilated1", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")


class InfLLM(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        minfence_patch = MInference("inf_llm", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
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


class Dilated2(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        minfence_patch = MInference("dilated2", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")


class YiStatic(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.config.static_pattern = True
        minfence_patch = MInference("minference", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")


class LlamaStatic(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.config.static_pattern = True
        minfence_patch = MInference("minference", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")


class MInferenceOP(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.config.static_pattern = True
        minfence_patch = MInference("minference", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")


class MInferenceOPYi(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.config.static_pattern = True
        minfence_patch = MInference("minference", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")


class OPYiHalfV2(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.config.static_pattern = True
        minfence_patch = MInference("minference", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")


class Streaming(MInferenceModel):
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.config.static_pattern = True
        minfence_patch = MInference("streaming", name_or_path, None, starting_layer=0)
        self.model = minfence_patch.patch_model(self.model)
        self.pipeline = None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")


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
