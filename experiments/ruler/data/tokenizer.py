# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]


import os
from typing import List

from tenacity import retry, stop_after_attempt, wait_fixed, wait_random


def select_tokenizer(tokenizer_type, tokenizer_path):
    if tokenizer_type == "nemo":
        return NeMoSentencePieceTokenizer(model_path=tokenizer_path)
    elif tokenizer_type == "hf":
        return HFTokenizer(model_path=tokenizer_path)
    elif tokenizer_type == "openai":
        return OpenAITokenizer(model_path=tokenizer_path)
    elif tokenizer_type == "gemini":
        return GeminiTokenizer(model_path=tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer_type {tokenizer_type}")


class NeMoSentencePieceTokenizer:
    """
    Tokenizer from NeMo SentencePieceTokenizer
    """

    def __init__(self, model_path) -> None:
        from nemo.collections.common.tokenizers.sentencepiece_tokenizer import (
            SentencePieceTokenizer,
        )

        self.tokenizer = SentencePieceTokenizer(model_path=model_path)

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.text_to_tokens(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.tokens_to_text(tokens)
        return text


class HFTokenizer:
    """
    Tokenizer from HF models
    """

    def __init__(self, model_path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text


class OpenAITokenizer:
    """
    Tokenizer from tiktoken
    """

    def __init__(self, model_path="cl100k_base") -> None:
        import tiktoken

        self.tokenizer = tiktoken.get_encoding(model_path)

    def text_to_tokens(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.decode(tokens)
        return text


class GeminiTokenizer:
    """
    Tokenizer from gemini
    """

    def __init__(self, model_path="gemini-1.5-pro-latest") -> None:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(model_path)

    @retry(wait=wait_fixed(60) + wait_random(0, 10), stop=stop_after_attempt(3))
    def text_to_tokens(self, text: str) -> List[int]:
        tokens = list(range(self.model.count_tokens(text).total_tokens))
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        pass
