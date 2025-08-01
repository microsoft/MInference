import torch
from datasets import load_dataset
from dataclasses import dataclass
from typing import Sequence, Dict

import torch
import transformers
from torch.utils.data import Dataset, IterableDataset
import os
import re

def get_dataset(dataset_name, split="train", size=None):
    dataset = load_dataset("json", data_files=dataset_name, split=split)
    if size is not None:
        dataset = dataset.select(range(size))
    return dataset


class MultiplePasskeyRetrievalDataset(Dataset):
    PASSKEY_ALPHABET = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliett",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
    ]

    ORDINAL_NUMBERS = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
    ]

    def __init__(
        self,
        haystack_dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length=None,
        passkey_length=32,
        num_passkeys=10,
        needle="Remeber this sequence of words, it's the {ordinal_number} passkey to the vault: ",
        retrieval_question="Based on the content of the book, what is the {ordinal_number} passkey to the vault?\nPasskey: ",
        prompt1="<|im_start|> This is a very long story book: <book> ",
        prompt2=" </book>.\n\n",
        buffer_size=300,
        seperator="\n\n",
        min_depth_ratio=0.1,
        max_depth_ratio=0.9,
        context_lengths_num_intervals=20,
        depth_ratio_num_intervals=20,
        context_length_min=None,
        context_length_max=None,
        pad_to_multiple_of=16,
    ):
        super(MultiplePasskeyRetrievalDataset, self).__init__()

        self.tokenizer = tokenizer

        self.max_length = (
            max_length if max_length is not None else tokenizer.model_max_length
        )
        self.max_depth_ratio = max_depth_ratio
        self.min_depth_ratio = min_depth_ratio
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.depth_ratio_num_intervals = depth_ratio_num_intervals

        if context_length_min is None or context_length_max is None:
            self.context_length_min = self.context_length_max = self.max_length
        else:
            self.context_length_min = context_length_min
            self.context_length_max = context_length_max

        self.context_length_intervals = torch.linspace(
            self.context_length_min,
            self.context_length_max,
            context_lengths_num_intervals,
            dtype=torch.int,
        )

        self.depth_ratio_intervals = torch.linspace(
            min_depth_ratio, max_depth_ratio, depth_ratio_num_intervals
        )

        self.passkey_length = passkey_length

        self.num_passkeys = num_passkeys

        self.haystack = ""

        for sample in haystack_dataset["text"]:
            if self._get_token_nums(self.haystack) >= self.context_length_max:
                break
            self.haystack += sample

        self.haystack = self._trim(self.haystack, self.context_length_max)

        self.needle = needle
        self.needle_tokens_list = [
            self.tokenizer.encode(
                self.needle.format(ordinal_number=ordinal_number),
                add_special_tokens=False,
            )
            for ordinal_number in self.ORDINAL_NUMBERS[: self.num_passkeys]
        ]
        self.retrieval_question_tokens_list = [
            self.tokenizer.encode(
                retrieval_question.format(ordinal_number=ordinal_number),
                add_special_tokens=False,
            )
            for ordinal_number in self.ORDINAL_NUMBERS[: self.num_passkeys]
        ]

        self.haystack_tokens = self.tokenizer.encode(
            self.haystack, add_special_tokens=False
        )
        self.seperator_tokens = self.tokenizer.encode(
            seperator, add_special_tokens=False
        )
        self.prompt1_tokens = self.tokenizer.encode(prompt1, add_special_tokens=True)
        self.prompt2_tokens = self.tokenizer.encode(prompt2, add_special_tokens=False)

        passkey = self._generate_passkey()
        passkey_tokens = self.tokenizer.encode(passkey, add_special_tokens=False)
        needle_tokens = self.needle_tokens_list[0] + passkey_tokens

        other_input_len = (
            len(self.prompt1_tokens)
            + len(self.prompt2_tokens)
            + (
                len(self.seperator_tokens)
                + len(needle_tokens)
                + len(self.seperator_tokens)
                + len(self.retrieval_question_tokens_list[0])
                + len(passkey_tokens)
            )
            * self.num_passkeys
        )
        if (
            len(self.haystack_tokens) + other_input_len
            > self.context_length_max - buffer_size
        ):
            self.haystack_tokens = self.haystack_tokens[
                : self.context_length_max - buffer_size - other_input_len
            ]

    def _generate_passkey(self):
        random_seq = torch.randint(
            0, len(self.PASSKEY_ALPHABET), (self.passkey_length,)
        )
        passkey = " ".join([self.PASSKEY_ALPHABET[i] for i in random_seq])
        return passkey

    def __len__(self):
        return len(self.context_length_intervals)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        context_length = self.context_length_intervals[i]
        # randomly sample self.num_passkeys depth ratios in self.depth_ratio_intervals
        depth_ratios = (
            self.depth_ratio_intervals[
                torch.randperm(self.depth_ratio_num_intervals)[: self.num_passkeys]
            ]
            .sort()
            .values
        )
        passkey_tokens_list = [
            self.tokenizer.encode(self._generate_passkey(), add_special_tokens=False)
            for _ in range(self.num_passkeys)
        ]
        context = self._insert_needle(context_length, depth_ratios, passkey_tokens_list)
        return self._construct_input(context, passkey_tokens_list)

    def _trim(self, context, context_length):
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) > context_length:
            context = self.tokenizer.decode(tokens[:context_length])
        return context

    def _get_token_nums(self, context):
        return len(self.tokenizer.encode(context))

    def _insert_needle(self, context_length, depth_ratios, passkey_tokens_list):
        haystack_tokens = self.haystack_tokens[:context_length]

        context = []
        last_insertion_point = 0

        for i, (depth_ratio, passkey_tokens) in enumerate(
            zip(depth_ratios, passkey_tokens_list)
        ):
            insertion_point = int(len(haystack_tokens) * depth_ratio)

            needle_tokens = self.needle_tokens_list[i] + passkey_tokens

            context += (
                haystack_tokens[last_insertion_point:insertion_point]
                + self.seperator_tokens
                + needle_tokens
                + self.seperator_tokens
            )
            last_insertion_point = insertion_point

        context += haystack_tokens[last_insertion_point:]

        return context

    def _construct_input(self, context_tokens, passkey_tokens_list):
        qa_tokens = []
        for i, (passkey_tokens, retrieval_question_tokens) in enumerate(
            zip(passkey_tokens_list, self.retrieval_question_tokens_list)
        ):
            qa_tokens += (
                retrieval_question_tokens + passkey_tokens + self.seperator_tokens
            )

        context_tokens = self.prompt1_tokens + context_tokens

        # pad to multiple of 16
        if len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens) % 16 != 0:
            pad_len = (
                16
                - (len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens)) % 16
            )
            context_tokens += self.haystack_tokens[-pad_len:]

        context_tokens += self.prompt2_tokens

        input_ids = torch.tensor(context_tokens + qa_tokens)

        assert input_ids.size(0) % 16 == 0

        labels = torch.tensor([-100] * len(context_tokens) + qa_tokens)
        length_context = len(context_tokens)
        # self.tokenizer.encode(input_text, return_tensors="pt").shape[-1]
        # labels = torch.tensor([-100] * length_context + [0] * (input_ids.shape[-1] - length_context))

        return dict(input_ids=input_ids, labels=labels, length_context=length_context)
        # return dict(input_ids=input_ids, labels=labels)
    
@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, length_context = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "length_context")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        ret_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            length_context=length_context
        )
        for key in instances[0].keys():
            if key not in ret_dict:
                ret_dict[key] = torch.stack([instance[key] for instance in instances])
        return ret_dict


def get_supervised_dataloader(
    dataset, tokenizer, batch_size, num_workers=4, shuffle=True, sampler=None
):
    collator = DataCollator(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=None if sampler is not None else shuffle,
        sampler=sampler,
    )
    return dataloader

import uuid
import random
from tqdm import tqdm
import json
import wonderwords
from nltk.tokenize import sent_tokenize
import numpy as np


nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
words = sorted(list(set(words)))

DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

class PasskeyRetrievalDataset(Dataset):
    
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        uuid_length=128,
        context_lengths_num_intervals=200,
        context_length_min=None,
        context_length_max=None,
        num_needle_k={"essay": 1, "needle": 1},
        num_needle_v={"essay": [4], "needle": [1]},
        num_needle_q={"essay": 1, "needle": 1},
        random_seed=42,
        template="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
            Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n
            {context}\n
            What are all the special magic {type_needle_v} for {query} mentioned in the provided text?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            The special magic {type_needle_v} for {query} mentioned in the provided text are """,
        tokens_to_generate=128,
        type_needle_v={"essay": "numbers", "needle": "uuids"},
        type_needle_k={"essay": "words", "needle": "uuids"},
        type_haystack=["needle", "essay"],
    ):
        super(PasskeyRetrievalDataset, self).__init__()

        self.tokenizer = tokenizer
        self.uuid_length = uuid_length
        self.num_needle_k = num_needle_k
        self.num_needle_v = num_needle_v
        self.num_needle_q = num_needle_q
        self.random_seed = random_seed
        self.template = template
        self.type_needle_v = type_needle_v
        self.type_needle_k = type_needle_k
        self.type_haystack = type_haystack

        self.context_length_min = context_length_min
        self.context_length_max = context_length_max

        self.context_length_intervals = torch.linspace(
            self.context_length_min,
            self.context_length_max,
            context_lengths_num_intervals,
            dtype=torch.int,
        )
        
        self.uuid_length = uuid_length
        self.haystack = {}
        needle = "One of the special magic {type_needle_v} for {key} is: {value}."
        if 'essay' in type_haystack:
            essay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/PaulGrahamEssays.json")
            essay = json.load(open(essay))['text']
            self.haystack['essay'] = re.sub(r'\s+', " ", essay).split(" ")
        if 'repeat' in type_haystack:
            self.haystack['repeat'] = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        if 'needle' in type_haystack:
            self.haystack['needle'] = needle
        self.needle = needle
    
        self.tokens_to_generate = tokens_to_generate

        self.incremental = {}

        if 'essay' in type_haystack:
            self.incremental['essay'] = 500
        if 'repeat' in type_haystack:
            self.incremental['repeat'] = 25
        if 'needle' in type_haystack:
            self.incremental['needle'] = 25
    
        self.num_haystack = {}

        for type in type_haystack: 
            num_haystack = self.incremental[type]
            total_tokens = 0 
            while total_tokens + tokens_to_generate < context_length_max :  
                input_text, answer = self.generate_input_output(num_haystack, template, type)
                total_tokens = self.tokenizer.encode(input_text + ' '.join(answer), return_tensors="pt").shape[-1]
                print(f'Max length {context_length_max} | Current length {total_tokens + tokens_to_generate} | Haystack: {num_haystack}')
                if total_tokens + tokens_to_generate > context_length_max:
                    num_haystack -= self.incremental[type]
                    break
                
                num_haystack += self.incremental[type]

            self.num_haystack[type] = num_haystack
            print('Num haystack of', type, ':', num_haystack)
        
    def __len__(self):
        return len(self.context_length_intervals) 
    
    def generate_random(self):
        return str(uuid.UUID(int=random.getrandbits(self.uuid_length), version=4))

    def generate_random_number(self, num_digits=7):
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        return str(random.randint(lower_bound, upper_bound))
    
    def generate_random_word(self):
        word = random.choice(words)
        return word
    
    def generate_random_uuid(self):
        return str(uuid.UUID(int=random.getrandbits(128), version=4))
    
    def generate_random(self, type_needle: str):
        if type_needle == 'numbers':
            return self.generate_random_number()
        elif type_needle == 'words':
            return self.generate_random_word()
        elif type_needle == 'uuids':
            return self.generate_random_uuid()
        else:
            raise NotImplementedError(f'{type_needle} is not implemented.')

    
    def generate_input_output(self, num_haystack, template, type):
        keys, values, needles = [], [], []
        for _ in range(self.num_needle_k[type]):
            keys.append(self.generate_random(self.type_needle_k[type]))
            value = []
            for _ in range(random.choice(self.num_needle_v[type])):
                value.append(self.generate_random(self.type_needle_v[type]))
                needles.append(self.needle.format(
                    type_needle_v=self.type_needle_v[type],
                    key=keys[-1], 
                    value=value[-1],
                ))
            values.append(value)
        
        random.Random(self.random_seed).shuffle(needles)

        if type == 'essay':
            text = " ".join(self.haystack[type][:num_haystack])
            document_sents = sent_tokenize(text.strip())
            insertion_positions = [0] + \
                                  sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(needles))]) + \
                                  [len(document_sents)]
            document_sents_list = []
            for i in range(1,len(insertion_positions)):
                last_pos = insertion_positions[i-1]
                next_pos = insertion_positions[i]
                document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
                if i-1 < len(needles):
                    document_sents_list.append(needles[i-1])
            context = " ".join(document_sents_list)
    
        else:
            if type == 'repeat':
                sentences = [self.haystack[type]] * num_haystack
            elif type == 'needle':
                sentences = [self.haystack[type].format(
                    type_needle_v=self.type_needle_v[type],
                    key=self.generate_random(self.type_needle_k[type]),
                    value=self.generate_random(self.type_needle_v[type]),
                ) for _ in range(num_haystack)]

            indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
            for index, element in zip(indexes, needles):
                sentences.insert(index, element)
            context = "\n".join(sentences)
        
        indices = random.sample(range(self.num_needle_k[type]), self.num_needle_q[type])
        queries = [keys[i] for i in indices]
        answers = [a for i in indices for a in values[i]]
        query = ', '.join(queries[:-1]) + ', and ' + queries[-1] if len(queries) > 1 else queries[0]

        type_needle_v = self.type_needle_v[type]
        if self.num_needle_q[type] * self.num_needle_v[type] == 1:
            template = template.replace('Some', 'A')
            template = template.replace('are all', 'is')
            template = template.replace('are', 'is')
            template = template.replace('answers', 'answer')
            type_needle_v = type_needle_v[:-1] # remove "s"

        input_text = template.format(
            type_needle_v=type_needle_v,
            context=context,
            query=query,
        )

        return input_text, answers

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        context_length = self.context_length_intervals[i]
        type = self.type_haystack[i % len(self.type_haystack)]
        used_haystack = self.num_haystack[type]
        while(True):
            try:
                input_text, answer = self.generate_input_output(used_haystack, self.template, type)
                length = self.tokenizer.encode(input_text, return_tensors="pt").shape[-1] + self.tokens_to_generate
                assert length <= context_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_haystack > self.incremental[type]:
                    used_haystack -= self.incremental[type]


        input_ids = self.tokenizer.encode(input_text + answer[0], return_tensors="pt")
        length_context = self.tokenizer.encode(input_text, return_tensors="pt").shape[-1]
        labels = torch.tensor([-100] * length_context + [0] * (input_ids.shape[-1] - length_context))

        return dict(input_ids=input_ids[0], labels=labels, length_context=length_context)
        
