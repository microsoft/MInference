import os
import logging
import argparse

from tqdm import tqdm
from typing import Dict
from streaming import StreamingDataset
from transformers import PreTrainedTokenizer
from datasets import Dataset, concatenate_datasets

from mtraining.utils.general import get_tokenizer
# ------------------------------------------------

logger = logging.getLogger(__name__)

LLAMA3_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
LLAMA_TOKENZIER = None

def tokenize(sample: Dict[str, str], tokenizer: PreTrainedTokenizer, seq_len: int=524288):
    text = sample['text']
    for token_k, token_v in LLAMA_TOKENZIER.special_tokens_map.items():
        if token_k in tokenizer.special_tokens_map:
            text = text.replace(token_v, tokenizer.special_tokens_map[token_k])

    input_ids = tokenizer.encode(
        text, 
        add_special_tokens=False,
        truncation=True,
        max_length=seq_len
    )
    return {"input_ids": input_ids, "length": len(input_ids)}   


DOMAINS = [
    "thestackv1_concat_by_repo-524288@0.15",
    "thestackv1_concat_by_repo-65536@0.15",
    "book-524288@0.05",
    "book-65536@0.25",
    "fineweb-edu@0.1",
    "fineweb-2023-50@0.1",
    "stackexchange@0.04",
    "dolmawiki@0.04",
    "tuluv2@0.03",
    "arxiv@0.03",
    "openwebmath@0.03",
    "textbooks@0.03",
]
FIXED_512K = [
    "thestackv1_concat_by_repo-524288",
    "book-524288"
]

DOMAIN_MIX_DICT = {
    "full": DOMAINS,
    "fixed_524288": FIXED_512K
}

def main(args):
    global LLAMA_TOKENZIER

    seq_len = args.sequence_length
    LLAMA_TOKENZIER = get_tokenizer(LLAMA3_MODEL_ID)  
    model_tokenizer = get_tokenizer(args.model_id)
    if model_tokenizer.bos_token is None:
        model_tokenizer.bos_token = "<|endoftext|>"

    domains = DOMAIN_MIX_DICT[args.dataset_mix]
    dataset_paths = [os.path.join(args.dataset_path, domain) for domain in domains]
    tokenized_datasets = []
    for idx, dataset_path in enumerate(dataset_paths):
        print('-' * 50)
        print(f"Processing {domains[idx]} from {dataset_path}...")
        texts = []
        dataset = StreamingDataset(
            local=dataset_path,
            remote=None,
            shuffle=False,
            batch_size=1,
        )

        for ix, sample in tqdm(enumerate(dataset)):
            if ix % args.sample_interval != 0: continue

            sample_input_ids = sample["input_ids"]
            sample_splits = [sample_input_ids[i:i+seq_len] for i in range(0, len(sample_input_ids), seq_len)]
            for sample_split in sample_splits:
                # De-tokenization
                text = LLAMA_TOKENZIER.decode(sample_split)
                texts.append(text)

        hf_dataset = Dataset.from_dict(
            {
                "text": texts
            }
        )

        tokenized_dataset = hf_dataset.map(
            tokenize,
            remove_columns=hf_dataset.column_names,
            num_proc=64,
            fn_kwargs={'tokenizer': model_tokenizer, 'seq_len': seq_len}
        )
        tokenized_datasets.append(tokenized_dataset)
    
    print('-' * 50)
    print(f"Concatenating and Saving tokenized datasets to {args.save_path}...")
    concat_dataset = concatenate_datasets(tokenized_datasets)
    filtered_concat_dataset = concat_dataset.filter(lambda x: x['length'] == seq_len, num_proc=128)
    filtered_concat_dataset.save_to_disk(args.save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument('--dataset_mix', type=str, default="fixed_524288")
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--sequence_length', type=int, default=524288)
    parser.add_argument("--sample_interval", type=int, default=1)
    args = parser.parse_args()

    main(args)
