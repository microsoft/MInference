#!/usr/bin/bash

# ------------------------------------------
# Download data
# Prerequisite: sudo apt-get install git-lfs && git lfs install
RAW_DATASET_DIR="/path/to/datasets"
git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-512K $RAW_DATASET_DIR/long-context-524288
cd $RAW_DATASET_DIR/long-context-524288
git lfs fetch
git lfs checkout


# ------------------------------------------
# Data Processing
cd /path/to/mtraining
MODEL_ID="Qwen/Qwen2.5-3B"
PROCESSED_DATA_DIR="/path/to/processed_dataset"
torchrun --nproc_per_node=4\
	utils/data_utils/prolong.py \
    --model_id $MODEL_ID \
    --dataset_mix fixed_524288 \
    --dataset_path $RAW_DATASET_DIR/long-context-524288 \
    --save_path $PROCESSED_DATA_DIR
