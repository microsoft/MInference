# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export TOKENIZERS_PARALLELISM=false

# Load Haystack
mkdir -p data
wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl -O ./data/pg19_mini.jsonl

# Run the Needle in A Haystack Test
python experiments/needle_in_a_haystack/needle_test.py \
    --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --max_length 1000000 \
    --min_length 1000 \
    --rounds 5 \
    --attn_type minference \
    --output_path ./needle \
    --run_name minference_LLaMA_1M \
    --jobs 0-4

python experiments/needle_in_a_haystack/needle_test.py \
    --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --max_length 1000000 \
    --min_length 1000 \
    --rounds 5 \
    --attn_type minference \
    --kv_cache_cpu \
    --output_path ./needle \
    --run_name minference_LLaMA_1M \
    --jobs 4-15

# Data Summary
python experiments/needle_in_a_haystack/needle_summary.py --output_path ./needle --run_name minference_LLaMA_1M

# Visualization
mkdir -p figures
python experiments/needle_in_a_haystack/needle_viz.py --res_file ./needle/minference_LLaMA_1M.json --model_name LLaMA-3-8B-Instruct-1M --mode ours
