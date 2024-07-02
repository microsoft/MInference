# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export TOKENIZERS_PARALLELISM=false

mkdir -p results/long-ppl/
python3.9 experiments/ppl/run_ppl.py \
    --model_name gradientai/Llama-3-8B-Instruct-262k \
    --attn_type minference \
    --intervals 19 \
    --num_eval_examples 500 \
    --output_path results/long-ppl/
