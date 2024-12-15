# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

python run_scbench.py \
    --task scbench_kv \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --data_dir ./data \
    --output_dir ./results \
    --rewrite \
    --attn_type minference \
    --kv_type snapkv \
    --use_chat_template \
    --trust_remote_code
