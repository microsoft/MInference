# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

TASKS=("scbench_kv" "scbench_prefix_suffix" "scbench_vt" "scbench_repoqa" "scbench_qa_eng" "scbench_qa_chn" "scbench_choice_eng"  "scbench_many_shot" "scbench_summary" "scbench_mf" "scbench_summary_with_needles" "scbench_repoqa_and_kv")
ATTN_KV_TYPES=(
    "vllm;dense" # FullAttention
    "vllm_minference;dense" "vllm_a_shape;dense" "vllm_tri_shape;dense" # 1) KV Cache Generation Stage
    "dense;streamingllm" "dense;snapkv" "dense;pyramidkv" "dense;kivi" # 2) KV Cache Compression Stage
    "vllm_blend;dense" # 3) KV Cache Retrieval Stage
    "dense;quest" "dense;retr_attn" # 4) KV Cache Loading Stage
)

for attn_kv_type in ${ATTN_KV_TYPES[@]}; do
IFS=';' read -r attn_type kv_type <<< "$attn_kv_type"
echo "attn_type: $attn_type, kv_type: $kv_type"
for task in ${TASKS[@]}; do
echo $task
python run_scbench.py \
    --task $task \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --data_dir ./data \
    --output_dir ./results \
    --rewrite \
    --attn_type $attn_type \
    --kv_type $kv_type \
    --use_chat_template \
    --trust_remote_code
done
done
