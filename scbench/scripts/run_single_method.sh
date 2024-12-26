# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

TASKS=("scbench_kv" "scbench_prefix_suffix" "scbench_vt" "scbench_repoqa" "scbench_qa_eng" "scbench_qa_chn" "scbench_choice_eng"  "scbench_many_shot" "scbench_summary" "scbench_mf" "scbench_summary_with_needles" "scbench_repoqa_and_kv")

# attn_type, kv_type
# ATTN_KV_TYPES=(
#     "vllm;dense" # FullAttention
#     "vllm_minference;dense" "vllm_a_shape;dense" "vllm_tri_shape;dense" # 1) KV Cache Generation Stage
#     "dense;streamingllm" "dense;snapkv" "dense;pyramidkv" "dense;kivi" # 2) KV Cache Compression Stage
#     "vllm_blend;dense" # 3) KV Cache Retrieval Stage
#     "dense;quest" "dense;retr_attn" # 4) KV Cache Loading Stage
# )
attn_type=$4
kv_type=$5

MODE=$3
if [ "$MODE" == "scdq" ]; then
    MODE="--same_context_different_query"
else
    MODE=""
fi

echo "attn_type: $attn_type, kv_type: $kv_type"
for task in ${TASKS[@]}; do
echo $task
python run_scbench.py \
    --task $task \
    --model_name_or_path $1 \
    --data_dir ./data \
    --output_dir ./results \
    --attn_type $attn_type \
    --kv_type $kv_type \
    --use_chat_template \
    --trust_remote_code \
    --max_seq_length 131_072 \
    --tensor_parallel_size $2 ${MODE}
done

# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_single_method.sh meta-llama/Llama-3.1-8B-Instruct 1 multi-turn vllm dense
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_single_method.sh meta-llama/Llama-3.1-8B-Instruct 2 multi-turn vllm dense
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_single_method.sh meta-llama/Llama-3.1-8B-Instruct 2 scdq vllm dense
