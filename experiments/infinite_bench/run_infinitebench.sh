# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

TASKS=("kv_retrieval" "longbook_choice_eng" "math_find" "longbook_qa_chn" "longbook_qa_eng" "longdialogue_qa_eng" "code_debug" "longbook_sum_eng" "number_string" "passkey")

export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR=$(dirname "$0")

for task in ${TASKS[@]}; do
echo $task
python "$SCRIPT_DIR/run_infinitebench.py" \
    --task $task \
    --model_name_or_path ${1} \
    --data_dir ./data \
    --output_dir ./results \
    --max_seq_length $2 \
    --rewrite \
    --num_eval_examples $3 --topk 1 --starting_layer 0 --attn_type $4
done

# bash run_infinitebench.sh gradientai/Llama-3-8B-Instruct-262k 160000 -1 minference
