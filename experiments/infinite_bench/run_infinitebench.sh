TASKS=("kv_retrieval" "longbook_choice_eng" "math_find" "longbook_qa_chn" "longbook_qa_eng" "longdialogue_qa_eng" "code_debug" "longbook_sum_eng" "number_string" "passkey")

for task in ${TASKS[@]}; do
echo $task
python run_infinitebench.py \
    --task $task \
    --model_name_or_path ${1} \
    --data_dir ./data \
    --output_dir ./results \
    --max_seq_length $2 \
    --rewrite \
    --num_eval_examples $3 --topk 1 --starting_layer 0 --attn_type $5
done

# bash run_infinitebench.sh gradientai/Llama-3-8B-Instruct-262k 160000 -1 minference
