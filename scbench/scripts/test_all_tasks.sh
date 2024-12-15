python run_scbench.py \
    --task scbench_choice_eng,scbench_qa_eng,scbench_qa_chn,scbench_kv,scbench_mf,scbench_repoqa,scbench_summary,scbench_vt,scbench_many_shot,scbench_summary_with_needles,scbench_repoqa_and_kv,scbench_prefix_suffix \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --data_dir ./data \
    --output_dir ./results \
    --rewrite \
    --attn_type dense \
    --kv_type dense \
    --use_chat_template \
    --trust_remote_code
