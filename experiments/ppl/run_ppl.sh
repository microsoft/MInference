mkdir -p results/long-ppl/
python3.9 experiments/ppl/run_ppl.py \
    --model_name gradientai/Llama-3-8B-Instruct-262k \
    --attn_type minference \
    --intervals 19 \
    --num_eval_examples 50 \
    --output_path results/long-ppl/
