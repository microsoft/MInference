# Load data
wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt

python experiments/benchmarks/benchmark_e2e.py \
    --attn_type minference \
    --context_window 300000
