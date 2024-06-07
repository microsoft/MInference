## MInference Benchmark Experiments

Note: All experiments were run on a single A100 GPU with 80GB of VRAM.

Environment parameters:
- CUDA 12.3
- Triton 2.1.0
- PyCuda 2023.1

### End-to-End Benchmark

To demonstrate the efficiency of our method, we conducted end-to-end latency tests using the [LLaMA-3-8B-Instruct-1M](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) model. The prompts were trimmed to different target token numbers, and we measured the pre-filling stage latency without using KV cache.

1. Download the prompt:

```bash
wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
```

2. Run a single context window size test using one method:

```bash
python experiments/benchmarks/benchmark_e2e.py --attn_type hf --context_window 1000000
```

3. Run all latency experiments using different methods:

```bash
python experiments/benchmarks/benchmark_e2e.py --run_benchmark
```

4. After that, you should get the end-to-end latency results like this:

```json
        FlashAttention-2 StreamingLLM   InfLLM          MInference
1K      0.54565         1.07110         2.94495         2.96450
10K     0.97590         1.18339         2.21052         2.77618
50K     8.52933         5.47972         14.63624        7.54537
100K    24.88319        10.86379        27.67215        13.98508
200K    79.39184        21.61490        55.64703        26.81303
300K    169.62441       32.44844        80.74326        41.09374
500K    456.78353       54.15910        167.91472       66.27691
1000K   1765.56387      107.85639       328.58551       179.12031
```

### Micro-Benchmark


## MInference Downstream Tasks Experiments

Note: All of these experiments were run on one A100 GPUs with 80GB of VRAM. You may need to modify commands to fit your own computing environment (e.g., changing the batch size, the max memory per GPU, the number of GPUs, etc)

### InfiniteBench

### RULER

### PPL

### Needle in A Haystack
