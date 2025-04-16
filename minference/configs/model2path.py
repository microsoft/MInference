# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL2PATH = {
    "gradientai/Llama-3-8B-Instruct-262k": os.path.join(
        BASE_DIR, "Llama_3_8B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "gradientai/Llama-3-8B-Instruct-Gradient-1048k": os.path.join(
        BASE_DIR, "Llama_3_8B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "gradientai/Llama-3-8B-Instruct-Gradient-4194k": os.path.join(
        BASE_DIR, "Llama_3_8B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "01-ai/Yi-9B-200K": os.path.join(
        BASE_DIR, "Yi_9B_200k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "microsoft/Phi-3-mini-128k-instruct": os.path.join(
        BASE_DIR, "Phi_3_mini_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
    ),
    "Qwen/Qwen2-7B-Instruct": os.path.join(
        BASE_DIR, "Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
    ),
    "Qwen/Qwen2.5-7B-Instruct": os.path.join(
        BASE_DIR,
        "Qwen2.5_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json",
    ),
    "Qwen/Qwen2.5-32B-Instruct": os.path.join(
        BASE_DIR,
        "Qwen2.5_32B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json",
    ),
    "Qwen/Qwen2.5-72B-Instruct": os.path.join(
        BASE_DIR,
        "Qwen2.5_72B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json",
    ),
    "Qwen/Qwen2.5-7B-Instruct-1M": os.path.join(
        BASE_DIR, "Qwen2.5_7B_Instruct_1M.json"
    ),
    "Qwen/Qwen2.5-14B-Instruct-1M": os.path.join(
        BASE_DIR, "Qwen2.5_14B_Instruct_1M.json"
    ),
    "THUDM/glm-4-9b-chat-1m": os.path.join(
        BASE_DIR, "GLM_4_9B_1M_instruct_kv_out_v32_fit_o_best_pattern.json"
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": os.path.join(
        BASE_DIR, "Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern_v2.json"
    ),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": os.path.join(
        BASE_DIR, "Llama_3.1_70B_Instruct_128k_kv_out_v32_fit_o_best_pattern_v2.json"
    ),
    "meta-llama/Llama-3.1-8B-Instruct": os.path.join(
        BASE_DIR, "Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern_v2.json"
    ),
    "meta-llama/Llama-3.1-70B-Instruct": os.path.join(
        BASE_DIR, "Llama_3.1_70B_Instruct_128k_kv_out_v32_fit_o_best_pattern_v2.json"
    ),
    "gradientai/Llama-3-70B-Instruct-Gradient-262k": os.path.join(
        BASE_DIR, "Llama_3_70B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "gradientai/Llama-3-70B-Instruct-Gradient-1048k": os.path.join(
        BASE_DIR, "Llama_3_70B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
}


def get_support_models():
    return list(MODEL2PATH.keys())


def check_path():
    for name, path in MODEL2PATH.items():
        assert os.path.exists(path), f"{name} Config does not exist! Please check it."


if __name__ == "__main__":
    check_path()
