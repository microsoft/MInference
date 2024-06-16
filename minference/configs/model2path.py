# Copyright (c) 2024 Microsoft
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
    "01-ai/Yi-9B-200K": os.path.join(
        BASE_DIR, "Yi_9B_200k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "microsoft/Phi-3-mini-128k-instruct": os.path.join(
        BASE_DIR, "Phi_3_mini_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
    ),
}
