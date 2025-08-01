# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from transformers import AutoModelForCausalLM, AutoTokenizer
from minference import MInference

prompt = open("narrativeqa_example.txt").read()

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",
    _attn_implementation="flash_attention_2",
)

# Patch model with dense prefill and leank decoding
minference_patch = MInference(
    attn_type="dense", model_name=model_name, kv_type="leank"
)
model = minference_patch(model)

batch_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print("Input sequence length", batch_inputs.input_ids.shape[1])
outputs = model.generate(**batch_inputs, max_new_tokens=512)
generated_text = tokenizer.decode(
    outputs[0][batch_inputs["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"Generated text: {generated_text!r}")
