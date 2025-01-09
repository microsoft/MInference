# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from transformers import AutoModelForCausalLM, AutoTokenizer

from minference import MInference

prompt = "Hello, my name is"

model_name = "microsoft/Phi-3-mini-128k-instruct"#"gradientai/Llama-3-8B-Instruct-262k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",
)

# Patch MInference Module
minference_patch = MInference("minference", model_name)
model = minference_patch(model)

batch_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**batch_inputs, max_length=10)
generated_text = tokenizer.decode(
    outputs[0][batch_inputs["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
