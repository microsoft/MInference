# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from vllm import LLM, SamplingParams

from minference import MInference

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=10,
)
model_name = "gradientai/Llama-3-8B-Instruct-262k"
llm = LLM(
    model_name,
    max_num_seqs=1,
    enforce_eager=True,
    max_model_len=128000,
)

# Patch MInference Module
minference_patch = MInference("vllm", model_name)
llm = minference_patch(llm)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
