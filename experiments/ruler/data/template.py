# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

Templates = {
    "base": "{task_template}",
    "meta-chat": "[INST] {task_template} [/INST]",
    "vicuna-chat": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {task_template} ASSISTANT:",
    "lwm-chat": "You are a helpful assistant. USER: {task_template} ASSISTANT: ",
    "command-r-chat": "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{task_template}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "chatglm-chat": "[gMASK]sop<|user|> \n {task_template}<|assistant|> \n ",
    "RWKV": "User: hi\n\nAssistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it\n\nUser: {task_template}\n\nAssistant:",
    "llama-3": "<|start_header_id|>user<|end_header_id|> {task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "glm-4": "[gMASK]<sop><|user|>\n {task_template}<|assistant|>\n",
}
