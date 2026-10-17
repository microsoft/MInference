<p align="center">
  <picture>
    <img alt="MInference" src="https://raw.githubusercontent.com/microsoft/MInference/main/images/MInference_logo.png" width=70%>
  </picture>
</p>

<h2 align="center">MInference：面向长上下文大语言模型的百万 Token 提示词极速预填充推理</h2>

<p align="center">
    | <a href="https://aka.ms/MInference"><b>项目主页</b></a> |
    <a href="https://arxiv.org/abs/2407.02490"><b>论文</b></a> |
    <a href="https://huggingface.co/spaces/microsoft/MInference"><b>HF Demo</b></a> |
    <a href="https://aka.ms/SCBench"><b>SCBench</b></a> |
    <a href="https://aka.ms/MMInference"><b>MMInference</b></a> |
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

https://github.com/microsoft/MInference/assets/30883354/52613efc-738f-4081-8367-7123c81d6b19

_现在，借助 LLaMA-3-8B-1M、GLM-4-1M 等长上下文大语言模型，您可以在**单张 A100 上以 10 倍速度处理 100 万（1M）Token 上下文**，甚至取得**更优的准确率**，立即体验 **MInference 1.0** 吧！_

## 📰 最新动态 (News)
- 🐝 [25/05/02] MMInference 已被 **ICML'25** 录用。
- 👨‍💻‍ [25/04/14] [SGLang](https://github.com/sgl-project/sglang/pull/5327) 与 [vLLM](https://github.com/vllm-project/flash-attention/pull/33) 已正式合入 MInference 稀疏注意力算子。_MInference 原生支持这些优化算子。_ 运行 `pip install sglang` 即可体验，可获得高达 **1.64× (64K)、2.4× (96K)、2.9× (128K)、5.2× (256K)、8× (512K) 以及 15× (1M)** 的加速。特别值得注意的是，SGLang 还针对 FlashAttention-3 进行了适配。特别感谢 @zhyncs 和 @yinfan98 的贡献！
- 👾 [25/04/23] 我们非常高兴地发布了多模态研究工作 [MMInference](https://aka.ms/2504.16083)，利用**模态感知重排稀疏注意力（Modality-Aware Permutation Sparse Attention）**加速长上下文多模态大模型（VLM）推理。我们将在 **ICLR'25** 的 **微软展台（Microsoft Booth）** 与 **FW-Wild** 进行展示，相约新加坡！
- 🤗 [25/01/27] MInference 已集成至 [Qwen2.5-1M](https://qwenlm.github.io/blog/qwen2.5-1m/) 及其线上服务。详情请参考[论文](https://arxiv.org/abs/2501.15383)与 [vLLM 实现](https://github.com/vllm-project/vllm/pull/11844)。
- 🪸 [25/01/23] SCBench 已被 **ICLR'25** 录用。
<details>
<summary>查看更多动态</summary>
 <ul>
  <li> 🍩 [24/12/13] 我们很高兴发布以 KV Cache 为核心的评测分析工作 <a href="https://aka.ms/SCBench">SCBench</a>，该工作从 KV 缓存全生命周期视角全面评估长上下文方法。</li>
  <li> 🧤 [24/09/26] MInference 被 <b>NeurIPS'24</b> 录用为 <b>Spotlight</b> 论文。相约温哥华！</li>
  <li> 👘 [24/09/16] 我们发布了 KV Cache 卸载研究工作 <a href="https://aka.ms/RetrievalAttention">RetrievalAttention</a>，通过向量检索显著加速长上下文 LLM 推理。</li>
  <li> 🥤 [24/07/24] MInference 现已支持 <a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct</a>。</li>
  <li> 🪗 [24/07/07] 感谢 @AK 赞助支持！您现在可以在包含 ZeroGPU 的 <a href="https://huggingface.co/spaces/microsoft/MInference">HF Demo</a> 中在线试用 MInference。</li>
  <li> 📃 [24/07/03] 论文现可通过该 <a href="https://export.arxiv.org/pdf/2407.02490">链接</a> 直接查阅。</li>
  <li> 🧩 [24/07/03] 我们将在 ICML'24 的 <b><i>微软展台</i></b> 与 <b><i>ES-FoMo</i></b> 研讨会展示 <b>MInference 1.0</b>。相约维也纳！</li>
</ul>
</details>

## 核心摘要 (TL;DR)

**MInference 1.0** 充分利用大语言模型注意力机制的动态稀疏特性（其在宏观上呈现出稳定的稀疏模式），以大幅加速长上下文 LLM 的预填充阶段（Pre-filling）。它首先在离线阶段确定各个注意力头所属的稀疏模式，随后在在线阶段近似求解稀疏索引，并通过最适配的定制化内核动态计算注意力。该方法在单张 A100 上实现预填充阶段高达 **10 倍的加速**，同时保持模型原本精度。

- [MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](https://arxiv.org/abs/2407.02490) (NeurIPS'24 **spotlight**, ES-FoMo @ ICML'24)<br>
  _Huiqiang Jiang†, Yucheng Li†, Chengruidong Zhang†, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang and Lili Qiu_

**SCBench** 从 **KV Cache 全生命周期视角**（涵盖 KV 缓存的生成、压缩、检索与加载）系统分析了各类长上下文优化方法。在两种共享上下文模式下评测了 12 项任务，覆盖字符串检索、语义检索、全局信息聚合与多任务场景四大类核心长上下文能力。

- [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](https://arxiv.org/abs/2412.10319) (ICLR'25, ENLSP @ NeurIPS'24)<br>
  _Yucheng Li, Huiqiang Jiang, Qianhui Wu, Xufang Luo, Surin Ahn, Chengruidong Zhang, Amir H. Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang and Lili Qiu_

**MMInference** 采用**模态感知重排稀疏注意力（Modality-Aware Permutation Sparse Attention）**技术，大幅加速长上下文多模态大模型在预填充阶段的推理。具体而言，我们结合 FlashAttention、FlashDecoding 和 PIT 实现了三种各具特色的基于重排的稀疏注意力机制，专门攻克视觉输入的网格模式与图文混合模态边界难题。

- [MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](https://arxiv.org/abs/2504.16083) (ICML'25, FM-Wild @ ICLR'25)<br>
  _Yucheng Li, Huiqiang Jiang, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Amir H. Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang and Lili Qiu_

## 🎥 架构总览 (Overview)

![Onepage of MInference](https://raw.githubusercontent.com/microsoft/MInference/main/images/MInference1_onepage.png)
![Onepage of SCBench](https://raw.githubusercontent.com/microsoft/MInference/main/images/SCBench_onepage.png)
![Onepage of MMInference](https://raw.githubusercontent.com/microsoft/MInference/main/images/MMInference_onepage.png)

## 🎯 快速上手 (Quick Start)

### 环境依赖 (Requirements)

- Torch
- FlashAttention-2（可选）
- Triton
- **Transformers >= 4.46.0**

使用 pip 安装 MInference：

```bash
pip install minference
```

### 支持的高效方法 (Supported Efficient Methods)

运行以下代码即可获取所有支持的高效方法列表：
```python
from minference import MInferenceConfig
supported_attn_types = MInferenceConfig.get_available_attn_types()
supported_kv_types = MInferenceConfig.get_available_kv_types()
```

目前支持的长上下文高效优化方法包括：

- **[① KV Cache 生成阶段]:** [MInference](https://arxiv.org/abs/2407.02490), [xAttention](https://arxiv.org/abs/2503.16428), [FlexPrefill](https://arxiv.org/abs/2502.20766), [A-shape](https://arxiv.org/abs/2309.17453), [Tri-shape](https://arxiv.org/abs/2412.10319), [MInference w/ static](https://arxiv.org/abs/2407.02490), [Dilated](https://arxiv.org/abs/2004.05150), [Strided](https://arxiv.org/abs/1904.10509)
- **[② KV Cache 压缩阶段]:** [StreamingLLM](https://arxiv.org/abs/2309.17453), [SnapKV](https://arxiv.org/abs/2404.14469), [PyramidKV](https://arxiv.org/abs/2406.02069), [KIVI](https://arxiv.org/abs/2402.02750)
- **[③ KV Cache 检索阶段]:** [CacheBlend](https://arxiv.org/abs/2405.16444)
- **[④ KV Cache 加载阶段]:** [Quest](https://arxiv.org/abs/2406.10774), [RetrievalAttention](https://arxiv.org/abs/2409.10516)

有关 KV Cache 生命周期的更多细节，请参阅 [**SCBench**](https://arxiv.org/abs/2412.10319)。请注意，部分模式受 vLLM 支持，而全部模式均由 Hugging Face 原生支持。

### 支持的模型系列 (Supported Models)

通用 *MInference* **支持任意自回归解码大语言模型**，包括 LLaMA 架构模型与 Phi 架构模型。
我们已经对市面上几乎所有主流开源长上下文 LLM 进行了适配。
如果您使用的模型不在已有支持列表中，欢迎在 Issues 中反馈，或参考[配置生成指南](https://github.com/microsoft/MInference/blob/main/experiments)手动生成稀疏头配置。

运行以下命令可获取受支持的完整模型清单：
```python
from minference import get_support_models
get_support_models()
```

目前支持的模型包括：
- Qwen2.5: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct), [Qwen/Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M), [Qwen/Qwen2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M)
- LLaMA-3.1: [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
- LLaMA-3: [gradientai/Llama-3-8B-Instruct-262k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k), [gradientai/Llama-3-8B-Instruct-Gradient-1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k), [gradientai/Llama-3-8B-Instruct-Gradient-4194k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-4194k), [gradientai/Llama-3-70B-Instruct-Gradient-262k](https://huggingface.co/gradientai/Llama-3-70B-Instruct-Gradient-262k), [gradientai/Llama-3-70B-Instruct-Gradient-1048k](https://huggingface.co/gradientai/Llama-3-70B-Instruct-Gradient-1048k)
- GLM-4: [THUDM/glm-4-9b-chat-1m](https://huggingface.co/THUDM/glm-4-9b-chat-1m)
- Yi: [01-ai/Yi-9B-200K](https://huggingface.co/01-ai/Yi-9B-200K)
- Phi-3: [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- Qwen2: [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

### 如何使用 MInference (How to use MInference)

> [!TIP]
> 为充分发挥高性能算子优势，建议安装 **SGLang** 或 **vLLM**。
> 对于 sglang：
> ```bash
> uv pip install "sglang[all]>=0.4.6.post4"
> ```
>
> 对于 vllm：
> ```bash
> uv pip install "vllm>=0.9.0"
> uv pip install git+https://github.com/vllm-project/flash-attention
> ```

对于 Hugging Face 原生推理：
```diff
from transformers import pipeline
+from minference import MInference

pipe = pipeline("text-generation", model=model_name, torch_dtype="auto", device_map="auto")

# 注入 MInference 模块补丁
# 若使用本地路径，初始化 MInference 时请使用对应的 Hugging Face 模型名称标识
+minference_patch = MInference("minference", model_name)
+pipe.model = minference_patch(pipe.model)

pipe(prompt, max_length=10)

# 使用稀疏 KV 缓存方法（如 snapkv, quest, retr_attn, kivi）
+minference_patch = MInference(attn_type="minference", model_name=model_name, kv_type="quest")
+pipe.model = minference_patch(pipe.model)

pipe(prompt, max_length=10)
```

对于 vLLM：
> 目前请使用 vllm>=0.4.1

```diff
from vllm import LLM, SamplingParams
+ from minference import MInference

llm = LLM(model_name, enforce_eager=True, max_model_len=128_000, enable_chunked_prefill=False)

# 注入 MInference 模块补丁
# 若使用本地路径，初始化 MInference 时请使用对应的 Hugging Face 模型名称标识
+minference_patch = MInference("vllm", model_name)
+llm = minference_patch(llm)

outputs = llm.generate(prompts, sampling_params)
```

对于带张量并行（Tensor Parallelism, TP）的 vLLM：

1. 将 `minference/patch.py` 中的 `minference_patch_vllm_tp` 和 `minference_patch_vllm_executor` 复制到 `vllm/worker/worker.py` 中 `Worker` 类的末尾。注意保持 `minference_patch_vllm_tp` 的代码缩进对齐。
2. 调用 vLLM 时，确保设置了 `enable_chunked_prefill=False`。
3. 参考运行脚本：https://github.com/microsoft/MInference/blob/main/experiments/benchmarks/run_e2e_vllm_tp.sh

```diff
from vllm import LLM, SamplingParams
+ from minference import MInference

llm = LLM(model_name, enforce_eager=True, max_model_len=128_000, enable_chunked_prefill=False, tensor_parallel_size=2)

# 注入 MInference 模块补丁
# 若使用本地路径，初始化 MInference 时请使用对应的 Hugging Face 模型名称标识
+minference_patch = MInference("vllm", model_name)
+llm = minference_patch(llm)

outputs = llm.generate(prompts, sampling_params)
```

仅单独调用底层稀疏注意力算子：
```python
from minference import vertical_slash_sparse_attention, block_sparse_attention, streaming_forward

attn_output = vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
attn_output = block_sparse_attention(q, k, v, topk)
attn_output = streaming_forward(q, k, v, init_num, local_window_num)
```

本地启动 Gradio 演示界面 <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>：

```bash
git clone https://huggingface.co/spaces/microsoft/MInference
cd MInference
pip install -r requirments.txt
pip install flash_attn
python app.py
```

更多细节请查阅 [Examples](https://github.com/microsoft/MInference/tree/main/examples) 与 [Experiments](https://github.com/microsoft/MInference/tree/main/experiments)。如需深入了解动态编译器 PIT，请参阅[论文](https://dl.acm.org/doi/10.1145/3600006.3613139)与 [GitHub 仓库](https://github.com/microsoft/SparTA/tree/pit_artifact)。

## SCBench 评测基准

> [!Note]
> - **datasets >= 2.15.0**

### 加载数据 (Load Data)
您可以通过 Hugging Face datasets 直接下载并加载 **SCBench** 数据集（[🤗 HF Repo](https://huggingface.co/datasets/microsoft/SCBench)）：
```python
from datasets import load_dataset

datasets = ["scbench_kv", "scbench_prefix_suffix", "scbench_vt", "scbench_repoqa", "scbench_qa_eng", "scbench_qa_chn", "scbench_choice_eng", "scbench_many_shot", "scbench_summary", "scbench_mf", "scbench_summary_with_needles", "scbench_repoqa_and_kv"]

for dataset in datasets:
    data = load_dataset("microsoft/SCBench", dataset, split="test")
```

### 数据格式 (Data Format)

**SCBench** 中的所有数据均已标准化为如下统一格式：

```json
{
    "id": "每条数据的随机唯一标识。",
    "context": "任务所需的超长上下文，例如代码库代码、长篇文档或多示例（many-shot）。",
    "multi_turns": [{"input": "多轮问题输入。", "answer": "多轮参考答案。"}],
}
```

### 评测实验 (Experiments)

我们在 [`GreedySearch`](https://github.com/microsoft/MInference/blob/yucheng/kvcompression/scbench/eval_utils.py#L1160) 与 [`GreedySearch_vllm`](https://github.com/microsoft/MInference/blob/yucheng/kvcompression/scbench/eval_utils.py#L1070) 两个类中结合 Hugging Face 与 vLLM 实现了**多轮对话（Multi-Turn）**与**多请求（Multi-Request）**两种评测模式。请参考以下脚本运行评测：

运行全部方法：
```bash
cd scbench
# 单 GPU，多轮对话模式
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_all_tasks.sh meta-llama/Llama-3.1-8B-Instruct 1 multi-turn
# 多 GPU，多轮对话模式
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_all_tasks.sh meta-llama/Llama-3.1-8B-Instruct 2 multi-turn
# 多 GPU，多请求模式
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_all_tasks.sh meta-llama/Llama-3.1-8B-Instruct 2 scdq
```

运行单个特定方法：
```bash
cd scbench
# 单 GPU，多轮对话模式，指定 attn_type: vllm, kv_type: dense
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_single_method.sh meta-llama/Llama-3.1-8B-Instruct 1 multi-turn vllm dense
# 多 GPU，多轮对话模式，指定 attn_type: vllm, kv_type: dense
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_single_method.sh meta-llama/Llama-3.1-8B-Instruct 2 multi-turn vllm dense
# 多 GPU，多请求模式，指定 attn_type: vllm, kv_type: dense
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/run_single_method.sh meta-llama/Llama-3.1-8B-Instruct 2 scdq vllm dense
```

有关 **attn_type** 与 **kv_type** 的参数详情，请参阅[支持的高效方法](https://github.com/microsoft/MInference/tree/main?tab=readme-ov-file#supported-efficient-methods)一节。

## 常见问题 (FAQ)

欲了解更多背景与技术细节，请参阅完整的 [FAQ 文档](https://github.com/microsoft/MInference/blob/main/Transparency_FAQ.md)。

**Q1：如何有效评测动态稀疏注意力对长上下文大模型能力的影响？**

为了系统评估基于 LLaMA-3-8B-Instruct-1M 和 GLM-4-9B-1M 等长上下文大模型的能力，我们评测了：1）利用 RULER 测试有效上下文窗口；2）利用 InfiniteBench 评测通用任务能力；3）利用大海捞针（Needle in a Haystack）测试信息检索能力；4）利用 PG-19 测试语言模型自身预测能力。<br/>
我们发现传统方法在精准检索任务中表现欠佳，其难度阶梯如下：<font color="#337ab7"><b>KV 检索 > 大海捞针 > 数字检索 (Retrieval.Number) > 密码锁检索 (Retrieval PassKey)</b></font>。核心挑战在于“目标针”与“背景干草堆”之间的语义差异程度。传统方法在此类差异极大时（如密码锁任务）表现良好；而 KV 检索对检索能力要求更高，因为任何 Key 都可能是检索目标，而多针检索任务的复杂度更进一步呈指数上升。<br/>
我们将在后续版本中持续加入更多模型与评测数据集的结果。

**Q2：这种动态稀疏注意力模式是否仅存在于未充分训练的长上下文 LLM 中？**

首先，注意力机制本身具有内在的动态稀疏性。我们选取了业内顶尖的长上下文大模型 GLM-4-9B-1M 与 LLaMA-3-8B-Instruct-1M，其实际有效上下文窗口分别为 64K 和 16K。而在接入 MInference 后，有效窗口分别显著拓展至 64K 与 32K。我们将继续将该方法适配至更多前沿长上下文模型并更新评测，同时持续探索此动态稀疏注意力模式的深层理论根基。

**Q3：这种动态稀疏注意力模式是否仅存在于自回归语言模型或基于 RoPE 的模型中？**

类似的垂直线（Vertical）与斜线（Slash）稀疏模式此前已在 BERT[1] 以及多模态大模型[2]中被发现。我们对 T5 注意力模式的分析（如下图所示）表明，即使在双向注意力机制中，此类稀疏模式依然跨不同注意力头稳定存在。<br/>
[1] SparseBERT: Rethinking the Importance Analysis in Self-Attention, ICML 2021.<br/>
[2] LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference, 2024.<br/>
<p align="center">
    <img src="https://raw.githubusercontent.com/microsoft/MInference/main/images/t5_sparse_pattern.png" width="600px" style="margin:auto;border-radius: 5px;display: inline-block;padding: 0 0 0 10px;" alt=''>
</p>
<p align="center">图 1：T5 Encoder 中的稀疏注意力模式</p>

**Q4：MInference、状态空间模型 (SSM)、线性注意力 (Linear Attention) 与传统稀疏注意力之间是什么关系？**

这四种方法（MInference、SSM、线性注意力与传统稀疏注意力）均致力于高效优化 Transformer 的计算复杂度，各自引入了不同的归纳偏置（Inductive Bias）。后三者往往需要从头训练模型。近期的前沿工作（如 Mamba-2 与统一隐式注意力表征）将 SSM 与线性注意力统一抽象为静态稀疏注意力，其中 Mamba-2 本身即是一种块级稀疏（Block-wise Sparse）方法。尽管由于注意力的稀疏冗余性，这些方法展现出巨大潜力，但静态稀疏注意力在应对复杂任务中的动态语义关联时常常面临挑战；相比之下，动态稀疏注意力更契合建模此类复杂的动态交互关联。

**Q5**：在 `_prepare_4d_causal_attention_mask_with_cache_position` 中出现 CUDA 显存不足（OOM）

_解决方法_：在模型初始化时传入 `_attn_implementation="flash_attention_2",` 参数，将 Hugging Face 模型的注意力计算后端指定为 FlashAttention-2。

**Q6**：在 `logits = self.lm_head(hidden_states[:, -logits_to_keep:, :])` 中出现 CUDA 显存不足（OOM）

_解决方法_：在模型前向传播（forward）时设置 `logits_to_keep=1`。

## 引用 (Citation)

如果您在研究或项目中发现 MInference 对您有所帮助，请引用我们的学术论文：

```bibtex
@inproceedings{jiang2024minference,
  author = {Huiqiang Jiang and Yucheng Li and Chengruidong Zhang and Qianhui Wu and Xufang Luo and Surin Ahn and Zhenhua Han and Amir H. Abdi and Dongsheng Li and Chin-Yew Lin and Yuqing Yang and Lili Qiu},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  title = {{MI}nference 1.0: Accelerating Pre-filling for Long-Context {LLM}s via Dynamic Sparse Attention},
  url = {https://openreview.net/forum?id=fPBACAbqSN},
  year = {2024}
}

@inproceedings{li2025scbench,
  title={{SCB}ench: A {KV} Cache-Centric Analysis of Long-Context Methods},
  author={Yucheng Li and Huiqiang Jiang and Qianhui Wu and Xufang Luo and Surin Ahn and Chengruidong Zhang and Amir H. Abdi and Dongsheng Li and Jianfeng Gao and Yuqing Yang and Lili Qiu},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=gkUyYcY1W9}
}

@inproceedings{li2025mminference,
    title={{MMI}ference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention},
    author={Li, Yucheng and Jiang, Huiqiang and Zhang, Chengruidong and Wu, Qianhui and Luo, Xufang and Ahn, Surin and Abdi, Amir H and Li, Dongsheng and Gao, Jianfeng and Yang, Yuqing and Qiu, Lili},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=me6PfbATWM}
}
```

## 参与贡献 (Contributing)

本项目诚挚欢迎社区贡献与建议！大多数贡献要求您签署贡献者许可协议（CLA），声明您有权并确实授予我们使用您贡献成果的权利。详情请访问 https://cla.opensource.microsoft.com。

当您提交 Pull Request 时，CLA 机器人会自动检查您是否需要签署 CLA 并为 PR 添加对应状态标记或说明，只需按照提示完成即可。在使用微软 CLA 的所有开源代码库中，您仅需签署一次。

本项目已采纳 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。欲了解更多信息，请参阅[行为准则常见问题解答](https://opensource.microsoft.com/codeofconduct/faq/)，若有任何疑问亦可发送邮件至 [opencode@microsoft.com](mailto:opencode@microsoft.com)。

## 商标声明 (Trademarks)

本项目可能包含项目、产品或服务的商标或标志。授权使用微软商标或标志需遵守并遵循 [Microsoft 商标与品牌指南](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general)。在修改后的项目版本中使用微软商标或标志不得造成混淆或暗示微软的官方赞助。任何第三方商标或标志的使用均受相应第三方的政策约束。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月18日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
