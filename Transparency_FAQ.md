# MInference's Responsible AI FAQ

## 🏮 What is MInference?

- MInference is a system designed for efficient inference of Large Language Models (LLMs) in long context scenarios. It significantly accelerates LLM inference through approximate-sparse calculation, reducing latency by 6x in the prefix stage and 8x in the decoding stage on an A100, while maintaining the same level of performance.

### 1. What can MInference do?

- MInference can effectively reduce the inference cost of LLMs in long-context scenarios, with minimal impact on performance.

### 2. What are MInference’s intended use(s)?

- LLMs deployers and users.

### 3. How was MInference evaluated? What metrics are used to measure performance?

- We using the long-context benchmark InfiniteBench and their evaluation metric to evaluate the MInference system.
- We have conducted extensive testing for accuracy across various scenarios, including multi-document QA, single document QA, retrieval-based methods, few-shot learning, and reasoning. The results show almost no change in accuracy. We have also tested on TruthfulQA, a benchmark for assessing susceptibility to harmful outputs, which further demonstrates that our method does not compromise accuracy. Finally, our method does not alter the inherent capabilities of LLMs, including their potential for harm.

### 4. What are the limitations of MInference? How can users minimize the impact of MInference’s limitations when using the system?

- The potential harmful, false or biased responses using the MInference would likely be unchanged. Thus using MInference has no inherent benefits or risks when it comes to those types of responsible AI issues.
- MInference may struggle to perform well at particularly high sparsity.

### 5. What operational factors and settings allow for effective and responsible use of MInference?

- Users can set parameters such as the sparsity, sink, and local window size used for MInference. Afterward, they can get the response for the LLMs.

## 🔎 The questions about Motivation and Applicability

### 1. How to effectively evaluate the impact of dynamic sparse attention on the capabilities of long-context LLMs?

To evaluate long-context LLM capabilities using models like LLaMA-3-8B-Instruct-1M and GLM-4-9B-1M, we tested: 1) context window with RULER, 2) general tasks with InfiniteBench, 3) retrieval tasks with Needle in a Haystack, and 4) language model prediction with PG-19.<br/>
We found traditional methods perform poorly in retrieval tasks, with difficulty levels as follows: <font color="#337ab7"><b>KV retrieval > Needle in a Haystack > Retrieval.Number > Retrieval PassKey</b></font>. The main challenge is the semantic difference between needles and the haystack. Traditional methods excel when this difference is larger, as in passkey tasks. KV retrieval requires higher retrieval capabilities since any key can be a target, and multi-needle tasks are even more complex.<br/>
We will continue to update our results with more models and datasets in future versions.

### 2. Does this dynamic sparse attention pattern only exist in long-context LLMs that are not fully trained?

Firstly, attention is dynamically sparse, a characteristic inherent to the mechanism. We selected state-of-the-art long-context LLMs, GLM-4-9B-1M and LLaMA-3-8B-Instruct-1M, with effective context windows of 64K and 16K. With MInference, these can be extended to 64K and 32K, respectively. We will continue to adapt our method to other advanced long-context LLMs and update our results, as well as explore the theoretical basis for this dynamic sparse attention pattern.

### 3. Does this dynamic sparse attention pattern only exist in Auto-regressive LMs or RoPE based LLMs?

Similar vertical and slash line sparse patterns have been discovered in BERT[1] and multi-modal LLMs[2]. Our analysis of T5's attention patterns, shown in the figure, reveals these patterns persist across different heads, even in bidirectional attention.<br/>
[1] SparseBERT: Rethinking the Importance Analysis in Self-Attention, ICML 2021.<br/>
[2] LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference, 2024.<br/>
<p align="center">
    <img src="images/t5_sparse_pattern.png" width="600px" style="margin:auto;border-radius: 5px;display: inline-block;padding: 0 0 0 10px;" alt=''>
</p>
<p align="center">Figure 1. The sparse pattern in T5 Encoder.</p>

### 4. What is the relationship between MInference, SSM, Linear Attention, and Sparse Attention?

All four approaches (MInference, SSM, Linear Attention, and Sparse Attention) efficiently optimize attention complexity in Transformers, each introducing inductive bias differently. The latter three require training from scratch. Recent works like Mamba-2 and Unified Implicit Attention Representation unify SSM and Linear Attention as static sparse attention, with Mamba-2 itself being a block-wise sparse method. While these approaches show potential due to sparse redundancy in attention, static sparse attention may struggle with dynamic semantic associations in complex tasks. In contrast, dynamic sparse attention is better suited for managing these relationships.

## 🛠 The questions about Usage

### 1. Error "RuntimeError: tensor does not have a device"

This issue is due to the current version of MInference being incompatible with torch and CUDA. Please reinstall MInference.
```bash
pip uninstall minference
pip install minference
```

## 2. How to use MInference with torch 2.3.x?

MInference supports various torch versions. However, due to certain issues with flash-attn in torch 2.3.x, please use flash-attn version <= 2.4.x.
