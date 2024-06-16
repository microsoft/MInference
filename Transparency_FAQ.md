# MInference's Responsible AI FAQ

## What is MInference?

- MInference is a system designed for efficient inference of Large Language Models (LLMs) in long context scenarios. It significantly accelerates LLM inference through approximate-sparse calculation, reducing latency by 6x in the prefix stage and 8x in the decoding stage on an A100, while maintaining the same level of performance.

## What can MInference do?

- MInference can effectively reduce the inference cost of LLMs in long-context scenarios, with minimal impact on performance.

## What are MInference’s intended use(s)?

- LLMs deployers and users.

## How was MInference evaluated? What metrics are used to measure performance?

- We using the long-context benchmark InfiniteBench and their evaluation metric to evaluate the MInference system.
- We have conducted extensive testing for accuracy across various scenarios, including multi-document QA, single document QA, retrieval-based methods, few-shot learning, and reasoning. The results show almost no change in accuracy. We have also tested on TruthfulQA, a benchmark for assessing susceptibility to harmful outputs, which further demonstrates that our method does not compromise accuracy. Finally, our method does not alter the inherent capabilities of LLMs, including their potential for harm.

## What are the limitations of MInference? How can users minimize the impact of MInference’s limitations when using the system?

- The potential harmful, false or biased responses using the MInference would likely be unchanged. Thus using MInference has no inherent benefits or risks when it comes to those types of responsible AI issues.
- MInference may struggle to perform well at particularly high sparsity.

## What operational factors and settings allow for effective and responsible use of MInference?

- Users can set parameters such as the sparsity, sink, and local window size used for MInference. Afterward, they can get the response for the LLMs.

## How to effectively evaluate the impact of dynamic sparse attention on the capabilities of long-context LLMs?

To effectively evaluate long-context LLM capabilities, we tested: 1) effective  context window with RULER, 2) general long-context tasks with InfiniteBench, 3) retrieval tasks across different contexts and positions with Needle in a Haystack, and 4) language model prediction with PG-19.<br/>
We found that traditional methods perform poorly in retrieval tasks, with difficulty levels varying as follows: KV retrieval (every key as a needle) > Needle in a Haystack > Retrieval.Number > Retrieval PassKey. The key challenge is the semantic difference between needles and the haystack. Traditional methods perform better when the semantic difference is larger, as in passkey tasks. KV retrieval demands higher retrieval capabilities since any key can be a target, and multi-needle tasks are even more complex.<br/>
We will continue to update our results with more models and datasets in future versions.

## Does this dynamic sparse attention pattern only exist in long-context LLMs that are not fully trained?

Firstly, attention is dynamically sparse, and this is true for both short- and long-contexts, a characteristic inherent to the attention mechanism.
Additionally, we selected the state-of-the-art open-source long-context LLM, LLaMA-3-8B-Instruct-1M, which has an effective context window size of 16K. With MInference, this can be extended to 32K.
We will continue to adapt our method to other advanced long-context LLMs and update our results. We will also explore the theoretical reasons behind this dynamic sparse attention pattern.

## What is the relationship between MInference, SSM, Linear Attention, and Sparse Attention?

All four approaches (MInference, SSM, Linear Attention, and Sparse Attention) are efficient solutions for optimizing the high complexity of attention in Transformers, each introducing inductive bias from different perspectives. Notably, the latter three require training from scratch.
Additionally, recent works like Mamba-2 and Unified Implicit Attention Representation unify SSM and Linear Attention as static sparse attention. Mamba-2 itself is a block-wise sparse attention method.
Intuitively, the significant sparse redundancy in attention suggests that these approaches have potential. However, static sparse attention may not handle dynamic semantic associations well, especially in complex tasks. Dynamic sparse attention, on the other hand, holds potential for better managing these dynamic relationships.
