## ABSTRACT

Large Language Models (LLMs) have shown strong performance in complex reasoning tasks, especially when guided by Chain-of-Thought (CoT) prompting. However, conventional CoT reasoning in the discrete token space suffers from high computational and memory costs due to verbose intermediate steps. Recent work has explored latent reasoning in the embedding space to improve efficiency, but often at the cost of clarity and performance. In this work, we propose Hybrid Reasoning (HyRea), a unified framework that enables LLMs to dynamically switch between explicit (token-based) and latent (embedding-based) reasoning during inference. To train the model to make these decisions effectively, we introduce a two-stage training pipeline: (1) a supervised cold-start phase that introduces latent reasoning by replacing low-entropy CoT steps with embeddings, and (2) a reinforcement learning phase using Group Relative Policy Optimization (GRPO) to fine-tune the model’s reasoning strategy based on task-specific rewards. Experiments on mathematical reasoning benchmarks show that HyRea achieves significant reductions in token usage while maintaining or improving accuracy, offering an effective and scalable solution for efficient multi-step reasoning in LLMs. Our code is publicly available at https://github.com/zhaoyiran924/HyRea.

## 1 INTRODUCTION

Large Language Models (LLMs) (OpenAI, 2023; Grattafiori et al., 2024; Team et al., 2023; 2024; Yang et al., 2024a; Comanici et al., 2025; OpenAI, 2025) have shown impressive capabilities in complex reasoning tasks (Wang et al., 2024; Hsiao et al., 2025; Shi et al., 2025; Qu et al., 2025), particularly when guided by Chain-of-Thought (CoT) prompting (Wei et al., 2022), which encourages step-by-step intermediate reasoning. However, conventional CoT reasoning operates entirely in the discrete token space, leading to inefficiencies in computation and memory due to verbose outputs. As long-context and cost-effective inference become increasingly important, especially in reinforcement learning (RL) settings for math tasks (Liu et al., 2024), methods like Deepseek-R1 (Guo et al., 2025) face high token costs and slow convergence. To tackle these challenges, recent approaches propose operating directly in the embedding space, bypassing tokenization altogether (Hao et al., 2024b; Yue et al., 2025; Zhang et al., 2025). These latent reasoning methods offer substantial token compression and improved efficiency. However, reasoning over embeddings remains inherently difficult and can lead to degraded model performance compared to traditional token-based inference, as certain tokens encode complex, nuanced information that cannot be fully preserved in compressed embeddings (Hao et al., 2024b; Shen et al., 2025). Moreover, current models lack the capability to selectively determine which tokens should be compacted and which should remain in their original form.

In this work, we propose a novel Hybrid Reasoning (HyRea) framework that combines explicit (token-based) and latent (embedding-based) reasoning<sup>1</sup>. Explicit reasoning offers interpretability through step-by-step generation but is inefficient, while latent reasoning improves efficiency by operating in embedding space, often sacrificing clarity and performance. As shown in Figure 1, HyRea enables LLMs to dynamically switch between these two modes during inference, selecting the most suitable reasoning strategy based on the context. This adaptive mechanism allows the model to maintain high accuracy while significantly reducing the number of generated tokens, yielding better trade-offs between performance and efficiency. To support this hybrid reasoning capability, we introduce a two-stage training framework. In the supervised cold-start phase, the model learns to use latent reasoning by partially replacing intermediate CoT steps with continuous embeddings. To guide this replacement, we prioritize steps with low entropy, which are more deterministic and thus easier for the model to learn in latent space. This encourages the model to interpret and generate latent computations within a broader reasoning trajectory. In the subsequent reinforcement learning phase, to further refine the model’s decision-making process. Specifically, we adopt Group Relative Policy Optimization (GRPO) (Shao et al., 2024), a RL technique that stabilizes learning by evaluating policies within sampled groups. This phase enables the model to learn when to invoke explicit versus latent reasoning based on downstream rewards, such as accuracy, format correctness, and efficiency. By integrating explicit and latent reasoning in a unified and learnable framework, HyRea provides a flexible and scalable approach to improve reasoning performance in LLMs, particularly in domains like mathematics where both precision and efficiency are critical.

![](images/12b99dd2f2bc67f916deb1ad2c543f2b266c7e6bf16ebdf5aef3aa408776c1c0.jpg)  
Figure 1: HyRea framework overview. The model dynamically switches between explicit (tokenbased) and latent (embedding-based) reasoning during inference. Explicit reasoning provides interpretability through step-by-step token generation, while latent reasoning operates in embedding space for greater efficiency. HyRea adaptively selects the optimal reasoning mode based on context, enabling a flexible and efficient hybrid approach.

We conduct comprehensive experiments to evaluate the performance of HyRea across various models and tasks. Our compression method was evaluated on two widely used open-source models, Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct (Yang et al., 2024a), and we demonstrate that it can reduce length of the answer to approximately 60% of its original length without any loss in mathematical reasoning capability. Specifically, on MATH-500 (Lightman et al., 2023), Minerva Math (Hendrycks et al., 2021), AMC23 (AMC, 2023), and Olympiad Bench (He et al., 2024), HyRea achieves competitive accuracies while generating solutions with an average of only 309, 419, 452, and 492 tokens under Qwen2.5-7B, respectively. In contrast, Qwen2.5-7B-Instruct trained using a conventional reinforcement learning approach produces substantially longer outputs, averaging 698, 671, 892, and 854 tokens on the same benchmarks. Furthermore, we evaluate the model trained with HyRea on general tasks beyond mathematics, including MMLU (Hendrycks et al., 2020) and GPQA (Rein et al., 2024). The results demonstrate strong generalization ability and a significant reduction in output length.

## 2 INTEGRATE EXPLICIT AND LATENT REASONING

## 2.1 EXPLICIT REASONING

Explicit reasoning refers to the process by which a model generates reasoning outputs in an autoregressive manner, decoding one token at a time. Formally, given an input query $x = [ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot , x _ { t } ]$ the model first maps each token to its corresponding embedding via the embedding layer, resulting in the sequence $\bar { E = [ { e ( x _ { 1 } ) , e ( x _ { 2 } ) , \cdot \cdot \cdot , e ( x _ { t } ) } ] }$ . These embeddings are then processed by a series of Transformer blocks as introduced by Vaswani et al. (2017):

$$
H _ {t} = \left[ h _ {1}, h _ {2}, \dots , h _ {t} \right] = \operatorname{Transformer} (E),\tag{1}
$$

where $h _ { i }$ denotes the hidden embeddings at the i-th position from the final layer of the Transformer. Furthermore, the hidden state $h _ { t }$ is transformed into a probability distribution over the vocabulary set V through the language modeling head, denoted as $\mathrm { L } \bar { \mathrm { M } } _ { \mathrm { h e a d } }$ . The next token $\hat { x } _ { t + 1 }$ is then selected by taking the argmax over the resulting logits:

$$
\hat {x} _ {t + 1} = \arg \max _ {\mathcal {V}} (\text { logits }) = \arg \max _ {\mathcal {V}} \bigl (\mathrm{LM} _ {\text { head }} (h _ {t}) \bigr).\tag{2}
$$

The input is then updated to $[ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot , x _ { t } , \hat { x } _ { t + 1 } ]$ , which is subsequently processed sequentially by Equation 1 followed by Equation 2.

## 2.2 LATENT REASONING

Instead of decoding token by token, Chain of Continuous Thought (Coconut) (Hao et al., 2024b) proposes to bypass tokenization entirely by operating directly on embeddings. Specifically, during decoding, the model feeds the hidden output from the final layer back into the first layer as input. That is, rather than decoding the next token using Equation 2, the model proceeds as follows:

$$
H _ {t + 1} = \left[ h _ {1}, h _ {2}, \dots , h _ {t}, h _ {t + 1} \right] = \operatorname{Transformer} (E \| h _ {t}),\tag{3}
$$

where E denotes the initial embedding sequence and $h _ { t }$ is the hidden state at time step t and ∥ represents concatenation. Compared to explicit reasoning, operating directly on embeddings allows for a more compact representation during inference, potentially reducing token consumption. However, reasoning over embeddings remains inherently difficult and can lead to degraded model performance compared to traditional token-based inference, as certain tokens encode complex, nuanced information that cannot be fully preserved in compressed embeddings. This loss of semantic fidelity can be particularly detrimental in tasks that require precise symbolic manipulation, such as mathematical reasoning or code generation, where even small distortions in representation may lead to incorrect conclusions. Moreover, current models lack the capability to selectively determine which tokens should be compacted and which should remain in their original form. Compression is typically applied uniformly or based on fixed heuristics, without accounting for the varying informational value of different tokens within a reasoning trace. This inability to adaptively balance compression and fidelity limits the effectiveness of latent reasoning and highlights the need for more principled, content-aware strategies for selective representation.

## 2.3 HYBRID REASONING

To improve the efficiency of the reasoning process while minimizing performance degradation, we aim for the model to flexibly switch between explicit and latent reasoning during inference. When decoding the next position, the model can autonomously decide whether to operate in the embedding space or the token space—that is, whether to perform latent reasoning or explicit reasoning. Formally, at position i, if the model chooses to employ the explicit reasoning mode, it follows Equation 2 to generate the next token. Conversely, if the model opts for the latent reasoning mode, it applies Equation 3, inserting a <start-latent> token at the beginning and a <end-latent> token at the end to mark the latent reasoning span. Formally, the ideal structure of hybrid reasoning can be represented as:

$$
[ \text {Question} ] [ \text {Step} _ {1} ] \dots <   \text {start - latent} > [ \text {latent} ] <   \text {end - latent} > \dots [ \text {Step} _ {N} ] \dots [ \text {Answer} ].
$$

## 3 TRAINING METHOD

In this section, we mainly introduce the training method to enable the model conduct hybrid reasoning.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Hybrid Reasoning Training (HyRea)

Input: Pretrained language model $\mathcal{LLM}$, CoT training data $\mathcal{D}_{\text{SFT}}$, RL training data $\mathcal{D}_{\text{RL}}$, max latent steps $S$, group size $G$, clipping threshold $\varepsilon$

Output: Hybrid reasoning model $\mathcal{LLM}$

1: // Stage 1: Cold-Start Supervised Fine-Tuning

2: for $s$ from 1 to $S$ do

3:    for each batch $(q, a)$ in $\mathcal{D}_{\text{SFT}}$ do

4:    Identify reasoning steps with low entropy

5:    Replace $s$ selected steps with [&lt;start-latent&gt; latent &lt;end-latent&gt;]

6:    Compute loss $\mathcal{L}_{\text{SFT}} = -\log \mathcal{LLM}(q \setminus [\text{latent}])$

7:    Update $\mathcal{LLM}$ using gradient descent on $\mathcal{L}_{\text{SFT}}$

8:    end for

9:    Incrementally introduce 10% new data into $\mathcal{D}_{\text{SFT}}$ at each iteration

10: end for

11: // Stage 2: Reinforcement Learning with GRPO

12: for each query $q$ in $\mathcal{D}_{\text{RL}}$ do

13:    Sample $G$ outputs $\{o_1, o_2, \ldots, o_G\} \sim \mathcal{LLM}_{\text{old}}(\cdot | q)$

14:    Evaluate rewards $\{r_1, \ldots, r_G\}$ (accuracy, formatting, latent usage)

15:    Compute GRPO loss by Equation 6

16:    Update $\mathcal{LLM}$ using gradient descent on $\mathcal{L}_{\text{GRPO}}$

17: end for
</div>

## 3.1 COLD START FOR REPLACE LOW-ENTROPY STEPS

To enable the model to perform hybrid reasoning, we adopt a cold-start approach inspired by Deepseek-R1 (Liu et al., 2024). In this initial phase, the model is trained using supervised fine-tuning (SFT) on chain-of-thought (CoT) (Wei et al., 2022) reasoning data to learn preliminary switching capabilities. This serves as a foundational step toward developing more advanced hybrid reasoning abilities. Specifically, the original CoT data is structured as a sequence:

$$
C := [ \text { Question } ], [ \text { Step } _ {1} ], [ \text { Step } _ {2} ], \ldots , [ \text { Step } _ {N} ], [ \text { Answer } ].\tag{4}
$$

To introduce latent reasoning, we select steps that have low entropy and replace them with a latent segment $[ \mathrm { L a t e n t } ] : = < \mathsf { s t a r t } - \mathrm { l a t e n t } > c \times [ \mathrm { l a t e n t } ] < \mathsf { e n d } - \mathrm { l a t e n t } >$ , where c denotes the number of latent tokens. This creates a hybrid sequence in which [Latent] serves as a placeholder for latent reasoning. Importantly, the entropy-based replacement strategy prevents the model from compacting tokens that encode complex or critical information that cannot be faithfully represented by latent embeddings. During training, the loss is computed only over the visible (non-latent) tokens, ensuring that the model focuses on learning the interpretable parts of the reasoning trace while implicitly handling the latent segments.

$$
\mathcal {L} _ {\text { cold   start }} := - \log \mathcal {L L M} (C \setminus [ \text { Latent } ]),\tag{5}
$$

where LLM(·) denotes the language model’s likelihood function. In addition, the number of steps replaced by [Latent] increases progressively during training, starting from 0 and gradually reaching a predefined maximum threshold S.

## 3.2 REINFORCEMENT LEARNING

We mainly employ the Group Relative Policy Optimization (GRPO) proposed by Shao et al. (2024)

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{GRPO}} (\theta) = \mathbb {E} _ {q \sim P (Q),   \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (O | q)} \\ \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \left(\min \left(\frac {\pi_ {\theta} (o _ {i} \mid q)}{\pi_ {\theta_ {\mathrm{old}}} (o _ {i} \mid q)} A _ {i},   \operatorname{clip} \left(\frac {\pi_ {\theta} (o _ {i} \mid q)}{\pi_ {\theta_ {\mathrm{old}}} (o _ {i} \mid q)}, 1 - \varepsilon ,   1 + \varepsilon\right) A _ {i}\right)\right) \right], \end{array}\tag{6}
$$

where

$$
A _ {i} = \frac {r _ {i} - \mathrm{mean} (\{r _ {1} , r _ {2} , \cdots , r _ {G} \})}{\mathrm{std} (\{r _ {1} , r _ {2} , \cdots , r _ {G} \})},\tag{7}
$$

and reward consists of both a format reward and an accuracy reward, as well as a latent reward. Specifically, we use the latent reward to guide the model in generating the token [Latent]. Additionally, during loss computation, we exclude the [Latent] token and calculate the loss only over the remaining token steps. The overall algorithm is further illustrated in Algorithm 1. Through reinforcement learning, this approach eliminates the need for manually constructing synthetic data to train the model’s switching capability. Instead, the model learns to perform hybrid reasoning in a self-supervised manner, guided by reward signals that reflect both correctness and efficiency.

## 4 EXPERIMENT

## 4.1 EVALUATION SETUP

Datasets We primarily utilize two large-scale mathematical reasoning datasets for cold start: NuminaMath-CoT (LI et al., 2024), which contains 860k diverse math problems ranging from high school exercises to international olympiad level questions formatted in a CoT style, and MetaMathQA (Yu et al., 2023), comprising 390k problem-solution pairs enhanced through various data augmentation techniques to promote diverse and robust reasoning pathways. Specifi cally, we split the answer by ‘\n’ and ‘.’, and ensure that each equation is kept as an independent step. Figure 2 presents a concrete example of the processed dataset. Additionally, in the reinforcement learning stage, we adopt Math-12k<sup>2</sup>, a challenging math dataset derived from (Lightman et al., 2023).

![](images/7a162eaaa6ea47eb5d00bce61c0dc0b34592aa0055d35c1c7e2c2df8d547c5ab.jpg)  
Figure 2: An example data point.

Backbone Models We evaluate two widely-used open-source LLMs as backbone models. Qwen2.5- 7B-Instruct and Qwen2.5-32B-Instruct (Yang et al., 2024a).

Baselines. We employ several state-of-the-art efficient reasoning method with less tokens (i) CoT (Wei et al., 2022); (ii) Reinforcement Learning (Guo et al., 2025), which optimizes reasoning strategies through trial-and-error by rewarding accurate outputs; (iii) Chain of Continuous Thought (Coconut) (Hao et al., 2024b), a novel paradigm that replaces discrete tokens with latent “continuous thoughts” by feeding the last hidden state directly back into the model, allowing reasoning in an unconstrained latent space and enabling breadth-first exploration over multiple reasoning paths; (iv) Soft thinking (Zhang et al., 2025), a training-free method that mimics human-like soft reasoning by generating probability-weighted concept tokens in a continuous concept space, capturing semantic ambiguity and enabling more abstract, flexible reasoning with significantly fewer tokens.

Benchmarks To assess mathematical reasoning proficiency, we employ four benchmarks: MATH 500 (Lightman et al., 2023), a 500-question subset of the MATH benchmark curated for rigorous; Minerva Math (Hendrycks et al., 2021); Olympiad Bench (He et al., 2024) as well as competitionlevel benchmarks such as AMC23 (AMC, 2023). We evaluate the performance of the model on each dataset using pass@1. Additionally, we analyze the length of each generated answer by counting the total number of tokens, including both standard tokens and the [Latent] token. Furthermore, we assess the frequency of switches between latent and explicit reasoning within each response.

Implement Details For the cold-start, we employed the LLaMA-Factory library (Zheng et al., 2024), a widely adopted GitHub-hosted framework for efficient large-model fine-tuning, to carry out all training procedures. Experiments are conduced on eight 140GB NVIDIA H200 GPUs, with learning rate as $4 \times 1 0 ^ { - 7 }$ , global batch size as 32. Furthermore, to reduce memory consumption during training, we applied ZeRO Stage-2 optimization and gradient checkpointing, both provided by the DeepSpeed library. Note that in the cold-start stage, the loss on <start-latent> and <end-latent> are scaled by a factor of 4 to emphasize their importance and help the model learn when to switch. For reinforcement learning training, we use the EasyR1 (Zheng et al., 2025) framework built on verl (Sheng et al., 2024), with specialized support for VLMs. Experiments are conducted using eight 140GB NVIDIA H200 GPUs with a global batch size of 128, a rollout batch size of 128, a rollout temperature of 1.0, a consistent learning rate of $1 \times 1 0 ^ { - 6 }$ , and 8 rollouts.

Table 1: Main results of HyRea across four math reasoning benchmarks. We report accuracy (Acc), average number of output tokens (#Tokens), and average number of latent-explicit switches (#Switch).

<table><tr><td rowspan="2"></td><td rowspan="2">Methods</td><td colspan="3">Math-500</td><td colspan="3">Minerva Math</td><td colspan="3">AMC23</td><td colspan="3">Olympiad Bench</td></tr><tr><td>Acc</td><td>#Tokens</td><td>#Switch</td><td>Acc</td><td>#Tokens</td><td>#Switch</td><td>Acc</td><td>#Tokens</td><td>#Switch</td><td>Acc</td><td>#Tokens</td><td>#Switch</td></tr><tr><td rowspan="6">Qwen2.5-7B-It</td><td>SFT</td><td>73.6</td><td>546</td><td>0</td><td>26.1</td><td>619</td><td>0</td><td>36.1</td><td>845</td><td>0</td><td>29.3</td><td>723</td><td>0</td></tr><tr><td>SFT + RL</td><td>84.2</td><td>698</td><td>0</td><td>26.8</td><td>671</td><td>0</td><td>48.2</td><td>892</td><td>0</td><td>40.0</td><td>854</td><td>0</td></tr><tr><td>Coconut</td><td>70.4</td><td>106</td><td>1</td><td>22.1</td><td>174</td><td>1</td><td>33.7</td><td>217</td><td>1</td><td>26.8</td><td>296</td><td>1</td></tr><tr><td>Soft Thinking</td><td>66.4</td><td>617</td><td>1</td><td>16.9</td><td>604</td><td>1</td><td>24.1</td><td>784</td><td>1</td><td>24.7</td><td>595</td><td>1</td></tr><tr><td>HyRea w/o RL</td><td>71.8</td><td>248</td><td>4.8</td><td>20.6</td><td>288</td><td>3.1</td><td>34.9</td><td>339</td><td>4.9</td><td>28.5</td><td>378</td><td>4.3</td></tr><tr><td>HyRea</td><td>83.6</td><td>387</td><td>4.4</td><td>27.2</td><td>425</td><td>3.6</td><td>48.2</td><td>526</td><td>4.7</td><td>39.6</td><td>583</td><td>3.8</td></tr><tr><td rowspan="6">Qwen2.5-32B-It</td><td>SFT</td><td>83.6</td><td>595</td><td>0</td><td>37.1</td><td>644</td><td>0</td><td>55.4</td><td>803</td><td>0</td><td>49.0</td><td>866</td><td>0</td></tr><tr><td>SFT + RL</td><td>85.2</td><td>588</td><td>0</td><td>39.7</td><td>608</td><td>0</td><td>61.4</td><td>905</td><td>0</td><td>49.5</td><td>899</td><td>0</td></tr><tr><td>Coconut</td><td>79.8</td><td>179</td><td>1</td><td>32.4</td><td>209</td><td>1</td><td>53.0</td><td>285</td><td>1</td><td>46.2</td><td>403</td><td>1</td></tr><tr><td>Soft Thinking</td><td>82.4</td><td>574</td><td>1</td><td>33.4</td><td>602</td><td>1</td><td>55.4</td><td>776</td><td>1</td><td>43.7</td><td>887</td><td>1</td></tr><tr><td>HyRea w/o RL</td><td>79.8</td><td>305</td><td>3.2</td><td>34.6</td><td>267</td><td>3.9</td><td>54.2</td><td>304</td><td>5.2</td><td>47.3</td><td>428</td><td>4.7</td></tr><tr><td>HyRea</td><td>84.4</td><td>369</td><td>4.1</td><td>38.6</td><td>381</td><td>3.7</td><td>57.8</td><td>498</td><td>4.9</td><td>48.9</td><td>563</td><td>5.3</td></tr></table>

## 4.2 MAIN RESULT

HyRea achieves strong accuracy–efficiency trade-offs. As shown in Table 1, our proposed hybrid reasoning framework HyRea consistently delivers strong performance across all four math reasoning benchmarks, demonstrating both high accuracy and efficient token usage. Under the Qwen2.5-7B-Instruct, HyRea achieves an accuracy of 83.6 on MATH-500, nearly matching the SFT+RL baseline (84.2), while using less than half the number of output tokens (387 vs. 698). Similar trends are observed across other benchmarks: on Minerva Math, HyRea slightly outperforms SFT+RL (27.2 vs. 26.8) with a substantial reduction in tokens (425 vs. 671); and for AMC23 and Olympiad Bench, it matches or closely approaches the best-performing baselines in accuracy (48.2 and 39.6, respectively), while reducing token usage by nearly 50%. When scaled up to Qwen2.5-32B-Instruct, HyRea continues to show competitive results, achieving 84.4 accuracy on MATH-500 with only 369 tokens, compared to 85.2 accuracy and 588 tokens for SFT+RL. On Minerva Math and AMC23, HyRea reaches 38.6 and 57.8 accuracy, respectively, again using significantly fewer tokens than the SFT+RL counterpart. Notably, HyRea also exhibits meaningful latent-explicit switching behavior, averaging 3.6 to 5.3 mode transitions per sample depending on the dataset and model size, compared to zero switches for all baselines and only one for Coconut and Soft Thinking. This switching capacity reflects HyRea’s ability to dynamically leverage both explicit and latent reasoning strategies. Importantly, HyRea outperforms token-efficient approaches like Coconut, which, despite producing shorter outputs (e.g., 106–179 tokens on MATH-500), suffers from significant accuracy degradation across all tasks. The ablation study (HyRea w/o RL) further highlights the benefits of reinforcement learning, as performance notably drops when RL is removed. Overall, these results underscore HyRea’s effectiveness as a scalable solution for multi-step reasoning—achieving high accuracy with compact solutions via strategic reasoning mode transitions.

Complementing this, Figure 3 shows that during training, the accuracy and latent rewards steadily improve and stabilize, while the format reward remains consistently high. This indicates that HyRea learns to balance correctness, structure, and reasoning mode usage effectively. Overall, these results underscore HyRea’s effectiveness as a scalable solution for multi-step reasoning—achieving high accuracy with compact solutions via strategic reasoning mode transitions.

![](images/13dc31b14ba33a157bd2ec04ff414b1a5650eb489106c1fae08691acb57d9139.jpg)

![](images/b5eb4fe36eb3a1b3644913007f7447e8b4ed5bfb3ab39b952474a174abc3fb5c.jpg)  
Figure 3: Reward over steps.

![](images/cb33827f64c66578b14ac4a2c1d02f2e6b2b2c0f3d9d8dd7bd6efa7ecc868dcb.jpg)

![](images/5be80615727933528fd73350e0830e30544d773b1f2d41fd848267e015baffb4.jpg)  
Figure 4: Concrete Example.

Table 2: Comparison of entropy based replacement and random sampling replacement.

<table><tr><td rowspan="2"></td><td colspan="2">Math</td><td colspan="2">Minerva</td><td colspan="2">AMC23</td><td colspan="2">Olympiad</td></tr><tr><td>Acc</td><td>#Token</td><td>Acc</td><td>#Token</td><td>Acc</td><td>#Token</td><td>Acc</td><td>#Token</td></tr><tr><td>SFT+RL</td><td>84.2</td><td>698</td><td>26.1</td><td>619</td><td>48.2</td><td>892</td><td>40.0</td><td>854</td></tr><tr><td>Random</td><td>83.4</td><td>309</td><td>26.5</td><td>419</td><td>49.4</td><td>452</td><td>39.6</td><td>492</td></tr><tr><td>Entropy</td><td>83.6</td><td>287</td><td>27.2</td><td>372</td><td>48.2</td><td>426</td><td>39.6</td><td>483</td></tr></table>

Concrete Examples Figure 4 exemplifies HyRea’s ability to abstract away low-utility, generic reasoning steps into latent space, effectively bypassing verbose yet semantically redundant content. By retaining only semantically salient computations and final outputs in explicit token space, the model achieves a more compact and efficient reasoning trace without compromising interpretability. This demonstrates HyRea’s capacity to balance informativeness and conciseness through dynamic reasoning mode selection.

## 4.3 ABLATION ANALYSIS

Random Replacement Table 2 compares two strategies for selecting intermediate reasoning steps to replace with latent embeddings: random sampling versus entropy-based selection. Both strategies significantly reduce token usage compared to the CoT+RL baseline while achieving comparable or better accuracy. Notably, the entropy-based method consistently produces shorter outputs across all benchmarks, with token counts of 287 on MATH-500, 372 on Minerva Math, 426 on AMC23, and 483 on Olympiad Bench—representing the lowest token usage among all methods evaluated. In contrast, random replacement uses slightly more tokens (e.g., 309, 419, 452, and 492, respectively), suggesting that entropy-based selection produces more compressible reasoning paths. In terms of accuracy, entropy-based replacement also demonstrates greater stability. On Minerva Math, it achieves an accuracy of 27.2 compared to 26.5 for random, and on MATH-500 and Olympiad Bench, it matches or slightly outperforms random selection. Although both strategies come close to the CoT+RL performance in accuracy, entropy-based replacement offers a more favorable trade-off between performance and efficiency. These results validate our design choice of using entropy as a principled criterion for identifying deterministic and compressible reasoning steps to encode in latent space.

As shown in Figure 5, entropy-based sampling significantly outperforms random sampling during the cold-start stage. It achieves faster performance gains, surpassing 80 points within 10 iterations, while sample-based selection improves more slowly and plateaus around 75. This demonstrates that uncertainty-aware sampling more effectively prioritizes informative examples early in training, accelerating convergence and yielding better final performance.

Replace Number of Latent Figure 7 presents an ablation study on the number of latent replacements, showing how varying the parameter c affects both accuracy and output length. As c increases from 1 to 8, accuracy (purple line) drops sharply—from over 80% to below 10%—indicating that replacing more latent components substantially degrades model performance. Conversely, output length (blue line) increases significantly, particularly after c = 4, suggesting that excessive replacement leads to verbosity or redundancy. This inverse relationship between c and performance suggests that the training stability of HyRea and hybrid reasoning deteriorates rapidly as c increases.

![](images/cc9b3e41f1a20e1586a278005b11a8a02e540005ae46097758c1a6c6a56241a6.jpg)  
Figure 5: Cold-start stage: entropy-based vs. random sampling.

![](images/97ba7699efd538d3121ddf13f7fab9c5b5f46764e216eefd7d393adb8a2169c8.jpg)  
Figure 6: Concrete Example.

![](images/f367634c6a6cb0a0686d3496ebd435cee1426b68a8a8bebe01088d72565f9c31.jpg)

Table 3: Generalization ability of HyRea trained model.

<table><tr><td rowspan="2"></td><td colspan="2">MMLU</td><td colspan="2">GPQA</td></tr><tr><td>Acc</td><td>#Token</td><td>Acc</td><td>#Token</td></tr><tr><td>SFT</td><td>70.2</td><td>84</td><td>23.7</td><td>837</td></tr><tr><td>SFT+RL</td><td>73.4</td><td>102</td><td>29.8</td><td>1083</td></tr><tr><td>HyRea</td><td>68.6</td><td>53</td><td>27.4</td><td>685</td></tr></table>

Figure 7: Ablation of replace number of latent.

## 5 FURTHER ANALYSIS

## 5.1 SWITCHING PATTERN

Figure 6 provides a fine-grained view of HyRea’s reasoning behavior via entropy over decoding steps. We observe that latent reasoning steps (marked with red crosses) tend to appear in low-entropy regions, indicating high model confidence in selecting the latent mode. Notably, these latent spans often occur at the beginning or end of the reasoning trajectory, suggesting HyRea prefers to compress either the problem setup or the final answer derivation when confident. Additionally, switching between modes is relatively infrequent, aligning with our earlier observation of around 3–5 switches per sample. However, each latent span typically contains multiple latent steps rather than isolated calls, indicating that HyRea strategically groups latent reasoning into longer, coherent segments. This behavior highlights HyRea’s ability to balance interpretability and efficiency by using explicit reasoning in uncertain mid-process steps, while leveraging latent modules for confident, compressed segments.

## 5.2 ASSESSING MODEL GENERALIZATION CAPABILITY

We evaluate the generalization ability of the HyRea-trained model on two benchmark datasets: MMLU (Hendrycks et al., 2020) and GPQA (Rein et al., 2024). As shown in Table 3, HyRea is compared against models trained with supervised fine-tuning (SFT) and reinforcement learning (SFT+RL). While SFT+RL achieves the highest accuracy (73.4 on MMLU and 29.8 on GPQA), it generates substantially longer responses (102 and 1083 tokens, respectively). In contrast, HyRea produces more concise outputs (53 tokens on MMLU and 685 on GPQA) while maintaining competi tive accuracy (68.6 and 27.4). This demonstrates HyRea’s strong generalization ability to untrained domains, balancing performance and efficiency without task-specific optimization. Such generaliza tion is crucial for deploying language models in diverse, real-world settings where robustness and adaptability are essential.

## 6 RELATED WORKS

Chain-of-Thought Reasoning Chain-of-Thought (CoT) prompting (Wei et al., 2022) improves model performance by guiding the generation of intermediate reasoning steps. Such behavior can be elicited directly through prompt engineering without additional training (Khot et al., 2022; Zhou et al., 2022). To further enhance reasoning quality, recent works employ supervised fine-tuning and reinforcement learning to explicitly optimize multi-step reasoning (Yue et al., 2023; Yu et al., 2023; Wang et al., 2023; Xiong et al., 2025). An important extension of CoT involves integrating it with tree search algorithms (Yao et al., 2023; Hao et al., 2024a; Xie et al., 2023; Liao et al., 2025b), allowing models to explore and evaluate multiple reasoning paths to achieve better performance on complex tasks. These advances have been formalized into test-time inference scaling laws (Snell et al., 2024; Wu et al., 2024), which provide theoretical grounding for the observed performance improvements. The success of advanced reasoning models such as OpenAI’s O1 (OpenAI et al., 2024) and DeepSeek’s R1 (DeepSeek-AI et al., 2025) has further intensified interest in test-time search (TTS) techniques. However, the deliberative nature of these methods often leads to inefficiencies in token usage. As a result, reasoning efficiency has become an increasingly important research focus (Liu et al., 2025; Xu et al., 2025; Liao et al., 2025a; Aggarwal & Welleck, 2025).

Latent Reasoning Recent advances in LLM reasoning in latent space (Yang et al., 2024b; Biran et al., 2024) underscore the significance of hidden computations. To better leverage these latent dynamics, a range of methods introduce special tokens to explicitly guide reasoning. For instance, <pause> tokens Goyal et al. (2023) are incorporated during both pretraining and downstream finetuning, yielding consistent improvements over standard training. Similarly, non-semantic filler tokens (e.g., ‘...’) Pfau et al. (2024) have been shown to enhance performance on certain reasoning tasks. Several works explore alternatives to explicit chain-of-thought (CoT) prompting. Yu et al. (2024) proposes implicit CoT, where models internalize intermediate reasoning steps distilled from explicit CoT supervision. Wang et al. (2023) introduces a hierarchical reasoning structure guided by planning tokens, which demarcate latent reasoning stages. More recently, Hao et al. (2024b) proposes COCONUT, replacing discrete CoT tokens with continuous latent embeddings, leading to improved reasoning across multiple tasks. Zhu et al. (2025) provides theoretical justification for the superiority of continuous CoTs over their discrete counterparts. In contrast to these approaches, our work focuses on selectively replacing low-entropy tokens with continuous embeddings, allowing the model to dynamically switch between discrete and continuous reasoning modes. We formulate this switching behavior as a reinforcement learning problem, enabling the model to learn when and how to invoke latent computation for improved reasoning quality.

## 7 CONCLUSION

We presented HyRea, a hybrid reasoning framework that allows large language models to dynamically alternate between explicit token-based reasoning and latent embedding-based reasoning, aiming to balance interpretability, accuracy, and efficiency. To train this capability, we proposed a two-stage approach that begins with entropy-guided supervised fine-tuning, where low-entropy reasoning steps are selectively replaced with latent representations, followed by reinforcement learning using Group Relative Policy Optimization to refine the model’s reasoning strategy based on task-specific rewards. Extensive experiments on challenging mathematical benchmarks demonstrate that HyRea achieves substantial reductions in output length while maintaining or improving accuracy compared to strong baselines. Furthermore, HyRea exhibits strong generalization ability on non-mathematical tasks, showing its potential as a versatile and scalable solution for efficient multi-step reasoning in large language models.

## REFERENCES

Mathematical association of america. american mathematics competitions (amc). 2023.

Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.

Eden Biran, Daniela Gottesman, Sohee Yang, Mor Geva, and Amir Globerson. Hopping too late: Exploring the limitations of large language models on multi-hop queries. arXiv preprint arXiv:2406.12775, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. arXiv preprint arXiv:2310.02226, 2023.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Shibo Hao, Yi Gu, Haotian Luo, Tianyang Liu, Xiyan Shao, Xinyuan Wang, Shuhua Xie, Haodi Ma, Adithya Samavedhi, Qiyue Gao, et al. Llm reasoners: New evaluation, library, and analysis of step-by-step reasoning with large language models. arXiv preprint arXiv:2404.05221, 2024a.