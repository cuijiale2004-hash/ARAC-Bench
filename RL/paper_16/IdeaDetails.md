**1. Research Background and Existing Pain Points**
Large Language Models (LLMs) have shown strong performance in complex reasoning tasks, especially when guided by Chain-of-Thought (CoT) prompting, which encourages step-by-step intermediate reasoning. However, conventional CoT reasoning operates entirely in the discrete token space, leading to inefficiencies in computation and memory due to verbose outputs. As long-context and cost-effective inference become increasingly important, especially in reinforcement learning (RL) settings for math tasks, methods like Deepseek-R1 face high token costs and slow convergence. To tackle these challenges, recent approaches propose operating directly in the embedding space, bypassing tokenization altogether. These latent reasoning methods offer substantial token compression and improved efficiency. However, reasoning over embeddings remains inherently difficult and can lead to degraded model performance compared to traditional token-based inference, as certain tokens encode complex, nuanced information that cannot be fully preserved in compressed embeddings. This loss of semantic fidelity can be particularly detrimental in tasks that require precise symbolic manipulation, such as mathematical reasoning or code generation, where even small distortions in representation may lead to incorrect conclusions. Moreover, current models lack the capability to selectively determine which tokens should be compacted and which should remain in their original form. Compression is typically applied uniformly or based on fixed heuristics, without accounting for the varying informational value of different tokens within a reasoning trace. This inability to adaptively balance compression and fidelity limits the effectiveness of latent reasoning and highlights the need for more principled, content-aware strategies for selective representation.

**2. Core Research Motivation and Scientific Questions**
The core research motivation stems from the need to improve the efficiency of the reasoning process while minimizing performance degradation. The central scientific questions addressed are: How can LLMs be enabled to flexibly switch between explicit (token-based) and latent (embedding-based) reasoning during inference? When decoding the next position, how can the model autonomously decide whether to operate in the embedding space or the token space—that is, whether to perform latent reasoning or explicit reasoning? How can the model be trained to make these decisions effectively, selecting the most suitable reasoning strategy based on the context? How to selectively determine which tokens should be compacted and which should remain in their original form, preventing the compaction of tokens that encode complex or critical information that cannot be faithfully represented by latent embeddings? These scientific questions drive the design of a hybrid reasoning framework, aiming to balance interpretability, accuracy, and efficiency by dynamically leveraging both explicit and latent reasoning strategies.

**3. Overall Core Idea and Design Philosophy**
The overall core idea is to propose a novel Hybrid Reasoning (HyRea) framework that combines explicit (token-based) and latent (embedding-based) reasoning. Explicit reasoning offers interpretability through step-by-step generation but is inefficient, while latent reasoning improves efficiency by operating in embedding space, often sacrificing clarity and performance. HyRea enables LLMs to dynamically switch between these two modes during inference, selecting the most suitable reasoning strategy based on the context. This adaptive mechanism allows the model to maintain high accuracy while significantly reducing the number of generated tokens, yielding better trade-offs between performance and efficiency. To support this hybrid reasoning capability, the design philosophy introduces a two-stage training framework. In the supervised cold-start phase, the model learns to use latent reasoning by partially replacing intermediate CoT steps with continuous embeddings. To guide this replacement, steps with low entropy are prioritized, which are more deterministic and thus easier for the model to learn in latent space. This encourages the model to interpret and generate latent computations within a broader reasoning trajectory. In the subsequent reinforcement learning phase, the model's decision-making process is further refined using Group Relative Policy Optimization (GRPO), a RL technique that stabilizes learning by evaluating policies within sampled groups. This phase enables the model to learn when to invoke explicit versus latent reasoning based on downstream rewards, such as accuracy, format correctness, and efficiency. By integrating explicit and latent reasoning in a unified and learnable framework, HyRea provides a flexible and scalable approach to improve reasoning performance in LLMs.

**4. Core Innovation Points**
*   **Dynamic Hybrid Reasoning Framework (HyRea):** A unified framework that enables LLMs to dynamically switch between explicit (token-based) and latent (embedding-based) reasoning during inference. The model autonomously decides whether to operate in the embedding space or the token space at each decoding position.
*   **Entropy-Guided Replacement Strategy:** In the cold-start phase, low-entropy CoT steps are selectively replaced with continuous embeddings. This prioritizes steps that are more deterministic and easier for the model to learn in latent space, preventing the compaction of tokens that encode complex, nuanced information that cannot be faithfully represented by latent embeddings.
*   **Two-Stage Training Pipeline:** A structured training approach consisting of a supervised cold-start phase to introduce preliminary switching capabilities and latent reasoning, followed by a reinforcement learning phase to refine the model's reasoning strategy and decision-making process based on task-specific rewards.
*   **Reinforcement Learning with Group Relative Policy Optimization (GRPO):** Adopting GRPO to fine-tune the model's reasoning strategy based on rewards reflecting correctness and efficiency. This eliminates the need for manually constructing synthetic data to train the model’s switching capability, allowing the model to learn to perform hybrid reasoning in a self-supervised manner.
*   **Adaptive Mode Switching with Special Tokens:** The use of `<start-latent>` and `<end-latent>` tokens to explicitly mark the latent reasoning span, enabling a clear structural representation of hybrid reasoning that separates interpretable token steps from compact latent segments.
*   **Reward-Driven Latent Usage:** Incorporating a specific latent reward within the reinforcement learning phase to guide the model in generating the `[Latent]` token, effectively balancing correctness, structure, and reasoning mode usage.

**5. Overview of the Overall Technical Solution**
The overall technical solution is built upon the integration of explicit and latent reasoning mechanisms. Explicit reasoning refers to the process by which a model generates reasoning outputs in an autoregressive manner, decoding one token at a time. Given an input query, the model maps each token to its corresponding embedding via the embedding layer. These embeddings are then processed by a series of Transformer blocks. The hidden state from the final layer is transformed into a probability distribution over the vocabulary set through the language modeling head, and the next token is selected by taking the argmax over the resulting logits. Latent reasoning bypasses tokenization entirely by operating directly on embeddings. During decoding, the model feeds the hidden output from the final layer back into the first layer as input. Rather than decoding the next token, the model concatenates the initial embedding sequence with the hidden state at time step t and processes it through the Transformer. To improve efficiency while minimizing performance degradation, the Hybrid Reasoning mechanism allows the model to flexibly switch between explicit and latent reasoning during inference. At position i, if the model chooses explicit reasoning mode, it generates the next token. If the model opts for latent reasoning mode, it applies latent reasoning, inserting a `<start-latent>` token at the beginning and a `<end-latent>` token at the end to mark the latent reasoning span. The training method consists of two stages. Stage 1 is a supervised cold-start phase where low-entropy reasoning steps are replaced with a latent segment comprising special tokens and continuous latent tokens. The loss is computed only over the visible (non-latent) tokens. The number of steps replaced increases progressively. Stage 2 is a reinforcement learning phase using GRPO. The model samples outputs, evaluates rewards based on accuracy, formatting, and latent usage, computes the GRPO loss, and updates the model. During loss computation, the `[Latent]` token is excluded.

**6. Detailed Module Design**
*   **Explicit Reasoning Module:**
    *   Input query definition: x = [x1, x2, · · · , xt]
    *   Embedding layer mapping: E = [e(x1), e(x2), · · · , e(xt)]
    *   Transformer processing: Ht = [h1, h2, · · · , ht] = Transformer(E), where hi denotes the hidden embeddings at the i-th position from the final layer of the Transformer.
    *   Language modeling head and next token selection: x̂t+1 = argmaxV (logits) = argmaxV ( LMhead(ht) ), where V is the vocabulary set and LMhead is the language modeling head.
    *   Input update mechanism: The input is then updated to [x1, x2, · · · , xt, x̂t+1], which is subsequently processed sequentially by the Transformer equation followed by the language modeling head equation.

*   **Latent Reasoning Module:**
    *   Hidden state feedback mechanism: Ht+1 = [h1, h2, · · · , ht, ht+1] = Transformer(E ∥ ht), where E denotes the initial embedding sequence and ht is the hidden state at time step t and ∥ represents concatenation.
    *   Operating directly on embeddings allows for a more compact representation during inference, potentially reducing token consumption, bypassing tokenization entirely.

*   **Hybrid Reasoning Switching Mechanism:**
    *   At position i, the model autonomously decides whether to employ the explicit reasoning mode (following the language modeling head equation) or the latent reasoning mode (applying the latent reasoning equation).
    *   If the model opts for the latent reasoning mode, it inserts a `<start-latent>` token at the beginning and a `<end-latent>` token at the end to mark the latent reasoning span.
    *   Ideal structure representation: [Question] [Step1] · · ·<start-latent> [latent] <end-latent> · · · [StepN ] · · · [Answer].

*   **Cold-Start Supervised Fine-Tuning Module:**
    *   Original CoT sequence definition: C := [Question], [Step1], [Step2], . . . , [StepN ], [Answer].
    *   Latent segment definition: [Latent] := <start-latent> c × [latent]<end-latent>, where c denotes the number of latent tokens.
    *   Entropy-based replacement strategy: Select steps that have low entropy and replace them with the latent segment. The entropy-based replacement strategy prevents the model from compacting tokens that encode complex or critical information that cannot be faithfully represented by latent embeddings.
    *   Loss computation: Lcold start := − logLLM(C \ [Latent]), where the loss is computed only over the visible (non-latent) tokens, ensuring that the model focuses on learning the interpretable parts of the reasoning trace while implicitly handling the latent segments.
    *   Progressive replacement constraint: The number of steps replaced by [Latent] increases progressively during training, starting from 0 and gradually reaching a predefined maximum threshold S.
    *   Loss scaling constraint: During the cold-start stage, the loss on `<start-latent>` and `<end-latent>` are scaled by a factor of 4 to emphasize their importance and help the model learn when to switch.

*   **Reinforcement Learning Module (GRPO):**
    *   Group Relative Policy Optimization (GRPO) loss function: LGRPO(θ) = Eq∼P (Q), {oi}Gi=1∼πθold (O|q)[ 1/G ∑Gi=1 ( min( πθ(oi | q)/πθold(oi | q) Ai, clip( πθ(oi | q)/πθold(oi | q), 1− ε, 1 + ε) Ai ) )]
    *   Advantage calculation: Ai = (ri −mean({r1, r2, · · · , rG})) / std({r1, r2, · · · , rG})
    *   Reward design: The reward consists of both a format reward and an accuracy reward, as well as a latent reward. Specifically, the latent reward is used to guide the model in generating the token [Latent].
    *   Loss computation constraint: During loss computation, the [Latent] token is excluded and the loss is calculated only over the remaining token steps.
    *   Training effect: Through reinforcement learning, this approach eliminates the need for manually constructing synthetic data to train the model’s switching capability. Instead, the model learns to perform hybrid reasoning in a self-supervised manner, guided by reward signals that reflect both correctness and efficiency.

**7. All Mathematical Formulas and Symbol Definitions**
*   Equation 1: Ht = [h1, h2, · · · , ht] = Transformer(E)
    *   E: embedding sequence [e(x1), e(x2), · · · , e(xt)]
    *   hi: hidden embeddings at the i-th position from the final layer of the Transformer
    *   Ht: sequence of hidden embeddings up to position t
*   Equation 2: x̂t+1 = argmaxV (logits) = argmaxV ( LMhead(ht) )
    *   x̂t+1: the next token
    *   V: vocabulary set
    *   LMhead: language modeling head
    *   ht: hidden state at time step t
*   Equation 3: Ht+1 = [h1, h2, · · · , ht, ht+1] = Transformer(E ∥ ht)
    *   E: initial embedding sequence
    *   ht: hidden state at time step t
    *   ∥: concatenation
    *   Ht+1: sequence of hidden embeddings up to position t+1
*   Equation 4: C := [Question], [Step1], [Step2], . . . , [StepN ], [Answer].
    *   C: original CoT data sequence
*   Equation 5: Lcold start := − logLLM(C \ [Latent])
    *   Lcold start: cold-start loss
    *   LLM(·): language model’s likelihood function
    *   C \ [Latent]: visible (non-latent) tokens of the sequence
    *   [Latent]: latent segment <start-latent> c × [latent]<end-latent>, where c denotes the number of latent tokens
*   Equation 6: LGRPO(θ) = Eq∼P (Q), {oi}Gi=1∼πθold (O|q)[ 1/G ∑Gi=1 ( min( πθ(oi | q)/πθold(oi | q) Ai, clip( πθ(oi | q)/πθold(oi | q), 1− ε, 1 + ε) Ai ) )]
    *   θ: policy parameters
    *   P(Q): query distribution
    *   πθold: old policy
    *   oi: i-th output sampled from the old policy
    *   G: group size
    *   ε: clipping threshold
    *   Ai: advantage
*   Equation 7: Ai = (ri −mean({r1, r2, · · · , rG})) / std({r1, r2, · · · , rG})
    *   ri: reward for the i-th output
    *   G: group size

**8. Algorithm Pseudocode**
Algorithm 1 Hybrid Reasoning Training (HyRea)
Input: Pretrained language model LLM, CoT training data DSFT, RL training data DRL, max latent steps S, group size G, clipping threshold ε
Output: Hybrid reasoning model LLM
1: // Stage 1: Cold-Start Supervised Fine-Tuning
2: for s from 1 to S do
3:   for each batch (q, a) in DSFT do
4:     Identify reasoning steps with low entropy
5:     Replace s selected steps with [<start-latent> latent <end-latent>]
6:     Compute loss LSFT = − logLLM(q \ [latent])
7:     Update LLM using gradient descent on LSFT
8:   end for
9:   Incrementally introduce 10% new data into DSFT at each iteration
10: end for
11: // Stage 2: Reinforcement Learning with GRPO
12: for each query q in DRL do
13:   Sample G outputs {o1, o2, . . . , oG} ∼ LLMold(·|q)
14:   Evaluate rewards {r1, . . . , rG} (accuracy, formatting, latent usage)
15:   Compute GRPO loss by Equation 6
16:   Update LLM using gradient descent on LGRPO
17: end for