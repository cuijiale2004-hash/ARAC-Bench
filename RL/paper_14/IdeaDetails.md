## 1. Research Background and Existing Pain Points

**Research Background:**
Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains, including natural language processing, question answering, and planning tasks. Their abundant prior knowledge enables them to tackle previously unattainable challenges, particularly in fields such as mathematical reasoning and code generation. While inherently powerful, LLMs often require post-training to better align their capabilities with downstream tasks. Post-training refines base models by strengthening specific abilities, addressing gaps in reasoning patterns, domain expertise, or human preferences. Among various post-training paradigms, reinforcement learning (RL) has recently emerged as a particularly promising approach, offering significant performance improvements for LLMs across downstream tasks. RL’s advantages are compelling: it eliminates the need for large-scale curated demonstrations and holds the potential to achieve superhuman performance in tasks where well-defined reward signals are available to guide optimization. To address the sample inefficiency of RL and the limitations of base models in discovering complex reasoning patterns, many post-training pipelines incorporate a supervised fine-tuning (SFT) cold-start phase that enables models to acquire essential reasoning patterns, such as long chain-of-thought (CoT) reasoning, before RL training begins.

**Existing Pain Points:**
1. RL algorithms are notoriously sample-inefficient, requiring extensive exploration to learn complex reasoning patterns from scratch. Moreover, RL’s effectiveness heavily depends on the quality and characteristics of the base model—models with limited reasoning capabilities may struggle to discover appropriate reasoning patterns and fail to benefit from RL training.
2. Despite the widespread adoption of cold-start phases, fundamental questions remain unanswered: How long should the cold-start phase last to optimally prepare base LLMs for subsequent RL training? 
3. Is the standard SFT objective of demonstration imitation well aligned with the goal of RL preparation? These open questions highlight the need for deeper understanding of effective cold-start design.
4. The SFT checkpoint with the highest evaluation performance often fails to maximize RL potential due to distributional forgetting—a phenomenon where the model drifts excessively away from the base model’s distribution even before traditional overfitting occurs. Deterioration of RL potential occurs before SFT overfitting, revealing a fundamental misalignment between cold-start objectives (preparing models for RL) and CE loss objectives (maximizing demonstration imitation).

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The objective of the cold-start phase is crucial for steering base models toward a promising starting point for RL training, particularly when models lack domain-specific knowledge or reasoning capabilities (e.g., long-CoT patterns). The investigation reveals that the objective of the cold-start phase does not necessarily align with the purpose to prepare base model for subsequent RL training, particularly when demonstration data is limited. While evaluation performance continues improving during the readaptation phase of cold-start, corresponding post-RL performance begins declining. This indicates that the RL potential deteriorates before overfitting to the cold-start dataset. There is a critical need to reframe the cold-start objective from complete demonstration imitation to achieving an optimal trade-off between learning new patterns and preserving distribution from the original base model. 

**Scientific Questions:**
1. How long should the SFT cold-start phase last, and what criteria should dictate early stopping to maximize RL potential?
2. Is the standard SFT objective truly aligned with the requirements for effective RL preparation?
3. How can the cold-start phase dynamically balance the acquisition of new reasoning patterns with the preservation of the base model's original distribution at a fine-grained level?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to optimize the cold-start SFT phase for RL preparation by shifting the training objective beyond simple dataset imitation. Instead of selecting the best-performing checkpoint after SFT cold-start based on evaluation metrics, the model should select the checkpoint corresponding to the diversity measurement turning point, which indicates the onset of base model distribution forgetting. Furthermore, to address the lack of flexibility in vanilla early stopping (which applies uniform stopping criteria across the entire dataset), an Adaptive Early-Stop Loss (AESL) is proposed to provide fine-grained trade-off control at the token and subsequence level. AESL dynamically balances new pattern acquisition with base distribution preservation, recognizing that overfitting and distribution forgetting occur at different rates for different tokens and contexts due to varying "distance" between the base model and cold-start dataset distributions.

**Design Philosophy:**
The design philosophy is rooted in the observation that diversity metrics, such as entropy and self-BLEU, serve as more reliable early-stopping criteria than standard performance-based checkpoint selection. The diversity peak represents an optimal balance point—a "sweet spot" where the model successfully acquires new reasoning patterns while sufficiently retaining distribution from the base model for effective RL training. The enhanced entropy at this point likely stems from the model maintaining dual distribution characteristics from both the original base model and the new dataset patterns. At the token level, the learning should be slowed when the model already assigns high probabilities to correct tokens, thereby preserving original knowledge. At the subsequence level, since tokens following high-probability prefixes contribute more to overall diversity, the token-level early stopping should be scaled with the average log-probability of the prefix context to maintain base distribution when the prefix already aligns closely with the dataset distribution.

## 4. Core Innovation Points

1. **Discovery of Distributional Forgetting:** The paper uncovers a key limitation where the SFT checkpoint with the highest evaluation performance often fails to maximize RL potential due to distributional forgetting—a phenomenon where the model drifts excessively away from the base model’s distribution even before traditional overfitting occurs. It demonstrates a fundamental misalignment between cold-start objectives (preparing models for RL) and CE loss objectives (maximizing demonstration imitation).

2. **Diversity as Early-Stopping Criteria:** It demonstrates that diversity metrics, such as entropy and self-BLEU, serve as superior early-stopping criteria compared to evaluation performance. Selecting checkpoints based on diversity consistently yields better RL performance than relying solely on evaluation metrics.

3. **Adaptive Early-Stop Loss (AESL):** Building on diversity insights, the paper proposes AESL, a novel cold-start method that dynamically balances new pattern acquisition with base distribution preservation at both token and subsequence levels, providing fine-grained control over the preparation process.

4. **Fine-Grained Token and Subsequence Level Control:** AESL operates at both the token and subsequence levels. At the token level, it gradually reduces the loss contribution for tokens where the ground truth already corresponds to high probability under the current policy. At the subsequence level, it scales the token-level early stopping with the average log-probability of the prefix context, recognizing that tokens following high-probability prefixes contribute more to overall sequence diversity.

## 5. Overview of the Overall Technical Solution

The overall technical solution integrates the Adaptive Early-Stop Loss (AESL) cold-start method into standard post-training pipelines to prepare LLMs for RL post-training. Instead of using the standard cross-entropy (CE) loss that aims for complete demonstration imitation, AESL employs a weighted version of the original CE loss. The adaptive weighting function is designed to reduce the loss contribution for tokens where the ground truth already has high probability under the current policy, scaled by the prediction confidence of the prefix context. 

During the SFT cold-start phase, the model is trained on a lightweight demonstration dataset using AESL. The weighting function explicitly accounts for prefix prediction confidence, preventing excessive drift from the base model's distribution. By steering the LLM toward a better initialization point where it maintains dual distribution characteristics (acquiring new patterns while preserving the base distribution), AESL ensures enhanced response diversity (higher entropy and lower self-BLEU). This superior starting point subsequently allows the RL phase (using algorithms like GRPO) to achieve significantly better sample efficiency and final performance compared to traditional CE-based cold-start strategies.

## 6. Detailed Module Design

**1. Reinforcement Learning Module (GRPO):**
The RL phase utilizes the Group Relative Policy Optimization (GRPO) algorithm to optimize the policy (the LLM) based on verifiable rewards. The objective is to maximize the expected reward from a verifier by optimizing a clipped surrogate objective while penalizing deviations from a reference policy via KL divergence.

**2. Standard Cold-Start Module (CE Loss):**
The traditional cold-start phase employs the cross-entropy (CE) loss as the training objective to align the model’s predictions with the provided demonstrations. This module suffers from distributional forgetting as it purely maximizes demonstration imitation without preserving the base model's prior distribution.

**3. Adaptive Early-Stop Loss (AESL) Module:**
This is the core proposed module that replaces the standard CE loss during the cold-start phase. It operates via two integrated mechanisms:
- **Token-Level Mechanism:** The weighting function gradually reduces the loss contribution for tokens where the ground truth already corresponds to high probability under the current policy. This mechanism slows learning when the model already assigns high probabilities to correct tokens, thereby preserving original knowledge while still allowing adaptation to new patterns.
- **Subsequence-Level Mechanism:** AESL incorporates subsequence-level considerations by scaling the token-level early stopping with the average log-probability of the prefix context. Because sequence diversity measured by entropy can be decomposed such that tokens following high-probability prefixes contribute more to overall diversity, this scaling encourages the model to maintain base distribution when the prefix already aligns closely with the dataset distribution, balancing adaptation to demonstrations with preservation of the distribution of the base models.

## 7. All Mathematical Formulas and Symbol Definitions

**1. GRPO Objective Function:**
$L_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}_{\text{RL}}, \{s_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(s|q)} \frac{1}{G} \sum_{i=1}^G \frac{1}{|s_i|} \sum_{t=1}^{|s_i|} \left\{ \min \left[ \frac{\pi_\theta(s_t^i|q, s_{<t}^i)}{\pi_{\theta_{\text{old}}}(s_t^i|q, s_{<t}^i)} \hat{A}_t^i, \text{clip}\left(\frac{\pi_\theta(s_t^i|q, s_{<t}^i)}{\pi_{\theta_{\text{old}}}(s_t^i|q, s_{<t}^i)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t^i \right] - \beta D_{\text{KL}}[\pi_\theta||\pi_{\text{ref}}] \right\}$

**2. Cross-Entropy (CE) Loss Function:**
$L_{\text{CE}}(\theta) = -\mathbb{E}_{q,s^* \sim \mathcal{D}_{\text{SFT}}} [\log \pi_\theta(s_t^*|q, s_{<t})]$

**3. Adaptive Early-Stop Loss (AESL) Function:**
$L_{\text{Ada-stop}}(\theta) = -\mathbb{E}_{q,s^* \sim \mathcal{D}_{\text{SFT}}} [p(q, s_t^*, \pi_\theta) \cdot \log \pi_\theta(s_t^*|q, s_{<t})]$

**4. AESL Adaptive Weighting Function:**
$p(q, s_t^*, \pi_\theta) = 1 - \text{softmax}\left[y(s_t^*|q, s_{<t}) - t_{\text{scaling}} \cdot \frac{1}{|t|} \sum_{i=1}^t \log \pi_\theta(s_i^*|q, s_{<i})\right]$
where $\text{softmax}[x_i] = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$

**5. Sequence Entropy and Diversity Decomposition:**
$H(s_t|q) = -\sum_{s_t} \pi(s_t|q) \log \pi(s_t|q)$
$= -\sum_{s_t} \pi(s_t|s_{t-1})\pi(s_{t-1}|q) \log [\pi(s_t|s_{t-1})\pi(s_{t-1}|q)]$
$= -\sum_{s_t} \pi(s_t|s_{t-1})\pi(s_{t-1}|q) \log \pi(s_t|s_{t-1}) - \sum_{s_t} \pi(s_t|s_{t-1})\pi(s_{t-1}|q) \log \pi(s_{t-1}|q)$
$= \sum_{s_{t-1}} \pi(s_{t-1}|q)H(s_t|s_{t-1}) + H(s_{t-1}|q)$

**Symbol Definitions:**
- $\pi_\theta(s_t|q, s_{<t})$: The policy (i.e., the LLM) representing the probability of the current token output $s_t$ given input question $q$ and the first $t-1$ generated tokens $s_{<t}$.
- $R(s_t, q)$: The verifier that provides reward signals.
- $\mathcal{D}_{\text{RL}}$: The question set used for RL.
- $G$: The group size in GRPO.
- $s_i$: A complete rollout sequence.
- $\pi_{\theta_{\text{old}}}$: The policy before the update.
- $\pi_{\text{ref}}$: The reference policy.
- $\hat{A}_t^i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$: The advantage estimate normalized within the group, where $r = R(s_t, q)$ is the verifier reward and $r_i$ is the reward for the $i$-th rollout.
- $\mathcal{D}_{\text{SFT}}$: The demonstration dataset.
- $y$: The output logits before softmax normalization.
- $t_{\text{scaling}}$: A hyperparameter that controls the balance between learning new reasoning patterns and preserving distribution from the base model.

## 8. Algorithm Pseudocode

The paper does not provide explicit algorithm pseudocode.