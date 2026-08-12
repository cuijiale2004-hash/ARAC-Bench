**1. Research Background and Existing Pain Points**

**Research Background:** Multimodal large language models (MLLMs) have achieved notable gains in various tasks by incorporating chain-of-thought (CoT) reasoning in language spaces. Recent work extends this direction by leveraging external tools for visual editing, thereby enhancing the visual signal along the reasoning trajectories. Current paradigms are primarily divided into "Thinking about Images" and "Thinking with Images." The "Thinking about Images" paradigm performs multimodal reasoning entirely in text space via CoT, where the LLM decomposes a query into intermediate steps and resolves each step while conditioning on static visual inputs. The "Thinking with Images" paradigm leverages external visual tools (such as drawing auxiliary lines, zooming in, shifting image styles, or highlighting sub-regions) to actively incorporate salient visual information throughout the reasoning process, injecting new image tokens as visual enhancements into subsequent textual reasoning.

**Existing Pain Points:** 
1. **Reasoning Confined to Language Space:** Both "Thinking about Images" and "Thinking with Images" remain fundamentally constrained because reasoning is still confined to the language space, with visual information treated as static preconditions or external augmentations rather than integral reasoning components.
2. **Textual Dominance and Modality Interference:** In the "Thinking about Images" paradigm, despite sophisticated visual encoders projecting visual information into text spaces, backbone LLMs often fail to capture the visual details most relevant to the text query due to modality projection bias, modality interference, and cross-modality attention bias. Excessive token generation may cause the textual context to dominate, overshadowing essential visual inputs.
3. **Tool Dependency and Bypassing:** The "Thinking with Images" paradigm relies on external tools to inject visual information during text generation. However, these approaches often bypass the newly injected sub-images due to training data bias, or remain constrained by the predefined operations of the tools. Tool APIs can be difficult to extend, and updates or changes often require substantial training effort.
4. **Information Loss in Omni-modality:** Omni-modality foundation models that generate explicit visual outputs for reasoning face issues where visual inputs, once decoded and re-encoded, may not faithfully preserve the original information. Additionally, relying on auxiliary images for supervision introduces extra labeling and pairing costs, limiting scalability.
5. **Limitation of Discrete Tokens:** Conventional LLMs are limited to operating on discrete tokens due to their next-token prediction training objective, restricting their ability to directly reason over continuous visual semantics.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:** Rethinking the input of MLLMs, where continuous visual tokens and discrete text tokens are embedded in a shared latent semantic space, the paper seeks to mimic human visual reasoning where humans can reason about images naturally without translating them into text. If visual and textual tokens are embedded in a joint semantic space within an MLLM, it is both natural and efficient to extend reasoning beyond discrete text tokens to include visual tokens that directly encode visual information. The motivation is to enable models to understand and reason directly in the visual space, unifying latent reasoning over visual inputs with text generation in the language space, thereby enabling deeper integration of visual and textual signals throughout the model’s reasoning process.

**Scientific Questions:** If visual and textual tokens are embedded in a joint semantic space within an MLLM, why not reason over both jointly as well? How can we enable autoregressive reasoning directly in the visual embedding space using the LLM's latent states, and how can we train such a model to stably and effectively balance latent visual reasoning with textual generation?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:** The paper introduces Latent Visual Reasoning (LVR), a new paradigm that enables autoregressive reasoning directly in the visual embedding space. A visual encoder first projects images into visual tokens within a joint semantic space shared with the language model. The language model is then trained to generate latent states that reconstruct key visual tokens critical for answering the query, constituting the process of latent visual reasoning. By interleaving LVR with standard text generation, the model achieves hybrid reasoning that alternates between reconstructing visual semantics and predicting discrete text tokens. 

**Design Philosophy:** The design philosophy centers on "Thinking via Visual Semantics Reconstruction." Unlike text-space CoT or external tool manipulation, LVR leverages the LLM’s latent space to reconstruct the semantics of Regions of Interest (ROIs), enabling seamless cross-modal reasoning. The hidden-state sequence produced during the LVR segment is regarded as an analogue of a human’s “visual thinking,” where query-relevant visual semantics are mentally reconstructed to enhance the precision of reasoning and answering. The model is designed to pass last hidden states rather than text tokens during LVR, enabling more efficient expression of complex thoughts. A two-stage training pipeline (SFT followed by RL) is designed to first instill the basic pattern of latent visual reasoning via teacher-forcing and then allow the model to self-evolve the latent reasoning process freely using reinforcement learning.

**4. Core Innovation Points**

1. **Novel Multimodal Reasoning Paradigm (LVR):** The paper proposes LATENT VISUAL REASONING, a novel multimodal reasoning paradigm that unifies latent reasoning over visual inputs with text generation in the language space. It extends the conventional Vision–Projector–LLM structure by allowing the LLM to perform hybrid reasoning that alternates between LVR (reconstructing visual semantics using last hidden states) and standard text generation, operating in an auto-regressive manner.
2. **Two-Stage Training Framework with Joint Objectives:** The paper introduces architectural innovations and training frameworks for stable and scalable training of MLLMs with LVR. The SFT stage combines a visual reconstruction loss (MSE) with next-token prediction (Cross-Entropy) to explicitly couple latent visual reasoning with downstream text generation, using bounding boxes to supervise the latent states.
3. **GRPOlatent for Reinforcement Learning on Latent Reasoning:** The paper adapts the GRPO algorithm to conduct reinforcement learning on latent reasoning, proposing GRPOlatent. This variant addresses the mismatch between policy-gradient loss (defined over token distributions) and the latent reasoning process (lacking explicit token distribution) by replaying recorded latent reasoning hidden states during the teacher-forcing log-probability pass for importance ratio computation.
4. **Decoding Strategies for LVR:** To address the challenge of determining when the model should exit the LVR process, the paper proposes and investigates three decoding strategies: Fixed Token (assigning a constant budget), Latent End Token (introducing a trainable tensor in the hidden state space), and Mode Switching Loss (adding an auxiliary BCE loss to supervise the token distribution toward the end token).

**5. Overview of the Overall Technical Solution**

The overall technical solution follows a standard MLLM design augmented with LVR capabilities, consisting of a vision encoder `vision(·)`, an LLM backbone `θ(·)`, and a multimodal projector `proj(·)`. Given an input image–question pair $(X_v, X_t)$, the vision encoder transforms the image into visual features $V = vision(X_v)$, and the textual question $X_t$ is embedded into language features $T$. The projector maps visual features into the language model's latent space: $VT = proj(V)$. 

During inference, the model operates in an interleaved manner. When the special token `<|lvr_start|>` is generated, the model enters a latent reasoning mode, where it reconstructs visual semantics in the space of $VT$. The hidden states produced in this mode are directly propagated as input embeddings to subsequent positions. When a stopping criterion is reached, the model generates `<|lvr_end|>` and resumes standard text generation. 

The training pipeline consists of two stages:
1. **Supervised Fine-Tuning (SFT):** Utilizes bounding box annotations to identify ROI patches and gather ground-truth visual embeddings. The model is trained with a joint loss combining Mean Squared Error (MSE) for visual reconstruction and Cross-Entropy (CE) for next-token prediction.
2. **Reinforcement Learning (RL):** Utilizes the GRPOlatent algorithm. No constraints are imposed on intermediate LVR outputs. Rewards are computed based on output accuracy and format adherence (presence of LVR special tokens). The policy gradient loss is computed by replaying the recorded hidden states from the rollout phase to ensure consistent conditional log-probabilities.

**6. Detailed Module Design**

**6.1. Input Embedding Module**
- **Visual Encoding:** The input image $X_v$ is processed by a vision encoder (ViT) to extract visual features $V = vision(X_v)$, which are then projected into the joint semantic space via a projector $VT = proj(V)$.
- **Text Embedding:** The textual question $X_t$ is embedded into language features $T$ by the LLM’s embedding layers.

**6.2. Latent Visual Reasoning (LVR) Module**
- **Triggering:** When the LLM generates the special token `<|lvr_start|>`, the model switches from standard text generation to latent reasoning mode.
- **Hidden State Propagation:** In this mode, instead of mapping hidden states to discrete tokens via the language modeling head, the last hidden states are directly passed forward as input embeddings to subsequent positions.
- **Reconstruction Target:** The hidden states are expected to encode the underlying visual semantics of the query-relevant regions (ROIs).
- **Termination:** The LVR process continues until a stopping criterion is met (e.g., a fixed number of steps is reached, or the `<|lvr_end|>` token is predicted), after which the model resumes standard text generation.

**6.3. SFT Training Pipeline**
- **ROI Selection:** For a given image, the model's image processor crops it into a grid of visual patches. Based on a pre-annotated bounding box, LVR selects the corresponding patches and retrieves their indices $I = \{I_1, I_2, I_3, \dots, I_{T_v}\}$ in $O(1)$ time.
- **Target Gathering:** During the forward pass, a subset of visual embeddings $\mathbf{v} = \{v_1, v_2, \dots, v_{T_v}\}$ is gathered using the index list $I$, corresponding to the visual patches within the ROI. These tokens are enclosed by `<|lvr_start|>` and `<|lvr_end|>`.
- **Joint Optimization:** The model is optimized using a joint loss $L = L_{NTP} + \lambda_{LVR} \cdot L_{LVR}$, balancing the Next-Token Prediction loss and the Visual Reconstruction Loss.

**6.4. RL Training Pipeline (GRPOlatent)**
- **Rollout and Recording:** The policy generates responses, and during rollout, the last hidden states of LVR processes $\tilde{h}^{latent}_i = \{h^{latent}_{i,1}, \dots, h^{latent}_{i,L}\}$ are recorded.
- **Reward Computation:** Rewards are derived solely from the text output $y$, consisting of a format reward (1 if response contains `<|lvr_start|>` and `<|lvr_end|>`, 0 otherwise) and an accuracy reward (1 if answer is correct, 0 otherwise).
- **Importance Ratio Calculation:** To evaluate importance ratios, a teacher-forcing forward pass is performed under both $\pi_\theta$ and $\pi_{\theta_{old}}$. The recorded hidden states $\tilde{h}^{latent}_i$ are patched into the latent-reasoning positions, restoring the exact context and ensuring consistent conditional log-probabilities.

**6.5. Decoding Strategies Module**
- **Fixed Token:** Assigns a constant budget of reasoning steps. Once the budget is reached, the model immediately exits latent reasoning mode. (Empirically found to achieve the best performance).
- **Latent End Token:** Introduces a trainable tensor in the hidden state space. When the last hidden state approaches this tensor (measured by distance metrics like cosine similarity, L1, or L2), the decoder resumes text-token generation. (Found to be unstable).
- **Mode Switching Loss:** Adds an auxiliary BCE loss during SFT that supervises the token distribution predicted by the language model head in the latent reasoning phase. It encourages the distribution of the final latent reasoning token toward `<|lvr_end|>` (close to 1), while all intermediate tokens are pushed away from `<|lvr_end|>` (close to 0). At inference, the model exits once `<|lvr_end|>` is predicted. (Failed to encode stopping conditions, collapsing to zero LVR steps).

**7. All Mathematical Formulas and Symbol Definitions**

- $X_v, X_t$: Input image and textual question pair.
- $V = vision(X_v)$: Visual features extracted by the vision encoder.
- $T$: Language features embedded by the LLM’s embedding layers.
- $VT = proj(V)$: Visual features projected into the representation aligned with the language model’s latent space.
- $I = \{I_1, I_2, I_3, \dots, I_{T_v}\}$: Indices of the flattened visual patches within the ROI.
- $\mathbf{v} = \{v_1, v_2, \dots, v_{T_v}\}$: Subset of visual embeddings gathered using the index list $I$, representing ground-truth visual embeddings.
- $\{h_t\}_{t=1}^{T_v}$: Sequence of the final hidden states predicted during latent reasoning.
- **Visual Reconstruction Loss ($L_{LVR}$):** Mean squared error (MSE) objective enforcing hidden states to approximate ground-truth visual embeddings:
$$L_{LVR} = \frac{1}{T_v} \sum_{t=1}^{T_v} \|h_t - v_t\|_2^2$$
- $\{y_t\}_{t=1}^{T_y}$: Final response tokens generated to answer the visual question.
- **Next-Token Prediction Loss ($L_{NTP}$):** Standard cross-entropy (CE) loss to maximize the likelihood of the ground-truth sequence:
$$L_{NTP} = - \frac{1}{T_y} \sum_{t=1}^{T_y} \log p_\theta(y_t | y_{<t}, h_{1:T_v})$$
- **Joint Objective ($L$):** Weighted sum of the two components:
$$L = L_{NTP} + \lambda_{LVR} \cdot L_{LVR}$$
- $\lambda_{LVR}$: Hyperparameter to balance reconstruction and generation signals.
- **GRPOlatent Objective ($J_{GRPOlatent}$):** The adapted GRPO algorithm objective function for latent reasoning:
$$J_{GRPOlatent}(\theta) = \mathbb{E}_{q,I, o \sim \pi_{\theta_{old}}} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) - \beta D_{KL}(\pi_\theta(\cdot | q, I) \| \pi_{ref}(\cdot | q, I)) \right]$$
- **Token-wise Importance Ratio ($r_{i,t}(\theta)$):** Computed using a teacher-forcing log-probability pass where latent reasoning hidden states are replayed:
$$r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} | q, I, \tilde{h}^{latent}_i, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | q, I, \tilde{h}^{latent}_i, y_{i,<t})}$$
- $\tilde{h}^{latent}_i = \{h^{latent}_{i,1}, \dots, h^{latent}_{i,L}\}$: Recorded last hidden states of LVR processes during rollout.
- **Group-normalized Reward ($\tilde{R}_i$):** Normalized reward derived solely from the text output $y$:
$$\tilde{R}_i = \frac{R(y_i) - \text{mean}(R(y_1), \dots, R(y_G))}{\text{std}(R(y_1), \dots, R(y_G))}$$
- **Advantage ($\hat{A}_{i,t}$):** Defined based on the group-normalized reward:
$$\hat{A}_{i,t} = \tilde{R}_i \quad (\forall t \in \{1, \dots, |y_i|\})$$

**8. Algorithm Pseudocode**

```algorithm
// SFT Training Step
Input: Image X_v, Question X_t, ROI bounding box, Response Y
1. V <- vision(X_v)
2. VT <- proj(V)
3. T <- EmbedText(X_t)
4. Identify ROI patch indices I from bounding box
5. v <- GatherVisualEmbeddings(VT, I) // {v_1, ..., v_{T_v}}
6. Construct Input Sequence: [VT, T, <|lvr_start|>, v, <|lvr_end|>, Y]
7. Forward pass through LLM theta
8. Extract hidden states {h_t} corresponding to LVR segment
9. Compute L_LVR <- (1/T_v) * SUM(||h_t - v_t||_2^2)
10. Compute L_NTP <- CrossEntropy(Y_predicted, Y)
11. Compute Loss L <- L_NTP + lambda_LVR * L_LVR
12. Backpropagate and update theta

// GRPOlatent Training Step
Input: Question q, Image I, Old policy pi_theta_old, Reference policy pi_ref
1. // Rollout Phase
2. For i = 1 to G: // Generate G responses
3.    o_i, h_tilde_latent_i <- Rollout(pi_theta_old, q, I) 
4.    R_i <- ComputeReward(o_i) // Format + Accuracy Reward
5. Compute group-normalized rewards R_tilde_i for all G samples
6. Set Advantage A_hat_i,t <- R_tilde_i
7. 
8. // Policy Gradient Phase
9. For i = 1 to G:
10.   // Teacher-forcing pass with replayed latent states
11.   Compute log_pi_theta <- LogProb(pi_theta, o_i, q, I, h_tilde_latent_i)
12.   Compute log_pi_theta_old <- LogProb(pi_theta_old, o_i, q, I, h_tilde_latent_i)
13.   Compute importance ratio r_i,t(theta) <- exp(log_pi_theta - log_pi_theta_old)
14.   Compute clipped objective term <- min(r_i,t(theta)*A_hat_i,t, clip(r_i,t(theta), 1-epsilon, 1+epsilon)*A_hat_i,t)
15. Compute J_GRPOlatent <- Mean(clipped objective terms) - beta * D_KL(pi_theta || pi_ref)
16. Update theta using gradient ascent on J_GRPOlatent
```