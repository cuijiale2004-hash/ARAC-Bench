1. **Research Background and Existing Pain Points**
Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive language models. A DLM defines a forward process that gradually corrupts text into a noise prior, and learns a reverse process to recover clean text. Unlike autoregressive models, DLMs do not commit to a fixed left-to-right order, offering greater flexibility in generation and an inherent ability to predict multiple tokens in parallel. A dominant variant is the mask-based DLM, where the noise is represented by a special mask token. Under this formulation, the model learns to recover masked tokens during training, while assuming that once tokens are unmasked, they are supposed to be correct without having to clean them later. 

The key existing pain point is that this assumption is problematic: the model may generate wrong tokens, which should be revealed and corrected in later steps when more contexts are available. However, most existing DLMs keep already unmasked tokens fixed, preventing them from being revised by self-reflecting on errors. In early stages, limited unmasked tokens often lead to unreliable predictions, resulting in errors that persist through the remainder of the generation process. As generation progresses, additional context may reveal these errors, but the current paradigm offers no way to correct them. Existing efforts to address this fall into two categories: 1) Applying predictor-corrector samplers without training (e.g., stochastically remasking a subset of tokens during inference), which lack a mechanism to identify which tokens are actually wrong and rely on many extra sampling steps that are inefficient and hard to optimize; 2) Modifying the diffusion process by combining mask diffusion with uniform diffusion or edit-based diffusion, but none of these approaches provides a principled way to detect and selectively correct errors during generation. Furthermore, methods like Seed Diffusion allow all tokens to be resampled at every step, but lack a mechanism to ensure that the number of mask tokens decreases monotonically—a key feature for diffusion models to ensure decreasing noise levels over steps so that mask tokens eventually vanish at the final step.

2. **Core Research Motivation and Scientific Questions**
The core research motivation is to address the limitation of existing mask-based DLMs that they cannot revise generated tokens. The paper aims to train DLMs with the ability of finding wrong tokens and turning them back to mask ones so that they can be resampled with richer context in later steps. The key challenge is to identify potential errors in the inputs and train the model how to remask incorrect tokens in a self-reflective manner. 

The scientific questions addressed are: How can a mask-based DLM identify which already unmasked tokens are incorrect and should be remasked? How can the model be trained to detect and remask incorrect tokens while still predicting mask tokens correctly? And how can the remasking process be controlled to ensure that the total number of mask tokens decreases monotonically over steps, fulfilling the requirement of the reverse diffusion process?

3. **Overall Core Idea and Design Philosophy**
The overall core idea is to propose a self-reflective remasking approach to train DLMs, introducing Remasking-enabled Diffusion Language Model (RemeDi). RemeDi incorporates self-reflective remasking to revise already generated but incorrect tokens. The design philosophy is that RemeDi should jointly predict token distributions and per-token confidence scores at each step. The confidence scores determine which tokens to be unmasked after the current step, allowing the model to identify tokens with low quality and remask them. These remasked tokens can be resampled with richer context in subsequent steps. 

To achieve this, the model extends the standard transformer into a dual-stream transformer architecture comprising a Token Prediction Stream (TPS) and an Unmasking Policy Stream (UPS). A remask-aware pipeline is designed to train this ability in two stages: 1) Remask SFT, which teaches the model to detect and remask incorrect tokens in addition to predict mask tokens by treating incorrect tokens as a second noise type and supervising the unmasking scores; 2) Remask RL, which further fine-tunes the model with outcome-based reinforcement learning to optimize full generation trajectories toward higher rewards by considering how to remask and predict tokens in each step.

4. **Core Innovation Points**
1) **Self-Reflective Remasking Mechanism**: Introduces remasking as another fundamental mechanism in mask-based DLMs, enabling more flexible text refinement. Unlike existing DLMs where unmasked tokens are fixed, RemeDi re-decides a token to be unmasked or (re-)masked at each step by its trained confidence score, allowing previously unmasked tokens to be remasked and resampled later.
2) **Dual-Stream Transformer Architecture**: Proposes a dual-stream transformer architecture consisting of the Token Prediction Stream (TPS) to predict probabilities for masked tokens, and the Unmasking Policy Stream (UPS) to output token-wise confidence scores. The two streams perform bidirectional feature sharing to enrich their representations.
3) **Remask SFT with Monotonic Noise Decrease**: Designs a Remask SFT algorithm that treats incorrect tokens as a second noise type. It constructs training samples by applying both mask ratio and incorrect token ratio, and strictly ensures that the noise level (number of mask tokens) decreases monotonically over steps by satisfying a specific inequality constraint.
4) **Remask RL with Joint Policy Optimization**: Develops a Remask RL algorithm using outcome-based reinforcement learning (GRPO). It models the generation process as a joint policy of an unmasking policy (using the Plackett–Luce model to sample positions to unmask) and a token prediction policy (sampling tokens at chosen positions), optimizing the entire generation trajectory toward final outputs with higher rewards.

5. **Overview of the Overall Technical Solution**
The overall technical solution is centered on the RemeDi model, a mask-based DLM that identifies and remasks low-confidence tokens during generation to enable iterative self-reflection. The model architecture is a dual-stream transformer. The Token Prediction Stream (TPS) is a stack of transformer blocks that predict probabilities for masked tokens as in a typical DLM. The Unmasking Policy Stream (UPS) is another stack of transformer blocks that output token-wise confidence scores. During inference, UPS is inserted periodically, receives hidden states from TPS, and produces an auxiliary representation for confidence scoring. The two streams perform bidirectional feature sharing: UPS layers are conditioned on fTPS, and their outputs also feed back into TPS.

During inference, given the input sequence at a step, UPS first predicts a confidence score at each position and selects a subset of positions to unmask. For the selected positions, if they have already been unmasked in the input, they remain unchanged; otherwise, they are sampled from the predicted probabilities by TPS. Tokens with low confidence are remasked. A noise schedule controls that the total number of unmasked tokens increases linearly from 0 to L, ensuring the number of mask tokens approaches zero at the final step.

The training pipeline consists of two stages: Remask SFT and Remask RL. In Remask SFT, the model learns to identify and remask incorrect tokens while predicting masked tokens. The training samples are constructed by randomly masking tokens and replacing a subset of the remaining tokens with random alternatives. The model is supervised with a binary cross-entropy objective for the unmasking score, where clean tokens receive a positive label, incorrect tokens receive a negative label, and mask tokens receive a soft label equal to the predicted probability of the ground-truth token. In Remask RL, the model is fine-tuned using GRPO, where the joint policy of unmasking and token prediction is optimized to maximize rewards based on task verifiability or human preference.

6. **Detailed Module Design**
**Token Prediction Stream (TPS)**: A stack of transformer blocks that predict probabilities $p_i^\theta(\cdot|x_t)$ for masked tokens as in a typical DLM. It produces the final representation $f_{TPS}$ using an independent linear head.

**Unmasking Policy Stream (UPS)**: A stack of transformer blocks that output token-wise confidence score $h_i^\theta$. It represents the model's confidence over the output tokens, indicating if they should be unmasked with high confidence. Otherwise, if the confidence is too low for a token, it should be kept masked or remasked. It produces the auxiliary representation $f_{UPS}$ using an independent linear & softmax head.

**Bidirectional Feature Sharing**: UPS layers are conditioned on $f_{TPS}$, and their outputs also feed back into TPS to enrich its representations. Specifically, the two streams are weakly coupled via bi-directional connections at TPS blocks 1, 11, 21, and 31. At each connection point, the output from the previous TPS block is added to the current UPS feature to form the input for the next UPS block, while the output of the current UPS block is added to the output of the corresponding TPS block before it is passed onward. To preserve the TPS’s original capability inherited from the pretrained weights, a zero-initialized projection is added on the connections from UPS to TPS.

**Input Sequence Construction (Remask SFT)**: To simulate inference inputs at a diffusion time $t$, training samples $x_t$ are constructed from clean text $x_0$ by applying two types of noise: given a randomly sampled diffusion time $t \in (0, 1)$, set the corresponding mask ratio $\rho_{t,mask}$, alongside the incorrect token ratio $\rho_{t,incorrect}$. With both ratios, randomly mask tokens with $\rho_{t,mask}$. Then, among the remaining unmasked positions, sample a subset with the ratio $\rho_{t,incorrect}$ and replace each selected token with a random alternative to simulate the incorrect tokens.

**Remask RL Policies**:
- **Unmasking policy**: The UPS produces a per-token confidence score $h_i^\theta$. During RL training, construct an unmasking policy to sample positions using the Plackett–Luce model: based on $h_\theta$, use a multinomial distribution and sequentially sample $K_n$ positions from $\{1, \ldots, L\}$ without replacement.
- **Token prediction policy**: For each selected position, if it is masked in the input, the model samples token from $p_i^\theta(\cdot|x^{t_{n-1}})$; otherwise, the token remains unchanged.

7. **All Mathematical Formulas and Symbol Definitions**
**Forward Process ODE**:
$$ \frac{dp_t}{dt} = Q_t p_t, \quad p_0 = p_{data}, \quad p_T = p_{prior} $$

**Diffusion Loss**:
$$ L_{diffusion}(\theta) = \mathbb{E}_{t, x_0, x_t} \left[ - \frac{1}{t} \sum_{i=1}^L \mathbb{1}(x_i^t = [M]) \log p_i^\theta(x_i^0|x_t) \right] $$

**Monotonic Noise Decrease Constraint**:
$$ \lceil \rho_{t,incorrect} \cdot (1 - \rho_{t,mask}) \cdot L \rceil < \lceil \rho_{t,mask} \cdot L \rceil $$
(Choosing $\rho_{t,mask} = t$ and $\rho_{t,incorrect} = 4r \cdot t(1-t)$ with $r=0.1$ satisfies this inequality on $t \in [0, 1]$.)

**Unmask Label Definition**:
- A clean token ($i \in S_{clean} = \{i | x_i^t = x_i^0\}$) receives a positive unmask label $y_i = 1$.
- An incorrect token ($i \in S_{incorrect} = \{i | x_i^t \neq x_i^0, x_i^t \neq [M]\}$) receives a negative unmask label $y_i = 0$.
- A mask token ($i \in S_{mask} = \{i | x_i^t = [M]\}$) is assigned a soft unmask label $y_i = p_i^\theta(x_i^0|x_t)$ (equal to the predicted probability of the ground-truth token $x_i^0$). In Algorithm 1, this is denoted as $y_i = \text{stopgrad}(p_i^\theta(x_i^0|x_t))$ for $i \in S_{mask}$.

**UPS Loss**:
$$ L_{UPS}(\theta) = \sum_i BCE(\sigma(h_i^\theta), y_i) $$
(where $\sigma(\cdot)$ is the sigmoid function)

**Total Remask SFT Objective**:
$$ L(\theta) = L_{diffusion}(\theta) + \lambda_{UPS} L_{UPS}(\theta) $$
(where $\lambda_{UPS}$ balances the two losses)

**Unmasking Policy Probability (Plackett-Luce)**:
$$ \pi_{\theta, n}^{unmask}(U_n | x^{t_{n-1}}) = \prod_{k=1}^{K_n} \frac{\exp(h_{\theta, n}^{u_n(k)})}{\sum_{j \notin \{u_n(1), \ldots, u_n(k-1)\}} \exp(h_{\theta, n}^j)} $$

**Token Prediction Policy Probability**:
$$ \pi_{\theta, n}^{token}(x^{t_n} | x^{t_{n-1}}, U_n) = \prod_{i \in U_n, x_i^{t_{n-1}} = [M]} p_i^\theta(x_i^{t_n} | x^{t_{n-1}}) $$

**Joint Policy Probability**:
$$ \pi_{\theta, n}(x^{t_n} | x^{t_{n-1}}) = \pi_{\theta, n}^{unmask}(U_n | x^{t_{n-1}}) \cdot \pi_{\theta, n}^{token}(x^{t_n} | x^{t_{n-1}}, U_n) $$

8. **Algorithm Pseudocode**

**Algorithm 1 Input sequence construction and loss calculation in Remask SFT**
Require: Clean sequence $x_0 = [x_0^1, \ldots, x_0^L]$ of length $L$. Model $M$ with learnable parameters $\theta$.
1: Sample $t \in (0, 1)$ according to the noise schedule, obtaining $\rho_{t,mask}$ and $\rho_{t,incorrect}$.
2: Construct noisy input $x_t$:
3: For each position $i$, replace $x_0^i$ with $[M]$ w.p. $\rho_{t,mask}$
4: Among remaining positions, replace $x_0^i$ with a random alternative token w.p. $\rho_{t,incorrect}$
5: Define index sets:
$S_{mask} = \{i | x_i^t = [M]\}$, $S_{incorrect} = \{i | x_i^t \neq x_i^0 \land x_i^t \neq [M]\}$, $S_{clean} = \{i | x_i^t = x_i^0\}$
6: Get model outputs: $[p_\theta, h_\theta] = M(x_t; \theta)$
7: Calculate the diffusion loss, on mask tokens only: $L_{diffusion}(\theta) = - \frac{L}{|S_{mask}|} \sum_{i \in S_{mask}} \log p_i^\theta(x_i^0|x_t)$
8: Get labels for UPS:
$$ y_i = \begin{cases} 
1 & i \in S_{clean} \\
0 & i \in S_{incorrect} \\
\text{stopgrad}(p_i^\theta(x_i^0|x_t)) & i \in S_{mask} 
\end{cases} $$
9: UPS BCE loss: \triangleright $\sigma(\cdot)$ represents the sigmoid function
$$ L_{UPS}(\theta) = - \frac{1}{L} \sum_{i=1}^L \left( y_i \log \sigma(h_i^\theta) + (1 - y_i) \log (1 - \sigma(h_i^\theta)) \right) $$
10: Total loss: $L(\theta) = L_{diffusion}(\theta) + \lambda_{UPS} L_{UPS}(\theta)$

**Algorithm 2 Remask RL**
Require: Model parameters $\theta$, a dataset $\mathcal{D}$, and reward func.
1: while $\theta$ not converged and maximum epochs not reached do
2: Sample questions $q \sim \mathcal{D}$
3: for $g = 1$ to $G$ do \triangleright Generate a group of $G$ trajectories
4: Initialize $x_{t_0}^g$ with $q$ and mask tokens.
5: for $n = 1$ to $N$ do \triangleright $N$ denotes the number of denoising steps
6: Calculate the ranking score $h_{\theta, n}$ for each token
7: Sample $K_n$ positions to unmask in this step: $U_n \sim \text{Plackett-Luce}(h_{\theta, n}, K_n)$
8: Sample $x_{i}^{t_n, g} \sim p_{i, \theta, n}(\cdot|x_{t_{n-1}}^g)$, $\forall i \in U_n$, $x_{i}^{t_{n-1}, g} = [M]$
9: end for
10: $r_g = \text{reward func}(q, x_{t_N}^g)$ \triangleright Compute the rewards
11: end for
12: for $g = 1$ to $G$ do \triangleright Compute the advantages as in GRPO
13: $A_g = \frac{r_g - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}$
14: end for
15: for $n = 1$ to $N$ do \triangleright Compute $\pi_\theta$ and losses for each denoising step
16: $\pi_{\theta, n}(x_{t_n}^g | x_{t_{n-1}}^g) = \pi_{\theta, n}^{unmask}(U_n^g | x_{t_{n-1}}^g) \cdot \pi_{\theta, n}^{token}(x_{t_n}^g | x_{t_{n-1}}^g, U_n^g)$ \triangleright see Eq. 8
17: $L_{\theta, n} = - \frac{1}{G} \sum_{g=1}^G \frac{\pi_{\theta, n}(x_{t_n}^g | x_{t_{n-1}}^g)}{\pi_{old, n}(x_{t_n}^g | x_{t_{n-1}}^g)} A_g$
18: Calculate the gradient $\nabla_\theta L_{\theta, n}$
19: end for
20: Update $\theta$ with accumulated gradients $\sum_{n=1}^N \nabla_\theta L_{\theta, n}$ along the descent direction
21: end while