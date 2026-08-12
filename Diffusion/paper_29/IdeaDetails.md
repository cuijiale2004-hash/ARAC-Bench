**1. Research Background and Existing Pain Points**

**Research Background:**
Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. Diffusion models, originally introduced for generation in continuous domains like images, have recently gained attention in text generation for their ability to maintain global coherence and enable iterative refinement. Prior works have explored continuous or latent diffusion for language generation, operating diffusion in latent spaces obtained from text autoencoders or token-embedding spaces. Existing works largely emphasize the parallelization properties of diffusion models or evaluate fluency in text generation. 

**Existing Pain Points:**
1. **Autoregressive Decoding Limitations:** LLM's autoregressive (AR) decoding may limit the ability to revisit and refine earlier tokens in a holistic manner. Their sequential nature prevents revising earlier tokens, making self-refinement inefficient and difficult. 
2. **Limited Reasoning Diversity:** AR models with discrete CoT generate a linear chain of thought, which limits reasoning diversity and restricts exploration of multiple valid solutions. AR models tend to collapse to similar trajectories and produce repetitive solutions.
3. **Semantic Refinement Inability of Discrete Diffusion:** Discrete diffusion language models merely transit into masked tokens and cannot achieve self-refinement at the semantic level in latent space, failing to correct earlier semantic reasoning errors iteratively.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
A more important direction is to ask: How can these approaches enhance the reasoning capabilities of LLMs? The focus is on one particularly promising capability: the ability to self-correct and refine reasoning chains at semantic levels in latent space. Such self-refinement cannot be achieved by discrete diffusion language models. Modeling reasoning at the semantic level, rather than at the token level, may lead to more faithful intermediate steps that accumulate into stronger final answers.

**Scientific Questions:**
How to unify the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM, while explicitly designing a framework for latent reasoning that learns causal dependencies across reasoning steps, propagates answer correctness signals back to latent tokens, and enables diverse exploration of reasoning trajectories?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
Propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that separates reasoning from answering. It unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. 

**Design Philosophy:**
1. Construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations.
2. Utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute.
3. Introduce a diversity-guidance mechanism during diffusion inference to push latent trajectories apart within a batch to explore multiple diverse reasoning paths.
4. Propagate answer correctness signals back to latent tokens via a two-stage training process, allowing answer supervision to backpropagate through the trajectory and directly shape latent predictions.

**4. Core Innovation Points**

1. **Iterative Refinement at Semantic Level:** Unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. Unlike prior latent diffusion works focusing on fluent text generation, this framework is explicitly designed for latent reasoning, enabling self-correction and refinement of reasoning chains at the semantic level rather than at the token level.
2. **Blockwise Latent Reasoning Space:** Constructs a latent space of intermediate reasoning steps using a VAE, encoding each step as a block of thought tokens. Adopts a hybrid attention mask where tokens attend bidirectionally within each block (enabling the model to internally reason over a horizon defined by the block size and capture richer local dependencies), while attention is strictly causal across blocks.
3. **Diversity Guidance Mechanism:** Introduces a diversity-guidance mechanism that applies repulsive forces during diffusion inference, pushing latent trajectories apart within a batch to explore multiple diverse reasoning paths, rather than producing repetitive solutions as often occurs in standard autoregressive sampling. This is combined with increased initial noise.
4. **Two-Stage Training with Rollout:** Introduces Stage 2 rollout training to mitigate the error accumulation issue between training (with oracle latents) and inference (with self-generated latents). It keeps the gradients on self-generated latents during denoising, allowing answer supervision to backpropagate through the trajectory and directly shape latent predictions.
5. **Adaptive Test-Time Compute:** Enables a better trade-off between accuracy and test-time compute, as additional denoising steps can be flexibly allocated to improve performance, presenting an alternative paradigm to long CoT reasoning.

**5. Overview of the Overall Technical Solution**

The approach separates reasoning from answering. A VAE constructs a latent space of intermediate reasoning steps, encoding each step as a block of thought tokens. A reasoning model predicts and refines thought tokens via latent diffusion, and then generates the final answer tokens conditioned on the denoised latent tokens.
1. **Blockization:** The text preceding the answer prefix is treated as CoT $c$, split into individual sentences, each treated as a block of latent tokens.
2. **VAE Construction:** A $\beta$-VAE is trained to encode each reasoning step into a structured latent representation aligned with the semantic space of the language model, utilizing robustness augmentations (Latent Gaussian noise and Input token substitution).
3. **Reasoning Model Training (Two Stages):** 
   - Stage 1: Teacher-forcing training where the model has access to oracle latent blocks, jointly optimizing flow matching on latent blocks and cross-entropy supervision on final answers and special tokens.
   - Stage 2: Rollout training where the model generates its own latents from random noise, keeping gradients on latents to allow answer supervision to backpropagate, avoiding latent collapse.
4. **Reasoning Model Inference:** Iterative denoising of latent blocks until the `<SOA>` token is predicted, followed by autoregressive answer generation. Diversity is improved in parallel via increased initial noise and diversity gradient guidance.

**6. Detailed Module Design**

**6.1 VAE Architecture**
- **Encoder:** Initialized from a pretrained LLM and fine-tuned with all parameters, along with $L_b$ learnable embeddings. The encoder’s last hidden state is passed through two linear projections to obtain the mean $\mu$ and variance $\sigma^2$, from which latent tokens are sampled.
- **Decoder:** A frozen pretrained LLM that conditions on the sampled latent tokens $Z^{(b)}$ to reconstruct the corresponding block of text.
- **Robustness Augmentations:**
  - *Latent Gaussian noise:* For each latent token $z^{(b)}_i$, inject isotropic Gaussian perturbations: $z'^{(b)}_i = z^{(b)}_i + \eta_i, \eta_i \sim \mathcal{N}(0, k^2I)$, where $k=3$.
  - *Input token substitution:* For the encoder input sequence, with probability $p=0.3$ replace a token with another randomly chosen token.

**6.2 Reasoning Model Architecture**
- Utilizes an existing LLM as the reasoning model.
- **Token Insertion:** After the input question $Q$, inserts a special token `<BOT>` to mark the start of a block, followed by the block tokens $Z^{(1)}$, and a token `<EOT>` to mark its end. For predicted blocks, adds a timestep embedding between `<BOT>` and $z^{(2)}_1$ to encode the timestep information. Appends a `<SOA>` token to indicate the start of the answer.
- **Attention Mask:** Adopts a hybrid attention mask $M$. Within each block, tokens attend bidirectionally. Across blocks, attention is strictly causal.

**6.3 Training Stages**
- **Stage 1: Teacher-Forcing Training:** The model is trained under a teacher-forcing regime where it has access to oracle latent blocks produced by the VAE encoder, denoted as $Z^{(1:B)}$. Oracle latents are concatenated between special tokens `<BOT>` and `<EOT>` and provided as context. The overall training objective jointly optimizes flow matching on latent blocks and cross-entropy supervision on both final answers and special tokens.
- **Stage 2: Rollout Training:** Keeps the same number of blocks $B$ as in the ground truth, but instead of conditioning on oracle latents, the model generates its own latents $\tilde{Z}^{(1:B)}$ from random noise using a fewer denoising steps (i.e., 50 $\rightarrow$ 10). Gradients are kept on $\tilde{Z}^{(1:B)}$ during denoising, allowing answer supervision to backpropagate through the trajectory. To avoid latent collapse, the flow matching loss is kept.

**6.4 Inference Process**
- **Iterative Denoising:** For each block, initialize with a Gaussian noise, and gradually transform the noise into a semantically coherent latent reasoning block $\hat{Z}^{(b)}$.
- **Stopping Criterion:** Latent block generation continues until the model explicitly predicts the special token `<SOA>`.
- **Answer Generation:** Conditioned on the generated latent reasoning sequence $\hat{Z}^{(1:\hat{B})}$ and the input question $x$, the model predicts output tokens autoregressively.
- **Diversity Improvement:**
  1. *Increased initial noise:* Sample with an increased variance $\tilde{\sigma}^2$ as initial noise at the first denoising step.
  2. *Diversity gradient guidance:* At each denoising step, enhance diversity by adding a repulsion term to push the latent tokens in a batch apart.

**7. All Mathematical Formulas and Symbol Definitions**

**7.1 Blockization Definition**
$c$ is treated as CoT, $y$ is treated as the final answer. $c$ is split into individual sentences, each treated as a block of latent tokens with block size $L_b$:
$$Z^{(b)} = \{z^{(b)}_1, \ldots, z^{(b)}_{L_b}\}, b = 1, \ldots, N$$

**7.2 $\beta$-VAE Formulations**
Let $x \in \mathbb{R}^{L \times d_x}$ denote a sequence of token embeddings with length $L$ and embedding dimension $d_x$, and $z \in \mathbb{R}^{M \times d_z}$ denote latent representations with $M$ latent tokens of dimension $d_z$. The $\beta$-VAE loss is:
$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}[- \log p_\theta(x|z)] + \beta \text{KL}(q_\phi(z|x) \| p(z))$$
where $q_\phi(z|x)$ is the encoder distribution parameterized by $\phi$, $p_\theta(x|z)$ is the decoder likelihood parameterized by $\theta$, and $p(z) = \mathcal{N}(0, I)$ is the prior distribution over latents.

Latent token sampling:
$$z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$$

**7.3 Flow Matching (Latent Diffusion) Formulations**
Path interpolating between clean data $z_0 \sim p_{\text{data}}$ and noise $\epsilon \sim \mathcal{N}(0, I)$:
$$z_t = (1 - t)z_0 + t\epsilon$$
Target velocity field $u^\star$:
$$u^\star(z_t, t) = \frac{dz_t}{dt} = \epsilon - z_0$$
Flow matching loss:
$$\mathcal{L}_{FM} = \mathbb{E}_{t \sim \mathcal{U}(0,1), z_0 \sim p_{\text{data}}, z_1 \sim \mathcal{N}(0,I)}[\|u_\theta(z_t, t) - u^\star(z_t, t)\|^2]$$
Inference ODE solver:
$$z_{t-\Delta t} = z_t - \Delta t u_\theta(z_t, t)$$

**7.4 Answer Token Loss**
Given the question $q$, the reasoning blocks $Z^{(\le B)}$, and the past answer tokens $y_{<w}$, the model predicts the next answer token $y_w$ with distribution $p_\psi(y_w | q, Z^{(\le B)}, y_{<w})$. The training objective for answer tokens is:
$$\mathcal{L}_{Ans} = -\sum_{w=1}^W \log p_\psi(y_t | q,Z^{(\le B)}, y_{<w})$$

**7.5 Special Token Loss**
Let $\tau$ index positions of `<EOT>` tokens in the output. For each $\tau$, the model produces a distribution $p_\psi(s_\tau | q, Z^{(\le B)}, y_{\le \tau})$, $s_\tau \in \{<SOA>, <BOT>\}$. The classification loss is:
$$\mathcal{L}_{Spec} = - \sum_{\tau \in T_{EOT}} \log p_\psi(s_\tau | q, Z^{(\le B)})$$

**7.6 Total Training Loss**
$$\mathcal{L} = \lambda_{FM} \mathcal{L}_{FM} + \lambda_{Ans} \mathcal{L}_{Ans} + \lambda_{Spec} \mathcal{L}_{Spec}$$

**7.7 Diversity Guidance Formulations**
Bandwidth parameter $\sigma$:
$$\sigma = \text{median}_{i < j} \|z_i - z_j\|_2$$
Repulsion force field for a latent token $z_i$:
$$F(z_i) = \sum_{j \neq i} \frac{2}{\sigma^2} \left(1 - \frac{\|z_i - z_j\|_2^2}{\sigma^2}\right) \exp\left(-\frac{\|z_i - z_j\|_2}{2\sigma^2}\right) (z_i - z_j), \forall j \le B$$
Time-dependent scale $\gamma_t$:
$$\gamma_t = \gamma_{\max} \left(\frac{t}{T}\right)$$
where $T$ is the total number of inference steps, $t$ decreases from $T$ to $0$, $\gamma_{\max}$ is the initial repulsion strength.
Diversity-guided prediction:
$$\hat{z}_{t-1} = f_\psi(x_t, t, x) + \gamma_t F(z)$$

**7.8 VAE Robustness Augmentation Formulation**
Latent Gaussian noise injection:
$$z'^{(b)}_i = z^{(b)}_i + \eta_i, \eta_i \sim \mathcal{N}(0, k^2I)$$

**8. Algorithm Pseudocode**
*(The paper does not provide explicit boxed algorithm pseudocode, but the iterative process is fully described in the Methodology and Inference sections. The logical procedural steps are extracted precisely as follows:)*

**Training Process:**
1. **VAE Training:** Train the $\beta$-VAE with robustness augmentations (Latent Gaussian noise and Input token substitution) to learn latent representations of thought tokens.
2. **Stage 1: Teacher-Forcing Training:**
   - Provide oracle latent blocks $Z^{(1:B)}$ from the VAE encoder.
   - Concatenate oracle latents between `<BOT>` and `<EOT>` as context.
   - Optimize the joint loss $\mathcal{L} = \lambda_{FM} \mathcal{L}_{FM} + \lambda_{Ans} \mathcal{L}_{Ans} + \lambda_{Spec} \mathcal{L}_{Spec}$.
3. **Stage 2: Rollout Training:**
   - Keep the same number of blocks $B$ as in the ground truth.
   - Instead of conditioning on oracle latents, generate own latents $\tilde{Z}^{(1:B)}$ from random noise using fewer denoising steps (50 $\rightarrow$ 10).
   - Keep gradients on $\tilde{Z}^{(1:B)}$ during denoising.
   - Optimize the same joint loss $\mathcal{L}$.

**Inference Process:**
1. Initialize with Gaussian noise of increased variance $\tilde{\sigma}^2$.
2. **For each latent block:**
   - **Iterative Denoising:** For $t$ from 1 to 0:
     - Compute base model prediction $f_\psi(x_t, t, x)$.
     - Compute bandwidth parameter $\sigma = \text{median}_{i < j} \|z_i - z_j\|_2$.
     - Compute repulsion force $F(z_i) = \sum_{j \neq i} \frac{2}{\sigma^2} \left(1 - \frac{\|z_i - z_j\|_2^2}{\sigma^2}\right) \exp\left(-\frac{\|z_i - z_j\|_2}{2\sigma^2}\right) (z_i - z_j)$.
     - Compute time-dependent scale $\gamma_t = \gamma_{\max} \left(\frac{t}{T}\right)$.
     - Update latent via diversity-guided prediction: $\hat{z}_{t-1} = f_\psi(x_t, t, x) + \gamma_t F(z)$.
   - Output denoised block $\hat{Z}^{(b)}$.
3. **Stopping Criterion:** Continue block generation until the model predicts the `<SOA>` token.
4. **Answer Generation:** Conditioned on generated latent sequence $\hat{Z}^{(1:\hat{B})}$ and input question $x$, predict output tokens $y = (y_1, \ldots, y_T)$ autoregressively.