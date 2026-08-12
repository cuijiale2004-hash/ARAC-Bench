## 1. Research Background and Existing Pain Points

**Research Background:**
Diffusion models have established themselves as a powerful class of generative models, demonstrating state-of-the-art performance across images and videos. In particular, Text-to-Video (T2V) diffusion models have received increasing attention for their ability to generate temporally coherent and visually rich video sequences. Most T2V model architectures extend Text-to-Image (T2I) diffusion backbones by incorporating temporal modules or motion-aware attention layers. Beyond architectural design, another promising direction lies in improving noise initialization at inference time for T2I and T2V generation. This aligns with the growing trend of inference-time scaling. Due to the iterative nature of the diffusion process, the choice of initial noise profoundly influences video quality, temporal consistency, and prompt alignment. The same prompt can lead to drastically different videos depending solely on the noise seed.

**Existing Pain Points:**
Recent approaches tackle noise initialization by introducing external noise priors. PYoCo enforces inter-frame dependent patterns for coherence but requires heavy fine-tuning. FreeNoise reschedules noise across time with a fusion strategy, FreeInit preserves low-frequency components via frequency filtering, and FreqPrior extends this with Gaussian-shaped priors and partial sampling. While effective, these methods rely on external priors and repeated full diffusion passes, leading to high computational cost. Crucially, they overlook internal model signals that identify inherently preferable seeds. They ignore the model's own uncertainty and confidence regarding different noise seeds, treating the model as a black box rather than leveraging its internal epistemic state.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To address the limitation of external priors, the research is motivated by the principle that "the model already knows the best noise." Instead of imposing external frequency constraints, the model's internal signals—specifically, attention-based uncertainty—can indicate inherently preferable seeds. By quantifying this uncertainty, one can select seeds that induce confident and consistent model behavior, leading to higher-quality generation without retraining or costly full diffusion passes.

**Scientific Questions:**
How can we quantify the epistemic uncertainty of a video diffusion model regarding a specific noise seed at inference time? How can we adapt uncertainty-based acquisition functions (like BALD) from classification settings to the attention space of generative diffusion models? How can we efficiently estimate this uncertainty without requiring multiple independent forward passes or evaluating all layers, ensuring the method is inference-friendly?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to propose a model-aware noise selection framework, ANSE (Active Noise Selection for Generation), grounded in Bayesian uncertainty. The framework selects high-quality seeds by quantifying attention-based uncertainty. At its core is BANSA (Bayesian Active Noise Selection via Attention), an acquisition function that identifies noise seeds inducing confident and consistent attention behaviors under stochastic perturbations. Unlike BALD, which estimates uncertainty from classification logits, BANSA measures entropy in attention maps, capturing both uncertainty and cross-pass disagreement. A lower BANSA score indicates more confident, consistent attention and empirically correlates with more coherent video generation.

**Design Philosophy:**
The design philosophy centers on transferring the BALD principle to the generative setting by operating in the attention space, where text and visual tokens align during generation. It leverages Bernoulli-masked attention for efficient stochastic approximation from a single forward pass and employs correlation-based layer selection to avoid redundant computation, ensuring that the framework improves video quality with only marginal inference overhead.

## 4. Core Innovation Points

1.  **First Active Noise Selection Framework for Video Diffusion:** The paper presents ANSE, the first active noise selection framework for video diffusion, grounded in a Bayesian formulation of attention-based uncertainty.
2.  **Novel Acquisition Function (BANSA):** The introduction of BANSA, an acquisition function that measures attention consistency under stochastic perturbations, enabling model-aware noise selection without retraining. It quantifies both the sharpness (confidence) and consistency (agreement) of attention behavior.
3.  **Efficient Stochastic Approximation:** The introduction of a Bernoulli-masked approximation of BANSA that estimates scores from a single diffusion step and a subset of informative attention layers, allowing K stochastic attention samples from a single forward pass.
4.  **Principled Layer Selection Mechanism:** A layer selection method via cumulative BANSA correlation that truncates computation at the smallest depth $d^*$ where the average BANSA up to $d$ remains highly correlated with the full-layer score, significantly reducing cost.
5.  **Theoretical Guarantee:** The BANSA Zero Condition (Proposition 1), which proves that the BANSA score becomes zero if and only if all attention samples are identical, confirming it strictly quantifies disagreement among sampled attention maps.

## 5. Overview of the Overall Technical Solution

The overall technical solution consists of a pipeline that evaluates a pool of candidate noise seeds and selects the optimal one based on the BANSA score. Given a text prompt $c$ and a noise pool $Z = \{z_1, \ldots, z_M\}$, the framework computes BANSA scores for each seed using Bernoulli-masked attention maps from selected layers at an early diffusion step. Instead of running multiple independent forward passes to sample attention, stochasticity is injected directly into the attention scores via Bernoulli masks, yielding multiple stochastic attention samples from a single pass. The framework computes the approximate BANSA score up to a truncated layer depth $d^*$ determined by correlation analysis. The seed with the lowest BANSA score, indicating the most certain and consistent attention behavior, is selected as $z^*$ and used for the full video generation sampling process.

## 6. Detailed Module Design

**Module 1: BANSA Score Definition**
BANSA is an acquisition function that estimates uncertainty in the attention space. It treats attention maps as stochastic predictions conditioned on the seed $z$, prompt $c$, and timestep $t$. It measures the difference between the entropy of the averaged attention maps and the average of the entropies of individual attention maps. This captures both the sharpness (confidence) and consistency (agreement) of attention behavior.

**Module 2: Bernoulli-Masked Attention Approximation**
To avoid the cost of $K$ independent passes per seed, this module draws $K$ stochastic attention samples from a single pass. For each sample, a binary mask $m_k$ is sampled i.i.d. from Bernoulli($p$). The masked attention map is computed by element-wise multiplication of the attention map and the mask, followed by row re-normalization. The approximate BANSA (BANSA-E) is then calculated using these masked samples.

**Module 3: Layer Selection via Cumulative BANSA Correlation**
Since BANSA can be computed at any layer but behavior differs across depth, this module truncates evaluation at a specific depth $d^*$. It computes the cumulative average BANSA score up to layer $d$ and finds the smallest $d$ such that the Pearson correlation between the partial average and the full-layer average exceeds a threshold $\tau$.

**Module 4: Noise Selection and Generation**
Using the computed BANSA-E scores up to layer $d^*$, the module selects the optimal noise seed that minimizes the score from the noise pool $Z$. This seed is then passed to the standard video diffusion sampling process.

## 7. All Mathematical Formulas and Symbol Definitions

**Forward Diffusion Process:**
$z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad t = 1, \ldots, T$
Where $\bar{\alpha}_t$ is a pre-defined variance schedule.

**Denoising Score Matching Loss:**
$\mathcal{L}_\theta = \mathbb{E}_{z_t, \epsilon, t} \left[ \| \epsilon_\theta(z_t, c, t) - \epsilon \|^2 \right]$
Where $c$ is the conditioning text.

**DDIM Sampling Update:**
$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{z}_0(t) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(z_t, c, t)$

**Tweedie's Formula for Denoised Latent Estimate:**
$\hat{z}_0(t) := \frac{z_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(z_t,c,t)}{\sqrt{\bar{\alpha}_t}}$

**BALD Definition:**
$\text{BALD}(x) = \mathbb{H}[p(y|x)] - \mathbb{E}_{p(\theta|\mathcal{D}_U)} [\mathbb{H}[p(y|x, \theta)]]$
Where $\mathbb{H}[p] = -\sum_y p(y) \log p(y)$ is Shannon entropy.

**BALD Approximation:**
$\hat{\text{BALD}}(x) = \mathbb{H} \left[ \frac{1}{K} \sum_{k=1}^K p^{(k)}(y|x) \right] - \frac{1}{K} \sum_{k=1}^K \mathbb{H} \left[ p^{(k)}(y|x) \right]$

**Attention Map Calculation:**
$A(z, c, t) := \text{Softmax} \left( Q(z, c, t)K(z, c, t)^\top \right) \in \mathbb{R}^{N \times N}$
Where $Q(z, c, t), K(z, c, t) \in \mathbb{R}^{N \times d}$ denote the query and key matrices.

**Entropy of Attention Map:**
$\mathbb{H}(A) := \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N -A_{ij} \log A_{ij}$

**BANSA Score Definition:**
$\text{BANSA}(z, c, t) := \mathbb{H} \left( \frac{1}{K} \sum_{k=1}^K A^{(k)} \right) - \frac{1}{K} \sum_{k=1}^K \mathbb{H}(A^{(k)})$
Where $A(z, c, t) = \{A^{(1)}, \ldots, A^{(K)}\}$ denotes a set of $K$ stochastic attention maps.

**Optimal Noise Seed Selection:**
$z^* := \text{argmin}_{z \in Z} \text{BANSA}(z, c, t)$

**Proposition 1 (BANSA Zero Condition):**
Let $A(z, c, t) = \{A^{(1)}, \ldots, A^{(K)}\}$ be a set of row-stochastic attention maps. Then:
$\text{BANSA}(z, c, t) = 0 \iff A^{(1)} = \cdots = A^{(K)}$

**Proof of Proposition 1:**
Since the Shannon entropy $\mathbb{H}(\cdot)$ is strictly concave over the probability simplex. Therefore, by Jensen’s inequality:
$\mathbb{H} \left( \frac{1}{K} \sum_{k=1}^K A^{(k)} \right) \ge \frac{1}{K} \sum_{k=1}^K \mathbb{H}(A^{(k)})$
with equality if and only if $A^{(1)} = \cdots = A^{(K)}$. Thus, $\text{BANSA}(z, c, t) = 0$ if and only if all attention maps are identical.

**Bernoulli-Masked Attention Map:**
$\hat{A}^{(k)}(z, c, t) := A(z, c, t) \odot m_k$
Where $m_k \in \{0, 1\}^{N \times N}$ is sampled i.i.d. from Bernoulli($p$), $\odot$ denotes element-wise multiplication, and rows are re-normalized.

**Approximate BANSA (BANSA-E):**
$\text{BANSA-E}(z, c, t) := \mathbb{H} \left( \frac{1}{K} \sum_{k=1}^K \hat{A}^{(k)}(z, c, t) \right) - \frac{1}{K} \sum_{k=1}^K \mathbb{H}(\hat{A}^{(k)}(z, c, t))$

**Cumulative Average BANSA-E:**
$\hat{\text{BANSA-E}}_{\le d}(z_i, c, t) := \frac{1}{d} \sum_{l=1}^d \text{BANSA-E}^{(l)}(z_i, c, t)$

**Layer Selection Constraint:**
$\text{Corr} \left( \hat{\text{BANSA-E}}_{\le d}, \hat{\text{BANSA-E}}_{\le L} \right) \ge \tau$
Where $\tau$ is the correlation threshold.

## 8. Algorithm Pseudocode

**Algorithm 1: Active Noise Selection with BANSA Score for Video Generation**
Input: Text prompt $c$, noise pool $Z = \{z_1, \ldots, z_M\}$, timestep $t$, cutoff layer $d^*$
1. foreach $z_i \in Z$ do
2. Compute BANSA score: $\hat{\text{BANSA-E}}_{\le d^*}(z_i, c, t)$ via Eq. (7);
3. Select optimal noise: $z^* = \text{argmin}_{z_i} \hat{\text{BANSA-E}}_{\le d^*}(z_i, c, t)$;
4. return Generate video: $\hat{v} = \text{SampleVideo}(z^*, c, t)$;