## 1. Research Background and Existing Pain Points

**Research Background:** Masked diffusion large language models (dLLMs) are emerging as promising alternatives to autoregressive LLMs, offering competitive performance while supporting unique generation capabilities such as inpainting via bidirectional attention. Recent research has shown that masked dLLMs such as LLaDA and Dream can achieve performance competitive with autoregressive LLMs of similar size. Their capabilities and performance can be further enhanced via RL post-training and ability to flexibly include multimodal data. Unlike autoregressive LLMs, which decode in a left-to-right manner, dLLMs iteratively unmask tokens in parallel. This brings potential for faster inference as shown in closed models like Mercury and Gemini Diffusion, along with a flexible inductive bias for operations such as inpainting, the ability to fill in missing content within existing text.

**Existing Pain Points:** Aligning LLMs with reinforcement learning faces an exploration challenge: sparse reward signals and sample waste when LLMs fail to discover correct solutions. While this inefficiency affects LLMs broadly, a fundamental exploration challenge persists: for challenging tasks, policies struggle to discover correct solutions and binary rewards provide minimal learning signal when most generated solutions are incorrect. This leads to substantial sample waste and poor training efficiency, exacerbating the computational costs of online RL. Specifically, in group-based policy optimization methods such as GRPO, exploration failures cause zero advantages and gradients (Zero-Advantage Dilemma). When all sampled responses in a group yield identical incorrect rewards, the group-normalized advantage collapses to zero, resulting in zero gradients. This zero-advantage scenario makes the policy gradient component degenerate. Specifically, the clipped surrogate objective collapses to zero regardless of whether the update lies in the clipped or unclipped region, since both terms contain zero advantage. As a result, no meaningful policy update can be extracted from the reward signal, wasting compute sampling these responses. Additionally, for SFT, there is a generation length mismatch across SFT, RL sampling, and evaluation phases. Popular reasoning SFT corpora contain verbose traces often exceeding 10k tokens, creating distribution mismatch and including repeated reflective behaviors unsuited for limited context.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** The bidirectional generative structure of diffusion LLMs provides a unique mechanism to address the exploration challenge. Since dLLMs are trained through stochastic masking patterns, they possess inherent capability for accepting externally provided partial hints through inpainting. The motivation is to leverage this ability to strategically guide exploration for dLLMs by injecting reasoning hints when answering difficult problems. Furthermore, to resolve the length mismatch in SFT, the motivation is to align the SFT data length with RL sampling and evaluation length via rewritten concise reasoning traces.

**Scientific Questions:** How can the unique inpainting capabilities of diffusion LLMs be utilized for reinforcement learning to alleviate the inefficiency of sparse verifiable rewards? How can partial hint injection bridge on-policy generation with ground truth guidance to stay closer to the policy distribution in online RL compared to full supervision? How can the zero-advantage dilemma in group-based policy optimization methods be resolved to restore gradient signals? How can Length-Aligned SFT improve RL initialization by avoiding implicit length compression?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:** Introduce IGPO (Inpainting Guided Policy Optimization), an RL framework that strategically injects partial ground-truth reasoning traces during online sampling. When the policy is unlikely to generate correct solutions, partial reasoning traces are injected into the generation region, and the dLLM is tasked with completing the remaining reasoning sequence and output final answer. The final answers are verified against ground truth, and only successful completions are used for downstream policy optimization. Augment IGPO with Length-Aligned SFT using synthetically rewritten, concise reasoning traces and entropy-based gradient filtering for injected tokens.

**Design Philosophy:** Interpolate between supervised and RL paradigms. Unlike providing full solutions, inpainting steers exploration toward promising trajectory spaces while preserving self-generated reasoning. The injected tokens act as conditioning context that steers the policy’s action distribution toward high-reward regions. Unlike pure SFT, which might suffer from distribution shift between data and policy rollouts, IGPO maintains on-policy generation for the non-injected tokens. Elastic inpainting is triggered only when all sampled responses in a group yield incorrect rewards to restore gradient signals. Partial inpainting outperforms full ground-truth inpainting by staying closer to the policy distribution. Entropy-based gradient filtering restricts learning to hint token positions where the model exhibits sufficient uncertainty to prevent training instability from off-policy tokens.

## 4. Core Innovation Points

1.  **IGPO:** The first work to utilize the unique inpainting capabilities of diffusion LLMs for reinforcement learning. By strategically injecting partial reasoning traces during exploration, IGPO alleviates the inefficiency of sparse verifiable rewards and mitigates the zero-advantage dilemma in group-based policy optimization methods, substantially reducing the proportion of all-wrong groups (by approximately 60%).
2.  **Length-Aligned SFT:** Propose a Length-Aligned SFT for full-attention based dLLMs using synthetically rewritten, concise reasoning traces. This design better aligns SFT data length with RL sampling and evaluation length, avoids the limitations of verbose traces, and provides stronger initialization for RL.
3.  **SoTA Performance:** The full training recipe achieves substantial improvements on mathematical benchmarks, including +5.3% on GSM8K, +8.4% on Math500, +11.4% on AMC, and +4.0% on Minerva relative to the LLaDA-Instruct, achieving SoTA performance among full-attention based dLLMs.
4.  **Mechanism Disentanglement:** Comprehensive ablation study that disentangles the mechanisms of IGPO. It shows that partial inpainting consistently outperforms full ground-truth inpainting by staying closer to the policy distribution in online RL, and proposes an entropy-based gradient filtering mechanism that stabilizes training dynamics.
5.  **Robustness and Theory:** Showing IGPO’s robustness to imperfect reasoning traces and providing theoretical analysis of gradient recovery in the zero-advantage regime.

## 5. Overview of the Overall Technical Solution

The overall technical solution consists of a two-stage pipeline:
**Stage 1: Supervised Fine-Tuning with Rewritten Traces.** We begin with Length-Aligned SFT on the LLaDA-8B-Instruct model using the OpenR1-Math-220K dataset’s default split, but with all reasoning traces rewritten into concise, structured forms that preserve logical flow while respecting dLLM computational limits. This ensures consistency between training distribution and downstream RL/evaluation phases by aligning trace lengths.
**Stage 2: Reinforcement Learning with IGPO.** Following Length-aligned SFT, we apply IGPO to further enhance reasoning capabilities through strategic inpainting-guided policy optimization. We utilize the reasoning traces from the MetaMathQA dataset for the elastic inpainting process. During RL, when detecting that all sampled responses for a query yield identical incorrect rewards (the zero-advantage case), an additional set of responses is generated through the inpainting process. Ground truth reasoning traces are segmented into variable-length contiguous chunks, and selected chunks are injected as fixed hints during generation while the model generates the remaining tokens. Correct responses generated through inpainting replace a fraction of the original incorrect responses, creating reward variance that enables non-zero advantages for effective policy gradient updates. Entropy-based gradient filtering is applied to hint tokens to restrict learning to high-uncertainty positions.

## 6. Detailed Module Design

**Zero-Advantage Dilemma Detection:** In the GRPO framework, when sampling G responses for a given prompt q, the advantage computation relies on reward variance across the group. When all responses receive identical rewards—either all correct or all incorrect—the advantages become zero. Specifically, the clipped surrogate objective collapses to zero regardless of whether the update lies in the clipped or unclipped region, since both terms contain zero advantage. The policy gradient for this prompt q therefore becomes zero. No meaningful policy update can be extracted from the reward signal, wasting compute sampling these responses.

**Masked dLLM Generation and Inpainting:** In full-attention masked dLLM generation, the model input at denoising step 0 is the concatenation $[q; z_{mask}]$, where q represents the prompt and $z_{mask}$ denotes a fully masked completion sequence of predetermined length L. Hint injection modifies this formulation by fixing selected positions of $z_{mask}$ to ground-truth tokens. We create a binary mask $m \in \{0, 1\}^L$ indicating which positions to inject as fixed hints. The masked dLLM then performs bidirectional denoising on $[q; z_{hint}]$ through the inpainting process, leveraging both the prompt and injected hint tokens to generate coherent responses. The injected hint tokens remain fixed throughout the iterative denoising steps.

**Hint Pattern Construction:** To construct meaningful hint patterns for the inpainting process, we segment the ground truth reasoning trace $y^*$ into variable-length contiguous chunks $C = \{c_1, c_2, \ldots, c_N\}$, where each chunk length $|c_j|$ is sampled from $U[s_{min}, s_{max}]$. We explicitly exclude the final answer tokens from chunking to prevent reward hacking behaviors. For a given hint injection ratio $\eta \in [0, 1]$, we randomly select $\lfloor \eta \cdot N \rfloor$ chunks and set their corresponding positions in the binary mask m to 1 for hint injection.

**Elastic Inpainting-Triggered Sampling:** With the above inpainting setup, we design IGPO to be elastic: hint injection is only triggered when all sampled responses in a group yield incorrect rewards (the zero-advantage case), and when activated, both the hint injection ratio $\eta$ and chunk sizes ($U[s_{min}, s_{max}]$) are randomized to provide diverse training signals. When detecting that all sampled responses for query q yield identical rewards, we generate an additional set of responses through the inpainting process. Each response is generated via inpainting with a distinct hint injection ratio to ensure diverse hint densities. Following inpainting generation, we evaluate the correctness and only use the correct ones for replacement. Specifically, we replace $K = \min(|\{\tilde{o}_i : r(\tilde{o}_i) = 1\}|, \lfloor \lambda G \rfloor)$ of the original incorrect responses with correct responses generated through inpainting, where $\lambda \in (0, 1)$ controls the replacement fraction.

**Entropy-based Gradient Filtering for Hint Tokens:** When applying IGPO to zero-advantage scenarios, the responses generated through inpainting contain ground truth reasoning chunks that originate from a different distribution than the current policy $\pi_\theta$. This creates an off-policy learning scenario where gradient updates from ground truth tokens can conflict with the model’s current beliefs, particularly at positions where the model has high confidence (low entropy). To mitigate potential training instability from this distribution mismatch, we implement an entropy-based filtering approach that restricts learning to hint token positions where the model exhibits sufficient uncertainty. Specifically, for each hint token position (i.e., positions with injected ground-truth tokens) we compute the entropy. We then apply gradient updates only to the top $\tau$ percentile of hint token positions with highest entropy values. This selective learning strategy serves two purposes: high-entropy positions represent genuine decision boundaries where the model is naturally uncertain and thus more receptive to external guidance, and they correspond to flatter probability distributions that yield more stable gradient updates when incorporating ground truth information.

**Length-Aligned SFT via Concise Reasoning Trace Rewriting:** To further strengthen our training recipe, we seek better RL initialization via SFT but identified generation length mismatches across SFT, RL sampling, and evaluation phases. Full-attention masked dLLMs like LLaDA lack KV cache optimization, requiring full-sequence attention at every denoising step, which dominates online RL training cost. As a result, we restrict RL rollouts to 256 tokens for faster convergence within a reduced exploration space, and evaluation setups typically use 256–1024 tokens. In contrast, popular reasoning SFT corpora contain verbose traces often exceeding 10k tokens, creating distribution mismatch. To resolve this, we systematically rewrite verbose traces into concise, structured forms that preserve logical flow while respecting dLLM computational limits. Using LLaMA-4-Maverick with prompts, we remove redundant reflections, condense multi-sentence elaborations into precise, mathematically rigorous statements, and retain essential reasoning. Our Length-Aligned SFT trains LLaDA solely on rewritten traces, improving RL initialization by avoiding implicit length compression and focusing learning on reasoning quality within fixed compute budgets.

## 7. All Mathematical Formulas and Symbol Definitions

**Masked dLLM NELBO Objective (Eq 1):**
$-\mathbb{E}_{t\sim U[0,1), x_0\sim p_{data}, x_t\sim q_{t|0}(x_t|x_0)} \left[ \frac{1}{t} \sum_{k=1}^{|x_t|} \mathbb{1}[x_t^k = \text{mask}] \log f_\theta(x_0^k | x_t) \right]$

**GRPO Advantage (Eq 2):**
$A_i = r(o_i) - \frac{1}{G}\sum_{j=1}^G r(o_j)$

**GRPO Objective (Eq 3):**
$\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{q\sim D} \mathbb{E}_{o_1,...,o_G\sim\pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{k=1}^{|o_i|} \min(\rho_i^k A_i, \text{clip}(\rho_i^k, 1-\epsilon, 1+\epsilon)A_i) - \beta D_{KL} [\pi_\theta(\cdot|q)\|\pi_{ref}(\cdot|q)] \right]$
Where $\rho_i^k = \frac{\pi_\theta(o_i^k |q, o_i^{<k})}{\pi_{\theta_{old}}(o_i^k |q, o_i^{<k})}$ is the probability ratio.

**Hint-injected Initialization (Eq 4):**
$z_{hint}[i] = \begin{cases} y_i^* & \text{if } m[i] = 1 \text{ and } i \le |y^*|, \\ \text{mask} & \text{otherwise} \end{cases}$

**IGPO Objective (Eq 5):**
$\mathcal{L}_{IGPO}(\theta) = \mathbb{E}_{q\sim D} \mathbb{E}_{\{o_1,...,o_{G-K}, \tilde{o}_1,...,\tilde{o}_K\}\sim \text{IGPO-Sample}(\pi_\theta, q, y^*)} \left[ \left( \frac{1}{G}\sum_{i=1}^G \frac{1}{L_i}\sum_{k=1}^{L_i} \min(\rho_i^k A_i^k, \text{clip}(\rho_i^k, 1-\epsilon, 1+\epsilon)A_i^k) \right) - \beta D_{KL} [\pi_\theta(\cdot|q)\|\pi_{ref}(\cdot|q)] \right]$

**Theoretical Analysis (Appendix K):**
**Expected reward (Eq 9):**
$J(\theta) = \mathbb{E}_x J(\pi_\theta;x), \quad J(\pi_\theta;x) = \mathbb{E}_{o\sim\pi_\theta(\cdot|x)}[r(o)]$

**Group-normalized advantages (Eq 10):**
$\bar{r} = \frac{1}{G}\sum_{i=1}^G r(o_i), \quad A_i = r(o_i) - \bar{r}$

**Per-query policy gradient (Eq 11):**
$\hat{g}_{GRPO}(x) = \frac{1}{G}\sum_{i=1}^G A_i \nabla_\theta \log \pi_\theta(o_i | x)$

**All-wrong event (Eq 12):**
$\mathcal{E}_{wrong} = \{ r(o_i) = 0 \forall i \in \{1, \ldots, G\} \}$

**Zero-gradient dilemma (Eq 13):**
$\hat{g}_{GRPO}(x) = \frac{1}{G}\sum_{i=1}^G 0 \cdot \nabla_\theta \log \pi_\theta(o_i | x) = 0$

**Effective replacement ratio (Eq 14):**
$\rho = \frac{K}{G} \in [0, \lambda]$

**Augmented group definition (Eq 15, 16):**
$S' = \{o'_1, \ldots, o'_G\}, \quad |\{i : r(o'_i) = 1\}| = K, \quad |\{i : r(o'_i) = 0\}| = G-K$
$I_{correct} = \{i : r(o'_i) = 1\}, \quad I_{wrong} = \{i : r(o'_i) = 0\}$

**New group-average reward (Eq 17):**
$\bar{r}_{new} = \frac{1}{G}\sum_{i=1}^G r(o'_i) = \frac{K}{G} = \rho$

**New advantages (Eq 18):**
$A'_i = \begin{cases} 1-\rho, & i \in I_{correct} \\ -\rho, & i \in I_{wrong} \end{cases}$

**Per-query IGPO gradient estimator (Eq 19):**
$\hat{g}_{IGPO}(x) = \frac{1}{G}\sum_{i=1}^G A'_i \nabla_\theta \log \pi_\theta(o'_i | x)$

**Mixture policy interpretation (Eq 20):**
$\pi_\alpha(\cdot | x) = (1-\alpha)\pi_\theta(\cdot | x) + \alpha \delta_{o^*}, \quad \alpha \in [0, 1]$

**Advantage definition (Eq 21):**
$A^{\pi_\theta}(x, o) = r(o) - J(\pi_\theta;x)$

**Performance-difference lemma (Eq 22):**
$J(\pi') - J(\pi_\theta) = \mathbb{E}_x \mathbb{E}_{o\sim\pi'(\cdot|x)} [A^{\pi_\theta}(x, o)]$

**Mixture expectation (Eq 23, 24):**
$\mathbb{E}_{o\sim\pi_\alpha(\cdot|x)} [A^{\pi_\theta}(x, o)] = (1-\alpha)\mathbb{E}_{o\sim\pi_\theta(\cdot|x)} [A^{\pi_\theta}(x, o)] + \alpha A^{\pi_\theta}(x, o^*) = (1-\alpha) \cdot 0 + \alpha (r(o^*) - J(\pi_\theta;x)) = \alpha (1 - J(\pi_\theta;x))$
$J(\pi_\alpha) - J(\pi_\theta) = \alpha \mathbb{E}_x [1 - J(\pi_\theta;x)]$

**Lemma 1 (Closed-form expected IGPO gradient). Define the gradient expectations under the correct and wrong distributions separately:**
$g_{correct}(x) = \mathbb{E}_{o'\sim\delta_{o^*}} [\nabla_\theta \log \pi_\theta(o' | x)] = \nabla_\theta \log \pi_\theta(o^* | x)$
$g_{wrong}(x) = \mathbb{E}_{o'\sim\pi_\theta(\cdot|x)} [\nabla_\theta \log \pi_\theta(o' | x)]$
Then
$g_{IGPO}(x) = \mathbb{E}[\hat{g}_{IGPO}(x)] = \rho(1-\rho) (g_{correct}(x) - g_{wrong}(x))$

**Proof:**
$o'_i \sim \pi_\alpha(\cdot | x) \text{ with } \alpha = \rho = K/G$
$g_{IGPO}(x) = \mathbb{E}_{S'} \left[ \frac{1}{G}\sum_{i=1}^G A'_i \nabla_\theta \log \pi_\theta(o'_i | x) \right] = \frac{1}{G}\sum_{i=1}^G \mathbb{E}_{o'_i\sim\pi_\alpha(\cdot|x)} [A'_i \nabla_\theta \log \pi_\theta(o'_i | x)] = \mathbb{E}_{o'\sim\pi_\alpha(\cdot|x)} [A'_i \nabla_\theta \log \pi_\theta(o' | x)]$
$\mathbb{E}_{o'\sim\pi_\alpha(\cdot|x)} [A'_i \nabla_\theta \log \pi_\theta(o' | x)] = (1-\rho)\mathbb{E}_{o'\sim\pi_\theta(\cdot|x)} [A'_i \nabla_\theta \log \pi_\theta(o' | x)] + \rho \mathbb{E}_{o'\sim\delta_{o^*}} [A'_i \nabla_\theta \log \pi_\theta(o' | x)]$
$(1-\rho)\mathbb{E}_{o'\sim\pi_\theta|r(o')=0} [A'_i \nabla_\theta \log \pi_\theta(o' | x)] = (1-\rho) \cdot (-\rho)\mathbb{E}_{o'\sim\pi_\theta|r(o')=0} [\nabla_\theta \log \pi_\theta(o' | x)] = -\rho(1-\rho) g_{wrong}(x)$
$\rho \mathbb{E}_{o'\sim\delta_{o^*}} [A'_i \nabla_\theta \log \pi_\theta(o' | x)] = \rho \cdot (1-\rho)\mathbb{E}_{o'\sim\delta_{o^*}} [\nabla_\theta \log \pi_\theta(o' | x)] = \rho(1-\rho) g_{correct}(x)$
$g_{IGPO}(x) = \rho(1-\rho) g_{correct}(x) - \rho(1-\rho) g_{wrong}(x) = \rho(1-\rho) (g_{correct}(x) - g_{wrong}(x))$

**Theorem 1 (Gradient recovery and policy improvement). Conditioned on $\mathcal{E}_{wrong}$ and for $0 < \rho < 1$, the expected IGPO gradient satisfies**
$g_{IGPO}(x) = \rho(1-\rho) (g_{correct}(x) - g_{wrong}(x))$
and is non-zero whenever $g_{correct}(x) \neq g_{wrong}(x)$. Furthermore, for the mixture policy $\pi_\alpha$ in equation 20,
$J(\pi_\alpha) - J(\pi_\theta) = \alpha \mathbb{E}_x [1 - J(\pi_\theta;x)] \ge 0$
with strict inequality whenever there exists a query $x$ such that $J(\pi_\theta;x) < 1$. The scalar factor $\rho(1-\rho)$ governing the gradient magnitude is maximized at $\rho = 1/2$.

**Theorem 2 (KL control via partial hint injection). For the mixture policy $\pi_\alpha$ in equation 20, the KL divergence to the current policy satisfies**
$D_{KL}(\pi_\alpha(\cdot | x) \| \pi_\theta(\cdot | x)) \le \alpha D_{KL}(\delta_{o^*} \| \pi_\theta(\cdot | x)) = -\alpha \log \pi_\theta(o^* | x)$
**Proof:**
$D_{KL}((1-\alpha)P_1 + \alpha P_2 \| Q) \le (1-\alpha)D_{KL}(P_1\|Q) + \alpha D_{KL}(P_2\|Q)$
$D_{KL}(\pi_\alpha(\cdot | x) \| \pi_\theta(\cdot | x)) \le (1-\alpha)D_{KL}(\pi_\theta(\cdot | x) \| \pi_\theta(\cdot | x)) + \alpha D_{KL}(\delta_{o^*} \| \pi_\theta(\cdot | x)) = \alpha D_{KL}(\delta_{o^*} \| \pi_\theta(\cdot | x))$
$D_{KL}(\delta_{o^*} \| \pi_\theta(\cdot | x)) = \sum_o \delta_{o^*}(o) \log \frac{\delta_{o^*}(o)}{\pi_\theta(o | x)} = \log \frac{1}{\pi_\theta(o^* | x)} = -\log \pi_\theta(o^* | x)$

## 8. Algorithm Pseudocode

**Algorithm 1 IGPO: Inpainting-Guided Policy Optimization for Masked dLLMs**
**Require:** Reference model $\pi_{ref}$, prompt distribution $D$, ground-truth reasoning traces $\{y^*\}$, number of completions per prompt $G$, number of inner updates $\mu$, hint injection ratio range $[\eta_{low}, \eta_{high}]$, replacement fraction $\lambda$, entropy filter threshold $\tau$, chunk size range $[s_{min}, s_{max}]$
1: Initialize $\pi_\theta \leftarrow \pi_{ref}$
2: while not converged do
3: $\pi_{old} \leftarrow \pi_\theta$; sample prompt $q\sim D$ and responses $o_{1:G}\sim\pi_{old}(\cdot |q)$; compute rewards $r_{1:G}$
4: if all $r_i = 0$ (zero-advantage case) then
5: Segment ground-truth reasoning $y^*$ into chunks $\{c_1, \ldots, c_N\}$ with $|c_j | \sim U [s_{min}, s_{max}]$
6: for $i = 1, \ldots, G$ do
7: Sample hint injection ratio $\eta \sim U [\eta_{low}, \eta_{high}]$ and select $\lfloor \eta N \rfloor$ chunks from $\{c_1, \ldots, c_N\}$ randomly
8: Inject selected chunk tokens as fixed hints at corresponding positions
9: Generate $\tilde{o}_i$ via inpainting: denoise only masked positions, keep hint tokens fixed
10: Evaluate rewards $r(\tilde{o}_i)$ and replace up to $\lfloor \lambda G \rfloor$ incorrect $o_i$ with correct $\tilde{o}_i$
11: Compute advantages $A_i$ on the updated response set
12: for $n = 1, \ldots, \mu$ do
13: Estimate $\log \pi_\theta$, $\log \pi_{old}$, $\log \pi_{ref}$; apply top-$\tau$ entropy filter on hint positions
14: Update $\pi_\theta$ via $\mathcal{L}_{IGPO}(\theta)$ (Eq. 5)
15: return $\pi_\theta$