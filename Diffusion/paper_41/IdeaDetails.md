### 1. Research Background and Existing Pain Points

**Research Background:**
Diffusion-based large language models (dLLMs) have recently emerged as promising alternatives to autoregressive (AR) models for language modeling tasks. Unlike AR models that generate tokens sequentially, dLLMs iteratively refine entire response sequences through a denoising process, offering significantly improved inference efficiency. Recent closed models (e.g., Mercury, Gemini Diffusion) and open-weight models (e.g., LLaDA-8B, Dream-7B) demonstrate competitive performance. While reinforcement learning (RL) has proven highly effective in aligning and improving the reasoning capabilities of AR models, applying RL to fine-tune dLLMs remains an open and important problem.

**Existing Pain Points:**
A key challenge in applying RL to dLLMs is the intractability of their likelihood functions ($\log \pi_\theta$). This necessitates approximation for policy optimization. Existing methods, such as diffusion-based GRPO (d1), suffer from several severe limitations:
1.  **Exponential Error Amplification:** Existing methods rely on approximating the current, old, and reference policy likelihoods to compute the policy ratio for importance sampling ($r_i^k(\theta) = \pi_\theta / \pi_{old} \approx \exp(\phi_{\pi_\theta} - \phi_{\pi_{old}})$). Approximation errors in the log-likelihood can be exponentially amplified in this ratio, leading to large variance and estimation error in the RL objective.
2.  **Bias and Variance Issues:** Methods using ELBO for likelihood approximation (e.g., UniGRPO) require a large sample size of $t$ to reduce variance, resulting in inefficiency. Biased but efficient methods (like d1, sampling at $t=1$) yield a biased ratio that deviates substantially from the ELBO, introducing systematic bias.
3.  **Computational Overhead:** GRPO requires applying the approximation function $\phi$ separately to three policies—$\pi_\theta, \pi_{old}$, and $\pi_{ref}$—at every training step, leading to significant computational overhead (increased FLOPs and NFEs). These issues are exacerbated as completion length and diffusion steps increase.
4.  **Suboptimal Sample Utilization:** Standard Weighted Log-Likelihood (WLL) objectives derived from reverse-KL constraints tend to assign vanishingly small weights to negative samples (low-advantage completions), failing to actively reduce the probability of detrimental outcomes.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To mitigate the bias, variance, and computational overhead associated with likelihood ratio estimation in dLLM policy optimization. The goal is to design an RL fine-tuning algorithm for dLLMs that dispenses with explicit policy ratios, relies on a single likelihood approximation (current policy only), actively penalizes low-advantage completions, and maintains theoretical soundness while improving reasoning capabilities efficiently.

**Scientific Questions:**
1.  How to derive a ratio-free RL objective for dLLMs that avoids the exponential amplification of approximation errors inherent in importance sampling ratios?
2.  How to ensure the objective actively unlearns or penalizes low-advantage completions rather than merely ignoring them?
3.  Can the ratio-free objective be theoretically grounded as a valid guided diffusion process, and can it leverage intermediate denoising steps to further accelerate convergence?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to reformulate the RL objective as a **ratio-free weighted log-likelihood** (wd1) derived from reverse-KL constrained policy optimization. Instead of computing policy ratios, the method minimizes the KL divergence between the current policy $\pi_\theta$ and the closed-form optimal policy $\pi^*$. To address the limitations of standard WLL (which ignores negative samples), wd1 introduces a complementary weight mechanism $(-w^+ + w^-)$ that balances positive reinforcement and negative penalization. The method is further extended to **wd1++**, which leverages intermediate denoising completions to construct a step-wise weighted policy optimization.

**Design Philosophy:**
1.  **Ratio-Free Optimization:** Eliminate importance sampling ratios to prevent exponential error amplification and reduce computational overhead by requiring only the approximation of the current policy $\pi_\theta$.
2.  **Symmetric Advantage Weighting:** Use a balanced combination of exponential advantage weights ($w^+$) and inverse exponential advantage weights ($w^-$) to explicitly reinforce high-advantage completions and actively unlearn low-advantage ones.
3.  **Step-wise Maximization:** Leverage the iterative refinement nature of dLLMs by utilizing all intermediate clean completions generated during the confidence-based remasking decoding process, not just the final output.

### 4. Core Innovation Points

1.  **Novel Ratio-Free RL Method (wd1):** Proposes wd1, a policy optimization approach with a weighted log-likelihood objective for dLLMs that dispenses with explicit policy ratios. The weight is defined as $(-w^+ + w^-)$, where $w^+ \propto \exp(A)$ increases the probability of higher-advantage completions, and $w^- \propto \exp(-A)$ decreases the probability of lower-advantage ones.
2.  **Theoretical Interpretation (Energy-Guided Diffusion + Unlearning):** Proves that wd1 can be interpreted as jointly training an energy-guided discrete diffusion model (guided by the advantage function) and unlearning low-advantage data. The negative penalty term corresponds to minimizing the ELBO of a Boltzmann distribution that places higher mass on lower advantage regions.
3.  **Step-wise Extension (wd1++):** Extends the method to wd1++ by leveraging intermediate completions generated in the decoding process. This expands the group of samples for advantage estimation and loss computation to include all clean completions at each denoising step, achieving rapid convergence (SOTA performance with only 20 RL training steps).
4.  **Efficiency and SFT Elimination:** Demonstrates that wd1 achieves significant accuracy improvements (up to +59% over d1) without requiring supervised fine-tuning (SFT) and with substantially lower computational cost (lower runtime, FLOPs, and NFEs per training step).

### 5. Overview of the Overall Technical Solution

The technical solution follows a rigorous derivation from constrained policy optimization to a ratio-free weighted loss:
1.  **Reverse-KL Objective Formulation:** Start with a reverse-KL regularized policy optimization objective (augmenting TRPO with reverse-KL and reference policy regularization).
2.  **Closed-Form Solution Derivation:** Derive the analytic form of the optimal policy $\pi^*$, which is a geometric mixture of the old policy, reference policy, and exponential advantage.
3.  **Weighted Log-Likelihood (WLL):** Minimize the KL divergence between $\pi^*$ and $\pi_\theta$ to yield a WLL loss. Approximate this using group-relative advantages and sample normalization.
4.  **wd1 Loss Construction:** Modify the WLL objective to explicitly penalize negative samples by introducing the complementary term $w^-$. The final weight becomes $(-w^+ + w^-)$, resulting in the wd1 loss.
5.  **wd1++ Extension:** Expand the sample group to include intermediate completions from the denoising trajectory. Compute step-wise weights and likelihoods for these intermediate samples.
6.  **Implementation:** Sample from the geometric mixture policy (approximated via logit addition) and apply the d1-based efficient likelihood approximation ($\log \pi_\theta(o|q) \approx \sum_k \log \pi_\theta(x_0^k|x_1, q')$).

### 6. Detailed Module Design

**1. Sampling Module (Geometric Mixture Policy):**
To avoid computing likelihoods for the reference policy during regularization, samples are drawn from a geometric mixture policy:
$\pi_{old}^{ref}(\cdot|q) \propto \pi_{old}(\cdot|q)^{\lambda/(\lambda+\beta)} \cdot \pi_{ref}(\cdot|q)^{\beta/(\lambda+\beta)}$
In implementation, since LLaDA parametrizes the clean token prediction, the logits of the denoising distribution are approximated as:
$\log \pi_{old}^{ref}(x_0^k|x_t, q) \approx \lambda \log \pi_{old}(x_0^k|x_t, q) + \beta \log \pi_{ref}(x_0^k|x_t, q)$

**2. Advantage and Weighting Module:**
Compute the group-relative advantage based on GRPO:
$\hat{A}_i = R(q, o_i) - mean(R(q, o_{1:G}))$
Define the positive and negative weights based on this advantage:
$w^+(q, o_i) = \frac{\exp(\psi \hat{A}_i)}{\sum_{j=1}^G \exp(\psi \hat{A}_j)}$
$w^-(q, o_i) = \frac{\exp(-\psi \hat{A}_i)}{\sum_{j=1}^G \exp(-\psi \hat{A}_j)}$
where $\psi = \frac{1}{\lambda + \eta}$. The normalization restricts gradient norm and stabilizes training.

**3. Loss Computation Module:**
The wd1 loss combines positive and negative weights to update the policy:
$\mathcal{L}_{wd1}(\theta) = E_{q \sim D, \{o_i\}_{i=1}^G \sim \pi_{old}^{ref}(\cdot|q)} [ \frac{1}{G} \sum_{i=1}^G ( - w^+(q, o_i) + w^-(q, o_i) ) \cdot \log \pi_\theta(o_i|q) ]$
The negative term $w^- \log \pi_\theta(o_i|q)$ minimizes the likelihood of low-advantage completions, inducing negative gradients. If all completions have identical advantages, $w^+ = w^-$, and optimization halts.

**4. Stepwise Extension Module (wd1++):**
During decoding with confidence-based remasking, intermediate clean completions $x_{0|l}$ are generated at step $l$. The group is expanded to $O_i = \{x_{0|l} | x_{0|l} \sim \pi_{old}^{ref}(\cdot | x_t, q), x_{0|L} = o_i, l = 1, \dots, L\}$. The loss becomes:
$\mathcal{L}_{wd1++}(\theta) = E_{q \sim D, \{O_i\}_{i=1}^G \sim \pi_{old}^{ref}(\cdot|q), l \sim Unif\{1,\dots,L\}} [ \frac{1}{L G_l} \sum_{i=1}^{G_l} \sum_{x_{0|l} \in O_i} ( - w^+(q, x_{0|l}) + w^-(q, x_{0|l}) ) \cdot \log \pi_\theta(x_{0|l}|x_l, q) ]$

**5. Likelihood Approximation Module:**
To compute $\log \pi_\theta(o_i|q)$ efficiently without full ELBO integration, the d1 approximation is used:
$\log \pi_\theta(x_0|q) \approx \sum_k \log \pi_\theta(x_0^k|x_1, q')$, where $q'$ is randomly masked from prompt $q$ at every gradient step.

**6. Theoretical Guidance Module:**
The method is theoretically interpreted as energy-guided diffusion. The target guided concrete score is approximated by minimizing the Advantage-Weighted Denoising Concrete Score Matching (AW-D-CSM) loss:
$\mathcal{L}_{AW-D-CSM} = E_{x_0 \sim p'_0(\cdot)} [ \underbrace{\exp(A(x_0))}_{\text{Advantage Weight}} \cdot E_{t \sim [0,T], p'_{t|0}(x_t|x_0)} [ \underbrace{\| s_\theta(x_t, t) - \frac{p'_0(\hat{x}_t|x_0)}{p'_0(x_t|x_0)} \|_2^2}_{\mathcal{L}_{D-CSM}(x_0)} ] ]$
This confirms that wd1 is equivalent to training an energy-guided diffusion where energy is negative advantage.

### 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $\pi_\theta$: Generation policy of dLLM.
*   $q \in D$: Prompt.
*   $o \in O$: Completions.
*   $R(q, o)$: Reward function.
*   $x_t$: Masked sequence at diffusion step $t \in [0,1]$.
*   $x_0$: Fully denoised sequence (completion $o$).
*   $K$: Length of sequence. $x_0^k$: $k$-th token of $x_0$.
*   $A^{\pi_{old}}$: Advantage function.
*   $\hat{A}_i = R(q, o_i) - mean(R(q, o_{1:G}))$: Group-relative advantage.
*   $p_{t|0}(x_t | x_0)$: Forward process probability.
*   $\pi_{old}^{ref}$: Geometric mixture policy $\propto \pi_{old}^{\lambda/(\lambda+\beta)} \pi_{ref}^{\beta/(\lambda+\beta)}$.

**Formulas:**

1.  **ELBO (Denoising Cross Entropy):**
    $\mathcal{L}(x_0) = -E_{t \sim U [0,1], x_t \sim p_{t|0}(x_t|x_0)} [ \frac{1}{t} \sum_{k=1}^K 1(x_t^k = [mask]) \log \pi_\theta(x_0^k | x_t) ]$

2.  **TRPO Objective:**
    $\max_\theta E_{q \sim D, o \sim \pi_\theta(\cdot|q)} [ A^{\pi_{old}}(q, o) - \lambda D_{KL}( \pi_{old}(\cdot|q) \| \pi_\theta(\cdot|q) ) ]$

3.  **Reverse-KL Regularized Policy Optimization:**
    $\max_\theta E_{q \in D, o \sim \pi_\theta(\cdot|q)} [ A^{\pi_{old}}(q, o) - \lambda D_{KL}( \pi_\theta(\cdot|q) \| \pi_{old}(\cdot|q) ) - \beta D_{KL}( \pi_\theta(\cdot|q) \| \pi_{ref}(\cdot|q) ) ]$

4.  **Optimal Policy $\pi^*$:**
    $\pi^*(\cdot|q) \propto \pi_{old}(\cdot|q)^{\lambda/(\lambda+\beta)} \cdot \pi_{ref}(\cdot|q)^{\beta/(\lambda+\beta)} \cdot \exp( \frac{A^{\pi_{old}}(q, \cdot)}{\lambda + \beta} )$

5.  **Weighted Log-Likelihood (WLL) Loss:**
    $\mathcal{L}_{WLL}(\theta) = E_{o \sim \pi_{old}^{ref}(\cdot|q)} [ - \exp(\psi A^{\pi_{old}}(q, o)) \cdot \log \pi_\theta(o|q) ]$
    $\approx E_{\{o_i\}_{i=1}^G \sim \pi_{old}^{ref}(\cdot|q)} [ \frac{1}{G} \sum_{i=1}^G - \frac{\exp(\psi \hat{A}_i)}{\sum_{j=1}^G \exp(\psi \hat{A}_j)} \log \pi_\theta(o_i|q) ]$

6.  **wd1 Loss:**
    $\mathcal{L}_{wd1}(\theta) = E_{q \sim D, \{o_i\}_{i=1}^G \sim \pi_{old}^{ref}(\cdot|q)} [ \frac{1}{G} \sum_{i=1}^G ( - w^+(q, o_i) + w^-(q, o_i) ) \cdot \log \pi_\theta(o_i|q) ]$
    Where:
    $w^+(q, o_i) = \frac{\exp(\psi \hat{A}_i)}{\sum_{j=1}^G \exp(\psi \hat{A}_j)} , \quad w^-(q, o_i) = \frac{\exp(-\psi \hat{A}_i)}{\sum_{j=1}^G \exp(-\psi \hat{A}_j)}$

7.  **wd1++ Loss:**
    $\mathcal{L}_{wd1++}(\theta) = E_{q \sim D, \{O_i\}_{i=1}^G \sim \pi_{old}^{ref}(\cdot|q), l \sim Unif\{1,\dots,L\}} [ \frac{1}{L G_l} \sum_{i=1}^{G_l} \sum_{x_{0|l} \in O_i} ( - w^+(q, x_{0|l}) + w^-(q, x_{0|l}) ) \cdot \log \pi_\theta(x_{0|l}|x_l, q) ]$

8.  **Energy-Guided Discrete Diffusion:**
    $p_{0|t}^*(x_0|x_t) \propto p'_{0|t}(x_0|x_t) \cdot \exp(A(x_0) - A_t(x_t))$
    Where $-A_t(x_t) = - \log E_{x_0 \sim p'_{0|t}(\cdot|x_t)}[\exp(A(x_0))]$ is intermediate energy function.

9.  **Concrete Score:**
    $s(x_t, t) \overset{\text{def}}{=} \frac{p(x_t^1, \dots, \hat{x}_t^i, \dots, x_t^d)}{p(x_t^1, \dots, x_t^i, \dots, x_t^d)}$

10. **Advantage-Weighted Denoising Concrete Score Matching (AW-D-CSM):**
    $\mathcal{L}_{AW-D-CSM} = E_{x_0 \sim p'_0(\cdot)} [ \underbrace{\exp(A(x_0))}_{\text{Advantage Weight}} \cdot E_{t \sim [0,T], p'_{t|0}(x_t|x_0)} [ \underbrace{\| s_\theta(x_t, t) - \frac{p'_0(\hat{x}_t|x_0)}{p'_0(x_t|x_0)} \|_2^2}_{\mathcal{L}_{D-CSM}(x_0)} ] ]$

11. **Advantage-Weighted Denoising Cross Entropy (AW-DCE):**
    $\mathcal{L}_{AW-DCE} = E_{x_0 \sim p'_0(\cdot)} [ \exp(A(x_0)) \cdot E_{t \sim [0,T], p'_{t|0}(x_t|x_0)} [ \sum_{x_t^i = [mask]} -\frac{1}{t} \log p_\theta(x_0^i | x_t^{UM}) ] ]$

### 8. Algorithm Pseudocode

**Algorithm 1: wd1: Weighted Policy Optimization for dLLMs**
Require: Reference model $\pi_{ref}$, prompt distribution $D$, group size $G$, reward function $R$, dLLM $\pi_\theta$, regularization hyperparameters $\lambda$ and $\beta$
1: Initialize $\pi_\theta \leftarrow \pi_{ref}$
2: while not converged do
3: $\pi_{old} \leftarrow \pi_\theta$
4: Sample prompt $q \sim D$ and $G$ completions $o_i \sim \pi_{old}(\cdot | q), \forall i \in [G]$
5: Compute advantage $\hat{A}_i = R(q, o_i) - mean(R(q, o_{1:G})), \forall i \in [G]$
6: Compute weights $w^+$ and $w^-$ in Equation (9), $\forall i \in [G]$
7: for gradient update iterations $n = 1, 2, \dots, \mu$ do
8: Compute approximated log-likelihood $\log \pi_\theta(o_i|q)$
9: Compute objective $\mathcal{L}_{wd1}(\theta)$ in Equation (8) or Equation (10) and update $\theta$
10: end for
11: end while
12: return $\pi_\theta$