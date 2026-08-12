# 1. Research Background and Existing Pain Points

**Research Background:**
Diffusion models have achieved state-of-the-art generative performance across a wide range of tasks, including image synthesis, text-to-image generation, text-to-3D generation, video generation, and audio generation. However, diffusion models typically require hundreds of iterative denoising steps and contain billions of parameters, resulting in high computational costs that hinder deployment in resource-constrained environments. To address this limitation, extensive efficiency techniques have been devoted to developing efficient model techniques for diffusion models. Among these techniques, network quantization and feature caching have emerged as two promising approaches. The former reduces data precision to low-bit representations to simultaneously shrink model size and accelerate inference, while the latter caches intermediate features to eliminate redundant computations across diffusion timesteps.

**Existing Pain Points:**
Although efficient methods effectively reduce overhead, they inevitably suffer from approximation errors between the efficient model and the original counterpart, which degrade the generation quality of the model. Prior studies have attempted to mitigate such errors through specialized mechanisms, such as timestep-wise quantization parameters and non-uniform caching strategies. Despite their effectiveness, these methods remain pre-deployment solutions that require the ability to re-execute the model-efficiency pipeline and the original model. In practice, such requirements often do not hold once a model has been deployed in edge or production environments. On the one hand, reapplying the model-efficiency pipeline and redeploying the model incurs significant engineering overhead, making it impractical. Also, deployed models are typically immutable due to storage limitations, deployment policies, or system design constraints. On the other hand, after being deployed, the original model is often irretrievable, making re-executing the model-efficiency pipeline unfeasible. Furthermore, through an analysis of error propagation across diffusion timesteps, it is revealed that these approximation errors can accumulate exponentially, severely impairing output quality.

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Inspired by recent advances in test-time scaling, where model behavior is adjusted at inference time without retraining, the research seeks to correct the performance degradation of efficient diffusion models post-deployment. Once deployed, models are difficult to correct as modifying the model is typically infeasible. There is a critical need for a method that can mitigate inference-time errors without requiring any retraining, access to the original model, or architectural changes.

**Scientific Questions:**
Is it possible to improve the performance of an already deployed diffusion model without repeating the model-efficiency pipeline? How can the exponential accumulation of approximation errors across diffusion timesteps be effectively suppressed and reduced to linear growth at test-time?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is Iterative Error Correction (IEC), a novel test-time method that mitigates inference-time errors by iteratively refining the model’s output. By analyzing error propagation across diffusion timesteps, the method reveals that approximation errors accumulate exponentially. IEC introduces correction steps within diffusion timesteps to iteratively refine the initial estimate. Theoretically, IEC reduces error accumulation from exponential to linear growth. 

**Design Philosophy:**
The design philosophy of IEC is to be a plug-and-play method that operates entirely at test-time. It requires no re-running the model-efficiency pipeline, no fine-tuning weights, and no changes in model architecture, making it compatible with deployed diffusion models. While IEC introduces additional computational overhead, it is highly flexible and can be selectively applied to a subset of all diffusion timesteps. Applying IEC to more timesteps yields greater quality improvements, while using fewer timesteps reduces computational overhead. Skipping IEC entirely preserves the model’s original performance. This flexibility provides users with fine-grained control over the trade-off between efficiency and generation quality.

# 4. Core Innovation Points

1. **Theoretical Analysis of Error Accumulation:** The paper provides a rigorous theoretical analysis of error propagation and accumulation across diffusion timesteps, revealing that approximation errors introduced by efficiency techniques accumulate exponentially, severely degrading output quality.
2. **Proposal of Iterative Error Correction (IEC):** A novel test-time method that mitigates these errors by iteratively refining the model’s output, breaking the exponential error growth trend during the accumulation process and reducing it to linear growth.
3. **Theoretical Guarantee via Fixed-Point Theory:** IEC is theoretically proven to converge using Banach's fixed-point theorem, demonstrating that the mapping is a contraction mapping with a Lipschitz constant less than 1, ensuring the propagated error at each timestep is bounded.
4. **Plug-and-Play and Flexible Trade-off:** IEC operates entirely at test-time without requiring re-training, fine-tuning, or architectural changes. It provides a flexible trade-off between performance and efficiency by allowing selective application to subsets of timesteps.

# 5. Overview of the Overall Technical Solution

The overall technical solution begins by analyzing the deterministic DDIM sampling process and modeling the errors introduced by efficient diffusion models (quantization or feature caching). It defines the discrepancy between perturbed input and ideal input, as well as the perturbation in the model’s prediction. By deriving the recursive relation for error propagation using first-order Taylor expansion and Jacobian matrices, it reveals the exponential growth nature of the accumulated error. To mitigate this, Iterative Error Correction (IEC) is introduced, which computes an initial estimate using the standard DDIM update and then iteratively refines this estimate using a specific update rule with a tunable hyperparameter $\lambda$. The convergence of IEC is mathematically justified using fixed-point theory, proving the mapping is a contraction. Finally, the error accumulation in IEC is analyzed, demonstrating that the total accumulated error grows only linearly. The method is implemented as a plug-and-play module that can be flexibly applied to any subset of timesteps.

# 6. Detailed Module Design

**Module 1: Analysis of Error Accumulation in Efficient Diffusion Models**
This module focuses on understanding how errors propagate through diffusion timesteps using the deterministic DDIM sampling procedure.
- **Preliminaries:** The deterministic DDIM sampling process is defined, and coefficients $A_t$ and $B_t$ are introduced to simplify the update rule to $x_{t-1} = A_t x_t + B_t \epsilon_\theta(x_t, t)$.
- **Modeling Error:** Two primary types of errors are modeled at each timestep. First, a discrepancy $\delta_t$ between the perturbed input $\tilde{x}_t$ and the ideal input $x_t$, defined as $\tilde{x}_t = x_t + \delta_t$. Second, a prediction error $\epsilon_\delta^\theta$ in the model’s prediction, defined as $\tilde{\epsilon}_\theta(x_t, t) = \epsilon_\theta(x_t, t) + \epsilon_\delta^\theta$.
- **Perturbed Update Rule:** Incorporating these errors, the perturbed update rule at timestep $t$ is derived using first-order Taylor expansion with the Jacobian $J_t$ of the model output.
- **Error Propagation:** By subtracting the ideal update from the perturbed update, the recursive relation for error propagation is derived: $\delta_{t-1} \approx (A_t + B_t J_t)\delta_t + B_t \epsilon_\delta^\theta$. Recursively expanding this from timestep $T$ to 0 gives the accumulated error at $t=0$.
- **Analysis of Error Growth:** The spectral norm $\|A_t + B_t J_t\|$ is analyzed to quantify the maximum amplification. If $\|A_t + B_t J_t\| > 1$, errors amplify and can grow exponentially. Empirical observations confirm this norm consistently exceeds 1, proving exponential error accumulation.

**Module 2: Iterative Error Correction (IEC)**
This module introduces the core mechanism to suppress error accumulation.
- **Core Mechanism:** At a timestep $t-1$, an initial estimate $x_{t-1}^{(0)}$ is computed using the standard DDIM update. We then iteratively refine this estimate by repeatedly applying the IEC update rule with hyperparameter $\lambda$.
- **Termination Condition:** The iteration proceeds until the difference $\|x_{t-1}^{(k+1)} - x_{t-1}^{(k)}\|$ falls below a predefined threshold $\tau$ or the maximum number of iterations $K$ is reached.
- **Convergence Analysis:** The update rule is reformulated into a mapping $G(x)$. Using Banach's fixed-point theorem, it is proven that the iterative procedure converges to a unique fixed point if $G(x)$ is a contraction mapping. The Lipschitz constant $L$ is computed as the spectral norm of the Jacobian of $G(x)$. Since $B_t < 0$, an appropriately chosen positive $\lambda$ ensures $L < 1$, guaranteeing convergence.
- **Error Accumulation in IEC:** The error at iteration $k+1$ is defined and bounded using the triangle inequality and Mean Value Inequality. The recursive bound shows the error converges to a bounded constant $C/(1-L)$, which is independent of previous timestep errors, thus transforming the total error growth from exponential to linear.

# 7. All Mathematical Formulas and Symbol Definitions

**Forward Process:**
$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$$
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, \beta_t I)$$
Where $\alpha_t = 1 - \beta_t$, $\beta_t$ is t-based parameters.

**Reverse Process:**
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \hat{\mu}_{\theta,t}(x_t), \hat{\beta}_t I)$$

**DDIM Sampling:**
$$x_{t-1} = \sqrt{\alpha_{t-1}} \frac{x_t - \sqrt{1-\alpha_t}\epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}} + \sqrt{1-\alpha_{t-1}}\epsilon_\theta(x_t, t)$$

**Simplified DDIM Update Rule:**
$$x_{t-1} = A_t x_t + B_t \epsilon_\theta(x_t, t)$$

**Coefficients Definitions:**
$$A_t = \frac{\sqrt{\alpha_{t-1}}}{\sqrt{\alpha_t}}$$
$$B_t = \sqrt{1-\alpha_{t-1}} - \frac{\sqrt{\alpha_{t-1}}\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}$$

**Error Modeling Definitions:**
- $\tilde{x}_t = x_t + \delta_t$ (Discrepancy between perturbed input and ideal input)
- $\tilde{\epsilon}_\theta(x_t, t) = \epsilon_\theta(x_t, t) + \epsilon_\delta^\theta$ (Perturbation from network quantization or feature caching)
- $J_t = \left. \frac{\partial \tilde{\epsilon}_\theta(x,t)}{\partial x} \right|_{x=x_t}$ (Jacobian of the model output with respect to its input)

**Perturbed Update Rule:**
$$\tilde{x}_{t-1} = A_t \tilde{x}_t + B_t \tilde{\epsilon}_\theta(\tilde{x}_t, t)$$
$$= A_t(x_t + \delta_t) + B_t \tilde{\epsilon}_\theta(x_t + \delta_t, t)$$
$$\approx A_t(x_t + \delta_t) + B_t (\tilde{\epsilon}_\theta(x_t, t) + J_t \delta_t)$$
$$= A_t(x_t + \delta_t) + B_t (\epsilon_\theta(x_t, t) + \epsilon_\delta^\theta + J_t \delta_t)$$

**Error Propagation Recursive Relation:**
$$\delta_{t-1} = \tilde{x}_{t-1} - x_{t-1}$$
$$\approx A_t \delta_t + B_t (\epsilon_\delta^\theta + J_t \delta_t)$$
$$= (A_t + B_t J_t)\delta_t + B_t \epsilon_\delta^\theta$$

**Accumulated Error at t=0:**
$$\delta_0 = \sum_{i=1}^T \left( \prod_{j=i+1}^T (A_j + B_j J_j) \right) (B_i \epsilon_\delta^\theta)$$

**IEC Update Rule:**
$$x_{t-1}^{(k+1)} = x_{t-1}^{(k)} + \lambda \left( A_t x_t + B_t \epsilon_\theta(x_{t-1}^{(k)}, t) - x_{t-1}^{(k)} \right), \quad k = 0, 1, 2, \ldots$$
Where $\lambda$ is a tunable hyperparameter.

**IEC Fixed-Point Reformulation:**
$$x_{t-1}^{(k+1)} = (1-\lambda)x_{t-1}^{(k)} + \lambda (A_t x_t + B_t \epsilon_\theta(x_{t-1}^{(k)}, t))$$

**Mapping G(x) Definition:**
$$G(x) = (1-\lambda)x + \lambda (A_t x_t + B_t \epsilon_\theta(x, t))$$

**Contraction Mapping Condition (Banach's fixed-point theorem):**
$$\|G(x)-G(y)\| \le L\|x-y\|, \forall x, y$$
Where $0 < L < 1$ is the Lipschitz constant.

**Jacobian of G(x):**
$$\nabla G(x) = (1-\lambda)I + \lambda B_t J_t$$

**Lipschitz Constant L:**
$$L = \|\nabla G(x)\| = \|(1-\lambda)I + \lambda B_t J_t\|$$

**IEC Error Definition:**
$$\delta_{t-1}^{(k+1)} = x_{t-1}^{(k+1)} - x_{t-1} = G(x_{t-1}^{(k)}) - x_{t-1}$$
$$= \underbrace{G(x_{t-1} + \delta_{t-1}^{(k)}) - G(x_{t-1})}_{\text{first term}} + \underbrace{G(x_{t-1}) - x_{t-1}}_{\text{second term}}$$

**Noisy Mapping G(x) for Error Accumulation:**
$$G(x) = (1-\lambda)x + \lambda(A_t \tilde{x}_t + B_t \tilde{\epsilon}_\theta(x, t))$$

**Second Term Approximation:**
$$G(x_{t-1}) - x_{t-1} = (1-\lambda)x_{t-1} + \lambda(A_t(x_t + \delta_t) + B_t(\epsilon_\theta(x_{t-1}, t) + \epsilon_\delta^\theta)) - x_{t-1}$$
$$= \lambda(A_t x_t + B_t \epsilon_\theta(x_{t-1}, t) - x_{t-1}) + \lambda(A_t \delta_t + B_t \epsilon_\delta^\theta)$$
$$= \lambda B_t(\epsilon_\theta(x_{t-1}, t) - \epsilon_\theta(x_t, t)) + \lambda(A_t \delta_t + B_t \epsilon_\delta^\theta)$$

**Error Norm Bound:**
$$\|\delta_{t-1}^{(k+1)}\| \le \|\nabla G(x_{t-1}) \cdot \delta_{t-1}^{(k)}\| + \lambda(\|B_t(\epsilon_\theta(x_{t-1}, t) - \epsilon_\theta(x_t, t))\| + \|A_t \delta_t + B_t \epsilon_\delta^\theta\|)$$
$$\le L \|\delta_{t-1}^{(k)}\| + C$$
Where $C = \lambda(\|B_t(\epsilon_\theta(x_{t-1}, t) - \epsilon_\theta(x_t, t))\| + \|A_t \delta_t + B_t \epsilon_\delta^\theta\|)$ is a bounded constant independent of $\delta_{t-1}^{(k)}$.

**Recursive Error Bound:**
$$\|\delta_{t-1}^{(k)}\| \le L^k \|\delta_{t-1}^{(0)}\| + C \frac{1-L^k}{1-L}$$

**Converged Error Bound:**
As $k \to \infty$, $L^k \to 0$, and the error converges to:
$$\|\delta_{t-1}^{(\infty)}\| \le \frac{C}{1-L}$$

**Linear Total Accumulated Error in IEC:**
$$\delta_0^{IEC} = \sum_{j=1}^T \delta_x^j$$
Where each $\delta_x^j$ is independently bounded.

# 8. Algorithm Pseudocode

**Algorithm 1 Iterative Error Correction (IEC)**
1: Input: $x_t$, timestep $t$, hyperparameter $\lambda$, threshold $\tau$ , max iterations $K$
2: Output: Final corrected estimate $x_{t-1}^*$
3: Initialize $x_{t-1}^{(0)}$ using Eq. 6
4: while $k < K$ do
5:     Obtain $x_{t-1}^{(k+1)}$ using Eq. 10
6:     if $\|x_{t-1}^{(k+1)} - x_{t-1}^{(k)}\| < \tau$ then
7:         break
8:     end if
9:     $k = k + 1$
10: end while
11: Return: $x_{t-1}^* = x_{t-1}^{(k+1)}$