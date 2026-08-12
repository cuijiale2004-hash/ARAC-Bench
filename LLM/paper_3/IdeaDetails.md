# 1. Research Background and Existing Pain Points

**Research Background**
As large language models (LLMs) evolve, Machine Unlearning (MU) has emerged to address growing concerns around user privacy, copyright infringement, and overall safety. Unlearning seeks to weaken a model's performance on undesired knowledge while preserving its original utility, serving as a practical alternative to retraining LLMs from scratch. The unlearning problem is typically cast as an optimization task that updates model parameters, splitting the dataset into a forget set and a retain set to balance unlearning effectiveness and utility preservation.

**Existing Pain Points**
Despite substantial research, state-of-the-art (SOTA) unlearning methods suffer from critical limitations:
1.  **Catastrophic Collapse and Metric Imbalance:** Existing methods often over-optimize one objective at the expense of others. For instance, Gradient Ascent (GA) and SAM-based methods experience a rapid loss of generalization where utility drops to near zero after specific training steps (catastrophic collapse). Conversely, methods like Task Vector and DOOR excel at preserving utility but sacrifice unlearning effectiveness, revealing a severe trade-off between forgetting strength and downstream utility.
2.  **Susceptibility to Jailbreak Attacks:** Current methods lack robustness against prompt manipulations (e.g., prefix injection, adaptive jailbreaks). Successful jailbreaks move the inner representations of harmful prompts toward the harmless anchor representation, thereby increasing the chance of generating undesired content. Small perturbations in the representation space can easily evade safety mechanisms.
3.  **Susceptibility to Relearning Attacks:** Unlearned models are vulnerable to adversaries who can recover deleted knowledge by fine-tuning the model on a tiny subset of the original forget set. This effectively undoes the unlearning process, as small parameter updates can revert the model to its pre-unlearning state.

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation**
The core motivation stems from the observation that while existing unlearning methods might achieve initial forgetting, they inherently lack robustness to adversarial perturbations and fine-tuning attacks. The geometric separability between harmful and harmless representations implies that enlarging the "margin" an attacker must breach is crucial for robustness. Similarly, flattening the parameter space can increase the "relearn margin." There is a pressing need for a unified framework that does not merely optimize for forgetting but enforces smoothness in both representation and parameter spaces to defend against diverse attacks while balancing the trade-off between forgetting and utility.

**Scientific Questions**
1.  How can we enlarge the representation-space margin to defend against jailbreak attacks that manipulate hidden states?
2.  How can we enlarge the parameter-space margin to mitigate relearning attacks that fine-tune model weights?
3.  How can we decouple the gradient conflicts between the forget and retain objectives to prevent catastrophic forgetting and balance unlearning metrics?
4.  Can a unified min-max optimization framework enforcing dual-space smoothness simultaneously improve robustness and metric balance in LLM unlearning?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea**
The paper proposes PRISM (Probe-guided Iterative Smoothness Minimization), a unified framework that enforces dual-space smoothness in representation and parameter spaces to improve robustness and balance unlearning metrics. The core idea is to cast unlearning as a min-max optimization game where the inner maximization searches for worst-case perturbations in both spaces, and the outer minimization updates parameters to enforce smoothness and widen attack margins.

**Design Philosophy**
1.  **Smoothness as Robustness:** Inspired by Sharpness-Aware Minimization (SAM), the design philosophy posits that promoting smoothness (flat minima) in the parameter space increases the relearn margin, and promoting smoothness (wide decision boundaries) in the representation space increases the jailbreak margin.
2.  **Decoupling for Balance:** Unlearning and retention are inherently conflicting objectives. The design employs gradient projection to orthogonalize the forget gradient against the retain gradient, ensuring first-order safety against utility degradation.
3.  **Adversarial Training in Representation Space:** Instead of relying on fragile safety tuning, the framework uses a robustly trained probe to guide the unlearning process. By training the probe against worst-case feature perturbations and then aligning forget representations with this probe's safe region, the model becomes robust to input-level and representation-level jailbreaks.

# 4. Core Innovation Points

1.  **Highlighting Limitations of Previous Methods:** The paper explicitly identifies and demonstrates the limitations of SOTA unlearning methods, including catastrophic forgetting, metric imbalance (trade-offs between utility and unlearning), and vulnerabilities to relearning and jailbreak attacks.
2.  **Dual-Space Smoothness Framework (PRISM):** Proposing a unified min-max unlearning framework that introduces dual-space smoothness. It simultaneously enforces smoothness in the representation space (via a robust probe) and the parameter space (via SAM-style optimization) to enhance robustness against diverse attacks.
3.  **Gradient Conflict Decoupling:** Introducing a gradient conflict decoupling mechanism through the lens of the min-max formulation. By projecting the forget gradient onto the orthogonal complement of the retain gradient, the method promotes balance among key unlearning metrics and mitigates catastrophic collapse.
4.  **Probe-Guided Adversarial Representation Smoothing:** Utilizing an adversarially trained probe as a smoothness driver in the representation space. The probe is trained with first-order worst-case perturbations (FGSM-style) to widen its decision boundary, and the base model is updated to align forget representations with this robust safe region, enlarging the jailbreak margin.

# 5. Overview of the Overall Technical Solution

The overall technical solution consists of two main smoothness optimization stages implemented via a min-max formulation:

1.  **Stage 1: Representation Space Smoothness (Robust Probe Training):** A linear or MLP probe is trained on the hidden states of a frozen base model to discriminate between harmful and benign representations. To ensure robustness, the probe is adversarially trained using a first-order inner maximization (FGSM) over an $\ell_\infty$ ball in the representation space. This creates a probe with a smooth, wide decision boundary.
2.  **Stage 2: Parameter Space Smoothness & Gradient Decoupling:** The unlearned model parameters are updated using a min-max objective over an $\ell_2$ ball in the parameter space (SAM). The inner maximization finds the worst-case parameter perturbation, and the outer minimization updates the weights to flatten the forget loss surface (increasing the relearn margin). Concurrently, the forget gradient is projected orthogonal to the retain gradient to decouple conflicts and preserve utility.

# 6. Detailed Module Design

**Module 1: Robust Probe Training**
A probe classifier $p_\phi$ is attached to a specific layer $L$ of the frozen base model. It takes the representation $z(x)$ as input and outputs class probabilities (harmless vs. undesired). To endow the probe with local robustness to jailbreak drifts and reduce loss sensitivity around $z$, the probe is trained on a mixture of clean and adversarially perturbed features.
*   **Perturbation Strategy:** For each pair $(x_i, y_i)$, the feature-space gradient $g(x_i; \phi)$ is computed. An adversarially perturbed representation $z_i^{\text{adv}}$ is constructed by solving a linearized inner maximization over an $\ell_\infty$ ball of radius $\epsilon$. The solution moves to the vertices of the hypercube (FGSM-style) to maximize the linearized loss.

**Module 2: Probe-Guided Forgetting**
During unlearning, the model parameters $\theta$ are updated while the robust probe $p_{\phi^*}$ is kept frozen. For each forget example $x \in \mathcal{D}_f$, the representation $h_{\theta, L}(x)$ is extracted. The model is optimized to minimize the negative log-likelihood of the harmless class $y=0$ under the probe. This pushes the forget representation deep into the probe's harmless region, enlarging the jailbreak margin.

**Module 3: Parameter-Space Smoothness Minimization**
To defend against relearning attacks, the forget objective is optimized under a min-max formulation over the parameter space. The inner maximization searches for a parameter perturbation $\delta$ within an $\ell_2$ ball of radius $\rho$ that maximizes the forget loss $\ell_f$. The outer minimization updates the parameters to minimize this worst-case loss. A first-order linear approximation is used to solve the inner problem efficiently, resulting in a perturbation direction aligned with the gradient and a smoothness penalty proportional to the gradient norm.

**Module 4: Gradient Conflict Decoupling (GCD)**
To prevent the smoothness optimization from over-weighting the forget objective and inducing catastrophic forgetting, the forget gradient $g_f$ is orthogonalized against the retain gradient $g_r$. The forget gradient is projected onto the orthogonal complement of $g_r$. This removes the component of $g_f$ that conflicts with $g_r$, providing first-order safety that the retain loss does not increase locally.

# 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions**
*   $\theta_0$: Pretrained model parameters.
*   $\mathcal{D}$: Training dataset, split into forget set $\mathcal{D}_f$ and retain set $\mathcal{D}_r$.
*   $\mathcal{L}_f, \mathcal{L}_r$: Forget loss and retain loss.
*   $\gamma$: Coefficient balancing $\mathcal{L}_f$ and $\mathcal{L}_r$.
*   $n$: Prompt length, $m$: Vocabulary size, $d$: Hidden dimension.
*   $f(\cdot)$: Model mapping prompt to representation.
*   $g(\cdot)$: PCA transformation.
*   $\mathcal{D}_a, \mathcal{D}_b$: Harmless and harmful anchor prompt sets.
*   $c_a, c_b$: Acceptance and refusal centers.
*   $e_a$: Acceptance direction.
*   $\theta_u$: Unlearned model parameters.
*   $\delta$: Parameter update variable introduced by relearning.
*   $h_{\theta, L}(x)$: Layer-$L$ representation of input $x$ under parameters $\theta$.
*   $\pi$: Pooling operator.
*   $p_\phi$: Probe classifier with parameters $\phi$.
*   $\epsilon$: Radius of $\ell_\infty$ ball for representation perturbation.
*   $\rho$: Radius of $\ell_2$ ball for parameter perturbation.
*   $\lambda$: Balances probe loss and generation loss.
*   $\mathcal{L}_{\text{gen}}$: Downweights preferences on $\mathcal{D}_f$ with reference model $\theta_{\text{ref}}$.
*   $J_\theta(x)$: Jacobian matrix mapping representation-space signals to parameter updates.
*   $g_h(x; \theta)$: Steepest-ascent direction w.r.t. the representation.
*   $P_r$: Retain projection operator.
*   $P_r^\perp$: Orthogonal projection operator.

**Mathematical Formulas**

1.  **General Unlearning Formulation:**
    $$\theta_u = \arg\min_\theta [\mathcal{L}_f(\theta; \mathcal{D}_f) + \gamma \mathcal{L}_r(\theta; \mathcal{D}_r)]$$

2.  **Jailbreak Attack Objective (Maximize projection onto acceptance direction):**
    $$\max_x L(x) := \langle g(f(x)) - g(f(x_0)), e_a \rangle$$
    where $e_a = \frac{c_a - c_b}{\|c_a - c_b\|_2} \in \mathbb{R}^d$.

3.  **Relearning Attack Formulation:**
    $$\min_{\theta, \delta} \ell_{\text{relearn}}(\theta | \mathcal{D}'_f) \quad \text{s.t.} \quad \theta = \theta_u + \delta, \theta(0) = \theta_u$$

4.  **Representation Extraction:**
    $$z(x) := h_{\theta_0, L}(x) \in \mathbb{R}^d := \pi(\text{hidden states}^{(L)}(x))$$

5.  **Probe Inner Maximization (Adversarial Perturbation):**
    $$\delta_i^* \in \arg\max_{\|\delta\|_\infty \leq \epsilon} g(x_i; \phi)^\top \delta, \quad z_i^{\text{adv}} = z(x_i) + \delta_i^*$$
    Closed-form solution:
    $$z_i^{\text{adv}} = z(x_i) + \epsilon \text{sign}(g(x_i; \phi))$$

6.  **Probe-Guided Forgetting Loss:**
    $$\mathcal{L}_{\text{probe}}(\theta; x) = -\log p_{\phi^*}(y=0 | h_{\theta, L}(x))$$

7.  **Gradient of Probe Loss:**
    $$\nabla_\theta \mathcal{L}_{\text{probe}}(\theta; x) = \left( \frac{\partial h_{\theta, L}(x)}{\partial \theta} \right)^\top \nabla_h [-\log p_{\phi^*}(0 | h)] \Big|_{h=h_{\theta, L}(x)} = J_\theta(x)^\top g_h(x; \theta)$$

8.  **Parameter-Space Smoothness Objective:**
    $$\min_\theta \left[ \max_{\|\delta\|_2 \leq \rho} \ell_f(\theta + \delta) \right], \quad \ell_f(\theta) = \lambda \mathcal{L}_{\text{probe}}(\theta; \mathcal{D}_f) + \mathcal{L}_{\text{gen}}(\theta; \mathcal{D}_f, \theta_{\text{ref}})$$

9.  **First-Order Approximation of Smoothness Minimization (LSM_f):**
    Define $\text{LSM}_f(\theta) := \ell_f(\theta + \delta(\theta))$ and $g(\theta) := \nabla_\theta \ell_f(\theta)$.
    $$\text{LSM}_f(\theta) \approx \ell_f\left(\theta + \arg\max_{\|\delta\|_2 \leq \rho} [\ell_f(\theta) + \delta^\top g(\theta)] \right)$$
    $$= \ell_f\left(\theta + \arg\max_{\|\delta\|_2 \leq \rho} \delta^\top g(\theta)\right)$$
    $$= \ell_f\left(\theta + \rho \frac{g(\theta)}{\|g(\theta)\|_2}\right)$$
    $$\approx \ell_f(\theta) + \rho \left\langle \frac{g(\theta)}{\|g(\theta)\|_2}, g(\theta) \right\rangle$$
    $$= \ell_f(\theta) + \rho \|g(\theta)\|_2$$

10. **Gradient Conflict Decoupling:**
    Forget projection operator: $P_r = \frac{g_r g_r^\top}{\|g_r\|_2^2}$
    Retain projection operator: $P_r^\perp = I - P_r$
    Orthogonalized forget gradient:
    $$g_f^\perp = g_f - \frac{\langle g_f, g_r \rangle}{\|g_r\|_2^2} g_r = P_r^\perp g_f$$

# 8. Algorithm Pseudocode

**Algorithm 1: Representation-level Unlearning**
Require: Xtrain, Ytrain, input dim d, batch size B, learning rate $\eta$, epochs T, FGSM radius $\epsilon$, adv weight $\alpha$, L1 weight $\lambda_1$
1 Construct mini-batch loader L from (Xtrain, Ytrain) with batch size B and shuffling
2 Initialize MLP probe $p_\phi$: d $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 2 with ReLU
3 for epoch = 1 to T do
4   foreach $(x_b, y_b) \in L$ do
5     mark $x_b$ as requiring gradients
6     $z_{\text{clean}} \leftarrow p_\theta(x_b)$
7     $\mathcal{L}_{\text{clean}} \leftarrow \ell(z_{\text{clean}}, y_b)$
8     $g_x \leftarrow \nabla_{x_b} \mathcal{L}_{\text{clean}}$
9     $x_{\text{adv}} \leftarrow x_b + \epsilon \cdot \text{sign}(g_x)$
10    $z_{\text{adv}} \leftarrow p_\theta(x_{\text{adv}})$
11    $\mathcal{L}_{\text{adv}} \leftarrow \ell(z_{\text{adv}}, y_b)$
12    $\mathcal{L}_1 \leftarrow \sum_{w \in \theta} |w|$
13    $\mathcal{L}_{\text{total}} \leftarrow \mathcal{L}_{\text{clean}} + \alpha \mathcal{L}_{\text{adv}} + \lambda_1 \mathcal{L}_1$
14    $\theta \leftarrow \text{AdamStep}(\theta, \nabla_\theta \mathcal{L}_{\text{total}})$
15 Return: trained probe $p_{\phi^*}$

**Algorithm 2: Smoothness Minimization**
Require: original model $\theta$, trained probe $p_{\phi^*}$, forget set $\mathcal{D}_f$, retain set $\mathcal{D}_r$, steps N, learning rate $\eta$, SM radius $\rho$, mixing weights $\lambda_f, \lambda_r$, adversarial schedule $\gamma_{\text{adv}}(i)$
1 $\theta_u \leftarrow \theta$
2 for i = 1 to N do
3   Sample $(x_f, y_f) \sim \mathcal{D}_f$
4   $\gamma \leftarrow \gamma_{\text{adv}}(i)$
5   $g \leftarrow \nabla_\theta [\ell_{\text{base}}(\theta_u; x_f) + \gamma \ell_{\text{probe}}(\theta_u; x_f)]$
6   $\delta \leftarrow \rho \cdot \frac{g}{\|g\|_2 + \epsilon}$
7   $g_f \leftarrow \nabla_\theta [\ell_{\text{base}}(\theta_u + \delta; x_f) + \gamma \ell_{\text{probe}}(\theta_u + \delta; x_f)]$
8   Sample $(x_r, y_r) \sim \mathcal{D}_r$
9   $g_r \leftarrow \nabla_\theta \ell_r(\theta_u; x_r)$
10  if $\|g_r\|_2 > 0$ then
11    $g_f \leftarrow g_f - \frac{\langle g_f, g_r \rangle}{\|g_r\|_2^2} g_r$
12  $\theta_u \leftarrow \theta_u - \eta (\lambda_f g_f + \lambda_r g_r)$
13 Return: $\theta_u$