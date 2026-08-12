**1. Research Background and Existing Pain Points**

Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation across diverse applications. However, their pre-training on massive data corpora raises growing concerns about safety, privacy, and trustworthiness. LLMs may inadvertently reproduce copyrighted content, expose personally identifiable information, or generate harmful instructions. To address these risks, LLM unlearning has emerged as a promising direction, aiming to remove the influence of undesired data, knowledge, and associated model capabilities without incurring the cost of retraining the entire model and preserving the model’s general utility.

Despite recent progress in developing LLM unlearning algorithms that achieve both effective forgetting and utility preservation, ensuring robust unlearning remains a significant challenge. Unlearning performance can quickly deteriorate under post-unlearning weight perturbations. Prior work shows that fine-tuning on even a small set of forgotten samples or semantically related texts can substantially reverse unlearning effects (relearning attacks), while model compression techniques such as quantization may also resurface erased content. Furthermore, when unlearned models are adapted to downstream tasks via fine-tuning, their unlearning guarantees often degrade.

Existing research on robust LLM unlearning has primarily focused on problem-level reformulations or algorithm-level modifications, often assuming a specific vulnerability source and tailoring the unlearning method accordingly. For instance, some works cast robust unlearning as a min–max problem against relearning-induced perturbations and adapt sharpness-aware minimization (SAM) to strengthen robustness. Others propose tamper-resistant unlearning via meta-learning, modeling the attacker as a weight-tampering adversary, or leverage invariant risk minimization (IRM) to regularize unlearning against degradation from irrelevant fine-tuning. While effective, these approaches rely on customized changes to unlearning objectives, thereby modifying the underlying optimization algorithm itself. In contrast, the role of the base optimizer, independent of any problem-wise and algorithm-level modifications, in shaping unlearning robustness remains largely unexplored. Notably, even heuristic optimizer adjustments, such as increasing the learning rate, have been observed to improve robustness against weight quantization, hinting at a deeper connection.

**2. Core Research Motivation and Scientific Questions**

The core research motivation stems from the gap in understanding how the choice of the base optimizer, independent of the unlearning objective formulation, influences the robustness of LLM unlearning against weight perturbations like relearning attacks and quantization. Since previous robustness methods require modifying the optimization problem itself, exploring the optimizer's inherent properties offers a novel, objective-agnostic pathway to enhance resilience.

This leads to the central research question of this work: 
(Q) How does the choice of optimizer influence the robustness of LLM unlearning, and what optimizers can improve robustness without sacrificing unlearning effectiveness?

To address this question, the work introduces the concept of optimizer "grade" for LLM unlearning, defined by the level of gradient information utilized by an optimizer. The motivation is that downgraded optimizers (using less precise gradient information) introduce higher optimization noise tolerance, naturally encouraging convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-training perturbations. 

**3. Overall Core Idea and Design Philosophy**

The overall core idea is "Downgrade to Upgrade": simplifying or downgrading the optimizer enhances the robustness of LLM unlearning. The design philosophy posits that a downgraded optimizer, which utilizes less precise gradient information (ranging from zeroth-order gradient-free methods to compressed-gradient variants like gradient sign-based optimizers), introduces inherent noise tolerance into the optimization process. While these optimizers produce noisier and less precise updates, they encourage convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-unlearning weight perturbations.

By connecting zeroth-order (ZO) methods with randomized smoothing (RS), the work highlights their natural advantage for robust unlearning, as ZO optimization inherently solves a randomized smoothing version of the original objective. To balance the trade-off where ZO optimizers exhibit weaker unlearning effectiveness due to variance but stronger robustness, and first-order (FO) optimizers exhibit the opposite, the work proposes a hybrid optimizer design. This hybrid approach integrates FO and ZO updates within a unified framework, modeled as a leader-follower (bi-level optimization) game, where the ZO optimizer acts as the "leader" driving robustness, and the FO optimizer acts as the "follower" providing high-precision initialization and reducing ZO variance.

**4. Core Innovation Points**

*   **First systematic study of optimizer choice in LLM unlearning:** The paper presents the first systematic investigation showing that downgrading the optimizer (via quantized or zeroth-order updates) can improve robustness against weight tampering. It establishes a clear link between optimizer grade and unlearning robustness grade.
*   **Rationale for robustness via noise tolerance:** The paper provides a technical rationale for the observed phenomenon: downgraded optimizers introduce higher optimization noise tolerance, making unlearned models more resilient to post-unlearning weight perturbations. Gradient compression acts as a "denoiser," while ZO estimation connects to randomized smoothing.
*   **FO–ZO hybrid optimization framework:** The paper proposes a novel unified framework that integrates FO and ZO optimizers, combining ZO-induced robustness with FO-driven unlearning effectiveness. This is conceptualized as a leader-follower game where ZO (leader) ensures robustness and FO (follower) ensures precision.
*   **Extensive empirical validation:** The findings are validated through extensive experiments across diverse unlearning tasks (MUSE, WMDP, TOFU) and methods, demonstrating a consistent link between optimizer grade and unlearning robustness, and the superior trade-off achieved by the hybrid method.

**5. Overview of the Overall Technical Solution**

The overall technical solution begins by formulating the LLM unlearning problem as a regularized optimization task involving a forget loss and a retain loss. It then defines the robustness challenges posed by relearning attacks and weight quantization. Introducing the concept of "optimizer grade," the solution categorizes optimizers based on the level of descent information they exploit, specifically looking at inter-order differences (second-order, first-order, zeroth-order) and intra-order differences (compressed vs. uncompressed gradients within the first order).

The solution demonstrates that intra-order downgrading via gradient compression (quantizing gradients into low-bit representations like signSGD) improves quantization robustness because the quantization operator effectively acts as a denoiser. For inter-order downgrading, the solution utilizes zeroth-order (ZO) optimization, which estimates gradients via finite differences of objective function values. Theoretically, the ZO gradient estimator is shown to be an unbiased estimator of the gradient of a smoothed version of the original objective, linking ZO optimization directly to randomized smoothing, which inherently incorporates noise and improves robustness.

Empirical analysis using Linear Mode Connectivity (LMC) reveals that FO and ZO optimizers converge to different basins: FO converges to basins with stronger unlearning but limited robustness, while ZO converges to separate basins with greater robustness but weaker unlearning effectiveness. To achieve the best of both worlds, the solution proposes a Hybrid FO-ZO optimization strategy. This hybrid approach alternates between FO updates (which provide high-quality initialization and reduce variance) and ZO updates (which steer the model toward robust basins), structured as a leader-follower game where ZO is the leader (prioritizing robustness) and FO is the follower (providing precision).

**6. Detailed Module Design**

*   **Optimizer Downgrade via Gradient Compression Module:** This module replaces the full-precision gradient with a quantized version using a quantization operator. The design is based on the premise that when a gradient compression-based optimizer is used for unlearning, it naturally improves tolerance to weight perturbations, as the quantization operator effectively acts as a "denoiser", mapping perturbed weights onto the same discrete bit values. Variants include signSGD, 8-bit Adam, and 1-bit Adam (signAdam).
*   **Optimizer Downgrade via ZO Gradient Estimation Module:** This module approximates gradients without backpropagation using finite differences of objective function values. It samples random direction vectors (e.g., uniformly from the unit sphere) to estimate the gradient. The design adopts the AdaZO optimizer to further reduce variance and improve convergence. The module inherently solves a randomized smoothing (RS) version of the unlearning objective, which incorporates random noise into the optimization process to enhance robustness.
*   **Linear Mode Connectivity (LMC) Analysis Module:** This module assesses whether two unlearned models (e.g., one trained with FO and one with ZO) can be connected by linear interpolation in parameter space. It validates that gradient-compressed optimizers display clear connectivity with standard FO (converging to the same basin), whereas ZO lacks LMC with FO, implying convergence to a separate basin that supports its distinctive robustness.
*   **Hybrid FO-ZO Optimization Module:** This module integrates FO and ZO optimizers in an alternating fashion. The mechanism applies FO optimization (Adam by default) to the pre-unlearned model for N steps, producing an intermediate model; then ZO optimization (AdaZO by default) continues for another N steps. This alternation repeats, ending on a ZO round. The design is grounded in a leader-follower game rationale: since unlearning robustness is the primary goal, the ZO optimizer is treated as the "leader" that drives the model toward robust basins. The FO optimizer acts as the "follower," providing a high-quality initialization for ZO and reducing the variance introduced by ZO gradient estimation. Ablation studies show that allocating an equal number of FO and ZO updates achieves the best balance, as more FO steps weaken the "leader" (reducing robustness) and more ZO steps weaken the "follower" (reducing unlearning effectiveness).

**7. All Mathematical Formulas and Symbol Definitions**

*   **LLM Unlearning Optimization Problem:**
    minimize_θ ℓf(θ|Df) + λℓr(θ|Dr)
    Where:
    θ: Model parameters.
    ℓf: Forget loss evaluated on the forget dataset Df.
    ℓr: Retain loss computed on the retain dataset Dr.
    λ ≥ 0: Regularization parameter balancing unlearning effectiveness and utility retention.

*   **Relearning Attack Formulation:**
    minimize_δ ℓrelearn (θu + δ | Drelearn)
    Where:
    θu: The unlearned model.
    δ: Weight perturbations induced by the attack.
    Drelearn: The relearn dataset (typically a subset of Df).

*   **Optimizer Downgrade via Gradient Compression:**
    θt+1 = θt − ηQ(mt;N); And if N = 1, then Q(mt; 1) = sign(mt)
    Where:
    mt: The descent direction used in the t-th update of a FO optimizer.
    η > 0: Learning rate.
    Q(·;N): Quantization operator using the gradient’s N-bit representation.
    sign(x): Element-wise sign of the vector x.

*   **ZO Gradient Estimation:**
    ∇̂f(x) = (1/q) Σ_{i=1}^q [ (f(x+ µui) − f(x− µui)) / (2µ) ] ui
    Where:
    ∇̂f(x): ZO approximation of the FO gradient ∇f(x).
    {ui}_{i=1}^q: Random direction vectors (e.g., sampled uniformly from the unit sphere).
    µ > 0: Perturbation size used for finite differences.
    q: Number of random direction vectors.

*   **Randomized Smoothing Connection:**
    fµ(x) := Eu[ f(x+ µu) ], with ∇fµ(x) = Eu[∇̂f(x)]
    Where:
    fµ(x): Smoothed version of the original objective function.
    Eu[·]: Expectation taken over the random direction vector u.
    ∇̂f(x): Serves as a stochastic gradient estimate of the smoothed objective.

**8. Algorithm Pseudocode**

```text
Algorithm: Hybrid FO-ZO Optimization for Robust LLM Unlearning

Input: Pre-unlearned model θ_0, Forget dataset Df, Retain dataset Dr, 
       Switch step N, Total alternating rounds K, 
       FO optimizer (Adam), ZO optimizer (AdaZO), Learning rates η_fo, η_zo.

Output: Robust unlearned model θ_final.

1:  Initialize model parameters θ = θ_0
2:  for k = 1 to K do
3:      // Follower Phase: First-Order Optimization
4:      for i = 1 to N do
5:          Compute FO gradient g_fo = ∇(ℓf(θ|Df) + λℓr(θ|Dr))
6:          Update θ using FO optimizer (Adam): θ = θ - η_fo * Update(g_fo)
7:      end for
8: 
9:      // Leader Phase: Zeroth-Order Optimization
10:     for j = 1 to N do
11:         Estimate ZO gradient g_zo = ∇̂(ℓf(θ|Df) + λℓr(θ|Dr)) 
                using finite differences and random directions
12:         Update θ using ZO optimizer (AdaZO): θ = θ - η_zo * Update(g_zo)
13:     end for
14: end for
15: Return θ_final = θ
```