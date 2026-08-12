**1. Research Background and Existing Pain Points**

Large language models (LLMs) have become a cornerstone of modern natural language processing, achieving remarkable progress across math, coding, and planning tasks. While autoregressive (AR) modeling has long dominated this field, recent advances in diffusion large language models (dLLMs) have demonstrated strong potential as an alternative formulation. By modeling language generation as an iterative denoising process, dLLMs bypass the left-to-right dependency of AR models and offer advantages in long context, multimodal, and fast inference. With the advent of powerful pretrained dLLMs, the next frontier lies in post-training to further enhance their capabilities. Among various post-training paradigms, reinforcement learning (RL) has emerged as a powerful approach that enables test-time scaling through verifiable rewards, yielding substantial gains on reasoning tasks in recent AR models. 

However, applying RL to dLLMs presents fundamental challenges. The core difficulty lies in likelihood approximation: while autoregressive models naturally provide token-level conditional probabilities essential for token-level RL objectives (e.g., GRPO), dLLMs generate sequences through iterative non-autoregressive denoising steps that lack this factorization. Mainstream RL algorithms in language modeling assume a left-to-right factorization of the sequence likelihood and rely on token-level importance ratios. In contrast, dLLMs generate sequences non-autoregressively, making such token-level conditionals either ill-defined or computationally expensive. 

Prior attempts to address this discrepancy have resorted to heuristic approximations—such as mean-field surrogates or token-level ELBO contributions—or computationally heavy trajectory-level formulations. Mean-field approximation ignores the context provided by other tokens in the sequence, making it fundamentally misaligned with the conditional denoising process. The token-decomposed ELBO term relies on bidirectional context but lacks formal probabilistic interpretation as a conditional likelihood, introducing unknown inconsistency. Trajectory-level formulations require T sequential network evaluations to compute the full trajectory probability, scaling compute cost linearly with the number of denoising steps (typically 50 to 1000), rendering it infeasible for large-scale post-training. None of these approaches fully resolves the mismatch between autoregressive RL objectives and the holistic generation process of dLLMs. The core issue is not merely about finding a better token-level proxy, but that the token-level decomposition itself fundamentally does not fit for dLLMs.

**2. Core Research Motivation and Scientific Questions**

The core research motivation stems from the fundamental conflict between the non-autoregressive nature of dLLMs and the autoregressive token-level design of standard RL algorithms. dLLMs should not be forced into an autoregressive token-level action space. Instead of adapting the dLLM model to fit the algorithm, we must adapt the algorithm to respect the holistic, sequence-level nature of the dLLM model. 

The scientific questions addressed are: How can we extend reinforcement learning to dLLMs in a principled manner that aligns with their non-autoregressive generation nature? How can we obtain a tractable sequence-level likelihood proxy for dLLMs that allows stable and efficient RL training without relying on heuristic token-level decompositions or prohibitive trajectory-level computations? How can we stabilize the sequence-level RL objective to prevent exploding or vanishing importance ratios and KL divergence gradients during large-scale training?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is to propose ELBO-based Sequence-level Policy Optimization (ESPO), a principled sequence-level reinforcement learning framework tailored for dLLMs. We treat the generation of an entire sequence as a single action, leveraging the ELBO as a tractable proxy for the intractable sequence log-likelihood. 

The design philosophy is that the ELBO is mathematically rigorous as a variational lower bound for the joint sequence log-likelihood. Substituting the ELBO into the sequence-level gradient formulation preserves the mathematical integrity of the objective, where the sequence-level ratio acts as a valid proxy for the global importance weight. Unlike token-level ELBO decompositions which are invalid proxies for AR conditionals, the sequence-level formulation correctly avoids splitting the ELBO. Furthermore, crucial stabilization techniques are incorporated, including per-token normalization of the ELBO ratio and robust KL-regularization using the k2 estimator, to ensure stable large-scale training and avoid the instability caused by exponential terms in sequence-level objectives.

**4. Core Innovation Points**

1. **Systematic Analysis of Incompatibility**: We provide a systematic analysis of why standard autoregressive RL objectives are incompatible with the non-autoregressive dLLMs, clarifying the limitations and inconsistencies of existing heuristic approaches such as mean-field surrogates and token-level ELBO decompositions.
2. **Principled Sequence-Level RL Framework ESPO**: We propose ESPO, a principled sequence-level RL framework that leverages the ELBO as a tractable proxy for sequence likelihood, treating the entire sequence generation as a single action to maintain the mathematical integrity of the objective.
3. **Stabilized Sequence-Level Importance Ratio**: We introduce per-token normalization of the ELBO difference to stabilize the sequence-level importance ratio. This transforms the unstable raw log-likelihood difference (which scales linearly with sequence length) into a stable, per-token scale to prevent astronomically large or infinitesimally small ratios.
4. **Robust KL-Divergence Estimation**: We incorporate robust KL-regularization using the k2 estimator instead of the unstable k3 estimator. The k2 estimator is a simple quadratic function of the ELBO difference, avoiding exponential instability and ensuring unbiased gradients for KL optimization.
5. **Comprehensive Empirical Validation**: We demonstrate through comprehensive experiments and ablation studies that ESPO yields stable optimization and consistent improvements across math, coding, and planning benchmarks, surpassing prior dLLM-RL methods, with dramatic improvements of 20-40 points on planning tasks.

**5. Overview of the Overall Technical Solution**

The overall technical solution of ESPO reformulates the RL objective to align with the nature of dLLM generation. It begins by treating the generation of the entire sequence as a single, atomic action, removing the token-level summation from the GRPO framework. Since the exact sequence log-likelihood is intractable in dLLMs, the ELBO is used as a tractable proxy to formulate the sequence-level importance ratio. To address the instability caused by the magnitude of the raw ELBO difference scaling linearly with sequence length, the log-ratio is normalized by the sequence length L, yielding a stabilized per-token scale importance ratio. For the KL-divergence term essential for regularizing policy updates, the k2 estimator is adopted instead of the k3 estimator to circumvent exponential instability and ensure correct, unbiased gradient estimation. Additionally, variance reduction techniques such as antithetic sampling through mask sharing and coupled sampling are incorporated to stabilize the Monte Carlo estimation of the ELBO. The complete objective optimizes this stabilized sequence-level ratio against group-relative advantages, penalized by the robust k2 KL-divergence estimate, enabling robust and efficient large-scale training.

**6. Detailed Module Design**

*6.1 Sequence-Level Policy Objective with ELBO Approximation Module*
Instead of viewing each token as an independent action, we treat the generation of the entire sequence y as a single, atomic action. This naturally leads to a sequence-level adaptation of the GRPO framework, where the token-level summation is removed. The objective now depends on a sequence-level importance ratio πθ(y(i)|x) / πθold(y(i)|x). By substituting the intractable log-likelihood log π(y(i)|x) with its ELBO approximation L(y(i)|x), we obtain the sequence-level ratio exp(Lθ(y(i)|x) − Lθold(y(i)|x)). Plugging this into the GRPO objective gives the vanilla sequence-level objective. While this sequence-level formulation correctly avoids splitting the ELBO at the token level, it is practically unusable because the magnitude of the raw ELBO difference typically scales linearly with the sequence length L, causing astronomically large or infinitesimally small ratios. To address this instability, the log-ratio is normalized by the sequence length L, transforming the unstable, raw log-likelihood difference into a stable, per-token scale. This final, stabilized importance ratio exp(1/L (Lθ(y(i)|x) − Lθold(y(i)|x))) enables effective training.

*6.2 Stable KL-Divergence Estimation Module*
The complete GRPO objective includes a KL-divergence term to regularize policy updates against a reference policy. A direct application of the k3 estimator to the sequence-level objective is highly problematic. The k3 estimator contains an exponential term exp(Lref(y(i)|x) − Lθ(y(i)|x)) − 1 − (Lref(y(i)|x) − Lθ(y(i)|x)), which reintroduces the unstable problem at the sequence level. Besides, while the k3 estimator is an unbiased estimate of KL-divergence, its gradient is an unbiased estimate of reverse-KL, leading to a different converged policy. To circumvent this exponential instability, we adopt the more robust k2 estimator. The k2 estimator is a simple quadratic function of the ELBO difference: 1/2 (Lθ(y(i)|x) − Lref(y(i)|x))². This polynomial form avoids the exponential term entirely, ensuring that the gradient signal from the KL regularizer remains stable and well-behaved, even for long sequences. Furthermore, the gradient of the k2 estimator is indeed unbiased for KL(πθ || πref), as opposed to the biased estimate in the k3 estimator.

*6.3 Variance Reduction Module*
To further stabilize training, two variance reduction techniques are incorporated for the Monte Carlo estimation of the ELBO:
- **Antithetic Sampling through Mask Sharing**: This is used when computing the difference between two ELBO terms. Concretely, we share the sampled timesteps and masked positions between the Monte Carlo estimators of the two policies, thereby reducing variance through negative correlation.
- **Coupled Sampling**: Based on the discrete ELBO formulation, we introduce a complementary masking strategy. For each sampled mask, we construct a complementary mask such that the two masks partition the token set: every token masked in the original mask is unmasked in the complementary mask, and vice versa. We then average the two losses. This construction guarantees that every token contributes at least once to the learning signal, and the estimator achieves lower variance.

**7. All Mathematical Formulas and Symbol Definitions**

*7.1 Diffusion Large Language Models*
- Forward process: qt(yt|y, x) = ∏Li=1 qt(yti|yi, x) and qt(yti|yi, x) = {1− t, yti = yi; t, yti = M}
- ELBO (Continuous): Lθ(y|x) ≜ Et∼U [0,1]Eyt∼qt(yt|y,x) [1/t ∑Li=1 1[yti = M] log pθ(yi|yt, x)] ≤ log πθ(y|x)
- ELBO (Discrete Variant): L′θ(y|x) ≜ El∼U({1,2,...,L})Eyl∼ql(yl|y,x) [L/l ∑Li=1 1[yli = M] log pθ(yi|yl, x)]

*7.2 Reinforcement Learning Baseline (GRPO)*
- GRPO Objective: JGRPO(πθ) = Ex∼D,y(1),...,y(G)∼πθold (·|x)[1/G ∑Gi=1 1/|y(i)| ∑|y(i)|k=1 min(ρk,(i)Â(i), clip(ρk,(i), 1− ϵ, 1 + ϵ)Â(i))− βDKL(πθ, πref)]
- Token-level importance ratio: ρk,(i) = πθ(yk,(i)|x,y<k,(i)) / πθold (yk,(i)|x,y<k,(i))
- Group-relative advantage: Â(i) = R(x, y(i)) − 1/G ∑Gj=1 R(x, y(j))

*7.3 Token-level ELBO Contribution*
- Lkθ(y|x) ≜ Et∼U [0,1]Eyt∼qt(yt|y,x) [1/t 1[ytk = M] log pθ(yk|yt, x)]

*7.4 Sequence-Level Policy Objective*
- Vanilla sequence-level ratio: ρ(i)seq = expLθ(y(i)|x) / expLθold(y(i)|x) = exp(Lθ(y(i)|x)− Lθold(y(i)|x))
- Vanilla sequence-level objective: Jseq(πθ) = Ex∼D,y(1),...,y(G)∼πθold (·|x) [1/G ∑Gi=1 min(ρ(i)seqÂ(i), clip(ρ(i)seq, 1− ϵ, 1 + ϵ)Â(i))]
- Stabilized sequence-level ratio: ρ(i)seq = exp(1/L (Lθ(y(i)|x)− Lθold(y(i)|x))) = exp(1/L ∑Lk=1 (Lkθ(y(i)|x)− Lkθold(y(i)|x)))

*7.5 KL-Divergence Estimators*
- k3 estimator: K̂Lk3 = exp(Lref(y(i)|x)− Lθ(y(i)|x)) − 1− (Lref(y(i)|x)− Lθ(y(i)|x))
- k2 estimator: K̂Lk2 = 1/2 (Lθ(y(i)|x)− Lref(y(i)|x))2

*7.6 Generic RL Gradient Derivations*
- Standard RL gradient: ∇θJ (πθ) = Ey∼πold(·|x) [πθ(y|x)/πold(y|x) ∇θ log πθ(y|x)R(x, y)] = Ey∼πold(·|x) [πθ(y|x)/πold(y|x) ∇θ log πθ(y|x)A(x, y)]
- Full sequence-level gradient (AR): Ey∼πold [∏di=1 πθ(yi|x, y<i)/∏di=1 πold(yi|x, y<i) ∑di=1 ∇θ log πθ(yi|x, y<i)A(x, y)]
- Token-level gradient (PPO/GRPO): ∇JPPO/GRPO = Ey∼πold [1/d ∑di=1 πθ(yi|x, y<i)/πold(yi|x, y<i) ∇θ log πθ(yi|x, y<i)A(x, y)]
- Sequence-level gradient (GSPO): ∇JGSPO = Ey∼πold [1/d (∏di=1 πθ(yi|x, y<i)/∏di=1 πold(yi|x, y<i))1/d ∑di=1 ∇θ log πθ(yi|x, y<i)A(x, y)]
- Trajectory-level gradient: ∇JTraj = Eπθ [∑Tt=1 ∇θ log πθ(yt−1|yt, x) R(x, y0)]

*7.7 Ablation Objective Variants*
- Token-level + Mean-field: J (x, y(i)|θ) = Â(i)/L ∑Lk=1 exp(log pθ(yk,(i)|x)− log pθold(yk,(i)|x))
- Sequence-level + Mean-field: J (x, y(i)|θ) = Â(i) · exp(1/L ∑Lk=1 [log pθ(yk,(i)|x)− log pθold(yk,(i)|x)])
- Token-level + ELBO: J (x, y(i)|θ) = Â(i)/L ∑Lk=1 exp(Lkθ(y(i)|x)− Lkθold(y(i)|x))
- Sequence-level + ELBO (Ours): J (x, y(i)|θ) = Â(i) · exp(1/L [Lθ(y(i)|x)− Lθold(y(i)|x)])

*7.8 Variance Reduction Formulas*
- Coupled Sampling loss: El∼U({0,1,2,...,L})Eyl∼q(yl|l,y,x) [ℓθ(yl, l, y|x) + ℓθ(ỹl, L− l, y|x) / 2]
- where ℓθ(yl, l, y|x) ≜ {L+ 1/l ∑Li=1 1[yli = M] log pθ(yi | yl, x), l > 0; 0, l = 0}

*7.9 FLOPs Calculation*
- Total FLOPs per sample with coupled sampling: Ftotal = 2ND(K + 6µM)

**8. Algorithm Pseudocode**

The paper does not provide explicit algorithm pseudocode in a box format. The algorithm steps are derived from the proposed ESPO method:
1. Given a prompt x, sample a group of G candidate completions {y(i)}Gi=1 from the old policy πθold.
2. Compute the reward R(x, y(i)) for each completion.
3. Compute the group-relative advantage Â(i) = R(x, y(i)) − 1/G ∑Gj=1 R(x, y(j)).
4. Estimate the ELBO Lθ(y(i)|x) and Lθold(y(i)|x) for each completion using Monte Carlo sampling with coupled sampling and antithetic mask sharing.
5. Compute the stabilized sequence-level importance ratio: ρ(i)seq = exp(1/L (Lθ(y(i)|x)− Lθold(y(i)|x))).
6. Estimate the KL-divergence using the k2 estimator: K̂Lk2 = 1/2 (Lθ(y(i)|x)− Lref(y(i)|x))2.
7. Update the policy πθ by maximizing the objective: J(πθ) = E[1/G ∑Gi=1 min(ρ(i)seqÂ(i), clip(ρ(i)seq, 1− ϵ, 1 + ϵ)Â(i)) − βK̂Lk2].