**Principle 1: First-Principles Derivation and Theoretical Justification of Policy Gradient Objectives for Non-Autoregressive Diffusion Language Models**

**Definition:**  
For reinforcement learning applied to non-autoregressive generative models such as diffusion large language models (dLLMs), the work must derive or rigorously justify its RL objective from the underlying probabilistic structure of the model class, rather than heuristically importing assumptions from autoregressive (AR) formulations. Because dLLMs lack the left-to-right causal factorization that makes token-level importance ratios mathematically well-defined, reviewers expect authors to explicitly address why standard AR policy gradients are incompatible, what the correct action space is (token-level, sequence-level, or trajectory-level), and why the chosen formulation is the only computationally tractable and theoretically valid option under the model's constraints. The work must situate its objective relative to exact trajectory-level formulations and token-level heuristics, ideally providing derivations or feasibility proofs that establish the principled nature of the approximation.

**Core Evaluation Criteria:**
- **Derivation from Model Structure**: Does the work derive the RL objective from the dLLM's actual generative process (e.g., iterative denoising, ELBO-based likelihood), or does it merely apply an AR algorithm with heuristic substitutions?
- **Invalidity of Token-Level Factorization**: Does the work rigorously explain why token-level conditional probabilities are ill-defined, computationally intractable, or probabilistically inconsistent for dLLMs, rather than assuming this is self-evident?
- **Tractability Arguments**: Does it prove or convincingly argue why the exact trajectory-level objective (backpropagating through all denoising steps) is computationally prohibitive, thereby motivating the need for a sequence-level surrogate?
- **Approximation Error Characterization**: Does the work quantify, bound, or at least qualitatively analyze the bias introduced by replacing the exact likelihood with a variational surrogate (e.g., ELBO) in the policy gradient?

---

**Principle 2: Rigorous Analysis of ELBO Tightness, Likelihood Surrogate Bias, and Variance in Sequence-Level Policy Optimization**

**Definition:**  
When exact sequence log-likelihoods are intractable—as is the case for dLLMs—and the evidence lower bound (ELBO) is used as a tractable proxy, reviewers demand rigorous analysis of how this surrogate affects policy optimization. The evaluation centers on whether the work adequately discusses ELBO tightness, whether the surrogate introduces systematic bias into the policy gradient, and whether performance gains stem from reduced bias, reduced variance, or both. A principled submission must not treat the ELBO as a black-box replacement for likelihood; it must analyze the validity of the proxy, its impact on importance ratio estimation, and how errors in the surrogate propagate into policy updates. This is especially critical because sequence-level ratios amplify any approximation errors across the entire token sequence.

**Core Evaluation Criteria:**
- **Surrogate Validity Discussion**: Does the work explicitly discuss the tightness of the ELBO (or alternative surrogate) as a proxy for the exact sequence likelihood, citing or providing evidence for its empirical or theoretical validity?
- **Bias-Variance Decomposition**: Does it provide quantitative or qualitative analysis comparing the bias-variance tradeoffs of the proposed sequence-level formulation against token-level alternatives, or does it rely solely on end-task performance to infer superiority?
- **Policy Impact Analysis**: Does the work analyze how errors or tightness gaps in the likelihood surrogate affect policy improvement, rather than assuming that a better ELBO automatically yields a better policy?
- **Isolation of Surrogate Effects**: Do ablations isolate the effect of the surrogate itself (e.g., sequence-level ELBO vs. mean-field approximation vs. token-level ELBO decomposition) from confounding factors such as variance reduction techniques or KL regularization?

---

**Principle 3: Mechanism Diagnosis and Robust Estimator Design for Training Stability in Sequence-Level RL with Importance Sampling**

**Definition:**  
Sequence-level policy optimization in dLLMs introduces severe numerical instability due to the exponential scaling of importance ratios over long sequences and the sensitivity of KL divergence estimation in high-dimensional spaces. Reviewers expect comprehensive diagnostic evidence—beyond final task accuracy—to demonstrate that the proposed method achieves stable training. This includes analyzing gradient norms, KL divergence trajectories, reward curves, and the mechanistic role of each stabilization component (e.g., per-token normalization of log-ratios, robust KL estimators). The work must justify why specific estimators (e.g., k₂ vs. k₁ vs. k₃) are chosen with both theoretical arguments and empirical diagnostics, and it must provide mechanistic explanations for anomalous training behaviors such as sudden performance jumps, collapse modes, or phase transitions.

**Core Evaluation Criteria:**
- **Multidimensional Stability Diagnostics**: Does the work provide training curves for rewards, gradient norms, and KL divergence to demonstrate stable optimization, rather than reporting only final aggregated metrics?
- **Estimator Justification**: Is the choice of robust estimators (e.g., k₂ KL estimator, length-normalized importance ratios) theoretically motivated and empirically validated against unstable alternatives (e.g., k₁, k₃) with head-to-head training dynamics?
- **Anomalous Behavior Explanation**: Does the work diagnose unexpected training phenomena (e.g., sudden performance spikes, collapse, stagnation) by relating them to underlying optimization signals such as gradient norm spikes or rapid KL divergence changes?
- **Causal Attribution of Stability**: Does the work establish that stability is caused by the proposed design choices (e.g., sequence-level action space + specific estimator) rather than by implicit regularization, hyperparameter tuning, or variance reduction tricks alone?

---

**Principle 4: Comprehensive Baseline Coverage and Fair Comparative Evaluation Across Diverse Task Domains and Model Architectures**

**Definition:**  
Empirical claims in this subfield must be supported by exhaustive and fair comparisons against all directly relevant prior methods, evaluated across multiple base model architectures and heterogeneous task domains. Reviewers scrutinize whether the experimental protocol covers the full landscape of existing approaches—including both token-level dLLM-RL baselines and closely related sequence-level or trajectory-level methods from parallel literature—and whether comparisons are conducted under fair conditions (e.g., identical hyperparameters such as PPO clipping thresholds, consistent fine-tuning regimes). The work must also demonstrate generalization across sequence lengths and task types (e.g., mathematical reasoning, coding, planning), and ablations must be sufficiently granular to isolate the contribution of each proposed component from confounding factors.

**Core Evaluation Criteria:**
- **Completeness of Baseline Comparisons**: Does the work compare against all directly relevant prior methods, including token-level dLLM-RL approaches and closely related works from the broader diffusion RL literature that may have been overlooked?
- **Cross-Model and Cross-Task Validation**: Are experiments conducted on multiple base models (e.g., LLaDA, Dream) and diverse task categories (reasoning, coding, planning) to demonstrate general applicability rather than isolated task-specific gains?
- **Fairness of Experimental Protocol**: Are hyperparameters (e.g., PPO ε, learning rate, KL coefficient) held constant or fairly tuned across all methods, and are evaluation conditions (generation length, decoding strategy) identical?
- **Granularity of Ablations**: Do ablation studies systematically isolate each proposed component (e.g., sequence-level objective, specific KL estimator, normalization strategy) to rule out alternative explanations for the observed improvements?

---

**Principle 5: Fundamental Necessity and Non-Incrementality of Sequence-Level Formulations Relative to Token-Level RL Frameworks**

**Definition:**  
A central concern raised by reviewers is whether the proposed method represents a fundamental methodological advance necessitated by the unique properties of dLLMs, or merely an incremental application of existing sequence-level RL frameworks (e.g., GSPO) to a new model class. The work must clearly articulate the structural properties of dLLMs—such as the intractability of token-level conditional likelihoods and the non-autoregressive nature of the denoising process—that make direct application of existing token-level or even sequence-level AR methods invalid or suboptimal. It must demonstrate that the proposed adaptations (e.g., ELBO-based sequence-level ratios, model-specific stabilization) are not cosmetic modifications but are fundamentally required to achieve stable and effective RL in this domain. The distinction between "beneficial for all LLMs" and "necessary for dLLMs" must be sharply drawn.

**Core Evaluation Criteria:**
- **Structural Necessity Argument**: Does the work identify specific structural properties of dLLMs (e.g., bidirectional context in denoising, lack of token-level factorization) that fundamentally invalidate token-level RL formulations, rather than arguing that sequence-level is merely "better"?
- **Distinction from Prior Frameworks**: Does it clearly differentiate its contribution from straightforward adaptations of existing sequence-level RL methods (e.g., GSPO) to diffusion models, highlighting algorithmic or theoretical innovations beyond domain transfer?
- **Invalidity of Simple Alternatives**: Does it empirically and theoretically demonstrate that token-level and trajectory-level alternatives are either fundamentally inconsistent or computationally infeasible, not just empirically weaker?
- **Evidence of Unique Necessity**: Does the evidence support the claim that the proposed method is uniquely critical for dLLMs (e.g., by showing token-level methods collapse while AR models tolerate them), rather than showing generic improvements that could apply to any model class?