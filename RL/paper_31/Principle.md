**Principle 1: Cross-Domain Empirical Validation of Heavy-Tailed Q-Bias Phenomenon in Offline-to-Online Transition**

**Definition:**  
In offline-to-online reinforcement learning, claims regarding the distributional nature of Q-value estimation bias must be substantiated across a heterogeneous set of control tasks and RL paradigms to rule out domain-specific artifacts. The evaluation should examine whether the heavy-tailed phenomenon is persistent and structurally distinct from transient biases observed in pure online training or conservative underestimation in offline training. Reviewers must assess whether the authors establish quantitative correlations between distributional shift severity and tail heaviness, and whether appropriate statistical metrics such as kurtosis, log-log tail indices, and distributional goodness-of-fit are employed. Merely observing outliers in a single domain is insufficient; the phenomenon must be characterized as an endemic property of the O2O transition arising from non-uniform distribution shift and max-operator amplification. Furthermore, the analysis should distinguish between environments with dense, sparse, and semi-sparse reward structures to confirm generalizability.

**Core Evaluation Criteria:**
- **Domain Coverage**: Does the study validate the heavy-tailed Q-bias across diverse environment types (e.g., locomotion, navigation, manipulation) and reward densities (dense, sparse, semi-sparse)?
- **Paradigm Specificity**: Does it clearly differentiate O2O-specific persistent heavy-tailedness from transient heavy tails in pure online RL or conservative-induced tails in offline RL?
- **Mechanistic Correlation**: Is there quantitative evidence linking the heavy-tailed bias magnitude to offline-to-online distribution shift distance metrics?
- **Statistical Rigor**: Are appropriate heavy-tailedness diagnostics (kurtosis, tail index, distributional fits) reported with sufficient statistical detail?

---

**Principle 2: Theoretical Justification and Comparative Analysis of Distributional Robust Loss Formulations**

**Definition:**  
Proposals to replace standard Bellman losses with distributionally robust alternatives require rigorous theoretical and empirical justification that extends beyond intuitive motivation. The research must formally derive the proposed loss from a probabilistic framework, such as KL divergence minimization between parametric likelihoods, and prove advantageous properties including reduced single-step estimation bias and variance relative to L2 under stated assumptions. Critically, the work must situate itself against structurally similar robust baselines—specifically L1 and Huber losses—to demonstrate that gains arise from preserving distributional information in both target and current Q-value estimates rather than from generic outlier suppression. Theoretical claims should be accompanied by empirical validation showing improved gradient boundedness and variance reduction. Finally, computational tractability and optimization stability of theoretically richer variants, such as asymmetric distributions, must be addressed to justify pragmatic modeling simplifications.

**Core Evaluation Criteria:**
- **Theoretical Distinctiveness**: Are there formal proofs showing the proposed loss reduces single-step estimation bias or variance compared to standard L2 under explicitly stated assumptions?
- **Alternative Comparison**: Does the work empirically compare against L1, Huber, and Cauchy losses under identical conditions to isolate the advantage of the specific distributional formulation?
- **Modeling Trade-offs**: Is the choice of symmetric versus asymmetric distribution justified by both empirical performance and computational stability analysis?
- **Bayesian Consistency**: Is the divergence or likelihood derivation clearly motivated in terms of preserving bias distributional information for both estimated Q-values and Bellman targets?

---

**Principle 3: Component Disentanglement and Isolated Contribution Analysis in Ensemble-Robust Hybrid Methods**

**Definition:**  
For hybrid methods combining parametric noise models with ensemble-based corrections, reviewers must demand granular ablation studies that isolate the mechanistic contribution of each submodule to both final performance and training dynamics. It is insufficient to report only end-task returns; the evaluation must include stability metrics such as Normalized Cumulative Performance Drop, return variance, and qualitative training curves to verify that each component addresses distinct pathologies. Specifically, the ensemble’s role in re-centering bias toward zero should be disentangled from the noise model’s role in reducing heavy-tailed variance and kurtosis, ensuring that neither component is masking the failure of the other. Additionally, ablations should control for confounding implementation details like high update-to-data ratios or large ensemble sizes to prevent attributing baseline infrastructure gains to the proposed technique. The analysis must ultimately establish whether the components operate synergistically or whether one dominates the overall improvement.

**Core Evaluation Criteria:**
- **Isolated Ablation Performance**: Are results reported for the full method minus each key component (e.g., without noise model, without ensemble)?
- **Stability Metrics**: Do ablation studies include training stability indicators (e.g., NCD, performance variance, training curves) rather than only final returns?
- **Distributional Impact**: Does the ablation analysis show how each component affects the empirical distribution of Q-bias (e.g., ensemble shifts mean, noise model reduces kurtosis/variance)?
- **Synergy Analysis**: Is there evidence that components interact synergistically rather than one dominating, and are confounders like high UTD ratios controlled?

---

**Principle 4: Plug-In Compatibility and Cross-Algorithm Generalization of Q-Value Robustification Techniques**

**Definition:**  
A robust Q-value estimation module intended for broad adoption in O2O RL must demonstrate plug-in compatibility across diverse offline pretraining backbones and online fine-tuning algorithms. The evaluation framework should require evidence that replacing the standard Q-loss with the proposed robust variant yields consistent improvements when integrated with substantially different base methods, such as conservative Q-learning, ensemble SAC, or TD3-based pipelines. Comparisons must be fair, matching ensemble sizes, update frequencies, and network capacities to ensure observed gains are attributable to the loss formulation rather than auxiliary computational budget increases. The method’s effectiveness should also be validated across different task modalities, including dense-reward locomotion and sparse-reward navigation, to confirm that robustification is not narrowly tailored to a single problem class. Ultimately, the principle demands that the technique function as a modular, minimally intrusive enhancement rather than a monolithic algorithmic overhaul.

**Core Evaluation Criteria:**
- **Cross-Backbone Integration**: Is the method evaluated when combined with multiple distinct base O2O algorithms showing consistent improvements?
- **Controlled Fairness**: Are comparisons performed under matched computational budgets (UTD, network size) to ensure gains are not artifacts of higher update frequencies?
- **Offline-to-Online Portability**: Does the method maintain effectiveness when offline pretraining backbones differ (e.g., conservative versus non-conservative)?
- **Minimal Intrusion**: Is the modification demonstrated to be lightweight and modular, requiring minimal hyperparameter retuning across base methods?

---

**Principle 5: Pragmatic Trade-off Assessment Between Asymmetric Modeling Fidelity and Optimization Stability**

**Definition:**  
When empirical Q-bias distributions exhibit complex characteristics such as skewness and asymmetry that conflict with idealized parametric assumptions, the research must explicitly confront and justify the resulting modeling trade-offs. Reviewers should evaluate whether the authors acknowledge the discrepancy between symmetric distributional assumptions and observed right-skewed heavy tails, and whether they provide empirical evidence that more flexible models introduce unacceptable optimization instability or approximation error. This includes comparing the proposed tractable formulation against theoretically richer but computationally burdensome alternatives, such as asymmetric distributions requiring Monte Carlo divergence approximations. The work must transparently communicate whether modeling choices represent deliberate empirical compromises between fidelity and trainability rather than claims of perfect distributional alignment. Such intellectual honesty is essential for distinguishing rigorous robust statistics from ad hoc heuristic design.

**Core Evaluation Criteria:**
- **Acknowledgment of Asymmetry**: Does the work explicitly test and report the impact of asymmetric modeling if the empirical data is skewed?
- **Stability-Cost Analysis**: Is there empirical evidence showing that more flexible models introduce optimization instability or approximation errors outweighing theoretical benefits?
- **Theoretical Tractability**: Are the computational and derivational barriers to more complex models (e.g., lack of closed-form divergence) explicitly analyzed?
- **Transparent Communication**: Is the final modeling choice framed as a deliberate empirical trade-off rather than presented as an unexamined or ideal assumption?