**Principle 1: Decoupled Value Maximization and Gradient Stability in Expressive Denoising-Chain Policies**

**Definition:**  
A central challenge in this subfield is that expressive policy classes—such as diffusion or flow-matching policies—are parameterized by long denoising chains or other complex transformations that hinder stable gradient propagation from action outputs to policy parameters when directly optimizing a value function. High-quality research must therefore demonstrate a principled mechanism for stable value maximization that avoids backpropagating high-variance Q-gradients through the full expressive architecture. The reviewer consensus indicates that simply applying standard actor-critic updates to the expressive base policy is insufficient; instead, effective methods decouple the stable training of the expressive base (e.g., via imitation learning) from the reward-maximizing optimization performed by a lightweight, structurally simple component. This principle ensures that the community distinguishes between methods that genuinely solve the instability problem and those that merely apply existing RL objectives to ill-suited policy parameterizations.

**Core Evaluation Criteria:**
- **Decoupling of Objectives:** Does the work explicitly separate the stable training objective of the expressive base policy from the online RL objective, thereby isolating the source of gradient instability?
- **Avoidance of Direct Expressive Gradients:** Does the method refrain from direct backpropagation of value gradients through the full denoising chain or expressive transformation, or if it does, does it rigorously demonstrate that instability is mitigated?
- **Generality Across Parameterizations:** Is the proposed mechanism agnostic to the specific expressive policy class (e.g., diffusion, flow-matching, autoregressive), or is it tightly coupled to a single architecture, limiting its applicability?

---

**Principle 2: Statistical Rigor and Reproducible Reporting in Offline-to-Online RL Experiments**

**Definition:**  
In offline-to-online and sample-efficient RL research, empirical claims regarding performance improvements are only as credible as the statistical and reporting standards underlying them. The review process revealed serious concerns when variability is conveyed solely through min/max bands across a small number of seeds (e.g., three), as this fails to establish statistical confidence in claimed gains. Furthermore, visual artifacts—such as smoothed performance curves appearing outside unsmoothed shaded regions—erode trust if not explicitly explained. This principle demands that experimental reporting meets standards sufficient to distinguish true algorithmic advances from random seed variation or presentation bias, including clear documentation of evaluation protocols, hyperparameters, and data-processing choices.

**Core Evaluation Criteria:**
- **Statistical Confidence Reporting:** Are performance differences supported by confidence intervals, standard errors, or appropriate statistical tests, rather than by min/max ranges that do not measure statistical significance?
- **Seed Robustness and Scale:** Is the evaluation conducted with a sufficient number of independent random seeds (commonly five or more) to justify claims of consistent and reliable improvement?
- **Transparency of Post-Processing:** Are smoothing procedures, outlier handling, evaluation frequencies, and exact dataset protocols explicitly documented so that the results are reproducible?

---

**Principle 3: Quantification of Wall-Time and Scaling Costs in Sampling-Based Action Maximization**

**Definition:**  
Methods that employ sampling-based extraction—such as best-of-N selection or on-the-fly (OTF) policy construction—inherently trade increased per-step computation for improved sample efficiency. The review highlighted that sample-efficiency claims can be misleading if decoupled from wall-time or computational cost: an algorithm requiring an order of magnitude more forward passes per gradient step may not be practically efficient, even if it reduces environment interactions. Consequently, rigorous evaluation in this subfield requires explicit profiling of compute overhead (e.g., per-step runtime, GPU hours, FLOPs) and an analysis of how performance scales with the sampling budget. Without this quantification, it remains unclear whether the method offers a genuine practical advantage or merely shifts the cost from data collection to inference.

**Core Evaluation Criteria:**
- **Wall-Time Profiling:** Are per-step runtimes, total GPU hours, or comparable hardware-aware metrics reported alongside sample-efficiency curves for the proposed method and key baselines?
- **Scaling Characterization:** Is the impact of the sampling budget (e.g., the number of candidate actions N) on both task performance and computational cost systematically analyzed?
- **Efficiency Mitigation Strategies:** Does the work propose or discuss concrete approximations—such as parallelization, distilled samplers, or reduced network sizes—to amortize inference-time overhead?

---

**Principle 4: Theoretical and Diagnostic Justification for Stability in Hierarchical Policy Updates**

**Definition:**  
When a method’s central contribution is “stable fine-tuning,” reviewers expect more than intuitive arguments or asymptotic performance curves. The critical feedback emphasized that stability claims must be substantiated through either formal analysis (e.g., convergence bounds, bounded-update theorems) or detailed empirical diagnostics (e.g., gradient norm trajectories, TD-error variance, critic loss oscillations). Moreover, hierarchical architectures—such as a frozen base policy paired with a learned edit policy—introduce coupled dynamics that are not theoretically trivial. This principle requires that authors rigorously validate their stability narrative against direct end-to-end optimization baselines and provide evidence that local, constrained edits are sufficient to reach high-value action modes without destabilizing training.

**Core Evaluation Criteria:**
- **Empirical Stability Diagnostics:** Are gradient norms, TD-error variance, or critic loss oscillations measured and compared against direct value-maximization baselines to validate “stable training” claims?
- **Formal Properties:** Are convergence guarantees, bounded-update assumptions, or error-propagation analyses provided for the coupled base-edit policy updates?
- **Local Edit Sufficiency:** Is the assumption that local edits (e.g., within a constrained radius) can capture high-value modes empirically or theoretically justified, rather than assumed?

---

**Principle 5: Comprehensiveness and Fairness of Baseline Comparisons in Expressive Policy Fine-Tuning**

**Definition:**  
Because this subfield sits at the intersection of expressive generative modeling and RL, baseline evaluation must encompass both state-of-the-art RL methods designed for expressive policies (e.g., diffusion-specific or flow-specific algorithms) and strong generic baselines (e.g., SAC, RLPD). Fairness further requires controlling for pretraining data, offline dataset retention, network capacity, and initialization quality. The reviews noted that omitting recent expressive-policy baselines (such as DIPO or DACER) weakens the empirical argument, as does failing to account for the fact that some baseline RL algorithms actively degrade a strong pretrained policy. This principle ensures that claimed improvements are attributable to the proposed algorithmic mechanism rather than to initialization or data asymmetries.

**Core Evaluation Criteria:**
- **Domain-Specific Baseline Coverage:** Are strong methods specifically designed for expressive policy classes (e.g., diffusion-based RL, flow-matching actor-critic methods) included in comparisons?
- **Controlled Experimental Conditions:** Do comparisons hold constant the pretraining data, offline dataset usage, policy architecture, and network capacity to isolate the contribution of the fine-tuning algorithm?
- **Offline-to-Online Degradation Analysis:** Does the evaluation explicitly report whether baseline methods cause performance collapse when fine-tuning from a pretrained policy, and does the proposed method demonstrably avoid such degradation?