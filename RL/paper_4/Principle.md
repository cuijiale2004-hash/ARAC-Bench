**Principle 1: Formal Convergence Guarantees for Adjoint-Matching-Based Policy Extraction in Flow-Matching Offline RL**

**Definition:**  
A method that proposes a new policy extraction objective for flow or diffusion policies must provide rigorous theoretical evidence that the critical points of its objective correspond to the desired optimal behavior-regularized policy (e.g., $\pi^* \propto \pi_\beta \exp(\tau Q)$). Intuitive connections to stochastic optimal control or generative modeling are insufficient; reviewers expect explicit propositions or theorems proving optimality under stated assumptions. This is crucial because the subfield contains many heuristic guidance and approximation-based methods (e.g., classifier-guidance variants, DAC) that lack such guarantees and may introduce subtle biases at convergence.

**Core Evaluation Criteria:**
- **Optimality Proof**: Does the work contain a formal result proving that the global or critical optimum of the policy extraction objective recovers the optimal behavior-regularized policy?
- **Assumption Transparency**: Are the necessary assumptions (e.g., exact critic gradients, memoryless SDE, infinite flow-model capacity, convergence of optimization) clearly stated and justified within the RL context?
- **Distinction from Heuristics**: Does the work explicitly contrast its guarantees with approximation-based alternatives (e.g., QSM, CGQL, DAC) that cannot ensure convergence to $\pi^*$?
- **Self-Contained Derivation**: Is the theoretical motivation accessible to the RL audience without requiring the reader to infer key steps from the generative modeling literature alone?

---

**Principle 2: Statistical Rigor and Fair Hyperparameter Protocols in Flow-Policy Offline-to-Online Benchmarking**

**Definition:**  
Empirical claims in this subfield must be supported by statistically valid experimental designs that avoid inflated performance estimates from small sample sizes or unfair tuning budgets. Reviewers scrutinize whether confidence intervals are computed from an adequate number of random seeds, whether hyperparameters were tuned with comparable search cardinalities across methods, and whether evaluation uses held-out tasks and seeds separate from tuning. This is especially important because flow-policy methods are sensitive to temperature coefficients, pessimism coefficients, and ensemble sizes, making rigorous protocols essential for credible ranking.

**Core Evaluation Criteria:**
- **Sample Size and Variance Reporting**: Are results aggregated over at least 8–12 seeds and sufficiently many tasks to produce reliable estimates of variance, avoiding bootstrap confidence intervals from fewer than 8 seeds?
- **Fair Tuning Protocol**: Is hyperparameter tuning performed on a representative subset of tasks with fixed computational budgets, and are the selected hyperparameters evaluated on held-out tasks with separate, unseen seeds?
- **Comparability**: Are hyperparameter search ranges and cardinalities comparable across all baselines to prevent over-tuning the proposed method?
- **Transparency**: Are tuning ranges, domain-specific choices, and sensitivity to key hyperparameters (e.g., temperature $\tau$, pessimistic coefficient $\rho$) fully documented?

---

**Principle 3: Comprehensive Baseline Coverage Across Gaussian, Backprop, and Guided Flow/Diffusion Policy Classes**

**Definition:**  
Because the policy-extraction landscape spans highly disparate algorithmic families, a credible empirical evaluation must fairly represent the full spectrum of alternatives: simple Gaussian policies (e.g., ReBRAC, IQL), backpropagation-based multi-step flow methods (e.g., FBRAC), distilled one-step policies (e.g., FQL), classifier-guidance approaches (e.g., CGQL variants), post-processing methods (e.g., FEdit, IFQL), and recent diffusion-RL baselines (e.g., QSM, DAC). Omitting any major family can lead to incorrect conclusions about the necessity of expressive policies or the utility of critic gradients. Furthermore, any modification to existing baseline implementations must be transparently reported and justified.

**Core Evaluation Criteria:**
- **Policy-Class Diversity**: Does the evaluation include strong Gaussian/simple baselines to test whether flow-policy expressivity is actually required for the tasks?
- **State-of-the-Art Diffusion/Flow Baselines**: Are recent and strong diffusion- or flow-specific RL methods (not just the authors' own prior work) included and fairly tuned?
- **Internal Ablations**: Are ablations provided for every major algorithmic novelty (e.g., “lean” vs. “basic” adjoint matching, ensemble size, gradient clipping) to isolate the source of improvement?
- **Modification Disclosure**: If existing methods are adapted to the benchmark, are those changes described and validated so that readers can assess whether the baseline is advantaged or handicapped?

---

**Principle 4: Mechanistic Verification of Training Stability in Multi-Step Denoising Continuous-Action Policy Optimization**

**Definition:**  
A central claim in this subfield is that backpropagating policy gradients through multi-step denoising processes is numerically unstable. If a method purports to solve this instability via techniques such as adjoint matching, it must furnish direct empirical evidence of improved stability—not merely higher final returns. Reviewers expect training loss curves, gradient norm trajectories, cross-seed variance analyses, and ablations that strip away the stabilization mechanism (e.g., using standard backprop or “basic” adjoint matching). Final performance alone is ambiguous because a method might score well due to other design choices while still suffering from instability during training.

**Core Evaluation Criteria:**
- **Direct Stability Metrics**: Does the work report training curves, gradient norm distributions, or variance across seeds to demonstrate that optimization is stable throughout training?
- **Causal Ablations**: Is there a direct comparison between the proposed stable objective and an unstable counterpart (e.g., QAM vs. BAM, or vs. FBRAC) showing that the stabilization mechanism is responsible for the improvement?
- **Failure Case Analysis**: Does the paper analyze conditions under which instability occurs (e.g., ill-conditioned critic gradients, high action dimensionality) and show that the proposed method mitigates them?
- **Separation of Stability and Performance**: Does the analysis clearly distinguish between “optimization stability” (smooth convergence, low variance) and “task performance” (high return), establishing a causal link rather than correlation?

---

**Principle 5: Robustness to Data Corruption, Dataset Scale, and Action Dimensionality in Flow-Policy Offline RL**

**Definition:**  
Methods must be stress-tested against realistic imperfections in offline data and task characteristics. Reviewers expect experiments on corrupted or low-quality datasets (e.g., noisy actions, reduced dataset sizes), high-dimensional action spaces, and diverse task types (e.g., locomotion vs. manipulation, sparse vs. dense rewards). If evaluation is confined to a single benchmark family (e.g., OGBench), authors must maximize internal diversity and explicitly acknowledge limitations regarding generalization to other settings (e.g., D4RL MuJoCo). Robustness analysis is critical because flow policies are often justified by their expressivity, but this expressivity can become a liability when data is scarce or noisy.

**Core Evaluation Criteria:**
- **Data Quality Stress Tests**: Are experiments conducted on corrupted, noisy, or reduced-size datasets to verify that the method does not collapse when offline data deviates from the clean, abundant regime?
- **Scalability to High Dimensions**: Does the evaluation include the highest action-dimensional tasks available (e.g., 21-D humanoid control) with explicit commentary on numerical stability and compute cost?
- **Task Diversity**: Does the benchmark coverage include qualitatively different control problems (e.g., navigation, manipulation, action chunking) to ensure the method is not overfitted to one task structure?
- **Limitations Acknowledgment**: If the study is restricted to one benchmark suite, does it frankly discuss the lack of dense-reward or alternative-domain experiments as a limitation?