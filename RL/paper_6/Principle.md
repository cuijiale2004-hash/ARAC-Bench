Principle 1: Causal Attribution of Performance Gains Under Hybrid Critique Reinforcement Learning to Genuine Policy Improvement Rather Than Distillation or Multi-Task Effects

**Definition:**  
In the subfield of large language model reasoning through hybrid reinforcement learning with auxiliary critique supervision, a primary evaluation priority is establishing whether reported performance improvements arise from the proposed mechanistic innovation—namely, learning to evaluate and reason about solution correctness—or from incidental confounds such as knowledge distillation from stronger teacher models, generic multi-task learning benefits, or simply increased training compute exposure. Because critique data often originates from external models or curated sources, reviewers must demand rigorous causal isolation through ablation studies that control for data volume, teacher model strength relative to the student, and the presence versus absence of imitation-style supervision. The principle emphasizes that genuine critique-based policy improvement should demonstrably function even when the model critiques its own on-policy solutions or when the external data generator is provably weaker than the base model, thereby ruling out distillation and establishing an independent learning signal. Furthermore, the evaluation should verify that the method is not merely a standard multi-task setup repackaged under a new name, but instead introduces a distinct reward-coupled reasoning dynamic that shapes the generation policy toward verification-oriented behavior. This requires careful examination of whether the auxiliary critique task interacts synergistically with the primary RL objective or merely adds parallel supervision.

**Core Evaluation Criteria:**
- **Distillation Control**: Has the work explicitly shown that the data-generating model used to produce critique candidates is not stronger than the base model being trained, or provided on-policy ablations where the model critiques itself?
- **Multi-Task Disambiguation**: Does the study isolate the critique mechanism from generic multi-task regularization by holding total training data volume constant and comparing against equivalent mixed-data baselines?
- **Causal Mechanism Evidence**: Are there experiments demonstrating that performance gains persist when the model evaluates weaker or self-generated solutions, confirming that improvement stems from evaluative reasoning rather than imitating superior content?
- **Compute Equivalency**: Is the total training compute and data volume matched between the proposed hybrid method and pure RL baselines, ensuring gains are not attributable to extended optimization?

---

Principle 2: Discrimination Between Training-Time Internalization of Critique Patterns and Test-Time Self-Critique Efficacy in Reflective Code Reasoning Models

**Definition:**  
A critical evaluative lens for critique-enhanced LLM training paradigms concerns the precise distinction between implicit training-time benefits and explicit test-time self-critique capabilities. Prior literature establishes that models generally cannot reliably correct their own final outputs without access to intermediate reasoning traces or external signals; therefore, reviewers must assess whether the paper clearly delineates that its gains originate from internalized critique-based judgment during the training process rather than from deployable post-hoc self-refinement. The work should be evaluated on whether it avoids misleading claims of "self-improvement" or "self-correction" when the actual mechanism is a policy shift toward more thorough reasoning before answer generation. An honest characterization of this limitation is essential, including explicit experimental documentation that feeding the model only its final solution at test time fails to yield performance improvements. This principle ensures that the community correctly understands critique-based RL as shaping generative behavior rather than enabling standalone verification modules.

**Core Evaluation Criteria:**
- **Conceptual Clarity**: Does the paper explicitly distinguish between internalized critique reasoning that improves generation quality and literal test-time self-critique that attempts to revise final outputs?
- **Empirical Honesty**: Are there direct experiments showing the failure or limitation of post-hoc self-critique when intermediate reasoning chains are withheld, and are these findings contextualized within established literature?
- **Claim Precision**: Does the work avoid conflating "enhanced reasoning" with "self-correction," using terminology that accurately reflects the observed training-time behavioral shift rather than overstating test-time capabilities?
- **Mechanistic Insight**: Does the analysis provide evidence—such as longer reasoning traces or increased reflective comments in generated code—that the model’s improvement is rooted in preemptive reasoning rather than externalized judgment?

---

Principle 3: Evaluation of Cross-Domain Transferability for Critique-Induced Reasoning Skills from Verifiable Code Domains to General Logical Reasoning Tasks

**Definition:**  
For methods that train critique capabilities within narrow, verifiable domains such as code generation—where correctness labels derive from executable test cases—it is essential to evaluate whether the acquired skills represent abstract, transferable reasoning competencies or merely domain-specific heuristics. Reviewers should expect systematic out-of-domain evaluation on tasks that lack automatic verification, such as logical reasoning, mathematical proof, or natural-language inference, to determine if the model has learned general evaluative and reflective patterns. The principle recognizes that a core promise of critique-based learning is the cultivation of broad critical thinking faculties that extend beyond the original training distribution. Consequently, evidence limited to in-domain coding benchmarks is insufficient; the work must demonstrate that critique supervision enhances performance on structurally dissimilar reasoning challenges where reward signals are unavailable or qualitatively different. Without such evidence, claims regarding the development of general critical thinking remain speculative and methodologically unsubstantiated.

**Core Evaluation Criteria:**
- **Out-of-Domain Benchmarking**: Are there experiments on non-code reasoning tasks (e.g., logical puzzles, mathematical reasoning, reading comprehension) that empirically verify cross-domain transfer?
- **Dissimilarity of Transfer Tasks**: Do the transfer tasks differ fundamentally from the training domain in terms of output format, verification mechanism, and knowledge requirements, ruling out superficial skill overlap?
- **Mechanistic Explanation of Transfer**: Does the paper offer a plausible account of why critique training in one domain generalizes, such as shared logical consistency checking or multi-step verification subroutines?
- **Comparative Transfer Analysis**: Does the hybrid RL+CRL model show larger transfer gains than a pure RL baseline trained for equivalent compute, confirming that critique supervision specifically drives generalization?

---

Principle 4: Assessment of Scalability Trends and Diminishing Returns Across Model Sizes in Joint Standard-and-Critique Reinforcement Learning

**Definition:**  
The practical viability of hybrid critique reinforcement learning depends critically on its behavior across different model scales, as auxiliary objectives may yield diminishing returns or even become negligible as base model capacity increases. Reviewers must scrutinize whether the method demonstrates consistent additive improvements over strong RL-only baselines when moving from smaller to larger parameter counts, or whether the gains attenuate due to the base model’s inherently stronger reasoning and self-correction capabilities. This evaluation should account for the quality of auxiliary data relative to the model scale; for instance, if critique candidates are generated by a fixed external model, larger base models may face a shrinking curriculum of learnable errors, artificially capping improvements. A thorough scalability analysis includes reporting absolute and relative gains across scales, investigating data-quality bottlenecks, and assessing whether the hybrid objective remains Pareto-optimal in the compute-quality tradeoff space at scale. Ultimately, the method’s contribution is weakened if its benefits are confined to small-scale regimes where base models are too weak to self-correct without explicit auxiliary signals.

**Core Evaluation Criteria:**
- **Multi-Scale Validation**: Are results reported for at least two distinct model sizes (e.g., 4B and 8B), with clear comparisons against same-scale RL-only baselines?
- **Diminishing Return Detection**: Does the work analyze whether absolute or relative performance improvements decrease as model size increases, and does it offer hypotheses or evidence for why?
- **Data-Quality Scaling Analysis**: Is the strength of the critique data generator evaluated relative to each model scale, ensuring that larger models are not handicapped by insufficiently challenging or informative critique material?
- **Compute-Gain Tradeoff**: Does the study justify the additional complexity of hybrid training by showing that the marginal gain over standard RL is favorable across scales, rather than being dominated by base model scaling alone?

---

Principle 5: Robustness and Stability of Multi-Objective Reward Aggregation Under Sparse Binary Verifiable Signals in Joint RL-and-CRL Optimization

**Definition:**  
When standard RL and auxiliary critique objectives are optimized jointly in a single policy, the design and aggregation of reward signals become a central object of review scrutiny, particularly in domains relying on sparse binary rewards such as test-case pass rates and critique correctness labels. The principle demands evaluation of whether the multi-objective setup introduces optimization instability, reward dominance by one signal, or gradient interference that could obscure the method’s true efficacy. Reviewers should expect detailed reporting on reward scaling coefficients, dynamic balancing strategies across training phases (e.g., length curriculum stages), and empirical metrics of training stability such as reward variance and convergence behavior. Moreover, the evaluation must confirm that the hybrid reward structure is robust to different mixing ratios and does not induce policy collapse or mode dropping that would not occur under pure RL optimization. A failure to demonstrate this robustness suggests that the hybrid objective is fragile and may not constitute a reliable training paradigm for large-scale deployment.

**Core Evaluation Criteria:**
- **Reward Balancing Transparency**: Does the paper disclose the specific mechanism for aggregating RL and CRL rewards (e.g., scaling coefficients, dynamic adjustment), and is this choice empirically justified?
- **Stability Metrics**: Are training dynamics reported, including mean and variance of rewards per objective, gradient norms, or policy entropy, to demonstrate stable co-optimization?
- **Ablation of Mixing Ratios**: Does the study systematically vary the proportion of critique versus standard RL data to identify the operating regime and detect performance degradation at extreme ratios?
- **Robustness to Reward Sparsity**: Given binary critique rewards, does the work analyze sensitivity to reward sparsity and verify that the joint objective does not suffer from variance issues or credit assignment problems?