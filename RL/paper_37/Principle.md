**Principle 1: Distinction Between Direct Adaptation and Domain-Specific Methodological Innovation in LLM-Style Post-Training for Traffic Simulation**

**Definition:**  
In evaluating research that migrates LLM post-training paradigms (e.g., R1-style SFT-RFT-SFT) to autonomous driving simulation, reviewers must assess whether the work represents a direct, off-the-shelf application or introduces substantive domain-specific methodological advances. The traffic simulation domain possesses unique characteristics—closed-loop multi-agent interaction, non-differentiable safety-critical metrics, and motion token distributions—that demand more than naive transfer. A high-quality contribution must clearly articulate novel algorithmic components tailored to these domain constraints and rigorously differentiate itself from concurrent and prior RL fine-tuning work in autonomous driving. Reviewers should examine whether the purported novelty lies in the cross-domain transfer insight itself or in domain-specific technical innovations, as application-area venues increasingly expect the latter to be explicitly disentangled from the former. Furthermore, the evaluation must verify that the authors engage deeply with domain-specific baselines rather than treating the new domain as a blank slate for existing LLM recipes.

**Core Evaluation Criteria:**
- **Explicit Differentiation from Prior Domain Work**: Does the paper clearly contrast with existing RL fine-tuning methods in autonomous driving and concurrent LLM post-training analyses, establishing unique technical contributions rather than application-only value?
- **Domain-Specific Algorithmic Design**: Are the proposed algorithmic modifications (e.g., advantage estimation, KL estimators, rollout strategies) fundamentally motivated by traffic simulation properties, or are they generic RL plug-ins?
- **Value of Cross-Domain Transfer Claim**: Is the claim that LLM-style post-training transfers to traffic simulation supported by mechanistic insight and ablations, or is it asserted post-hoc based on end results?
- **Positioning Clarity**: Does the manuscript explicitly discuss its relationship to the closest prior engineering baselines and explain why the new pipeline is non-trivial?

---

**Principle 2: Integrity of Metric-as-Reward Design and Cross-Metric Behavioral Generalization in Closed-Loop Simulation**

**Definition:**  
When evaluation metrics (e.g., WOSAC Realism Meta) are directly used as reward signals for reinforcement fine-tuning, there is an inherent risk of benchmark overfitting, where the policy optimizes superficial statistical correlations rather than genuine behavioral realism. Reviewers must evaluate whether the reward design is theoretically justified as a comprehensive proxy for realism, whether improvements on the target metric transfer to auxiliary or held-out behavioral measures, and whether the work explicitly guards against reward hacking or metric exploitation in closed-loop multi-agent settings. Because traffic simulation involves diverse, safety-critical behaviors, a reward function that collapses multi-dimensional realism into a single scalar must be scrutinized for unintended optimization shortcuts. High-quality research should demonstrate that optimizing the metric yields qualitatively better driving behaviors across diverse scenarios, not merely higher leaderboard scores. Additionally, reviewers must assess whether computationally efficient reward approximations maintain fidelity to the intended optimization objective.

**Core Evaluation Criteria:**
- **Theoretical Justification of Reward Proxy**: Does the paper rigorously argue why the chosen composite metric adequately captures human driving preferences, rather than treating it as an arbitrary optimization target?
- **Cross-Metric Consistency**: Do improvements on the primary realism meta score correlate with gains on disaggregated sub-metrics (kinematic, interactive, map adherence) and qualitative behavioral plausibility, or do they trade off against other realism dimensions?
- **Overfitting Controls**: Are there experiments or arguments demonstrating that the policy does not exploit loopholes in the metric computation (e.g., unrealistic trajectory modes that statistically satisfy the metric)?
- **Sample Efficiency of Reward Computation**: If the metric is expensive to compute (e.g., requiring many rollouts), does the work propose and validate computationally efficient approximations without sacrificing optimization fidelity?

---

**Principle 3: Stability Mechanisms and Catastrophic Forgetting Mitigation in Iterative SFT-RFT Cycles for Traffic Simulation**

**Definition:**  
The iterative interleaving of supervised fine-tuning and reinforcement fine-tuning (SFT-RFT-SFT) in non-language domains introduces distinct stability challenges, including catastrophic forgetting of behavioral priors, policy collapse due to sparse non-differentiable rewards, and sensitivity to KL regularization strength. Reviewers must evaluate whether the multi-stage pipeline is explicitly designed to preserve the pre-trained distribution while enabling metric-driven policy updates, and whether ablations substantiate the claim that intermediate SFT re-anchors mitigate forgetting more effectively than single-stage alternatives. Unlike language domains where post-training recipes are well-calibrated, traffic simulation involves continuous physical state spaces where distributional drift can manifest as unsafe or unstable driving behaviors. Therefore, the evaluation must extend beyond final performance to examine training curves, gradient norms, and the mechanistic role of each stage. A rigorous submission should provide empirical evidence that the final SFT stage restores behavioral diversity and distributional alignment compromised during RFT, rather than asserting this as a theoretical given. The choice and estimation of KL penalties require particular scrutiny because they serve as the primary mechanism preventing policy collapse in the absence of explicit value networks.

**Core Evaluation Criteria:**
- **Forgetting Quantification**: Does the paper measure or ablate the degradation of pre-trained/SFT-acquired capabilities after RFT, and demonstrate that the final SFT stage genuinely recovers distributional fidelity?
- **KL Regularization Rigor**: Is the choice of KL estimator (e.g., unbiased minimum-variance vs. naive ratio-based) empirically justified with ablations, and is the KL coefficient sensitivity analyzed across a meaningful range?
- **Stage-wise Necessity**: Are ablations provided for each stage of the pipeline (e.g., SFT-only, RFT-only, SFT-RFT without final SFT, consecutive SFT phases) to isolate the contribution of the iterative design?
- **Mechanistic Explanation**: Does the work provide a causal or mechanistic account of why interleaving SFT and RFT improves final performance, beyond simply showing higher leaderboard scores?

---

**Principle 4: Comprehensive Reproducibility and Computational Resource Transparency in Large-Scale Multi-Agent Simulation Training**

**Definition:**  
Given the substantial computational demands of closed-loop multi-agent simulation and multi-stage post-training on large-scale driving datasets, reproducibility hinges on detailed disclosure of training configurations, hardware resources, wall-clock times, inference latency, and code availability. Reviewers must assess whether the experimental setup is sufficiently documented to enable independent replication, whether hardware and time costs are reported to contextualize accessibility, and whether constraints on code release are transparently communicated with concrete commitments. The complexity of combining CAT-K rollouts, RL fine-tuning, and iterative stages creates numerous implementation details that, if omitted, prevent community verification. Transparency regarding computational budget also allows the community to judge whether performance gains are practically realizable outside well-resourced industrial labs. Finally, clear documentation of inference latency per simulation step is essential for assessing whether the method meets real-time requirements of downstream autonomous driving validation pipelines.

**Core Evaluation Criteria:**
- **Resource Disclosure Completeness**: Are training wall-clock times, GPU counts/types, memory requirements, and per-step inference latencies explicitly reported?
- **Implementation Clarity**: Are pipeline details (e.g., CAT-K rollout definitions, reward assignment across agents, exact stage transitions) described with sufficient precision to replicate without guesswork?
- **Code and Artifact Availability**: Does the submission include anonymous code or detailed supplementary material, and if industry constraints prevent release, is there a clear path and commitment to future open-sourcing?
- **Cost-Performance Trade-off Transparency**: Is the computational overhead of each stage (especially expensive metric computation and RFT rollouts) justified relative to performance gains?

---

**Principle 5: Scope Awareness and Generalization Assessment Beyond NTP Paradigms and Single-Leaderboard Benchmarks**

**Definition:**  
Research that achieves state-of-the-art results on a single benchmark (e.g., WOSAC) using a specific model paradigm (e.g., next-token prediction) must be evaluated on whether its claims are scoped appropriately and whether evidence supports broader applicability. Reviewers should examine whether the method is fundamentally coupled to the NTP architecture, whether it generalizes to other simulator types (e.g., diffusion-based), and whether distinctions between simulation and downstream tasks (e.g., end-to-end planning) are clearly articulated rather than conflated. Because the autonomous driving ecosystem encompasses diverse generative paradigms and evaluation benchmarks, claims of universal realism improvement must be carefully separated from leaderboard-specific optimization. A well-scoped submission explicitly analyzes architectural dependencies—such as whether the RFT reward assignment requires autoregressive tokenization—and discusses the theoretical barriers to transferring the method to continuous trajectory spaces. Moreover, the paper should avoid implying planning capabilities from simulation results without establishing the requisite problem formulation alignments, such as ego-centric goal conditioning and route constraints. Honest delineation of scope strengthens the contribution by showing the authors understand the boundaries of their paradigm rather than overstating generalization.

**Core Evaluation Criteria:**
- **Paradigm Coupling Analysis**: Does the paper acknowledge and analyze whether its post-training innovations are inherently tied to autoregressive tokenization, and discuss what adaptations would be required for continuous or diffusion-based simulators?
- **Cross-Benchmark Scope**: Are claims about realism validated beyond a single benchmark's metric suite, or does the work conflate leaderboard optimization with universal simulation quality?
- **Task Boundary Clarity**: Does the paper clearly delineate the differences between multi-agent traffic simulation (joint distribution modeling) and ego-centric motion planning, avoiding unsupported claims about planning applicability?
- **Future Work and Limitations Honesty**: Does the discussion forthrightly address scope limitations (e.g., no planning benchmarks, no diffusion experiments) rather than deferring them as mere future directions without analytical justification?