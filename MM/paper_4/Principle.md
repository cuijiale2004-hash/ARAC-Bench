Principle 1: Closed-Loop Interaction Design and Adaptive Termination in MLLM-Based Interactive Segmentation Agents

**Definition:**  
This principle evaluates whether the proposed agent establishes a genuine closed-loop interaction paradigm between the multimodal large language model and the visual foundation model, moving beyond fixed-step or open-loop pipelines to enable dynamic, state-dependent decision-making. It focuses on the completeness of the workflow, encompassing initial grounding, iterative mask refinement via positive and negative points, and crucially, adaptive termination conditioned on the current mask quality rather than predetermined iteration counts. The evaluation emphasizes whether the action space is sufficiently expressive to emulate human-like SAM usage, including the ability to judge when further refinement is unnecessary. Reviewers must assess whether the termination policy is learned end-to-end through reinforcement learning or relies on hand-crafted heuristics, and whether qualitative evidence supports the claim of human-like behavior. Furthermore, the principle demands clear differentiation from prior approaches that perform iterative refinement without integrated bounding box generation or without adaptive stopping mechanisms. Ultimately, the work must demonstrate that the closed-loop formulation yields tangible benefits in segmentation accuracy and interaction efficiency compared to decoupled or fixed-iteration alternatives.

**Core Evaluation Criteria:**
- **Closed-Loop Completeness**: Does the method integrate bounding box generation, iterative point-based refinement, and adaptive termination within a unified MDP framework?
- **Adaptive Termination Evidence**: Is there empirical proof (e.g., termination ratio curves across steps) that the model learns to stop conditionally based on mask quality?
- **Human-Like Behavioral Support**: Are qualitative visualizations or user studies provided to substantiate claims of human-like interaction, beyond mere performance metrics?
- **Distinction from Fixed-Step Baselines**: Is there direct comparison with methods employing fixed refinement steps to demonstrate the value of adaptive decision-making?

---

Principle 2: Stabilization Mechanisms and Reward Engineering in Multi-Task RL Training for Visual Grounding Agents

**Definition:**  
This principle assesses the scientific rigor and stability of the multi-task reinforcement learning framework employed to train the agent across heterogeneous objectives including textual grounding, mask comprehension, and auxiliary tasks. It recognizes that jointly optimizing multiple vision-language tasks via RL is inherently susceptible to gradient instability, mode collapse, reward hacking, and inter-task interference, necessitating carefully designed stabilization strategies. The evaluation focuses on whether the reward functions provide dense, non-misleading signals that enable reliable credit assignment for both continuous coordinates and discrete decisions. Reviewers should examine whether techniques such as dynamic sampling, global batch normalization, task-specific data balancing, and advantage estimation are principled and experimentally justified. Crucially, the work must provide transparent evidence of stable convergence through training curves and ablation studies that isolate the contribution of each stabilization component. Without such evidence, claims of effective multi-task RL training in the visual domain remain unverified.

**Core Evaluation Criteria:**
- **Reward Design Rigor**: Are reward components (IoU-based, decision-based, improvement-based) carefully shaped to avoid sparsity and provide clear learning signals for each sub-task?
- **Stabilization Strategies**: Does the work explicitly justify and ablate mechanisms like dynamic sampling, NMS-based diversity enforcement, global normalization, and cross-task data balancing?
- **Convergence Documentation**: Are training dynamics (reward progression, entropy loss, gradient norms) reported to demonstrate the absence of collapse, oscillation, or divergence?
- **Hyperparameter Robustness**: Is there systematic sensitivity analysis of key thresholds (e.g., IoU cutoffs, rollout counts, distance tolerances) with performance trade-offs quantified?

---

Principle 3: Generalization Preservation and Catastrophic Forgetting Mitigation in RL Fine-Tuning of Multimodal Foundation Models

**Definition:**  
This principle examines whether reinforcement learning fine-tuning preserves the broad multimodal reasoning capabilities of the base MLLM, contrasting this outcome with the catastrophic forgetting typically observed in supervised fine-tuning approaches for segmentation tasks. It is grounded in the understanding that interactive segmentation agents depend on general vision-language knowledge for query parsing and reasoning, making the retention of pre-trained competencies as important as task-specific gains. The evaluation requires rigorous empirical verification through standardized general-purpose MLLM benchmarks administered both before and after task-specific optimization. Reviewers must scrutinize whether the paper provides direct comparative evidence against SFT baselines on identical benchmarks to substantiate claims of superior generalization preservation. Additionally, the principle extends to out-of-domain segmentation performance across diverse datasets with varying query granularities and target complexities. Honest reporting of any residual capability degradation or trade-offs is essential for a credible assessment of the RL paradigm's practical impact.

**Core Evaluation Criteria:**
- **General Benchmark Retention**: Does the paper evaluate post-training performance on diverse general MLLM benchmarks (e.g., SEEDBench, TextVQA, MMStar, MME) to verify preservation of base capabilities?
- **SFT-Forgetting Comparison**: Is there head-to-head comparison with supervised fine-tuning methods on general benchmarks to empirically validate that RL mitigates catastrophic forgetting?
- **Cross-Dataset Segmentation Generalization**: Are multiple out-of-domain reasoning segmentation datasets (e.g., ReasonSeg, MMR, MUSE) used to test robustness beyond the training distribution?
- **Limitation Disclosure**: Does the work transparently acknowledge any remaining forgetting or capability erosion rather than selectively reporting only positive results?

---

Principle 4: Efficiency-Quality Trade-Off Analysis and Scalability Verification from 7B to 32B in Iterative Segmentation Systems

**Definition:**  
This principle evaluates the practical deployability and computational feasibility of iterative segmentation agents by demanding rigorous analysis of inference latency, training resource consumption, and scalability across model sizes. Given that iterative refinement introduces inherent computational overhead compared to single-shot methods, the work must demonstrate that its quality gains are not offset by prohibitive deployment costs or fixed multi-step inefficiencies. The evaluation encompasses absolute measurements such as average inference steps per sample, per-sample time costs under different backends, and total GPU-hour requirements for training. Furthermore, scalability analysis should confirm that the framework effectively utilizes increased model capacity, for instance by validating consistent improvements when scaling from 7 billion to 32 billion parameters. Reviewers should verify that efficiency claims are benchmarked against relevant baselines including both single-step methods and fixed-iteration competitors under comparable hardware conditions. Comprehensive cost transparency enables the community to assess whether the method represents a practical advance or merely an expensive proof of concept.

**Core Evaluation Criteria:**
- **Inference Cost Transparency**: Are average MLLM inference steps, per-sample latency, and backend acceleration (e.g., vLLM) explicitly reported and compared with single-step and fixed-step baselines?
- **Training Resource Disclosure**: Are absolute training costs (GPU count, memory per device, total hours) clearly stated to support reproducibility and resource planning?
- **Quality-Efficiency Trade-off**: Does the method achieve superior segmentation quality relative to single-step approaches while maintaining substantially lower latency than fixed multi-step pipelines?
- **Model Scalability Evidence**: Is there validation across different model scales (e.g., 7B to 32B) showing that the RL framework benefits from increased capacity without destabilizing?

---

Principle 5: Human-Preference Alignment and Behavioral Authenticity Validation in Interactive Segmentation Agents

**Definition:**  
This principle scrutinizes whether claims of human-like behavior in interactive segmentation workflows are substantiated through empirical validation rather than inferred solely from task performance metrics. Because adaptive termination and iterative point selection are ostensibly designed to mirror human annotator strategies, reviewers must demand evidence—such as user studies, human preference comparisons, or detailed qualitative trajectory analysis—that the agent's decisions align with or outperform human interaction patterns in a meaningfully similar way. The evaluation extends to whether failure modes and behavioral edge cases are transparently reported, including cases where the model produces redundant points, fails to terminate, or misinterprets visual inputs due to mask overlay artifacts. Without such behavioral scrutiny, assertions of human likeness risk being dismissed as marketing language rather than scientific contributions. Ultimately, the work should clearly delineate which aspects of the interaction are learned and which remain constrained by the predefined action space or prompt templates.

**Core Evaluation Criteria:**
- **Human-Preference Evidence**: Does the work include user studies, human evaluations, or preference-based metrics to validate that the agent's termination and refinement behavior aligns with human judgment?
- **Qualitative Trajectory Analysis**: Are detailed visualizations and case studies provided to compare the agent's step-by-step decisions against human-like interaction patterns, rather than only showing final mask outcomes?
- **Failure Mode Transparency**: Does the paper explicitly analyze and categorize failure cases (e.g., over-refinement, misgrounding, color confusion from mask overlays) to expose the boundaries of purportedly human-like behavior?
- **Action Space Limitations**: Is there honest discussion of constraints that prevent truly free-form human-like interaction, such as inability to revise bounding boxes or revoke previous actions?