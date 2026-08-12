**Principle 1: Stability Diagnosis and Collapse Mechanism Analysis in Iterative Self-Training LLMs**

**Definition:**  
In iterative self-evolving LLM frameworks, training stability over multiple cycles is a critical concern, as the system may suffer from performance collapse due to compounding pseudo-label noise, distribution shift, or degenerative feedback loops. Reviewers must evaluate whether the work merely reports accuracy curves or provides a mechanistic diagnosis of collapse triggers, such as degradation of self-generated label quality, loss of output diversity, or policy update biases. The assessment should examine whether the authors disentangle label-noise effects from fundamental model-collapse phenomena, analyze resilience across model scales, and situate the observed instability within the broader literature on self-consuming training loops. A high-quality study should not only document collapse but also offer empirical evidence on its root causes and potential mitigation strategies.

**Core Evaluation Criteria:**
- **Mechanistic Diagnosis**: Does the work identify specific root causes of performance collapse (e.g., pseudo-label accuracy degradation, diversity collapse) rather than only reporting the phenomenon?
- **Empirical Tracking of Signal Quality**: Are metrics such as pseudo-label accuracy, data diversity, or policy entropy tracked across iterations to explain performance trajectories?
- **Scale-Resilience Analysis**: Does the study examine how model capacity affects the onset and severity of collapse, and whether larger models fundamentally prevent or merely delay degradation?
- **Distinction from Surface-Level Reporting**: Does the analysis avoid conflating final benchmark scores with stability, instead establishing a causal link between training dynamics and eventual degradation?

---

**Principle 2: Baseline Comparability and Supervision-Assumption Taxonomy in Data-Free LLM Reasoning**

**Definition:**  
Research claiming to eliminate human data must be evaluated on the clarity of its supervision assumptions and the appropriateness of its baselines, as the space spans a continuum from fully data-free (no questions, no labels) to label-free (seed questions only) to externally verifiable (code executors, game engines). Reviewers should assess whether the authors rigorously distinguish their setting from related paradigms and select baselines that match their supervision budget, or provide well-justified explanations when direct comparison is impossible. This principle ensures that claims of autonomy are not inflated by comparing against weaker or mismatched settings, and that the community can clearly locate the work within the landscape of self-supervised reasoning enhancement.

**Core Evaluation Criteria:**
- **Supervision Clarity**: Does the paper explicitly define its position on the data-free vs. label-free vs. verifier-assisted spectrum, and avoid ambiguous terminology?
- **Baseline Appropriateness**: Are the selected baselines operating under comparable assumptions, or does the paper provide convincing justification for excluding mismatched methods (e.g., methods requiring seed questions or coding environments)?
- **Empirical Inclusion of Strong Comparators**: Where feasible, does the work include high-performing baselines from adjacent categories (e.g., Absolute Zero) with clear caveats about differing prerequisites?
- **Avoidance of Strawman Comparisons**: Does the work refrain from comparing only against the base model or weak variants, instead demonstrating superiority over the strongest available methods under matched or fairly characterized conditions?

---

**Principle 3: Role Separation and Co-Evolutionary Curriculum Dynamics in Dual-Agent LLM Self-Play**

**Definition:**  
In co-evolving multi-agent LLM frameworks, the architectural decision to separate or unify distinct functional roles—such as task generator and task solver—fundamentally determines the curriculum quality and training stability. Reviewers must evaluate whether the role separation is empirically validated against unified alternatives, and whether the generated curriculum genuinely progresses in difficulty and diversity as intended. The assessment should extend beyond final benchmark scores to examine the dynamics of the co-evolutionary loop itself, including whether the task generator adapts to the solver's frontier, whether diversity mechanisms prevent mode collapse, and whether the curriculum filtering successfully maintains tasks in the optimal difficulty zone.

**Core Evaluation Criteria:**
- **Ablation on Parameter Sharing**: Is there a controlled comparison between independent models and shared-parameter variants to validate the necessity of role separation?
- **Curriculum Progression Evidence**: Does the work empirically demonstrate that generated tasks increase in difficulty (e.g., via oracle evaluation or solver consistency metrics) across iterations?
- **Diversity and Redundancy Control**: Are mechanisms like repetition penalties or clustering evaluated for their impact on question diversity and downstream solver performance?
- **Filtering Efficacy**: Is the task-selection mechanism (e.g., consistency-based filtering) shown to successfully exclude trivial or unsolvable tasks, and is its contribution isolated via ablation?

---

**Principle 4: Verifiability Constraints and Cross-Domain Generalization in Self-Generated LLM Training**

**Definition:**  
Self-evolving LLM methods frequently rely on objective, verifiable correctness signals—such as exact answer matching or code execution—to bootstrap training without human labels, which inherently biases applicability toward structured domains like mathematics or coding. Reviewers must scrutinize whether the work acknowledges these domain constraints, evaluates generalization beyond the training domain, and honestly addresses limitations regarding open-ended or subjective tasks. A thorough study should test whether improvements stem from narrow domain alignment or genuine reasoning transfer, and should explicitly discuss the barriers to extending the approach to settings lacking unambiguous ground truth.

**Core Evaluation Criteria:**
- **Domain-Boundary Acknowledgment**: Does the paper clearly state its reliance on verifiable reward signals and the resulting limitations for open-ended, creative, or preference-based tasks?
- **Out-of-Domain Generalization Testing**: Are models trained in one domain (e.g., math) evaluated on distinct general-reasoning benchmarks to assess transferable capability gains rather than dataset-specific overfitting?
- **Relaxation of Domain Constraints**: Does the work include experiments that relax training-domain restrictions (e.g., removing math-specific prompts) to validate robustness to heterogeneous data?
- **Honest Limitation Disclosure**: Is there a candid discussion of why majority-vote pseudo-labeling or string-matching verification fails in subjective domains, and what external modules would be required?

---

**Principle 5: Practical Cost Efficiency and Synergy with Supervised Pipelines in Autonomous LLM Training**

**Definition:**  
Even methodologically novel data-free frameworks must be evaluated on their practical deployability, including computational overhead, hyperparameter robustness, and compatibility with existing supervised fine-tuning pipelines. Reviewers should assess whether the authors provide transparent cost analyses—distinguishing between GPU training, CPU preprocessing, and generation overhead—and examine whether the method adds value in realistic scenarios where some labeled data eventually becomes available. This principle ensures that autonomous training is assessed not merely as a theoretical construct but as a viable mid-training or pre-training strategy that practitioners can integrate with standard RLVR and SFT workflows.

**Core Evaluation Criteria:**
- **Computational Transparency**: Does the work report wall-clock time, resource overhead, and break down costs for each component (e.g., challenger training, solver training, filtering)?
- **Hyperparameter Sensitivity**: Are key hyperparameters justified (e.g., drawn from prior work or defaults) and is the method's sensitivity to them experimentally characterized or at least discussed?
- **Integration with Supervised Data**: Does the study evaluate whether the method serves as an effective pre-alignment or mid-training stage before SFT, and compare sequential versus concurrent integration strategies?
- **Practical Value Proposition**: Does the paper articulate a clear scenario where the method is preferable to standard supervised training, accounting for the total computational budget and data scarcity constraints?