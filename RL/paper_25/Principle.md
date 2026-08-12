**Principle 1: Theoretical Distinction and Novelty Attribution Between Process-Dense Supervision and Outcome-Driven Curriculum Learning in Expert-Trajectory-Based Reasoning**

**Definition:**  
In the subfield of leveraging expert demonstrations to train reasoning models, it is crucial to rigorously differentiate whether a proposed method provides intrinsic process-level supervision at each intermediate step or merely facilitates exploration toward a sparse final-outcome reward through curriculum heuristics. Many approaches decompose expert trajectories into steps, yet conflate dense reward shaping with mere reverse curriculum learning that still optimizes for answer correctness. A clear theoretical distinction must establish that the learning signal teaches the model *how* to reason rather than simply *where* to end up, particularly for hard problems where pass@k is near zero. Without this clarity, methodological contributions risk being perceived as incremental repackaging of existing Learning from Demonstration or R3-style techniques. Reviewers must assess whether the authors explicitly articulate the causal mechanism linking step-wise rewards to policy improvement independent of final-answer verification. This distinction is foundational because process-driven supervision and outcome-driven exploration exhibit fundamentally different scaling properties, error modes, and synergies with downstream online RL refinement.

**Core Evaluation Criteria:**
- **Clarity of Mechanistic Differentiation**: Does the work explicitly contrast its learning objective with outcome-driven reverse curriculum methods (e.g., R3) and per-token imitation (SFT), demonstrating that step-wise similarity rewards provide a distinct gradient signal?
- **Empirical Separation on Hard Tasks**: Are there controlled experiments showing the proposed method outperforms trajectory-decomposition baselines specifically on tasks where outcome rewards are extremely sparse (e.g., AIME24/25), not just standard benchmarks?
- **Causal Attribution**: Does the analysis isolate the effect of process supervision from exploration facilitation, proving that performance gains stem from dense intermediate feedback rather than easier access to positive outcome labels?
- **Intellectual Contribution Boundaries**: Does the paper honestly acknowledge prior art in both classical RL (e.g., Learning from Demonstration) and contemporary LLM reasoning (e.g., PRMs), clearly demarcating its unique theoretical niche?

---

**Principle 2: Automation, Scalability, and Objective Reproducibility of Expert Trajectory Decomposition into Logical Actions**

**Definition:**  
A critical evaluation standard for step-wise reasoning frameworks is the rigor and transparency of the data pipeline that converts raw expert demonstrations into structured training instances. Manual parsing or heuristic decomposition introduces subjective bias, irreproducibility, and severe scalability bottlenecks that undermine claims of general applicability. High-quality research must provide fully automated mechanisms—such as LLM-based rewriters with disclosed prompts and verifiable formatting checks—to transform unstructured trajectories into logical actions without domain-specific manual engineering. The pipeline must be shown to handle diverse trajectory styles and be robust to formatting failures that would otherwise contaminate training or require costly human annotation. Furthermore, the action granularity must be well-defined and justified, as arbitrary step boundaries can fragment coherent reasoning or conflate multiple logical operations. Reproducibility demands that every parsing decision, filtering heuristic, and failure mode is documented so that independent researchers can replicate the exact training distribution.

**Core Evaluation Criteria:**
- **Automation Transparency**: Are the decomposition algorithms, prompts, and parsing heuristics fully disclosed in sufficient detail to enable exact replication, including handling of malformed outputs?
- **Scalability Evidence**: Does the work demonstrate that the pipeline operates on existing datasets without requiring naturally step-wise annotations, and discuss throughput or cost relative to manual curation?
- **Bias and Error Analysis**: Is there quantification of parsing failure rates, inter-annotator agreement (if human-in-the-loop), or sensitivity to the choice of decomposition model, ensuring subjective biases are minimized?
- **Action Granularity Justification**: Is the definition of an "action" or "step" explicitly motivated by task structure rather than convenience, with ablations showing the impact of granularity on learning?

---

**Principle 3: Rational Design, Robustness to Near-Misses, and Mitigation of False Positives in Syntax- or Semantic-Based Intermediate Rewards**

**Definition:**  
The design of intermediate reward functions fundamentally determines the reliability of step-wise supervision and the risk of reinforcing erroneous behaviors. Reviewers must scrutinize whether the chosen similarity metric—whether syntactic (e.g., edit distance), embedding-based, or model-based—provides a smooth, meaningful learning signal that correlates with actual step correctness rather than surface-form mimicry. A robust reward must assign partial credit to semantically valid "near-miss" actions and avoid false positives when the model reproduces a flawed or suboptimal teacher step. The theoretical rationale for preferring one similarity function over alternatives (e.g., KL divergence, exact match, LLM-as-Judge) must be articulated, especially regarding computational constraints and gradient properties. Additionally, because imperfect teachers inevitably generate incorrect intermediate steps, the framework must include empirical evidence—such as downstream RLVR unlearning experiments or heuristic filtering—that demonstrates the mitigation of teacher-error propagation. Without such scrutiny, dense rewards may inadvertently bake in systematic biases that are difficult to remove in later training stages.

**Core Evaluation Criteria:**
- **Reward Function Justification**: Is the selection of the similarity metric theoretically grounded and empirically compared against plausible alternatives, with trade-offs between computational cost, granularity, and semantic fidelity explicitly discussed?
- **Near-Miss Robustness**: Does the work analyze reward landscapes for semantically equivalent but syntactically divergent actions, ensuring the signal does not collapse to sparse binary rewards or demand rigid token mimicry?
- **Teacher Error Propagation**: Are there experiments or analyses showing that the model can overcome suboptimal or erroneous teacher steps, for example through subsequent RLVR fine-tuning or data-cleaning heuristics?
- **Reward Granularity Ablations**: Does the paper isolate the benefit of multi-step dense rewards against holistic single-step similarity rewards and final-answer-only rewards, validating that step-level granularity is essential?

---

**Principle 4: Empirical Validation of Scalability Across Model Capacities and Cross-Domain Transfer Beyond Structured Reasoning**

**Definition:**  
Training frameworks that claim to unlock reasoning capabilities in small or mid-sized models must provide convincing evidence that their benefits are not artifacts of a specific model scale or narrow domain. The field demands evaluation across multiple parameter regimes to test whether the supervision mechanism remains effective as model capacity increases, or whether larger models render the technique redundant by solving tasks through standard exploration. Equally important is cross-domain generalization: a robust method should transfer from structured domains like mathematics to semi-structured or agentic tasks such as software engineering, where action definitions and environment feedback differ substantially. Experiments must be designed to show that gains are most pronounced when task difficulty exceeds the base model's capability frontier, confirming the method's role as a bridge for hard problems. Limitations regarding compute constraints or missing large-scale experiments must be frankly acknowledged rather than obscured. This principle ensures that contributions are evaluated as general training paradigms rather than narrow, scale-specific engineering tricks.

**Core Evaluation Criteria:**
- **Multi-Scale Validation**: Are results reported for at least two distinct model sizes (e.g., 3B and 7B), with discussion of how the difficulty frontier shifts and whether the method's relative advantage persists or grows?
- **Cross-Domain Benchmarking**: Does the work evaluate beyond a single domain, preferably including both reasoning (math) and agentic tasks (SWE), to demonstrate domain-agnostic applicability?
- **Capability-Relative Efficacy**: Is there analysis showing that performance gains are correlated with task difficulty relative to the base model's pass@k, rather than uniform across easy and hard problems?
- **Limitation Disclosure**: Does the paper transparently discuss computational barriers to scaling (e.g., 14B+ models) and hypothesize about expected behavior without overstating claims?

---

**Principle 5: Rigor of Curriculum Design and Phase Transition Mechanisms Between Supervised Initialization and Online Reinforcement Refinement**

**Definition:**  
When a training pipeline combines an imitation-style initialization with subsequent online RL refinement, the curriculum design and transition logic become central to the paper's scientific contribution. Reviewers must evaluate whether the initial supervised phase provides a demonstrably superior prior compared to standard SFT, specifically in terms of enabling exploration, preserving reasoning flexibility, and avoiding rigid mimicry. The transition mechanism—whether fixed, performance-gated, or variance-based—must be justified, and dynamic sampling or filtering strategies must be shown to meaningfully accelerate learning rather than serving as inert engineering details. Behavioral analysis should reveal emergent capabilities attributable to the curriculum, such as interleaved planning, reflection, or verification, rather than mere length expansion or surface pattern matching. Finally, the downstream RL phase must be proven capable of overriding teacher biases and suboptimalities introduced during initialization, establishing a true synergistic pipeline rather than a naive concatenation of two training stages. This principle discriminates between superficial multi-stage training and principled curriculum learning.

**Core Evaluation Criteria:**
- **Initialization Superiority**: Are there direct comparisons showing that the supervised RL initialization outperforms SFT initialization before online RL, measured by final performance, sample efficiency, or behavioral metrics?
- **Dynamic Sampling Rigor**: Is the variance-based filtering or dynamic sampling strategy ablated, with evidence that removing low-variance samples materially improves convergence or stability?
- **Behavioral Emergence Analysis**: Does the work provide qualitative and quantitative evidence of sophisticated reasoning patterns (e.g., structured planning, mid-stream self-correction) induced by the curriculum, beyond simple output length increases?
- **Downstream Unlearning Capability**: Does the online RL refinement phase demonstrate the ability to deviate from and improve upon the teacher's trajectory, proving that the initialization does not irreversibly lock the model into suboptimal behaviors?