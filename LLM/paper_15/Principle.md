**Principle 1: Actionable Environment Augmentation Quality and Automated Pedagogical Feedback Generation**

**Definition:**  
This principle evaluates the design and implementation of environment-side feedback augmentation that converts raw, ambiguous error signals into pedagogical, actionable hints for the agent. In multi-turn tool-use domains, standard environment feedback is often cryptic—returning generic error codes or vague messages—leading to inefficient blind exploration and training instability. High-quality augmentation must instead reveal root causes, inter-tool dependencies, and operational constraints without collapsing the agent's exploration space or leaking exact solutions. The evaluation must scrutinize whether the feedback generation process is automated and scalable—ideally via an LLM-based "Generator-Judge" workflow with constitutional safeguards—rather than relying on brittle, manually crafted rules per error type. Furthermore, the principle demands rigorous evidence that augmentation genuinely accelerates learning beyond surface-level performance gains, including isolation of its quantitative contribution via ablations and verification that it preserves the agent's ability to discover novel strategies. Ultimately, this criterion distinguishes superficial "hint injection" from principled environment engineering that fosters causal reasoning and generalizable skill acquisition.

**Core Evaluation Criteria:**
- **Prevention of Solution Leakage**: Does the augmented feedback avoid prescribing exact function calls, parameter values, or operation sequences, instead providing only directional guidance (e.g., "You may need to retrieve the correct code first")?
- **Preservation of Exploration Space**: Are mechanisms in place (e.g., a Constitutional Judge) to ensure hints do not enforce a single mandatory solution path or exclude reasonable alternative strategies?
- **Automation and Scalability**: Is the feedback pipeline automated via LLMs with minimal manual engineering (e.g., 3–5 seed examples per domain), or does it require extensive human-authored templates for each new task?
- **Isolated Quantitative Impact**: Do ablation studies compare training with and without augmentation under controlled conditions, demonstrating clear performance deltas and stability improvements across diverse data splits?

---

**Principle 2: Structured Curriculum Design and Validated Stage Transition Protocols for Long-Horizon RL**

**Definition:**  
This principle assesses the scientific validity and robustness of the staged curriculum used to decompose complex multi-turn RL training into progressively harder phases. In long-horizon sparse-reward settings, naive single-stage RL often suffers from cold-start problems and gradient explosion; a well-designed curriculum must build cumulative, transferable skills where each stage is a logical prerequisite for the next (e.g., syntax mastery preceding multi-step reasoning). The evaluation must examine whether stage transitions are governed by principled, empirically validated criteria—such as the joint satisfaction of validation accuracy plateau and gradient norm stability—rather than arbitrary, heuristic step counts. Additionally, the curriculum must be thoroughly ablated against direct full-dataset training to prove that the staging itself is essential for preventing collapse and enabling monotonic improvement. A strong curriculum should also demonstrate that the agent internalizes foundational skills rather than memorizing stage-specific scaffolds, evidenced by stable or improved performance when training wheels are removed in the final stage.

**Core Evaluation Criteria:**
- **Logical Necessity of Stage Ordering**: Does the curriculum progress from foundational prerequisites to advanced skills (e.g., format correctness → core reasoning → exception handling → unassisted execution), and is it shown that reordering disrupts learning?
- **Rigorous Transition Criteria**: Are transitions triggered by converged validation performance combined with optimization stability metrics (e.g., local minima in gradient norm), mitigating the risk of premature advancement into overly complex tasks?
- **Ablative Validation Against Single-Stage RL**: Is there direct experimental evidence that training on the full dataset without staging leads to instability, collapse, or significantly inferior final performance?
- **Scaffold Internalization and Generalization**: Does disabling training scaffolds (e.g., augmented feedback) in the final stage yield maintained or improved performance, proving that the agent has internalized skills rather than overfitted to hints?

---

**Principle 3: Fine-Grained Progress Reward Design and Credit Assignment Efficacy in Sparse-Reward Multi-Turn Tasks**

**Definition:**  
This principle focuses on the design and effectiveness of dense, fine-grained reward signals that replace sparse binary outcomes in multi-turn agent training. Standard trajectory-level rewards provide insufficient credit assignment over long interaction chains, frequently causing complete training failure in complex scenarios; an effective progress reward must decompose task success into turn-level state and execution evaluations to deliver continuous learning gradients. The evaluation must critically examine whether the reward mechanism appropriately handles credit assignment across the trajectory—particularly whether late-stage errors are adequately penalized or diluted by averaging—and whether the reward framework is domain-agnostic. Moreover, the novelty and practicality of the turn-level evaluator must be established, clarifying whether it constructs signals de novo from environment states and ground-truth comparisons rather than merely consuming pre-existing benchmark scores. Finally, the principle requires empirical proof that this dense reward makes complex splits tractable where binary rewards cause complete collapse, thereby validating its role as a core enabler of long-horizon RL.

**Core Evaluation Criteria:**
- **Tractability in Complex Scenarios**: Do ablation studies demonstrate that fine-grained progress rewards enable successful training on challenging task splits (e.g., Missing Functions, Long Context) where sparse binary rewards lead to near-zero performance?
- **Credit Assignment Rigor**: Does the reward formulation avoid diluting the penalty for catastrophic late-trajectory errors? If averaging is used, is the trade-off acknowledged and mitigated (e.g., via curriculum shortening horizons)?
- **Domain Agnosticism and Reusability**: Is the progress reward framework designed to generalize across tasks without redesign, relying on generic state/execution comparisons rather than task-specific heuristics?
- **Evaluator Novelty and Ground-Truth Dependency**: Is the turn-level evaluator constructed as a new technical contribution, or does it merely repackage existing benchmark signals? Is there a clear, reproducible pipeline from environment state to reward signal?

---

**Principle 4: Out-of-Distribution Generalization Mechanisms and Cross-Benchmark Validation Rigor**

**Definition:**  
This principle evaluates whether claimed out-of-distribution generalization reflects genuine, transferable problem-solving capabilities rather than dataset memorization or benchmark-specific biases. In data-scarce agent training, supervised fine-tuning often collapses on OOD tasks, but RL-based methods must also prove that their gains are not artifacts of overfitting to training environment idiosyncrasies or annotator patterns. The evaluation demands rigorous testing across multiple independent benchmarks spanning diverse domains (e.g., web search, memory management, telecom, airline, retail) with explicit domain shifts from the training data. It further requires head-to-head comparisons against both SFT baselines (to expose overfitting) and vanilla RL baselines (to isolate the specific contribution of environment tuning components). Beyond aggregate metrics, high-quality research must provide mechanistic evidence—through case studies or stage-wise OOD diagnostics—that the agent has acquired procedural meta-skills (e.g., causal attribution, structured recovery) rather than surface-level statistical correlations.

**Core Evaluation Criteria:**
- **Diverse and Independent OOD Benchmarking**: Are OOD evaluations conducted on multiple benchmarks from distinct sources and domains, with explicit justification that they test capabilities absent from the training distribution?
- **Comprehensive Baseline Comparisons**: Does the work compare against both strong SFT models (to demonstrate resistance to overfitting) and direct RL applications (to prove that environment tuning specifically drives generalization)?
- **Mechanistic Understanding of Generalization**: Are there qualitative case studies or quantitative analyses showing that the agent applies transferable procedural strategies (e.g., "verify→diagnose→resolve") on OOD tasks rather than memorizing training trajectories?
- **Robustness to Benchmark Bias**: Is there evidence that performance gains are not attributable to shared annotator biases or overlapping distributions between training and test sets (e.g., via cross-benchmark validation)?

---

**Principle 5: System Complexity, Error Propagation Risks, and Real-World Deployment Portability**

**Definition:**  
This principle assesses the practical feasibility and architectural hygiene of deploying a multi-component environment tuning pipeline in real-world or realistic settings. While simulated training environments are standard, a method's value is contingent on whether its components—particularly the augmented feedback generator, constitutional judge, and reward evaluator—introduce prohibitive complexity, latency, or error propagation risks that undermine training stability. The evaluation must verify that environment modifications are non-invasive (e.g., external wrappers intercepting API feedback) and do not require altering underlying tool logic, ensuring portability across diverse real-world APIs. It must also scrutinize the robustness of automated LLM-based workflows, including safeguards against hallucinated or inconsistent feedback and the manual effort required to bootstrap new domains. Ultimately, the principle distinguishes research prototypes that rely on heavy, domain-specific engineering from practical paradigms that minimize human intervention while maintaining training stability and agent performance.

**Core Evaluation Criteria:**
- **Non-Invasive Architectural Design**: Is the environment augmentation implemented as a decoupled external wrapper that intercepts and translates feedback, requiring no modification to the internal logic of target tools or APIs?
- **Error Propagation Safeguards**: Does the automated pipeline include validation mechanisms (e.g., a Constitutional Judge, retry limits, or confidence filtering) to prevent low-quality or inconsistent augmented feedback from corrupting the training process?
- **Manual Engineering Overhead**: Is the cost of adapting the method to new domains quantified and minimized (e.g., requiring only a handful of seed examples), or does it demand extensive per-domain prompt engineering and human annotation?
- **Real-World Applicability Argument**: Does the work provide a credible pathway from simulation to deployment (e.g., wrapper portability to real APIs), or does it remain limited to controllable synthetic environments without discussion of practical constraints?