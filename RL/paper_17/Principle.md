**Principle 1: Theoretical Grounding and Optimality Preservation in Automated Reward Machine Densification**

**Definition:**  
When foundation models are used to automatically generate dense, structured reward signals through automata-based formalisms, it is essential to provide theoretical guarantees that these intermediate rewards preserve the optimal policy of the original sparse-reward task. Without such formal grounding, the method risks being perceived as heuristic reward engineering that may inadvertently introduce spurious local optima or reward hacking. This principle demands that the work prove conditions under which the augmented reward structure is equivalent to potential-based shaping or satisfies no-positive-cycle properties, thereby ensuring that the FM-generated guidance strictly facilitates exploration without altering the global optimization objective. Theoretical rigor separates principled reward decomposition from ad-hoc bonus injection.

**Core Evaluation Criteria:**
- **Formal Optimality Proof:** Does the work contain a formal proposition or theorem demonstrating that optimal policies under the dense RM objective remain optimal for the original sparse MDP?
- **Precondition Verification:** Is there empirical or analytical evidence that generated artifacts satisfy necessary theoretical preconditions (e.g., absence of positive reward cycles, terminal reward dominance)?
- **Local Minima Analysis:** Does the work analyze whether intermediate rewards can create unintended local minima or bias the policy away from globally optimal behavior?
- **Conceptual Clarity on Shaping:** Is the relationship between dense sub-goal rewards and the sparse terminal reward explicitly characterized (e.g., as potential-based shaping)?

---

**Principle 2: Transparency and Quantification of Human-in-the-Loop Interventions in FM-Driven Reward Generation**

**Definition:**  
Claims of full automation in FM-generated reward design must be scrutinized for actual human involvement across the generation pipeline. This principle evaluates whether authors transparently report the frequency, nature, and resolution of human interventions, and whether such involvement is framed as a necessary feature—such as an interpretable refinement interface—or as a hidden limitation. High-quality research in this subfield must demonstrate scalability across hundreds or thousands of tasks with minimal per-task human engineering, and must provide detailed failure mode analysis rather than obscuring automation gaps behind aggregate performance metrics.

**Core Evaluation Criteria:**
- **Intervention Rate Disclosure:** Are the exact frequency and context of human interventions quantified (e.g., number of tasks corrected out of total generated)?
- **Failure Mode Taxonomy:** Are the types of failures (e.g., logical edge cases, overly sparse decompositions, local minima traps) clearly described along with their resolution mechanisms?
- **Role of Human Feedback:** Is the distinction made between passive human verification and active human-as-critic replacement, and is the feedback channel interpretable and efficient?
- **Scalability Evidence:** Does the work demonstrate batch generation success rates across diverse tasks and FM scales, using automated verification (e.g., LLM-as-judge) where possible?

---

**Principle 3: Semantic Embedding Fidelity and Topology-Aware Generalization in Language-Conditioned Compositional RL**

**Definition:**  
A central claim of language-aligned reward machines is that semantic embeddings of automaton states enable zero-shot generalization and cross-task skill reuse. This principle assesses whether the embedding space genuinely captures meaningful sub-goal semantics and whether it can adequately represent complex automaton structures—including conditional branches and loops—beyond simple linear sequences. Reviewers must evaluate whether similarity in embedding space reliably predicts behavioral transfer, and whether the method avoids catastrophic interference when task logic diverges into disjoint branches or requires non-sequential composition.

**Core Evaluation Criteria:**
- **Embedding Geometry Evidence:** Is there empirical analysis showing that semantically similar sub-goals cluster in embedding space and that proximity correlates with successful transfer?
- **Non-Sequential Task Evaluation:** Does the evaluation include compositional tasks with conditional or branching logic, rather than relying solely on linear sub-goal chains?
- **Isolated Ablation Studies:** Are the contributions of language embeddings, dense intermediate rewards, and automaton memory disentangled through controlled ablations?
- **Structural Representation Limitations:** Does the work acknowledge the limitations of pure semantic embeddings for conveying complex automaton topology and discuss complementarity with topology-aware alternatives?

---

**Principle 4: Architectural Modularity and Pathway from Privileged Symbolic State to Perceptual Grounding**

**Definition:**  
Many FM-generated reward frameworks rely on privileged simulator state for labeling functions, creating a significant deployment gap for raw-sensory domains. This principle evaluates whether authors explicitly acknowledge this reliance and architecturally separate high-level task logic from low-level perception, providing a credible roadmap toward vision-based or sensor-based predicates. The research must demonstrate that the framework is designed to accommodate learned or queried perceptual modules rather than being fundamentally bound to symbolic state access, and should justify symbolic experiments as an isolation of variables rather than an inherent system constraint.

**Core Evaluation Criteria:**
- **Privilege Explicitness:** Are labeling functions explicitly classified as relying on privileged simulator state versus observable sensory input?
- **Interface Modularity:** Does the framework exhibit a clean architectural separation between the high-level reward machine and low-level event detection, enabling pluggable perceptual modules?
- **Perceptual Replacement Roadmap:** Is there concrete discussion of how symbolic predicates could be replaced (e.g., via VLM queries or learned classifiers), including computational and reliability tradeoffs?
- **Experimental Justification:** If experiments remain symbolic, is this justified as isolating the core contribution, accompanied by a credible argument or preliminary evidence for real-world pluggability?

---

**Principle 5: Distinctiveness and Rigor in Positioning Against Reward Machine and FM-Guided RL Paradigms**

**Definition:**  
Given the maturity of both Reward Machines and FM-guided RL, this principle demands rigorous positioning against prior art to avoid perceptions of incrementalism. Reviewers assess whether the work clearly distinguishes itself from manual RM design, demonstration-driven automata learning, and FM agents that act as black-box reward generators or in-context reasoners. The contribution must articulate a unique interface—generating explicit, compositional, verifiable automata with semantic embeddings—and support this with comprehensive baseline comparisons that fairly account for domain assumptions, supervision requirements, and output interpretability.

**Core Evaluation Criteria:**
- **Unique Technical Insight:** Does the work articulate a specific insight beyond concatenating FMs and RMs (e.g., semantic state embeddings as a shared, generalizable skill space)?
- **Comprehensive Prior Art Comparison:** Are detailed comparison tables or discussions provided for both automata synthesis methods and FM-guided RL frameworks, highlighting differences in supervision, assumptions, and outputs?
- **Baseline Fairness and Justification:** Are strong, applicable baselines selected with clear justification for exclusions based on domain mismatch or computational constraints?
- **Interpretability Advantage:** Is there evidence that the structured RM representation yields tangible advantages in verifiability, debuggability, or generalization over end-to-end or opaque FM alternatives?