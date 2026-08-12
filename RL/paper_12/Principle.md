**Principle 1: Scientific Framing and Research Question Articulation in End-to-End System Papers**

**Definition:**  
In the domain of large-scale agent development, papers often present tightly integrated systems that risk being dismissed as pure engineering artifacts. This principle evaluates whether authors successfully distill generalizable scientific insights from system construction by isolating specific variables—such as data topology, environmental stability, or reward integrity—and framing them as explicit, falsifiable research questions. It demands that the contribution be structured around testable hypotheses, where final benchmark performance serves as evidence for broader principles rather than an end in itself. This is crucial because the LLM field increasingly blurs the line between engineering and science; without clear problem articulation, the community cannot extract transferable knowledge, build upon the work, or distinguish methodological innovation from opportunistic assembly of existing techniques.

**Core Evaluation Criteria:**
- **Clarity of Research Questions:** Are explicit, falsifiable RQs stated early, and does each experimental subsection map cleanly to validating a specific RQ rather than presenting an undifferentiated results dump?
- **Causal Framing vs. Performance Reporting:** Does the paper explain *why* the method works (underlying mechanisms) rather than merely *that* it works (SOTA scores)?
- **Generalizability of Insights:** Are the conclusions framed as principles transferable to other models, scales, or agentic domains, or are they overly specific to the final artifact?
- **Structural Signposting:** Are contributions enumerated as scientific claims, and is the paper organized so that figures and tables directly answer the posed research questions?

---

**Principle 2: Mechanistic Design and Validation of Uncertainty Taxonomy in Synthetic Knowledge-Graph Data for Agentic Reasoning**

**Definition:**  
Synthetic data quality is a primary determinant of agentic capability, yet vague claims of “diversity” or “sophistication” are scientifically insufficient. This principle assesses whether the paper provides a rigorous, mechanistic account of how uncertainty and structural complexity are injected into training data to target specific reasoning deficits. It requires explicit definitions of uncertainty categories, empirical or theoretical justification linking each category to required agent behaviors, and topological complexity beyond surface-level obfuscation. This is vital because LLM agents frequently fail on compositional reasoning, error correction, and global planning; without precise data taxonomies and structural analyses, reviewers cannot attribute performance gains to data design versus model scale, initialization luck, or training compute.

**Core Evaluation Criteria:**
- **Explicit Uncertainty Taxonomy:** Does the paper define distinct, non-overlapping uncertainty types with concrete generation templates, intended cognitive demands, and pipeline injection points?
- **Structural Complexity Coverage:** Does the data topology move beyond tree-like or linear expansion to include cyclic dependencies, cut vertices, or non-isomorphic subgraphs that force global, topology-aware reasoning?
- **Empirical Link to Behavior:** Is there evidence—such as case studies, error analyses, or targeted probe tasks—showing that the injected uncertainties actually elicit the hypothesized reasoning patterns rather than shallow pattern matching?
- **Distinguishability from Baselines:** Can the reviewer clearly differentiate the proposed data construction from prior “easy-to-hard” or simple obfuscation strategies in terms of both mechanism and empirical effect?

---

**Principle 3: Dynamics Alignment and Staged Validation of Simulator-to-Real Dual Environments in Web-Agent RL**

**Definition:**  
Training web agents via on-policy RL in live environments is prohibitively expensive and unstable due to API volatility, latency, and inconsistent outputs. This principle evaluates the scientific rigor of dual-environment frameworks that leverage simulation to prototype algorithms before real-world deployment. Rather than demanding exact content replication—which is often infeasible for live web data—reviewers should assess whether the paper demonstrates *dynamic alignment* between simulator and real environment, such as correlation of reward trends, pass-rate trajectories, or failure mode distributions. The principle also examines whether the staging strategy is justified by evidence that simulator stability translates to real-world convergence. This perspective is critical because it legitimizes simulation as a controlled “wind tunnel” while preventing unfounded claims of sim-to-real transfer.

**Core Evaluation Criteria:**
- **Dynamics Correlation Evidence:** Does the paper provide quantitative evidence that training dynamics in simulation mirror those in the real environment, rather than asserting fidelity without validation?
- **Staging Strategy Justification:** Is the allocation of rollouts between simulator and real world clearly explained by phase or purpose, and is the rationale for not interleaving them experimentally grounded?
- **Real-World Robustness Mechanisms:** Does the real-environment interface include explicit engineering for stability, and is the impact of these mechanisms on training signal quality discussed?
- **Honest Scoping of Simulation:** Does the paper explicitly delineate what the simulator captures versus what it cannot, avoiding conflation of algorithmic stability with content realism?

---

**Principle 4: Component Attribution and Ablation Rigor Under Computational Constraints in Large-Scale Agent Training**

**Definition:**  
End-to-end agent pipelines involving large MoE models and millions of tool calls frequently render exhaustive, factorial ablation studies economically or temporally impractical. This principle evaluates whether the paper still provides convincing attribution of performance gains to specific components through alternative evidence when full ablation is infeasible. It requires transparency about computational constraints, use of available partial evidence, and careful calibration of causal claims. This is essential in LLM agent research to prevent unfounded credit assignment, where a complex pipeline is presented as a monolithic block and the community cannot identify which design decisions actually matter.

**Core Evaluation Criteria:**
- **Stage-wise Decomposition:** Are there intermediate performance checkpoints that isolate the marginal contribution of major pipeline stages despite the absence of full factorial ablation?
- **Qualitative or Proxy Evidence:** When full ablations are impossible, does the paper provide mechanistic analysis, data-source breakdowns, or smaller-scale proxy studies to support component claims?
- **Transparency on Constraints:** Does the paper explicitly state the computational or budgetary limits preventing exhaustive ablation, rather than omitting them without acknowledgment?
- **Claim Calibration:** Are causal claims carefully scoped to avoid over-attribution, distinguishing between “X contributes Y points” and “X is the sole driver of generalization”?

---

**Principle 5: Training Stability and Reward Signal Integrity in Long-Horizon On-Policy RL with Tool-Using Agents**

**Definition:**  
In long-horizon, tool-using agent training, environmental stochasticity, sparse rewards, and large action spaces create unique stability risks distinct from traditional RL domains. This principle evaluates whether the paper adequately diagnoses and mitigates failure modes such as format collapse, policy oscillation, and destructive updates caused by noisy tool returns or ambiguous negative trajectories. It emphasizes the integrity of the reward signal—how negative trajectories are filtered or penalized, how exploration is sustained, and how the training interface isolates the learning loop from real-world volatility. This is crucial because on-policy RL for web agents is highly fragile; without explicit stability mechanisms and diagnostics, performance gains may reflect transient luck or overfitting to environment noise rather than genuine policy improvement.

**Core Evaluation Criteria:**
- **Stability Diagnostics:** Does the paper report training dynamics such as reward curves, entropy trends, gradient norms, or format-collapse frequency that reveal the stability of the RL process?
- **Negative Trajectory Handling:** Is there a principled strategy for dealing with failed trajectories, and is the sensitivity of training to this filter demonstrated?
- **Environmental Noise Insulation:** Does the training architecture explicitly decouple the policy loop from external API volatility, and is the impact on convergence analyzed?
- **Distinguishing Algorithmic vs. Environmental Effects:** Does the analysis separate performance improvements due to algorithmic innovations from those due to environmental stabilization or data curation?