
### Principle 1: Theoretical Rigor and Scope Definition for Generalizing Diffusion Models to Discrete and Non-Standard Data Domains

**Definition:**  
When evaluating research that attempts to generalize diffusion models beyond standard continuous image generation (e.g., to discrete tokens, graphs, or categorical data), reviewers must rigorously assess whether the theoretical framework explicitly addresses the fundamental mismatch between continuous stochastic differential equations (SDEs) and discrete state spaces. It is insufficient to merely adapt continuous noise schedules; the work must define valid forward and reverse processes that respect the topology and probability mass constraints of the target domain. This principle ensures that generalizations are mathematically sound rather than heuristic adaptations that may fail silently or lack convergence guarantees. Reviewers should verify if the authors provide formal proofs or rigorous derivations showing that the proposed generalized process converges to the true data distribution, distinguishing genuine theoretical extensions from superficial architectural modifications.

**Core Evaluation Criteria:**
-   **Mathematical Consistency**: Does the paper formally define the forward noising and reverse denoising processes for the non-standard domain, ensuring they satisfy necessary conditions (e.g., detailed balance, ergodicity) absent in naive continuous relaxations?
-   **Handling of Discreteness**: For discrete domains, does the method avoid invalid continuous assumptions (e.g., Gaussian noise on categorical variables) or clearly justify any continuous relaxation with error bounds?
-   **Convergence Guarantees**: Are there theoretical arguments or proofs demonstrating that the generalized sampler converges to the target distribution, or at least bounded divergence, as opposed to relying solely on empirical metrics?
-   **Scope Clarity**: Is the boundary of the generalization clearly articulated? Does the paper distinguish between universal claims and specific instantiations, avoiding overclaiming applicability to all "non-image" data without evidence?
-   **Comparison to Domain-Specific Baselines**: Does the theoretical formulation account for why existing domain-specific generative models (e.g., autoregressive for text) might be preferred, and does it theoretically justify the advantage of the diffusion approach in this new setting?

---

### Principle 2: Comprehensive Empirical Validation Strategy for Verifying Theoretical Generalizations Across Diverse Data Modalities

**Definition:**  
For papers proposing generalized diffusion frameworks, experimental validation must transcend single-domain benchmarks to demonstrate true transferability and robustness of the theoretical claims. Reviewers must evaluate whether the experimental design systematically tests the method across multiple distinct data modalities (e.g., images, text, molecules, tabular data) that stress-test different aspects of the generalization. A method claiming "generalization" cannot be accepted based on performance parity with SOTA in only one well-tuned domain; it must show consistent behavior across diverse settings where the underlying data structure varies significantly. This criterion filters out methods that are effectively overfitted to a specific dataset while masquerading as general theories.

**Core Evaluation Criteria:**
-   **Multi-Modal Coverage**: Do experiments span at least three fundamentally different data types or structures to validate the "generalization" claim, rather than just variations within a single modality?
-   **Stress Testing of Assumptions**: Are there specific experiments designed to break the model’s assumptions (e.g., varying dimensionality, sparsity, or dependency structure) to identify failure modes predicted by the theory?
-   **Fair Baseline Selection**: Are comparisons made against both strong domain-specific baselines (e.g., GPT for text, E(3)-equivariant nets for molecules) *and* other generalized generative models, ensuring the method isn't just beating weak generic baselines?
-   **Metric Appropriateness**: Are evaluation metrics tailored to each specific domain’s notion of quality (e.g., FID for images, perplexity/BLEU for text, validity/uniqueness for molecules), rather than using a single proxy metric across all domains?
-   **Reproducibility and Openness**: Given the complexity of generalized frameworks, is code, hyperparameter configurations, and data preprocessing pipelines fully disclosed to allow verification of cross-domain claims?

---

### Principle 3: Critical Assessment of Computational Efficiency and Scalability Trade-offs in Unified Generative Frameworks

**Definition:**  
Generalized diffusion models often incur significant computational overhead compared to specialized architectures due to their need to handle diverse data structures through unified mechanisms. Reviewers must critically evaluate whether the paper transparently reports and analyzes these efficiency trade-offs, including training time, inference latency, memory footprint, and sample efficiency relative to domain-specific alternatives. A theoretically elegant generalization that requires orders of magnitude more compute to match specialized performance has limited practical value and questionable scientific merit unless the efficiency cost is justified by unique capabilities. This principle ensures that "generality" does not become a euphemism for "inefficiency," and that scalability is treated as a first-class evaluation dimension alongside generation quality.

**Core Evaluation Criteria:**
-   **Transparent Resource Reporting**: Does the paper report wall-clock training/inference time, GPU memory usage, and FLOPs for all compared methods under identical hardware conditions?
-   **Sample Efficiency Analysis**: How many function evaluations (NFE) or diffusion steps are required to reach competitive performance, and how does this scale with data dimensionality or complexity?
-   **Pareto Frontier Analysis**: Does the work analyze the trade-off curve between generality/performance and computational cost, identifying regimes where the generalized approach is actually preferable?
-   **Scalability Trends**: Are there scaling laws or extrapolation analyses showing how performance and cost evolve with model size or dataset scale, particularly for the new domains introduced?
-   **Justification of Overhead**: If the method is slower or larger, is there a clear articulation of what unique capability (e.g., zero-shot transfer, unified representation) justifies this cost, backed by evidence?

---

### Principle 4: Depth of Mechanistic Interpretation Linking Generalized Theory to Observed Generation Behavior in Novel Domains

**Definition:**  
Beyond reporting final performance metrics, high-quality generalization research must provide mechanistic insights explaining *how* and *why* the adapted diffusion process succeeds or fails in new domains. Reviewers should demand analysis that connects theoretical constructs (e.g., score matching errors, noise schedule effects, latent space geometry) to observable generation phenomena (e.g., mode collapse, artifact patterns, diversity limitations). This prevents acceptance of methods that work empirically but lack understanding, which hinders future progress. The review must assess whether the authors have conducted diagnostic experiments (e.g., visualizing learned scores, analyzing gradient norms, probing latent representations) that validate the theoretical intuition behind the generalization.

**Core Evaluation Criteria:**
-   **Theory-Practice Alignment**: Do diagnostic experiments directly test predictions made by the theoretical framework (e.g., if theory predicts X causes Y, is there an experiment isolating X and measuring Y)?
-   **Failure Mode Characterization**: Does the paper systematically categorize and explain failure cases in new domains, linking them back to limitations in the generalization strategy rather than treating them as unexplained anomalies?
-   **Latent Space/Score Analysis**: Are there visualizations or quantitative analyses of the learned score functions or latent representations that reveal how the model adapts to the new data structure?
-   **Ablation of Theoretical Components**: Are key theoretical innovations (e.g., modified noise schedule, custom encoder/decoder, adjusted loss weighting) ablated to confirm their individual contribution to the observed behavior?
-   **Qualitative Insight Generation**: Does the analysis yield actionable insights for practitioners applying this framework to *other* unseen domains, beyond just validating the current results?

---

### Principle 5: Evaluation of Rebuttal Responsiveness and Clarification Quality Regarding Theoretical Ambiguities and Experimental Gaps

**Definition:**  
In the context of complex theoretical generalizations, the rebuttal phase is critical for resolving ambiguities about mathematical correctness, scope limitations, and missing experimental validations that are common in initial submissions. Reviewers and ACs must evaluate not just whether authors added new experiments, but whether they provided precise, technically accurate clarifications that address the *root cause* of reviewer concerns regarding theoretical soundness or empirical completeness. A strong rebuttal for this subfield should include formal corrections or supplements to proofs, targeted additional experiments that directly test disputed claims, and honest acknowledgment of remaining limitations. Vague promises, defensive rhetoric without substance, or misinterpretation of technical critiques should be weighted negatively.

**Core Evaluation Criteria:**
-   **Precision in Theoretical Clarification**: Do responses to theoretical concerns include exact mathematical statements, corrected derivations, or citations to supporting literature, rather than hand-wavy explanations?
-   **Targeted New Evidence**: Are new experiments or analyses specifically designed to address the exact gap identified by reviewers, rather than generic additional results that don’t resolve the core concern?
-   **Honesty About Limitations**: Do authors clearly delineate what their rebuttal *does not* fix, and provide reasoned arguments for why certain requests are out of scope or infeasible, rather than ignoring difficult questions?
-   **Consistency with Original Claims**: Do clarifications remain consistent with the paper’s original narrative, or do they implicitly retract central claims while pretending to uphold them?
-   **Constructive Engagement**: Do authors engage substantively with critical feedback, even when disagreeing, providing evidence-based counterarguments rather than dismissals or appeals to authority?