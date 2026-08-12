**Principle 1: Low-Dimensional Theoretical Analysis as a Proxy for High-Dimensional Discrete Diffusion Dynamics**

**Definition:**  
In discrete diffusion research, rigorous low-dimensional (e.g., 1D/2D) analysis is often used to derive insights about high-dimensional behavior. Reviewers must evaluate whether the author explicitly acknowledges the limitations of this approach, whether the theoretical derivations are exact and interpretable, and whether the authors provide a clear conceptual bridge—via structural insights, scaling arguments, or empirical validation—showing that low-dimensional phenomena (such as accelerated unmasking or schedule-dependent interpolation) are predictive of high-dimensional behavior. This principle demands that authors not treat low-dimensional results as automatic guarantees for realistic continuous-time Markov chains (CTMCs), but rather as principled intuition generators whose predictions are systematically tested in high-dimensional regimes.

**Core Evaluation Criteria:**
- **Scope Acknowledgment:** Does the work clearly delimit the scope of low-dimensional theory and avoid overclaiming automatic high-dimensional guarantees?
- **Structural Insight:** Are closed-form expressions derived in a way that reveals structural mechanisms (e.g., how partition functions control unmasking rates) rather than opaque numerical simulations?
- **Empirical Bridge:** Is there a deliberate effort to validate low-dimensional predictions in high-dimensional settings (e.g., via scaling studies, ablations, or cross-domain tests)?
- **Positioning:** Does the author situate their low-dimensional contribution as a first theoretical step, acknowledging the gap to full high-dimensional theory?

---

**Principle 2: Mechanistic Attribution and Principled Correction of Guidance-Induced Transition Imbalances**

**Definition:**  
A central concern in discrete CFG is whether guidance mechanisms inadvertently distort the reverse process beyond their intended effect. Reviewers should assess whether the work provides a mechanistic decomposition of the guided rate matrix into jump rates and jump distributions, whether it identifies pathologies such as excessive unmasking speed or stiffness caused by partition-function scaling, and whether the proposed remedy (e.g., column normalization) is derived from this mechanistic understanding rather than empirical trial-and-error. The evaluation must verify that the fix decouples rate rescaling from distribution biasing, preserves the intended conditional signal, and does not introduce new numerical or sampling instabilities.

**Core Evaluation Criteria:**
- **Mechanistic Decomposition:** Does the work decompose the guided transition mechanism to isolate unintended rate rescaling from intended distribution biasing?
- **Root-Cause Linkage:** Is the identified pathology (e.g., rapid unmasking, stiffness) rigorously linked to a specific mathematical term (e.g., normalizing constant $\mathcal{Z}_w$)?
- **Derivation of Correction:** Is the proposed correction derived from this decomposition, and does it provably decouple jump rates from jump distributions?
- **Causal Empiricism:** Are the empirical improvements shown to stem from the corrected mechanism rather than from incidental interactions with the sampler or step count?

---

**Principle 3: Cross-Modal Generalization and Comprehensive Metric Coverage for Guidance Evaluation**

**Definition:**  
Because discrete diffusion spans text, image, molecular, and multimodal domains, reviewers must evaluate whether the proposed guidance mechanism is validated across diverse data modalities and model architectures. This includes testing on both masked and non-masked discrete processes, evaluating on standard benchmarks (e.g., ImageNet FID, GenEval, MATH-500, QM9), and reporting multidimensional metrics that disentangle fidelity from diversity (e.g., precision/recall, ImageReward, HPSv2). The principle requires that authors demonstrate their method's efficacy is rooted in fundamental sampling dynamics rather than dataset-specific or architecture-specific artifacts.

**Core Evaluation Criteria:**
- **Domain Breadth:** Does the evaluation span multiple discrete domains (e.g., image, text, molecular) or at least justify why a single domain is sufficient?
- **Metric Completeness:** Are both quality and diversity metrics reported, and do they disentangle fidelity from mode coverage?
- **Architecture Robustness:** Is the method tested on multiple architectures or sampler types to rule out model-specific artifacts?
- **Responsiveness:** Does the rebuttal or revision address reviewer requests for additional benchmarks (e.g., GenEval, T2I-CompBench, precision/recall)?

---

**Principle 4: Internal Consistency Between Theoretical Predictions and Empirical Schedule Design**

**Definition:**  
For work proposing dynamic guidance schedules, reviewers must scrutinize whether theoretical conclusions about schedule effectiveness (e.g., early vs. late guidance) are internally consistent across all sections of the paper and align with empirical results. This includes verifying that temporal variables (forward diffusion time vs. generation time) are clearly distinguished, that schedule recommendations derived from theory match the ablation studies, and that any apparent contradictions are resolved through precise definitions rather than retrospective corrections. The principle also demands that notation for conditional/unconditional distributions, state spaces, and guidance targets be explicitly defined and consistent with established literature.

**Core Evaluation Criteria:**
- **Claim Consistency:** Are theoretical claims about schedule effectiveness (early vs. late guidance) consistent across the abstract, theory sections, and experiments?
- **Temporal Clarity:** Are temporal concepts (forward process time, generation time, sampling steps) explicitly defined and kept distinct?
- **Notational Rigor:** Is notation introduced prior to use, and are conditional/unconditional distributions clearly distinguished from state-space members?
- **Contradiction Resolution:** If contradictions are identified, are they resolved through precise redefinitions rather than vague restatements?

---

**Principle 5: Inference-Time Efficiency and Integration Cost of Guidance Modifications**

**Definition:**  
Given that CFG is applied at inference time, reviewers must evaluate whether proposed modifications preserve computational efficiency and are practically deployable. This principle assesses whether the method introduces measurable overhead in wall-clock time, GPU memory, or step count; whether it requires architectural changes or hyperparameter tuning; and whether its benefits persist across different samplers (e.g., Euler, τ-leaping) and step budgets. A method that achieves superior quality through a simple, theoretically motivated code change without additional cost is preferred over complex alternatives, provided the efficiency claims are empirically verified.

**Core Evaluation Criteria:**
- **Overhead Quantification:** Is the computational overhead of the proposed modification quantified (wall-clock time, GPU memory, step count)?
- **Deployability:** Does the method require architectural retraining, or is it a drop-in inference-time modification?
- **Cross-Configuration Validation:** Are efficiency claims validated across different guidance strengths and sampler configurations?
- **Implementation Fidelity:** Is the implementation simplicity (e.g., "one-line change") actually realized in pseudocode, and is its correctness verifiable?