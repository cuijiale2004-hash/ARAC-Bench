
**Principle 1: Rationale and Scope of Architectural Simplifications in Theoretical Transformer Optimization**

**Definition:**  
When analyzing transformer optimization through the teacher-student lens, authors frequently rely on simplified architectures—such as position-only attention, single-layer structures, or fixed orthogonal positional encodings—to enable tractable theoretical analysis. This principle demands that such simplifications be rigorously justified, either by empirical evidence (e.g., observing that full-attention training collapses to position-dominated dynamics) or by alignment with established theoretical conventions. It requires authors to explicitly delineate whether the simplification preserves the essential non-convexity and coupling of the original learning problem, or whether it inadvertently reduces the task to a trivial regime (e.g., strongly convex or input-independent). Furthermore, the principle evaluates whether the authors transparently compare their simplified setting against prior works that employ different architectural assumptions, clarifying whether claimed generalizations hold under comparable problem difficulties or whether they stem from altered problem formulations. Ultimately, the goal is to ensure that theoretical insights are not artifacts of oversimplification but instead reflect genuine properties of attention-based learning.

**Core Evaluation Criteria:**
- Is the architectural simplification (e.g., position-only attention, fixed encodings) empirically observed or theoretically justified relative to full softmax attention?
- Does the simplified setting preserve non-trivial optimization challenges (e.g., non-convexity, softmax coupling) or collapse to trivial or degenerate regimes?
- Are distinctions from prior works with differing assumptions (e.g., input-dependent vs. position-only attention) clearly and honestly articulated?
- Is the comparison with prior convergence rates fair given the differences in problem formulation and task difficulty?

---

**Principle 2: Tightness and Rigor of Non-Convex Convergence Guarantees for Attention Dynamics**

**Definition:**  
A central contribution in this subfield is the derivation of convergence rates for gradient descent on population loss in highly non-convex landscapes induced by softmax attention. This principle evaluates whether the paper establishes tight, information-theoretically optimal rates—ideally with matching upper and lower bounds—rather than loose or purely asymptotic guarantees. It scrutinizes whether the analysis provides a mechanistic, trajectory-level understanding of the optimization process, such as decomposing high-dimensional matrix dynamics into interpretable scalar evolutions or identifying distinct training phases (e.g., burn-in, tensor power, saturation). The principle also requires authors to justify polynomial dependencies on problem parameters (e.g., sequence length) by arguing their optimality or intrinsic necessity via lower bounds. Finally, it checks that proof techniques do not surreptitiously rely on standard regularity conditions—such as the Polyak-Łojasiewicz inequality or strong convexity—that fail to hold in the attention landscape, but instead exploit the specific bilinear or symmetric structure of the learning task.

**Core Evaluation Criteria:**
- Does the work prove tight convergence rates with matching upper and lower bounds (e.g., Θ(1/T))?
- Are training dynamics characterized mechanistically (e.g., via scalar factorization, phase transitions) rather than through black-box inequalities?
- Are scaling laws with respect to key parameters (e.g., sequence length, hidden dimension) derived and argued to be optimal or unavoidable?
- Does the proof genuinely address non-convexity without invoking inappropriate regularity conditions?

---

**Principle 3: Fair Comparative Positioning and Distinction of Problem Formulations**

**Definition:**  
The teacher-student transformer literature contains nuanced variations in problem setup—such as whether target token indices are fixed by the learning objective or provided as input queries, or whether tasks are formulated as regression versus classification—that fundamentally alter optimization difficulty. This principle assesses whether the paper accurately positions its contributions against closely related prior work, explicitly highlighting subtle but consequential differences in formulation rather than obscuring them. It demands that authors avoid misleading comparisons of convergence rates across incomparable settings (e.g., claiming improvement over a prior rate when the underlying task is strictly easier or differently structured). Reviewers must evaluate whether the authors acknowledge when their setting is complementary rather than strictly generalizing, and whether they provide formal reductions or equivalence demonstrations to support comparative claims. Intellectual honesty in delineating the scope of "unification"—whether the bilinear framework genuinely subsumes prior tasks or only parallels them—is paramount under this principle.

**Core Evaluation Criteria:**
- Are subtle differences in problem setup (e.g., fixed vs. input-varying targets, query column structure) explicitly stated?
- Are convergence rate comparisons with prior work conducted on equivalent or fairly comparable problem instantiations?
- Does the paper honestly assess whether its task is harder, easier, or orthogonal to related settings?
- Are claims of "generalizing" prior results backed by formal reduction or clear equivalence arguments?

---

**Principle 4: Empirical Corroboration and Robustness Beyond Idealized Assumptions**

**Definition:**  
Although grounded in theory, papers in this domain must bridge the gap between idealized assumptions and practical reality through carefully designed experiments. This principle evaluates whether synthetic experiments validate predicted convergence slopes and parameter alignment under the theoretical regime (e.g., Gaussian data, population loss), and whether additional experiments probe robustness beyond these assumptions. Specifically, it asks whether authors test non-Gaussian or finite-sample empirical loss settings, validate out-of-distribution generalization claims under distributional shift, and demonstrate applicability to real-world data despite theoretical mismatches. The principle also values explicit discussion of limitations—such as the absence of sample complexity bounds or the reliance on population loss—and whether the authors outline plausible pathways for extension. Strong empirical work in this area serves not merely as illustration but as evidence that theoretical mechanisms are not artifacts of the Gaussian-population idealization.

**Core Evaluation Criteria:**
- Do experiments match theoretical predictions (e.g., loss slopes, cosine similarity, attention heatmaps) under the assumed regime?
- Are results validated under relaxed conditions (e.g., non-Gaussian data, finite-sample/empirical loss)?
- Are out-of-distribution generalization claims supported by experiments or rigorous discussion of distributional shifts?
- Does the paper acknowledge theory-practice gaps (e.g., learned vs. orthogonal positional encodings) and discuss limitations transparently?

---

**Principle 5: Accessibility and Completeness of Complex Theoretical Derivations**

**Definition:**  
Given the inherent notational density and lengthy derivations in transformer optimization theory, this principle demands that papers remain accessible and verifiable for the review community. It requires authors to provide intuitive proof sketches in the main text, clearly explaining the role of critical lemmas and the evolution of key scalar quantities, rather than burying insight in opaque calculations. A comprehensive notation table and careful structuring of appendices are essential to allow reviewers to trace correctness without deciphering convoluted notation. Additionally, the principle insists on completeness: authors must analyze or at least discuss edge cases and degenerate regimes (e.g., full averaging, zero sparsity, or near-uniform attention) where standard bounds may break down or behavior qualitatively changes. Addressing reviewer questions about specific derivations (e.g., constant tracking in inequalities, failure modes) with clarity and precision is a critical component of this evaluative dimension.

**Core Evaluation Criteria:**
- Are proof sketches and intuitive explanations provided for major results and critical lemmas?
- Is notation sufficiently organized (e.g., summary tables) to enable verification?
- Are edge cases and degenerate parameter regimes analyzed or explicitly discussed?
- Are dense derivations broken down with clear logical steps rather than hand-waved computation?