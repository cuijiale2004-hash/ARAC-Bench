**Principle 1: Non-Asymptotic Statistical and Architectural Separation Between Ground-Truth and Empirical Score Functions in Diffusion Models**

**Definition:**  
The work must rigorously establish whether memorization in diffusion models arises from fundamental gaps that persist in finite-sample, realistic deep-network regimes. Reviewers should assess whether the paper provides formally non-asymptotic bounds distinguishing the ground-truth score function from the empirical score minimizer, and whether the network complexity required to represent the empirical score is proven to scale with sample size. The evaluation should focus on whether these separations transcend classical bias-variance decompositions by operating under general sub-Gaussian mixture distributions and deep ReLU architectures. Particular attention must be paid to whether the analysis introduces novel analytical techniques—such as isolating dominant mixture components or characterizing Fisher divergence in the small-$t$ regime—rather than restating known overfitting phenomena.

**Core Evaluation Criteria:**
- **Novelty beyond classical frameworks:** Does the analysis avoid reducing to standard generalization bounds or bias-variance trade-offs, and instead explicitly characterize the loss gap under the empirical distribution (e.g., via Fisher divergence) and architectural requirements as distinct phenomena?
- **Technical rigor under realistic settings:** Are the bounds established for deep ReLU networks and general sub-Gaussian mixtures, or do they rely on restrictive settings such as 2-layer random-feature denoisers or simple Gaussian parametric families?
- **Non-asymptotic sharpness:** Do the bounds explicitly quantify how the gap depends on sample size $n$, dimension $d$, and noise level $t$, particularly in the small-$t$ regime where the separation is most pronounced?

---

**Principle 2: Clarity of Theoretical Hypothesis and Explicit Causal Mechanism for Memorization Emergence**

**Definition:**  
The paper must articulate a precise, testable hypothesis about why memorization occurs, distinguishing between statistical estimation error and network approximation capacity. Reviewers should evaluate whether the central claim is clearly stated upfront—for example, whether the authors identify a specific quantitative regime (such as $\log n = \mathcal{O}(d)$) where memorization is inevitable, and whether they trace the causal chain from finite-sample loss minimization to empirical score fitting. The work should avoid vague or tautological reasoning (e.g., "proper width prevents memorization" without specifying quantitative relationships) and must explicitly connect its theorems to observed empirical phenomena, making it unambiguous what each experiment is designed to validate.

**Core Evaluation Criteria:**
- **Precision of central hypothesis:** Is the core claim explicitly formalized (e.g., "when $\log n = \mathcal{O}(d)$, the ground-truth score does not minimize empirical denoising loss") rather than left implicit or buried in technical derivations?
- **Causal chain clarity:** Does the paper trace a clear path from assumption $\to$ theorem $\to$ mechanistic insight (e.g., small $t$ amplifies Fisher divergence $\to$ strong optimizer drives network toward empirical score $\to$ memorization)?
- **Distinction from prior intuition:** Does the work explicitly differentiate its contributions from well-known overfitting phenomena, and clarify what is non-trivial about the non-asymptotic analysis under realistic architectures?

---

**Principle 3: Appropriateness of Empirical Validation Scale and Fidelity to Theoretical Predictions**

**Definition:**  
Given that the primary contribution is theoretical, experiments should serve as targeted demonstrations of theoretical insights rather than claims of state-of-the-art practical performance. Reviewers must assess whether the experimental design is commensurate with the theory—e.g., whether varying sample size, network width, and weight decay directly probes the quantitative predictions of the theorems. The evaluation should examine whether the metrics (memorization ratio, precision/recall, FID) are computed with documented methodologies, whether the datasets and model scales are sufficient to illustrate the theoretical phenomena, and whether the authors avoid overstating practical applicability when experiments are intentionally small-scale or synthetic.

**Core Evaluation Criteria:**
- **Theory-experiment alignment:** Do the experiments directly test specific theoretical predictions (e.g., sharp transition from generalization to memorization as $n$ increases, or thresholds in network width)?
- **Metric rigor and transparency:** Are memorization detection criteria, precision/recall computations, and statistical protocols clearly defined, reproducible, and placed in appropriate appendices?
- **Scale appropriateness and boundary-setting:** If small models or synthetic Gaussian mixtures are used, does the paper justify this as sufficient for theoretical illustration, while acknowledging limitations regarding large-scale diffusion transformers or complex real-world distributions?

---

**Principle 4: Theoretical Grounding and Explicit Derivability of Proposed Memorization Mitigation Strategies**

**Definition:**  
Any proposed mitigation method—such as pruning, weight decay, or deduplication—must be demonstrably derived from the theoretical framework rather than presented as an orthogonal engineering heuristic. Reviewers should evaluate whether the mitigation strategy directly operationalizes the theoretical insights (e.g., reducing network expressiveness to prevent empirical score approximation, or emphasizing the small-$t$ regime where the loss gap is largest). The connection should be explicit: the paper must show how a specific theorem motivates each algorithmic design choice and why the simplified method is sufficient for validating the theory. If the method is intentionally basic, the authors must convincingly argue that it serves as empirical proof-of-concept rather than a practical SOTA technique, and must compare against random baselines to isolate the value of theory-driven design.

**Core Evaluation Criteria:**
- **Explicit theoretical motivation:** Is each step of the mitigation method (e.g., pruning attention heads with low importance scores sampled from a $\mathrm{Beta}(0.8, 2)$ distribution over $t$) directly linked to a specific theorem or insight about network capacity or loss gaps?
- **Avoidance of heuristic blur:** Does the paper maintain a clear boundary between theoretical validation and practical SOTA claims, resisting the temptation to conflate memory reduction with generic network compression?
- **Baseline isolation:** Does the work include random pruning or other ablations to demonstrate that the theory-informed selection criterion outperforms arbitrary capacity reduction, thereby validating the mechanism rather than merely the intervention?

---

**Principle 5: Justification of Distributional Assumptions and Explicit Discussion of Scalability to Large-Scale Regimes**

**Definition:**  
Theoretical results on diffusion memorization often rely on structural assumptions—such as sub-Gaussianity, Hölder smoothness, well-separated mixture components, or small-$t$ dominance—that may not hold in real-world, heavy-tailed, high-dimensional image distributions. Reviewers should assess whether the authors clearly state their assumptions, provide justification for their realism (e.g., arguing that image classes correspond to mixture components with sub-Gaussian tails), and honestly discuss limitations. Furthermore, the paper should address how the theoretical predictions might scale to billion-parameter models and web-scale datasets, or explicitly identify which aspects of the theory are invariant to scale and which require extension. This includes discussing whether the identified sample-complexity regimes (e.g., $\log n = \mathcal{O}(d)$) are relevant to modern training pipelines.

**Core Evaluation Criteria:**
- **Assumption transparency and realism:** Are assumptions like sub-Gaussianity, bounded support, or small-$t$ focus explicitly listed, and are arguments provided for their relevance to real data (e.g., multi-class image distributions as mixtures)?
- **Limitation acknowledgment:** Does the paper discuss failure modes or inapplicability (e.g., heavy-tailed distributions, conditional text-to-image models, alternative noise schedulers, partial memorization in large models) rather than silently ignoring them?
- **Scalability discussion:** Does the work address whether the derived regimes are relevant to modern large-scale training (e.g., LAION-scale datasets, Stable Diffusion), and does it engage with empirical observations of memorization in such settings to ground its theoretical claims?