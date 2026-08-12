Based on a thorough analysis of the review materials, the following five core evaluation principles have been distilled from this study. These principles focus on **Minimax Complexity of GNNs** (26 characters). ---

Principle 1: Asymptotic Non-Vacuity and Structural Regime Characterization of Spectral-Homophily Conditions in Minimax Lower Bounds

**Definition:**
When evaluating theoretical contributions that derive minimax lower bounds under structural graph assumptions, reviewers must carefully assess whether the imposed conditions are asymptotically non-vacuous and capable of meaningfully isolating the intended graph regime. A condition such as a spectral-homophily bound should not be trivially satisfiable across all inputs, nor should it hold only in limited or uninteresting parameter ranges. Instead, the paper must formally demonstrate that the assumption excludes important alternative regimes, such as expander graphs with non-vanishing spectral gaps, and that it becomes progressively more restrictive as the problem scale increases. The authors should also clarify the role of any constants appearing in the assumption, distinguishing between constants that determine admissible graph families and those that influence the asymptotic rate. Without such justification, sharper minimax rates risk being dismissed as artifacts of loose or hidden assumptions rather than genuine consequences of graph structure. Ultimately, non-vacuity proofs and regime characterizations are essential for establishing that the theoretical results capture practically relevant structural phenomena.

**Core Evaluation Criteria:**
- **Formal Non-Vacuity Proof**: Does the paper provide a rigorous proof that the structural assumption is asymptotically non-vacuous and excludes graph families with non-vanishing spectral gaps?
- **Role of Constants**: Are the constants in the structural assumption clearly distinguished as class definers rather than rate shapers, with explicit justification for their asymptotic behavior?
- **Regime Discrimination**: Does the work sharply characterize which graphs fall under the assumption and which do not, clarifying the boundary between the general worst-case regime and the structured regime?
- **Asymptotic Restrictiveness**: Is it demonstrated that the condition becomes more restrictive as \(n\) grows, rather than trivially holding for all sufficiently large inputs?

---

Principle 2: Direct Constant-Level Empirical Validation of Minimax Rates via Ratio Diagnostics and Controlled Synthetic Constructions

**Definition:**
Empirical validation of minimax lower bounds must transcend heuristic curve-fitting to candidate functional forms, which can conflate noise, architecture bias, and optimization variance. Reviewers should expect direct, nonparametric comparisons between empirical test error and the theoretical rate expressed with explicit constants, such as ratio diagnostics that normalize observed error by the predicted bound. The work should also estimate empirical constants from these ratios to demonstrate finite-sample tightness and stability across orders of magnitude in sample size. Complementing real-world benchmarks, authors must provide controlled synthetic datasets that exactly instantiate the theoretical worst-case or structured constructions used in the proofs, thereby enabling bidirectional validation of the predicted scaling laws. These synthetic settings serve as ground-truth sanity checks, confirming that the empirical methodology recovers the correct regime when the underlying assumptions are precisely met. Together, these practices ensure that experimental evidence supports constant-level agreement with theory and can robustly discriminate between competing minimax regimes.

**Core Evaluation Criteria:**
- **Ratio Diagnostics**: Does the empirical section report direct error-to-bound ratios to test constant-level tightness without relying on parametric curve fits?
- **Empirical Constant Estimation**: Are empirical constants estimated from the ratio plots and shown to be stable across multiple orders of magnitude in sample size?
- **Controlled Synthetic Validation**: Are there synthetic datasets that exactly instantiate the theoretical constructions for both regimes, confirming that the empirical methodology recovers the predicted rate under ground-truth conditions?
- **Discrimination of Regimes**: Does the evidence clearly distinguish between competing scaling laws and demonstrate which regime governs each dataset?

---

Principle 3: Scope Alignment and Task Regime Consistency Between Theoretical Claims and Experimental Evaluation

**Definition:**
The experimental evaluation must maintain strict alignment with the theoretical contributions, with every empirical task directly corresponding to a formally proven result. Reviewers should scrutinize whether the paper clearly separates distinct statistical regimes, such as inductive graph-level prediction versus transductive node-level prediction, and whether the experiments respect these boundaries. If a task lacks a corresponding theoretical bound, its inclusion risks overstating the scope of the work and confusing readers about the actual coverage of the analysis. Authors should remove or reframe any empirical component that extends beyond the proven theoretical framework, ensuring that benchmarks and synthetic tests function as direct probes of specific theorems. This discipline prevents scope mismatch and guarantees that the empirical section reinforces rather than dilutes the theoretical narrative. Scope alignment is therefore a cornerstone of rigorous theory–experiment integration.

**Core Evaluation Criteria:**
- **Task-Theorem Correspondence**: Is every empirical task explicitly linked to a specific theoretical result, with no experiment extending beyond the proven scope?
- **Regime Separation**: Are the inductive and transductive settings clearly separated in both theory and experiment, avoiding conflation of graph-level and node-level statistics?
- **Removal of Mismatched Components**: Did the authors eliminate tasks or datasets that lack theoretical support, or rigorously justify their inclusion if retained?
- **Coherent Narrative**: Does the empirical section read as a direct validation of the theoretical claims rather than a broad survey of miscellaneous benchmark results?

---

Principle 4: Granularity, Scale, and Structural Verification of Empirical Scaling Studies

**Definition:**
Scaling studies must be evaluated on the density of their sampled configurations, the breadth of scale covered, and the statistical robustness of their measurements. Reviewers should expect experiments to span multiple orders of magnitude in sample size, from small to realistically large training sets, with densely spaced points that resolve the asymptotic transition region. Multiple random seeds and full reporting of variability are necessary to ensure that observed trends are not artifacts of sparsity or noise. Beyond raw scaling curves, authors must compute and report structural statistics for every dataset, such as spectral gap, homophily, and empirical constants, to verify that real-world benchmarks genuinely satisfy the theoretical premises. Large-scale public benchmarks should accompany controlled synthetic constructions to stress-test structural claims across diverse geometries. Only through such comprehensive granularity and verification can empirical work convincingly support asymptotic theoretical predictions.

**Core Evaluation Criteria:**
- **Sampling Density and Range**: Do scaling experiments use a dense grid of sample sizes spanning small to large \(n\) across multiple orders of magnitude?
- **Statistical Robustness**: Are results averaged over sufficient independent seeds with full mean and standard deviation reported for each configuration?
- **Structural Premise Verification**: Does the paper compute and report spectral gaps, homophily scores, and empirical structural constants for all real and synthetic datasets to confirm they satisfy the theoretical conditions?
- **Benchmark Diversity**: Are large-scale realistic benchmarks complemented by controlled synthetic worst-case constructions that stress-test the structural assumptions?

---

Principle 5: Accessibility, Self-Containment, and Pedagogical Clarity of Advanced Theoretical Machinery

**Definition:**
Papers deploying sophisticated information-theoretic or statistical machinery must be judged on their accessibility and self-containment for the broader research community. Reviewers should verify that all nontrivial technical tools, including Fano's inequality, packing numbers, and KL divergence bounds, are explicitly stated with standard references rather than treated as implicit folklore. Tutorial-style appendices that connect general minimax formulations to the specific problem at hand can significantly broaden the work's audience and impact. Additionally, every norm, metric, and regularity condition should be unambiguously defined to avoid confusion arising from conflicting conventions across subfields. The authors should also provide clear intuitive explanations linking abstract mathematical conditions to concrete structural properties, such as relating spectral gap bounds to graph bottleneckedness and slow mixing. Pedagogical clarity ensures that theoretical advances are verifiable, reproducible, and influential beyond a narrow circle of specialists.

**Core Evaluation Criteria:**
- **Explicit Tool Definitions**: Are all nontrivial technical tools explicitly stated and referenced with standard citations?
- **Tutorial Accessibility**: Does the paper or its appendices provide primer material connecting general minimax theory to the specific problem setting for non-expert readers?
- **Notation Consistency**: Are all norms, metrics, and regularity conditions unambiguously defined to prevent confusion from conflicting conventions?
- **Intuitive Structural Interpretation**: Does the work explain abstract mathematical conditions in terms of concrete graph properties, such as linking spectral gap to bottleneckedness and slow mixing?