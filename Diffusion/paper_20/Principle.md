**Principle 1: Formal Correctness and Non-Heuristic Limit Analysis in Cross-Paradigm Diffusion Unification**

**Definition:**  
The work must establish mathematically rigorous connections between discrete, Gaussian, and simplicial diffusion models without relying on conjectural or heuristic arguments. This includes formally proving limiting behaviors (e.g., large-population limits), correctly handling edge cases such as zero-temperature transitions or Markov property preservation, and ensuring that claimed equivalences between paradigms hold under precise mathematical conditions. The unification should resolve prior theoretical gaps or errors in the literature rather than merely observing similarities, and must provide verifiable derivations for likelihood comparisons, hyperparameter mappings, and embedding correspondences across paradigms.

**Core Evaluation Criteria:**
- **Proof Rigor**: Are all convergence claims (e.g., discrete to Gaussian limits) accompanied by formal theorems with complete proofs, or are they presented as conjectures or heuristics?
- **Edge Case Handling**: Does the work correctly analyze boundary behaviors (e.g., $t \to 0$ singularities, zero-temperature softmax limits, Markov property failures in prior work)?
- **Correction of Prior Errors**: Does the work identify and rectify mathematical errors or misconceptions in existing unification attempts?
- **Hyperparameter and Likelihood Comparability**: Are the mappings between hyperparameters (e.g., mutation rates vs. embeddings) and likelihood objectives formally derived and computationally meaningful?

---

**Principle 2: Cross-Domain Empirical Validation Beyond Single-Modality Benchmarks**

**Definition:**  
For a framework claiming domain-agnostic unification, the experimental evaluation must span multiple distinct data modalities (e.g., biological sequences, language, images) to substantiate generalizability claims. The work should demonstrate that the unified approach maintains competitive performance across all supported diffusion paradigms and data types, rather than excelling in only one niche domain. Evaluations must include appropriate domain-specific metrics, fair computational budgets across compared methods, and coverage of both likelihood-based and sample-quality assessments. Limited single-domain validation raises serious concerns about whether the theoretical framework transfers to diverse practical settings.

**Core Evaluation Criteria:**
- **Modality Diversity**: Are experiments conducted on at least three distinct domains (e.g., protein/DNA, language, vision) with domain-appropriate metrics?
- **Paradigm Competitiveness**: Does the unified model match or exceed individually trained discrete, Gaussian, and simplicial models within each domain under comparable training budgets?
- **Sample Quality vs. Likelihood**: Are both quantitative likelihoods and qualitative sample quality reported and shown to be consistent across domains?
- **Scalability Indicators**: Do experiments use realistic sequence lengths and model scales for each domain, or are they artificially constrained to small-scale settings?

---

**Principle 3: Practical Utility and Cost-Benefit Analysis of Unified Training vs. Specialized Models**

**Definition:**  
The work must clearly demonstrate that training a single unified model provides practical advantages over the baseline approach of independently training domain-specific or paradigm-specific models and selecting the best performer. This requires explicit analysis of computational overhead (or lack thereof), training efficiency, and the practical scenarios where practitioners benefit from avoiding pre-commitment to a single diffusion paradigm. The justification should address whether the unified model's flexibility at inference time compensates for any potential trade-offs in peak performance or training complexity, and must honestly characterize when unified training is preferable to specialization.

**Core Evaluation Criteria:**
- **Computational Overhead Analysis**: Is the computational cost of the unified framework explicitly compared against training three separate models, and is any overhead rigorously justified or shown to be negligible?
- **Inference-Time Flexibility**: Does the work demonstrate meaningful benefits from being able to switch between discrete, Gaussian, and simplicial modes at test time without retraining?
- **Performance Trade-offs**: Are there domains where the unified model underperforms compared to the best specialized model, and are these trade-offs honestly characterized?
- **Practitioner Value Proposition**: Does the work identify concrete downstream tasks where unified training offers unique advantages unavailable to specialized models?

---

**Principle 4: Numerical Stability and Algorithmic Tractability of Simplicial Diffusion**

**Definition:**  
Given that simplicial diffusion has historically suffered from numerical instabilities and expensive sampling procedures, any work in this area must rigorously address computational and numerical robustness. This includes providing stable algorithms for forward process sampling and loss computation (especially at small diffusion times), grounding numerical approximations in established mathematical theory, and experimentally validating stability across the full diffusion time horizon. The work should demonstrate that proposed stability fixes do not sacrifice the ability to compute likelihoods or access standard diffusion algorithms like classifier guidance, unlike flow-matching alternatives that abandon probabilistic objectives.

**Core Evaluation Criteria:**
- **Low-Time Stability**: Are numerical instabilities at small $t$ (e.g., infinite series divergence, SDE discretization errors) theoretically analyzed and practically resolved?
- **Sampling Efficiency**: Is the proposed sampling algorithm faster and more stable than prior SDE-based or approximate methods, with empirical timing and accuracy measurements?
- **Likelihood Preservability**: Do stability improvements maintain access to closed-form ELBOs and standard diffusion training objectives, unlike flow-matching alternatives?
- **Precision and Approximation Validity**: Are numerical approximations (e.g., central limit approximations, high-precision arithmetic) justified with error bounds and empirical validation?

---

**Principle 5: Accessibility of Interdisciplinary Intuitions and Theoretical Presentation Clarity**

**Definition:**  
When connecting machine learning to external disciplines such as population genetics, the work must make these interdisciplinary insights accessible to the core ML audience without sacrificing mathematical precision. This includes providing clear intuitive explanations for borrowed concepts (e.g., genetic drift, Wright-Fisher reproduction), effective visualizations that illustrate how different diffusion paradigms emerge from the unified process, and structuring dense theoretical content so that both the high-level insights and technical details are comprehensible. The presentation should enable reviewers to verify that the interdisciplinary connection is substantive and technically exploited rather than merely metaphorical or decorative.

**Core Evaluation Criteria:**
- **Intuitive Explanation Quality**: Are concepts from external fields (e.g., Wright-Fisher dynamics, genetic drift) explained in terms accessible to ML researchers without domain expertise?
- **Visualization Effectiveness**: Do figures clearly illustrate the unification mechanism (e.g., how population size controls discrete-to-Gaussian transitions) and stability improvements?
- **Theorem Accessibility**: Are dense theoretical results supplemented with proof sketches, intuitive interpretations, and clear statements of assumptions?
- **Interdisciplinary Substantiveness**: Is the connection to external fields (e.g., mathematical genetics literature) leveraged for concrete technical advances rather than used as superficial motivation?