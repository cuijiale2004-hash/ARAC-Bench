**Principle 1: Fidelity of Synthetic Corruption Protocols to Real-World Shortcut Phenomena in Multimodal Benchmarks**

**Definition:**  
Methods that refine multimodal benchmarks by filtering shortcut questions must demonstrate that their detection or simulation mechanisms reflect the true distribution of low-quality items found in real-world evaluations. Many approaches rely on synthetic corruptions—such as fully swapping images or text—to simulate shortcuts, yet actual benchmark contamination is frequently subtle, involving partial textual leakage, visually answerable items, or ambiguous questions that admit unimodal solutions. If the evaluation protocol only validates against extreme artificial mismatches, the method may fail to generalize to nuanced, naturally occurring shortcuts that compromise ranking reliability. Consequently, reviewers should assess whether the proposed corruption or detection strategy spans a realistic spectrum of shortcut severity and diversity, and whether empirical validation includes qualitative or quantitative evidence on existing benchmark artifacts rather than purely synthetic ones. This principle ensures that benchmark refinement tools are robust to the kinds of low-quality questions practitioners actually encounter.

**Core Evaluation Criteria:**
- **Spectrum of Shortcut Realism**: Does the shortcut generation or detection protocol address subtle real-world artifacts (e.g., answerable-by-text alone, visual redundancy, partial modal overlap) in addition to extreme mismatches?
- **Evidence on Natural Contamination**: Does the work provide empirical analysis demonstrating detection of naturally occurring low-quality questions in existing popular benchmarks, rather than relying solely on synthetic validation?
- **Generalization Across Shortcut Types**: Is the method evaluated against diverse shortcut categories (e.g., option-only solvable, caption-dependent, noisy correlation) that reflect the heterogeneity of actual benchmark noise?
- **Qualitative Grounding**: Are concrete examples of identified shortcuts presented and discussed to substantiate that the model's notion of "low-quality" aligns with human-expert intuition?

---

**Principle 2: External Validation and Human Cognitive Alignment of Latent Cross-Modal Difficulty Constructs**

**Definition:**  
Psychometric frameworks for multimodal model evaluation introduce latent constructs such as cross-modal ability or cross-modal difficulty that are not directly observable; their scientific value depends on whether they capture meaningful, externally verifiable properties of reasoning rather than merely fitting response patterns. If a framework decomposes performance into image-only, text-only, and cross-modal components, reviewers must demand evidence that these components correspond to independently measurable phenomena—such as human accuracy decline when modalities are masked, expert judgments of item necessity, or established cognitive theories of multimodal integration. Without external validation, the parameters risk being mathematical artifacts that optimize likelihood but lack interpretive validity, undermining claims that the method assesses genuine cross-modal reasoning. The work should therefore include validation studies, even if imperfect, and honestly report the strength and limitations of construct alignment. This principle is essential to prevent circular reasoning in which model parameters are used to prove the very constructs they define.

**Core Evaluation Criteria:**
- **Human Modality-Ablation Correlation**: Do the estimated cross-modal parameters correlate with human performance degradation when single modalities are removed, or with expert ratings of cross-modal necessity?
- **Construct Interpretability**: Are the latent parameters qualitatively inspected and shown to partition items in a way that aligns with intuitive notions of cross-modal demand (e.g., items requiring joint image-text inference score high)?
- **Independent Grounding**: Is the construct validated against any external signal beyond the VLM response matrix used for training (e.g., human annotations, auxiliary task performance, theoretical predictions)?
- **Honest Reporting of Misalignment**: Does the work transparently discuss cases or benchmarks where external validation is weak or fails, rather than overstating construct validity?

---

**Principle 3: Temporal Stability and Cohort Invariance of Psychometric Parameters Across Evolving Model Populations**

**Definition:**  
Item Response Theory and its extensions promise parameter invariance—item difficulties should not depend on the specific sample of subjects, and abilities should generalize across item sets. In the context of rapidly evolving LLM and VLM ecosystems, however, these theoretical guarantees are often compromised by limited cohort sizes, heterogeneous model populations, and non-stationary ability distributions. A core evaluation criterion is therefore whether the proposed psychometric framework produces stable estimates when the set of evaluated models changes, when benchmarks are subsampled, or when new state-of-the-art models are introduced over time. If parameters shift dramatically with different model cohorts or benchmark sizes, rankings become incomparable across studies and the framework cannot serve as a standardized evaluation tool. The research must provide empirical stability analyses and specify practical protocols—such as test equating, linking, or Bayesian updating—to maintain longitudinal comparability.

**Core Evaluation Criteria:**
- **Empirical Subsampling Stability**: Are RMSE or variance statistics reported for parameter estimates under random subsampling of both items and models?
- **Cohort Independence**: Do item parameters remain consistent when estimated from different subsets or generations of VLMs, or do they exhibit strong relativity to the specific model cohort?
- **Longitudinal Comparability Protocol**: Does the work propose concrete mechanisms (e.g., equating, linking, anchor items) for maintaining scale continuity when new models are evaluated after initial parameter estimation?
- **Scale Robustness Across Benchmark Sizes**: Is sensitivity to the number of test items characterized, particularly for heterogeneous benchmarks where small samples may fail to cover the ability spectrum?

---

**Principle 4: Practical Cost-Benefit Trade-offs and Deployment Feasibility of Auxiliary Psychometric Training**

**Definition:**  
Auxiliary frameworks that train latent-variable models to compress benchmarks or improve evaluation quality introduce their own computational and data-collection costs, which must be justified against the promised savings and reliability gains. As base model inference costs continue to decline, the net practical value of training a separate psychometric model depends on training time, hyperparameter sensitivity, data efficiency, and the magnitude of benchmark compression achieved without ranking distortion. Reviewers must evaluate whether the authors have conducted rigorous cost accounting—including GPU hours for response matrix construction, CPU training time for the auxiliary model, and sensitivity to sparse observations—and compared these against simpler baselines like standard IRT or random subset selection. A method that requires dense response matrices, extensive hyperparameter tuning, or opaque training procedures may be theoretically elegant but operationally infeasible for routine benchmark maintenance. This principle ensures that proposed tools are not merely algorithmic novelties but practical contributions to the evaluation ecosystem.

**Core Evaluation Criteria:**
- **Net Computational Accounting**: Are the full costs of auxiliary model training and inference explicitly quantified and compared against the computational savings from benchmark subset selection?
- **Data Efficiency and Sparsity**: Can the method operate reliably with sparse response matrices (e.g., 10% of model-question pairs), and is the minimal data requirement for stable estimation characterized?
- **Hyperparameter Robustness**: Is the method's performance (ranking fidelity, contamination filtering) shown to be stable across reasonable ranges of key hyperparameters?
- **Incremental Value over Baselines**: Does the method demonstrably outperform simple alternatives such as classical IRT, random sampling, or information-theoretic baselines in cost-adjusted terms?

---

**Principle 5: Theoretical Justification and Empirical Testing of Structural Decomposition Assumptions**

**Definition:**  
Multidimensional psychometric extensions applied to multimodal models typically impose strong structural assumptions—such as linear additivity, independence of latent traits, or separable modality contributions—that may not hold for complex neural reasoning processes. Cross-modal integration in VLMs can involve synergistic, redundant, or suppressive interactions where text guides visual attention or images disambiguate linguistic ambiguity, dynamics that a purely additive model cannot capture. Reviewers should therefore assess whether the authors have theoretically motivated their chosen decomposition for the multimodal domain and empirically tested its robustness against plausible alternatives, such as non-linear interaction terms or correlated latent dimensions. If the conclusions about ranking fidelity or shortcut detection change materially under relaxed assumptions, the original model's validity is suspect. This principle guards against convenient but domain-inappropriate mathematical simplifications that compromise scientific inference.

**Core Evaluation Criteria:**
- **Alternative Model Benchmarking**: Does the work explicitly compare the proposed linear/additive structure against non-linear variants (e.g., multiplicative cross terms, neural interaction layers) and report differences in downstream conclusions?
- **Domain-Theoretic Motivation**: Are the independence or additivity assumptions justified by cognitive or machine-learning theories of multimodal reasoning, rather than inherited uncritically from classical psychometrics?
- **Sensitivity of Practical Conclusions**: Do key outcomes—such as model rankings, selected subsets, or identified shortcuts—remain stable when structural assumptions are relaxed, or are they highly sensitive to model specification?
- **Transparency Regarding Synergy/Redundancy**: Does the work acknowledge and discuss modalities' potential non-linear interactions (e.g., text-guided visual attention) and the limitations imposed by the chosen decomposition?