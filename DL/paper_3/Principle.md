**Principle 1: Rigorous Perturbation Analysis and Notational Precision in Subspace Clustering Preservation Bounds**

**Definition:**  
This principle requires that theoretical claims regarding the preservation of geometric structure in neural representations be supported by derivations that are not only mathematically correct but also presented with unambiguous notation and explicit operator definitions. In the context of large language models, where representation spaces are high-dimensional and nonlinear transformations are complex, even minor notational ambiguities or unverified proof steps can propagate into fundamentally flawed conclusions about how semantic clusters are maintained or distorted across layers. Reviewers must assess whether matrix norms, projection operators, and perturbation bounds are clearly defined, whether each logical step can be independently verified, and whether the authors can adequately defend their derivations against technical challenges. This is particularly vital because LLM interpretability increasingly relies on geometric and algebraic characterizations of latent spaces, and the community cannot build upon theoretical foundations that contain hidden gaps or implicit assumptions. The principle also demands that authors explicitly state what is assumed versus what is proven, distinguishing between structural assumptions and derived guarantees.

**Core Evaluation Criteria:**
- **Notational Rigor**: Are all operators, norms, and matrix decompositions explicitly defined, and are they consistent with standard linear algebra conventions used in the target community?
- **Proof Verifiability**: Can each lemma and theorem be checked step-by-step, and do the authors provide intermediate derivations when requested, or do they rely on opaque appeals to standard results?
- **Assumption Transparency**: Are the sufficient conditions (e.g., spectral-gap requirements, noise boundedness) explicitly listed, and is their necessity or tightness discussed?
- **Defense Against Critique**: When reviewers identify potential technical errors, do the authors provide direct, granular clarifications (e.g., explicit trace manipulations) rather than vague assurances?

---

**Principle 2: Empirical Validation of Spectral-Gap and Union-of-Subspaces Conditions Under Realistic Data and Noise Settings**

**Definition:**  
This principle evaluates whether the sufficient conditions derived theoretically—such as large spectral gaps, union-of-subspaces data models, and bounded additive noise—are meaningfully connected to empirical observations across both controlled synthetic settings and diverse real-world datasets. For large language models, theoretical bounds that only hold under idealized conditions are insufficient; reviewers must assess whether authors demonstrate that structure preservation remains stable when theoretical conditions are relaxed, as real-world text distributions rarely satisfy clean subspace assumptions or isotropic noise. The principle demands that experiments validate not just downstream accuracy but the mechanistic conditions themselves (e.g., measuring projection distances, singular value gaps) to establish a credible bridge between abstract guarantees and practical network behavior. Furthermore, the evaluation should examine whether the authors acknowledge the limitations of their data model and discuss how their bounds degrade with noise, dataset diversity, and distributional shift, which is essential for LLMs trained on web-scale heterogeneous corpora.

**Core Evaluation Criteria:**
- **Synthetic-to-Real Generalization**: Do experiments progress from controlled synthetic data to a broad suite of real-world datasets, and do the authors discuss which theoretical conditions hold or fail in each regime?
- **Condition Monitoring**: Are the theoretical quantities (e.g., spectral gaps, perturbation bounds) actually measured and reported during training, or are they merely assumed to exist?
- **Robustness Under Relaxation**: Do the authors empirically demonstrate that structure preservation is stable under weaker conditions than the theoretical bound strictly requires?
- **Noise and Distribution Analysis**: Is the analysis limited to additive noise, and if so, is the impossibility of handling data-dependent or non-additive noise honestly acknowledged as a scope limitation?

---

**Principle 3: Justification of Structure-Preservation Metrics Versus Downstream Clustering Performance Indicators**

**Definition:**  
This principle focuses on the critical distinction between metrics that measure the preservation of underlying geometric structure (e.g., projection distances between principal row-spaces) and conventional clustering performance metrics (e.g., ACC, NMI, ARI). In large language model research, it is common for models to achieve near-perfect task accuracy while employing fundamentally different internal mechanisms; therefore, relying solely on downstream accuracy can obscure whether a network is preserving semantic subspace structures or reconstructing them through alternative, potentially less interpretable means. Reviewers must assess whether the chosen evaluation metrics are aligned with the paper’s core research question—namely, understanding *how* representations cluster rather than *whether* they cluster accurately. The principle requires authors to explicitly justify their metric selection, demonstrate that the metrics are sensitive to the hypothesized structural phenomena, and avoid conflating structural preservation with end-task performance, which is especially important when diagnosing representation collapse or emergent organization in LLM hidden states.

**Core Evaluation Criteria:**
- **Metric-Question Alignment**: Do the chosen metrics directly quantify the theoretical phenomenon under study (e.g., projection operator similarity), or do they only measure proxy tasks?
- **Complementary Metric Inclusion**: When standard clustering metrics are omitted, is there a clear scientific justification (e.g., all baselines saturate accuracy), and are these metrics provided in appendices for completeness?
- **Mechanistic Insight Value**: Do the reported metrics reveal differences between architectures (e.g., ReLU MLPs vs. Transformers) that accuracy alone cannot explain?
- **Avoidance of Performance Conflation**: Does the work explicitly distinguish between “clustering accurately” and “preserving original subspace structure,” and does it investigate cases where high accuracy coexists with broken structure?

---

**Principle 4: Architectural Scope Delineation and Multi-Layer Extension Validity Beyond ReLU Feedforward Networks**

**Definition:**  
This principle requires authors to clearly delineate the architectural scope of their theoretical claims and to rigorously justify any extensions from simple building blocks to deep, multi-layer networks or alternative architectures. Given that large language models are predominantly based on Transformer architectures with attention mechanisms and non-linearities beyond piecewise-linear activations, theoretical work focused on single-layer ReLU networks must be explicit about where its guarantees apply and where they break down. Reviewers should evaluate whether multi-layer extensions are derived through rigorous composition (e.g., union bounds, triangle inequalities with per-layer error terms) rather than asserted as trivial generalizations, and whether the authors empirically validate structure preservation across depth. Moreover, when experiments show that certain architectures (e.g., CNNs, Transformers) fail to preserve the studied structure, the principle demands an honest discussion of whether this constitutes a limitation of the theory or an intentional design choice of those architectures, and what mechanisms might replace subspace preservation in such models.

**Core Evaluation Criteria:**
- **Multi-Layer Rigor**: Is the extension to deep networks derived with explicit error propagation bounds (e.g., per-layer ε and cumulative layer-wise bounds), or is it hand-waved?
- **Architectural Boundary Setting**: Does the work clearly state which activation functions and layer types are covered (ReLU, GELU, SiLU, LSTM) and which are not (convolution, self-attention)?
- **Empirical Depth Validation**: Do experiments include deep networks (e.g., 5-layer feedforward) and report structure preservation at each layer over the training trajectory?
- **Failure Mode Interpretation**: When architectures like Transformers or CNNs exhibit large projection distances, does the paper interpret this as breaking subspace structure and hypothesize alternative clustering mechanisms, rather than ignoring the discrepancy?

---

**Principle 5: Conceptual Integration of Classical Subspace Clustering Theory with Deep Neural Network Representation Geometry**

**Definition:**  
This principle assesses the conceptual novelty and theoretical significance of connecting long-standing empirical observations in deep learning—such as the tendency of networks to preserve cluster structure—to formal, classical frameworks like subspace clustering, perturbation theory, and principal component analysis. For the large language model community, which often relies on phenomenological descriptions of emergent capabilities, rigorous theoretical bridges to classical learning theory are essential because they provide transferable insights into initialization strategies, regularization, and architecture design. Reviewers must evaluate whether the problem formulation addresses a genuine foundational gap, whether the work is situated comprehensively within related theoretical literature (including contemporaneous approaches like sparse rate reduction or deep self-expressive learning), and whether the findings offer actionable guidance beyond explaining a single phenomenon. The principle also values whether the authors articulate why certain data modalities benefit from structure-preserving mechanisms, as this has direct implications for understanding how LLMs process heterogeneous, inherently clustered data such as language, code, and multimodal content.

**Core Evaluation Criteria:**
- **Foundational Gap Addressing**: Does the paper formalize a conjecture or empirical phenomenon that was previously only observed heuristically, and is the theoretical answer non-obvious?
- **Related Work Integration**: Are connections to closely related theoretical frameworks (e.g., union-of-subspaces methods, sparse rate reduction, self-expressive learning) thoroughly discussed and appropriately differentiated?
- **Actionable Theoretical Insights**: Do the theoretical results yield concrete, testable recommendations for practice (e.g., non-negative initialization to preserve structure, implications for feature encoding)?
- **Cross-Domain Explanatory Power**: Does the analysis explain why deep learning excels on specific data types (images, text, audio) in terms of structural preservation, and are these explanations extensible to LLM training dynamics?