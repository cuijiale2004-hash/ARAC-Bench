**Principle 1: Distinguishing RAG-Specific Unfaithfulness from Generic Hallucination in Problem Formulation and Methodological Design**

**Definition:**  
When evaluating research on hallucination detection in retrieval-augmented or context-conditioned generation, reviewers must assess whether the work clearly delineates why unfaithfulness to provided context constitutes a distinct problem from generic parametric hallucination. The evaluation should verify that the method is not merely applying existing hallucination detection techniques to RAG datasets, but rather exploits the unique structure of context-conditioned generation—such as the interaction between retrieved evidence and generated output, or the self-contained nature of faithfulness relative to context. Authors should explicitly articulate how their approach captures evidence-contradiction or unsupported extrapolation dynamics that are specific to RAG, and demonstrate that the method's design (e.g., which representations are probed, how context is implicitly or explicitly encoded) is tailored to these dynamics. Without such differentiation, the contribution risks being perceived as an incremental application of existing tools to a relabeled task. The principle demands that both the problem motivation and the technical mechanism reflect RAG-specific challenges rather than generic factuality verification.

**Core Evaluation Criteria:**
- **Problem Distinctiveness:** Does the paper explicitly define what makes RAG hallucination (unfaithfulness to context) different from generic factual hallucination, and justify why existing generic detectors are insufficient?
- **Context-Conditioning Exploitation:** Does the method leverage the fact that generated outputs are conditioned on provided context, and does the paper empirically show that the detector responds to context-output mismatches rather than generic token patterns?
- **Differentiation from Prior Art:** Are prior works on generic hallucination detection (including those using SAEs) adequately cited and contrasted, with clear technical distinctions drawn regarding how the proposed approach addresses RAG-specific interaction dynamics?
- **Task Scope Clarity:** If the method applies broadly to context-conditioned generation (e.g., summarization, data-to-text), does the paper transparently discuss this scope rather than narrowly branding it as RAG without justification?

---

**Principle 2: Causal and Mechanistic Attribution of Sparse Autoencoder Features to Hallucination Modes Beyond Correlational Evidence**

**Definition:**  
In research employing sparse autoencoders for hallucination detection, it is insufficient to demonstrate that selected features correlate with hallucination labels or improve classifier accuracy. Reviewers must evaluate whether the work establishes a mechanistic link between specific SAE features and hallucination behavior, moving beyond "black-box" performance gains to demonstrate that the features themselves encode semantically coherent hallucination-related concepts. This requires evidence that manipulating identified features (via activation steering or suppression) causally influences the model's tendency to hallucinate, and that these features respond predictably to counterfactual perturbations of the retrieval context. The work should also disentangle the unique contribution of SAE disentanglement from the predictive capacity of the downstream classifier by comparing against raw hidden-state baselines. Ultimately, the principle ensures that interpretability claims are grounded in causal, not just correlational, evidence.

**Core Evaluation Criteria:**
- **Causal Intervention Evidence:** Does the paper include controlled experiments (e.g., feature amplification/suppression) showing that altering specific SAE feature values changes hallucination behavior, and does it acknowledge limitations where features activate too late for prevention?
- **Feature-Label Mechanistic Link:** Is there empirical demonstration that identified SAE features correspond to coherent, human-interpretable hallucination modes (e.g., unsupported numerics), validated through qualitative case studies and cross-corpus consistency checks?
- **Disentanglement Verification:** Does the work explicitly compare SAE features against raw hidden states (with identical classifiers and feature selection protocols) to prove that SAE disentanglement concentrates hallucination signals into fewer, more interpretable dimensions?
- **Counterfactual Responsiveness:** Are features shown to dynamically change activation levels when retrieval contexts are perturbed (added, removed, or contradicted), confirming they detect context-output mismatches rather than static output patterns?

---

**Principle 3: Cross-Domain, Cross-Task, and Cross-Model Generalization Robustness of Internal-Representation-Based Detectors**

**Definition:**  
Internal-representation-based hallucination detectors are inherently tied to the model that produced the activations, raising critical questions about their applicability across different LLMs, datasets, and tasks. Reviewers must assess whether the work rigorously evaluates generalization beyond the training distribution, including cross-dataset robustness (training on one benchmark, testing on others), cross-task transfer (e.g., from summarization to QA), and cross-model applicability (whether detectors trained on one model's internals can inform detection for other models, or whether each model requires bespoke training). Because label distributions and task structures vary, evaluation should use metrics robust to imbalance (e.g., AUROC) and clearly distinguish between zero-shot transfer and retrained adaptation. The principle ensures that claims of practicality and scalability are supported by evidence of robustness to distribution shifts.

**Core Evaluation Criteria:**
- **Cross-Dataset Validation:** Does the paper evaluate out-of-distribution performance by training detectors on one dataset and testing on others without retraining, using appropriate metrics like AUROC rather than accuracy alone?
- **Cross-Task Transfer:** Are experiments conducted across diverse task types (e.g., summarization, QA, data-to-text) to determine whether learned features capture generalizable RAG signals or overfit to specific task formats?
- **Cross-Model Scope:** Does the work clarify whether SAE features or downstream classifiers transfer across LLM architectures, and does it distinguish between same-model detection (internal activations) and cross-model detection (e.g., using text-based proxy signals)?
- **Diversity-Aware Analysis:** Does the paper analyze how training data diversity affects generalization, and discuss failure modes where domain or task shifts degrade performance?

---

**Principle 4: Quantitative Validation of Interpretability Fidelity and Semantic Consistency for SAE-Derived Features**

**Definition:**  
Interpretability claims based on sparse autoencoders require rigorous validation beyond anecdotal visualization of top-activated examples. Reviewers must evaluate whether the work quantitatively verifies that human-readable explanations of SAE features faithfully predict feature activations across held-out examples, and whether feature semantics remain consistent across datasets, languages, or corpora. This includes measuring correlations between explanation-derived predictions and actual activations, testing for dataset-specific biases in explanation generation, and acknowledging the prevalence of polysemantic or generic features. The principle ensures that interpretability is treated as an empirically testable property rather than a post-hoc narrative, and that the limitations of current SAE feature clarity are transparently discussed.

**Core Evaluation Criteria:**
- **Explanation Predictive Accuracy:** Does the paper quantitatively assess how well natural-language explanations predict feature activation levels on held-out data (e.g., via Pearson correlation or human/LLM rating consistency)?
- **Cross-Corpus Semantic Stability:** Are features validated on data outside the training distribution (e.g., pretraining corpora or different benchmarks) to confirm that their semantics generalize and are not dataset-specific artifacts?
- **Bias and Limitation Acknowledgment:** Does the work explicitly discuss risks of cherry-picking activation cases, the potential for polysemantic features, and the limitations of current SAE architectures in achieving clean monosemanticity?
- **Global vs. Local Explanation Balance:** Does the paper provide both instance-level feature attributions and global, instance-invariant semantic summaries, with validation for each level?

---

**Principle 5: Comprehensive Ablation and Baseline Rigor for Compositional Pipelines in Interpretability-Driven Detection**

**Definition:**  
Research proposing multi-stage pipelines—such as combining SAE encoding, token-level pooling, information-based feature selection, and additive modeling—must rigorously justify each design choice through ablation studies and comparisons against strong, standard baselines. Reviewers must evaluate whether the work includes all relevant baselines (including standard factuality metrics like FActScore and FactCC, as well as internal-state probes), and whether ablations systematically isolate the contribution of each component (e.g., SAE vs. raw hidden states, max-pooling vs. last-token features, MI-based selection vs. random selection, GAM vs. MLP/XGBoost/LR). Without such dissection, it remains unclear whether performance gains stem from the novel combination or from a single powerful component. This principle ensures that compositional claims are empirically decomposed and that fair comparisons establish true state-of-the-art status.

**Core Evaluation Criteria:**
- **Standard Baseline Completeness:** Does the paper compare against established hallucination detection methods across all relevant paradigms (prompt-based, uncertainty-based, internal-state-based, and reference-based factuality metrics)?
- **Component-wise Ablation:** Are systematic ablations provided for each pipeline stage (feature source, pooling strategy, feature selection method, classifier architecture) to demonstrate that every component contributes to the final performance?
- **Novelty Attribution:** Does the work clearly identify which components are existing techniques and which constitute new insights or adaptations, avoiding conflation of pipeline assembly with fundamental methodological innovation?
- **Fair Comparison Controls:** Are baseline methods evaluated under comparable conditions (same backbone LLM, same datasets, same train/test splits), and are metric choices justified for imbalanced or multi-domain settings?