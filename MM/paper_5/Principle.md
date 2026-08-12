**Principle 1: Theoretical Distinctiveness and Cross-Modal Motivation of Latent Visual Reasoning Paradigms**

**Definition:**  
The work must clearly establish that latent visual reasoning is not a trivial extension of textual latent reasoning (e.g., Coconut) nor a repackaging of external visual tool pipelines. It must articulate a fundamentally distinct motivation centered on bridging the modality gap through direct cross-modal reasoning in a shared embedding space, rather than compressing reasoning traces or invoking discrete visual operations. The paradigm should be positioned as expanding the reasoning search space across modalities, enabling the model to actively integrate visual semantics during intermediate steps. Reviewers must assess whether the proposed method offers a genuinely novel reasoning primitive or merely applies existing latent reasoning techniques to multimodal inputs with minimal adaptation. This distinction is crucial because conflating visual grounding with latent visual reasoning obscures the actual mechanistic contribution and limits scientific understanding of cross-modal cognition in foundation models. Proper positioning ensures the community recognizes whether the work opens a new research direction or simply optimizes an existing pipeline.

**Core Evaluation Criteria:**
- **Differentiation from Textual Latent Reasoning**: Does the work explicitly contrast its objectives with text-only latent reasoning (e.g., trajectory compression vs. cross-modal integration) in motivation, learning objective, and training methodology?
- **Rigor of Baseline Positioning**: Are head-to-head comparisons provided against both “think about images” (text CoT) and “think with images” (external tool) paradigms using identical backbone architectures, data scales, and compute budgets?
- **Architectural Novelty Beyond Adaptation**: Is the architectural contribution justified as a new reasoning primitive, or does it merely transplant existing latent reasoning mechanisms into multimodal inputs without structural innovation?
- **Clarity from Visual Grounding**: Does the paper establish a clear mechanistic boundary between latent visual reasoning and visual grounding or ROI detection, avoiding conflation of reconstruction supervision with cross-modal reasoning?

---

**Principle 2: Mechanistic Interpretability and Semantic Grounding of Continuous Latent Reasoning Trajectories**

**Definition:**  
Because reasoning occurs in an uninterpretable continuous hidden-state space, the work must provide concrete empirical evidence that generated latent states actually encode query-relevant visual semantics rather than spurious correlations or degenerate solutions. This requirement moves beyond end-task accuracy to demonstrate that latent trajectories selectively attend to and reconstruct task-critical visual features, thereby validating the claimed “reasoning” process. Without such evidence, latent reasoning risks becoming a black-box performance enhancer whose internal dynamics remain opaque to both reviewers and future practitioners. The evaluation should therefore treat interpretability not as an optional visualization supplement, but as a core validation mechanism for the paradigm’s scientific validity. Establishing this mechanistic link is particularly important in multimodal settings where visual semantics are dense and high-dimensional, making it easy for models to exploit superficial statistical cues. Ultimately, interpretability analyses determine whether the model is genuinely reasoning over visual content or merely performing dataset-biased feature matching.

**Core Evaluation Criteria:**
- **Qualitative and Quantitative Introspection**: Are visualizations (e.g., attention heatmaps, t-SNE projections, reconstruction fidelity metrics) provided that directly link latent tokens to specific query-relevant image regions or visual concepts?
- **Distinction Between Reasoning and Memorization**: Does the analysis demonstrate that latent states generalize to novel visual configurations, rather than memorizing training-set bounding-box patterns or dataset biases?
- **Dynamic Query Adaptation**: Is there evidence that latent trajectories adapt conditionally to the textual query, or do they follow static, query-agnostic visual biases?
- **Failure Case Analysis**: Does the work systematically analyze cases where latent reasoning drifts, attends to irrelevant regions, or collapses, providing diagnostic insight into the mechanism’s limitations?

---

**Principle 3: Algorithmic Soundness and Marginal Efficacy of Reinforcement Learning in Latent Visual Spaces**

**Definition:**  
Adapting policy-gradient RL to latent visual reasoning introduces unique algorithmic challenges because intermediate latent steps lack explicit token distributions and direct gradient pathways. The work must rigorously justify how RL optimizes latent trajectories indirectly via downstream text generation, ensuring that policy gradients propagate through attention and feed-forward layers to influence latent state formation. Reviewers should evaluate whether the RL stage yields measurable improvements in reasoning quality—such as more structured latent trajectories or better cross-modal alignment—rather than merely providing marginal accuracy gains over supervised fine-tuning. Technical correctness in replaying hidden states and computing importance ratios across modality boundaries is essential for reproducibility and algorithmic validity. Furthermore, the reward design must be scrutinized to ensure it incentivizes genuine latent reasoning behavior rather than degenerate shortcuts that bypass visual processing. A sound RL formulation for continuous visual latent spaces represents a significant methodological advance, whereas a poorly justified adaptation risks adding unnecessary complexity without principled benefit.

**Core Evaluation Criteria:**
- **Technical Correctness of Latent RL**: Is the RL formulation (e.g., adapted GRPO) technically sound in handling non-token latent sequences, particularly regarding hidden-state replay, credit assignment, and importance ratio computation across modality boundaries?
- **Isolated Contribution of RL**: Are ablations provided comparing SFT-only and SFT+RL pipelines to isolate whether RL improves latent reasoning quality beyond what reconstruction supervision alone achieves?
- **Reward Design Validation**: Does the reward structure explicitly encourage meaningful latent reasoning (e.g., format rewards for reasoning tokens) rather than exploiting shortcuts that circumvent visual-latent processing?
- **Training Stability and Convergence**: Is the stability of RL training analyzed, especially regarding the balance between latent reasoning length, text generation quality, and avoidance of policy collapse?

---

**Principle 4: Cross-Task Generalization and Architectural Portability of Latent Multimodal Reasoning**

**Definition:**  
A latent visual reasoning paradigm must demonstrate that its benefits are not confined to perception-heavy, single-image benchmarks where visual detail search dominates. The work should validate generalization to complex reasoning tasks requiring abstract logical deduction, mathematical reasoning, and multi-image compositional understanding, ensuring that latent states support cognition beyond pixel-level discrimination. Evaluation on out-of-distribution benchmarks is essential to rule out dataset memorization, especially when training relies on specific visual grounding annotations like bounding boxes. Architectural portability—applying the same paradigm to different foundation models or visual encoders—further distinguishes a universal reasoning primitive from a task-specific engineering trick. Without such breadth, the method may be overfitting to narrow perceptual patterns rather than learning a generalizable cross-modal reasoning mechanism. Comprehensive generalization studies thus serve as the ultimate test of whether latent visual reasoning constitutes a scalable paradigm or a limited performance booster.

**Core Evaluation Criteria:**
- **Complex Reasoning Beyond Perception**: Does the evaluation include tasks that de-emphasize pure perception in favor of symbolic, mathematical, or multi-step reasoning (e.g., MathVista, geometric proofs)?
- **Out-of-Distribution Robustness**: Is there evidence of strong performance on benchmarks drawn from entirely different sources or annotation protocols than the training data, ruling out memorization of dataset-specific patterns?
- **Multi-Image and Compositional Scenarios**: Does the work test scenarios requiring integration across multiple visual inputs to verify that latent states support compositional cognition?
- **Architecture-Agnostic Validation**: Are results reported on more than one foundation model or visual encoder to support claims that the paradigm is universally applicable rather than tied to a specific backbone?

---

**Principle 5: Reliability and Controllability of Variable-Length Latent Decoding and Stopping Mechanisms**

**Definition:**  
A critical challenge in latent reasoning is determining when to terminate the latent phase and resume text generation, yet many proposed adaptive mechanisms suffer from instability or collapse. The work must rigorously evaluate the reliability of variable-length decoding strategies, as fixed-length budgets may waste computation on simple queries or truncate complex reasoning prematurely. Adaptive mechanisms such as latent end tokens, confidence thresholds, or mode-switching losses must be shown to terminate robustly without degenerating into behaviors like immediate exit or infinite loops. Reviewers should examine whether the stopping criterion generalizes across task complexities and input distributions, or whether it requires task-specific tuning that undermines practical deployment. Inference efficiency analyses are equally important, as latent reasoning must not introduce prohibitive latency or memory overhead relative to standard autoregressive decoding. Honest reporting of current limitations in stopping mechanisms, paired with principled proposals for future improvement, signals scientific maturity and helps guide the community toward reliable latent reasoning systems.

**Core Evaluation Criteria:**
- **Comparative Stability Analysis**: Are variable-length decoding strategies systematically compared against fixed-length baselines, with explicit analysis of failure modes such as non-termination, premature exit, or zero-step collapse?
- **Cross-Task Stopping Robustness**: Does the proposed stopping criterion exhibit reliable behavior across diverse task complexities and visual distributions, or does its reliability degrade without task-specific tuning?
- **Inference Overhead and Latency**: Is the computational and memory overhead of latent decoding quantified and shown to be negligible or acceptable relative to standard autoregressive generation?
- **Transparency of Limitations**: Does the work candidly report the instability or failure of proposed adaptive mechanisms and offer principled directions for improvement rather than obscuring these weaknesses?