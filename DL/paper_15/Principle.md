**Principle 1: Comprehensive Cross-Architecture Benchmarking Against ANN SSL, Biological Plasticity, and Scaled SNN Backbones on Large-Scale Video**

**Definition:**  
In unsupervised SNN video representation learning, claims of advancement must be evaluated against rigorous baselines spanning ANN counterparts under identical spatiotemporal protocols, biologically inspired methods (e.g., STDP-based learning), and scaled SNN architectures. The field demands explicit positioning against ImageNet-pretrained supervised ANNs and identically configured ANN SSL models to isolate the unique value of spiking temporal dynamics. Without such contextualization, it is impossible to determine whether observed gains stem from the proposed mechanism, the backbone architecture, or the training data scale.

**Core Evaluation Criteria:**
- **Parity of experimental conditions:** Are ANN baselines trained with identical clip lengths, temporal strides, augmentations, and projection heads to ensure a fair comparison?
- **Inclusion of biological baselines:** Are SNN-native unsupervised methods such as STDP or predictive plasticity included, with explicit acknowledgment of architectural and protocol differences?
- **Scaling evidence:** Does the work validate effectiveness on deeper or wider SNN backbones (e.g., ResNet-34/50, spiking transformers) beyond shallow demonstration networks?
- **Dataset scale adequacy:** Are evaluations performed on temporally rich, large-scale video benchmarks rather than small static or DVS datasets alone?

---

**Principle 2: Discrimination Between Genuine Long-Range Temporal Dependency Learning and Local Feature Smoothing in SNN Pretext Tasks**

**Definition:**  
SNNs inherently possess temporal integration via membrane dynamics; auxiliary self-supervised objectives must therefore be scrutinized to determine whether they capture meaningful long-range causal structure or merely induce short-term feature smoothing. Reviewers must require evidence that temporal prediction extends beyond adjacent-timestep interpolation and preserves discriminative motion or action cues. Claims regarding "long-range temporal dependencies" must be calibrated against the actual prediction horizon and the temporal receptive field of the objective.

**Core Evaluation Criteria:**
- **Temporal horizon characterization:** Is performance analyzed across varying prediction steps (e.g., m=1 vs. m>1) to reveal the effective temporal receptive field and identify performance collapse at longer ranges?
- **Claim calibration:** Are statements about long-range dependencies empirically matched, or should they be qualified as short-to-mid-range consistency enhancement?
- **Over-smoothing mitigation:** Does the method include mechanisms (e.g., loss weighting, cross-view targets) to prevent suppression of salient temporal variations critical for downstream recognition?
- **Comparison to forced consistency:** Is explicit prediction shown to outperform naive feature-similarity constraints, proving that the mechanism extracts semantic structure rather than enforcing superficial uniformity?

---

**Principle 3: Mechanistic and Information-Theoretic Justification for Temporal Prediction Objectives in Unsupervised SNN Training**

**Definition:**  
Beyond empirical performance gains, unsupervised SNN methods leveraging temporal prediction must provide principled arguments—grounded in predictive coding, mutual information, or slow-feature analysis—for why future prediction yields semantically stable representations. This includes explaining how the objective disentangles slowly varying semantic content from rapidly decaying noise without inducing representational collapse. A purely empirical demonstration of improved accuracy or consistency is insufficient without a causal explanatory framework.

**Core Evaluation Criteria:**
- **Theoretical grounding:** Does the work present a formal or conceptual framework (e.g., mutual information maximization between current and future features) linking the prediction objective to representation quality?
- **Semantic-noise decomposition:** Is there an analysis showing how predictable semantic signals are separated from unpredictable low-level noise (e.g., illumination variations, camera shake)?
- **Collapse avoidance:** Is there a mechanistic explanation for why cross-view prediction prevents representational collapse compared to same-view prediction or forced consistency?
- **Clarity of causality:** Does the work rigorously distinguish correlation (temporal consistency co-occurs with performance) from causation (prediction directly drives better features)?

---

**Principle 4: Transparent Quantification of Computational Overhead and Hardware Scalability for Auxiliary SNN Training Modules**

**Definition:**  
The addition of plug-and-play modules to SNN self-supervised pipelines must be accompanied by precise reporting of computational resource consumption, including GPU memory peaks, wall-clock training time, throughput, and FLOPs. Because SNNs' recurrent temporal dimension already incurs significant memory costs due to timestep unfolding, auxiliary objectives must demonstrate that their overhead is marginal and does not create scalability bottlenecks for deeper networks or larger datasets.

**Core Evaluation Criteria:**
- **Resource granularity:** Are GPU hours, peak memory usage, training throughput (frames/sec), and device requirements reported per configuration?
- **Incremental cost analysis:** Is the overhead of auxiliary components (e.g., prediction heads) isolated from backbone costs to prove modular efficiency?
- **Inference cost clarity:** Are the auxiliary modules truly training-only, and is inference latency or spike-sparsity energy cost unaffected?
- **Scalability projection:** Does the resource analysis support application to larger models or datasets, or does it identify fundamental bottlenecks?

---

**Principle 5: Robustness of Modular Integration and Hyperparameter Sensitivity Across Diverse Self-Supervised Learning Paradigms**

**Definition:**  
Because proposed modules are designed to integrate with multiple SSL frameworks (contrastive, non-contrastive, redundancy-reduction), their robustness must be evaluated through comprehensive ablations and sensitivity analyses. Key balancing coefficients—such as the weighting term between the original SSL loss and the auxiliary prediction loss—require systematic sweeps to establish stable operating regimes and avoid suppression of the base objective. A module that only functions within a narrow hyperparameter window or within a single SSL paradigm has limited practical utility.

**Core Evaluation Criteria:**
- **Cross-paradigm validation:** Are consistent improvements demonstrated across multiple SSL families (e.g., SimCLR, MoCo, BYOL, SimSiam, Barlow Twins)?
- **Component ablation:** Are the individual contributions of sub-modules (e.g., step prediction vs. clip prediction, cross-view vs. same-view design) isolated via controlled ablation?
- **Sensitivity profiling:** Is the key hyperparameter swept across a meaningful range to reveal optimal values, instability thresholds, and convergence behavior?
- **Interference analysis:** Is there evidence that the auxiliary objective complements rather than dominates or destabilizes the base SSL loss at extreme weightings?