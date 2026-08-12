### Principle 1: Architectural Rationality Verification for Replacing Heavy 3D Spatio-Temporal Attention with Lightweight Hybrid Designs in Video Restoration

**Definition:**
When a study proposes replacing computationally intensive 3D spatio-temporal attention mechanisms (e.g., 3D DiT) with lightweight hybrid architectures (e.g., "2D Spatial + 1D Temporal") for video super-resolution, reviewers must rigorously assess whether this structural simplification is theoretically justified and empirically sufficient. The evaluation should focus on whether the authors provide a clear rationale explaining why heavy global modeling is redundant given the low-resolution input priors, and whether the lightweight temporal modules (e.g., 1D convolutions) are proven to be adequate for maintaining inter-frame coherence without introducing flickering. Crucially, this verification cannot rely solely on synthetic benchmarks; it demands comprehensive validation on large-scale real-world datasets to ensure that the efficiency gains do not come at the cost of catastrophic failure in handling complex real-world degradations or motion dynamics.

**Core Evaluation Criteria:**
*   **Justification of Redundancy Removal:** Does the paper clearly articulate *why* 3D attention is unnecessary for the specific restoration task (e.g., arguing that LR inputs already provide sufficient structural/temporal priors), rather than treating the architectural change as merely an empirical trick?
*   **Real-World Sufficiency Validation:** Are ablation studies comparing the lightweight student against the heavy teacher conducted on *real-world* benchmarks (not just synthetic ones like UDM10) to prove that performance degradation is within an acceptable margin despite massive parameter reduction (e.g., >90%)?
*   **Temporal Module Adequacy:** Is there explicit evidence (e.g., warping error metrics, temporal profile visualization) demonstrating that the chosen lightweight temporal module effectively suppresses flickering compared to a pure 2D baseline, and is comparable to or better than heavier alternatives?
*   **Capacity Boundary Analysis:** Has the author explored the lower bound of model capacity (e.g., via pruning ratio ablations) to demonstrate that the chosen architecture size represents a validated Pareto-optimal point rather than an arbitrary selection?

---

### Principle 2: Disentanglement Mechanism Validity for Conflicting Optimization Objectives in Adversarial Video Generation Distillation

**Definition:**
In adversarial distillation for video generation where spatial detail richness and temporal consistency are inherently conflicting objectives, reviewers must evaluate whether the proposed disentanglement mechanism (e.g., dual-head discriminators) effectively prevents gradient interference and optimization collapse. It is insufficient to simply claim that separate heads optimize separate goals; the work must demonstrate that the design enforces independent gradient flows and avoids the common pitfall where one objective dominates (typically details over consistency). The evaluation requires both formal justification of how the loss function decouples these objectives and rigorous ablation proving that the disentangled design outperforms entangled baselines across *both* conflicting dimensions simultaneously, rather than improving one at the expense of the other.

**Core Evaluation Criteria:**
*   **Gradient Independence Justification:** Does the paper provide a formal or intuitive explanation of how the dual-head/multi-objective design prevents the "seesaw effect" where optimizing details degrades consistency, distinguishing it from standard single-head adversarial learning?
*   **Simultaneous Improvement Verification:** Do ablation studies show that the disentangled method achieves superior scores on *both* detail-related metrics (e.g., LPIPS, MUSIQ) and consistency-related metrics (e.g., E_warp, DOVER) compared to single-head or mixed-objective baselines?
*   **Shared vs. Independent Representation Analysis:** If a shared backbone is used for multiple heads, is there justification or evidence that this sharing acts as beneficial regularization (forcing feature alignment) rather than harmful interference, particularly when supervision signals for different attributes have unequal strength?
*   **Robustness to Label Design:** Is the labeling strategy for training discriminators (e.g., leaving real video details unlabeled to avoid quality bias) empirically validated through ablation, ensuring that the disentanglement relies on principled data curation rather than fragile hyperparameter tuning?

---

### Principle 3: Comprehensive Efficiency-Quality Trade-off Assessment Beyond Parameter Counting in Generative Model Compression

**Definition:**
For research claiming significant efficiency improvements in generative video models, reviewers must demand a multidimensional efficiency assessment that extends far beyond simple parameter counting or inference latency. True efficiency evaluation in this subfield requires reporting FLOPs, GPU memory usage, and actual throughput under standardized hardware conditions, alongside a complete suite of quality metrics covering fidelity, perceptual realism, and temporal stability. Furthermore, the trade-off analysis must include comparisons against diverse baselines (including non-diffusion CNN/Transformer methods and larger backbone variants) to establish whether the compression achieves a genuine Pareto frontier improvement or merely shifts the operating point along an existing curve. The assessment must also verify that efficiency claims hold under varied resolutions and sequence lengths, not just fixed benchmark settings.

**Core Evaluation Criteria:**
*   **Multidimensional Efficiency Metrics:** Does the paper report FLOPs, memory footprint, and wall-clock time alongside parameter counts, providing a holistic view of deployment feasibility?
*   **Diverse Baseline Comparison:** Are efficiency-quality trade-offs compared against both generative (diffusion/GAN) and non-generative (CNN/Transformer) approaches to contextualize the absolute performance-efficiency position?
*   **Scalability Verification:** Are efficiency and quality evaluated across varying resolutions and video lengths (e.g., >100 frames) to ensure that the lightweight design does not suffer from hidden scaling bottlenecks or temporal drift?
*   **Backbone Sensitivity Analysis:** Has the author tested alternative backbone scales (e.g., SD2.1 vs. SDXL) to quantify the marginal quality gain per unit of efficiency cost, validating the chosen backbone as an optimal trade-off point?

---

### Principle 4: Empirical Grounding of Non-Intuitive Adversarial Training Data Curation and Labeling Strategies

**Definition:**
Adversarial distillation schemes often employ non-intuitive data curation or labeling strategies (e.g., using shuffled videos as negative samples, leaving certain real data unlabeled, or mixing image/video priors) to stabilize training or enforce specific attributes. Reviewers must require that such design choices be grounded in empirical evidence rather than heuristic intuition alone. Each non-standard labeling decision must be accompanied by ablation studies demonstrating its necessity and showing that more intuitive alternatives (e.g., labeling all real data as "real") lead to measurable degradation. The evaluation should also assess whether the authors provide post-hoc analysis explaining *why* the non-intuitive strategy works (e.g., identifying dataset quality biases or manifold mismatches), transforming engineering tricks into transferable knowledge.

**Core Evaluation Criteria:**
*   **Ablation of Labeling Choices:** Are all non-standard labeling decisions (e.g., unlabeled real video details, shuffled frame negatives) individually ablated to prove their contribution to final performance?
*   **Failure Mode Analysis of Intuitive Alternatives:** Does the paper explicitly demonstrate and explain why the "obvious" or intuitive labeling approach fails (e.g., showing metric drops or visual artifacts when real video details are labeled as positive)?
*   **Dataset Bias Identification:** When mixing data sources (e.g., high-quality images + lower-quality videos), is there explicit analysis of quality distribution differences and how the labeling strategy mitigates potential bias propagation?
*   **Transferability of Curation Logic:** Is the rationale for data curation presented as a generalizable principle applicable to similar distillation tasks, or is it overly specific to the current dataset/model combination without broader insight?

---

### Principle 5: Temporal Consistency Validation Through Dynamic Visualization and Long-Horizon Testing in Compressed Video Models

**Definition:**
For any video generation or restoration method employing compressed or simplified temporal modeling, static frame comparisons and short-sequence metrics are insufficient for validating temporal consistency. Reviewers must insist on dynamic validation through video demos, temporal profile slices, and testing on extended sequences (>100 frames) to expose latent instabilities that only manifest over longer horizons. The evaluation should specifically probe for failure modes unique to compressed architectures, such as gradual drift, periodic flickering, or inconsistency under complex motion, which may be masked in standard 25-frame benchmark clips. Additionally, qualitative temporal assessment must go beyond cherry-picked examples to include systematic visualization of temporal coherence across diverse content types and degradation levels.

**Core Evaluation Criteria:**
*   **Dynamic Evidence Requirement:** Does the submission include video demos or temporal profile visualizations that allow direct assessment of flickering and coherence, rather than relying solely on scalar metrics like E_warp?
*   **Long-Horizon Stability Testing:** Are experiments conducted on sequences significantly longer than standard benchmarks (e.g., >100 frames) to verify that temporal consistency does not degrade with accumulated inference steps?
*   **Complex Motion Stress Testing:** Is temporal consistency evaluated specifically on challenging scenarios (e.g., fast motion, occlusions, texture-rich regions) where lightweight temporal modules are most likely to fail?
*   **Comparative Temporal Profiling:** Are temporal profiles shown side-by-side with both the uncompressed teacher and non-temporal baselines to visually demonstrate the specific contribution of the temporal module versus the generative prior?