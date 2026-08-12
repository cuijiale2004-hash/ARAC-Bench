**Principle 1: Mechanistic Diagnosis and Intervention Verification of Cross-Layer Attention Reservoirs in Diffusion Language Models**

**Definition:**  
Research proposing inference optimizations for diffusion LLMs must provide rigorous, mechanistic evidence for its characterization of internal information flow, particularly when claiming that certain token regions (e.g., suffixes) function as transient memory buffers or "scratchpads." It is insufficient to rely solely on attention score correlations or qualitative observations. The work must empirically establish whether these tokens serve as essential cross-layer or cross-step information reservoirs—distinct from semantic prediction targets—through controlled intervention experiments that selectively disable or restore specific communication channels.

**Core Evaluation Criteria:**
- **Causal Verification via Intervention**: Does the work include targeted ablations (e.g., masking suffix tokens, blocking cross-step transmission, or restoring memory channels) to prove that the identified mechanism is *necessary* for performance, rather than merely correlated with it? For instance, does disabling the reservoir cause catastrophic failure on tasks requiring global consistency?
- **Dimensional Analysis of Information Flow**: Does the work decompose the mechanism into interpretable sub-components (e.g., cross-layer vs. cross-step transmission, write-path vs. read-path) and evaluate their individual contributions to generation quality?
- **Distinction from Semantic Content**: Does the analysis clearly differentiate between tokens carrying semantic ground-truth information and those serving as latent computational scratchpads? This requires showing that the reservoir tokens exhibit low semantic entropy but high functional utility for denoising.

---

**Principle 2: Training-Inference Distribution Robustness in Training-Free Structured Sparsity for Diffusion LLMs**

**Definition:**  
When a method introduces structured sparsity or context restrictions during inference that were not present during pre-training or fine-tuning, reviewers must scrutinize the robustness of the model to this training-inference distribution mismatch. The method cannot assume that a model trained with full, dense attention will seamlessly adapt to aggressive token dropout without degradation, especially in long-context regimes. Claims of "training-free" acceleration must be validated against the risk of distributional shift, and ideally supported by preliminary evidence that the model can adapt to restricted windows without retraining—or explicit characterization of when such shifts cause harm.

**Core Evaluation Criteria:**
- **Characterization of Failure Modes**: Does the work explicitly identify under what conditions (e.g., sequence length, model architecture, task type) the training-inference mismatch degrades performance? Does it avoid reporting only average-case improvements while hiding worst-case regressions?
- **Empirical Validation of Adaptability**: Does the work provide evidence (e.g., via supervised fine-tuning experiments or zero-shot robustness checks) that the model's internal representations are compatible with the proposed sparse inference pattern? If SFT experiments are conducted, do they confirm that the restricted window does not harm convergence?
- **Length-Dependent Sensitivity Analysis**: Does the analysis demonstrate whether performance degradation emerges as sequence length scales, given that longer contexts amplify the disparity between the dense training distribution and sparse inference conditions?

---

**Principle 3: Proactive Pre-Attention Pruning Paradigm vs. Reactive Post-Attention Eviction in Suffix Context Compression**

**Definition:**  
Research on context compression or token pruning for diffusion LLMs must clearly articulate and empirically validate its operational paradigm: whether it operates *proactively* (excluding tokens before attention computation, rendering them invisible to the computational graph) or *reactively* (computing full attention first, then evicting low-saliency tokens). This distinction is fundamental because it determines memory complexity bounds, peak allocation requirements, and runtime overhead. A proactive approach should achieve strictly lower peak memory and zero pruning overhead, while a reactive approach may suffer from "illusory" savings due to pre-eviction memory spikes.

**Core Evaluation Criteria:**
- **Paradigm Clarity and Complexity Claims**: Does the work explicitly state the computational complexity of its pruning strategy relative to sequence length (e.g., $O(1)$ vs. $O(L)$ for suffix memory)? Are these claims theoretically grounded and empirically verified?
- **Peak vs. Active Memory Distinction**: Does the evaluation report *peak memory* usage (system allocation requirement) in addition to active memory? Reactive methods may show low active memory but identical peak memory to dense baselines, which is a critical limitation for system deployment.
- **Head-to-Head Quantitative Comparison**: Does the work provide direct, controlled comparisons against reactive eviction baselines (e.g., attention-score-based pruning) on matched long-sequence regimes, reporting latency, throughput, and memory metrics under identical conditions?

---

**Principle 4: Position-Adaptive Stochastic Dropout Design for Maintaining Global Coverage Under Local Sparsity Constraints**

**Definition:**  
When employing structured dropout on positional token sequences, the design must balance two competing requirements: (1) exploiting local sparsity by concentrating retention probability on nearby tokens, and (2) avoiding permanent "blind spots" that rigid deterministic cutoffs would create. A high-quality method should demonstrate that its dropout strategy is *position-adaptive*—meaning information is preserved locally but can relocate to neighboring slots when specific positions are pruned—and that stochastic resampling across steps ensures probabilistic global coverage. The method must empirically validate why deterministic or fixed-window alternatives fail.

**Core Evaluation Criteria:**
- **Evidence Against Deterministic Baselines**: Does the work compare its stochastic strategy against deterministic alternatives (e.g., fixed nearby window, fixed distant window) and demonstrate catastrophic performance drops or blind-spot artifacts in the deterministic cases?
- **Reallocation and Adaptivity Verification**: Does the work provide empirical evidence (e.g., attention shift analysis) that when high-attention tokens are pruned, the model adaptively reallocates information to adjacent positions, confirming that functional roles are preserved even as specific storage locations change?
- **Dynamic Resampling Rationale**: Is the stochastic mask resampled dynamically across decoding steps or blocks, rather than fixed for the entire sequence? Is there justification for why temporal variation prevents systematic exclusion of distant-but-critical information?

---

**Principle 5: Orthogonal Composability and Cross-Architecture Generalization in Long-Context Diffusion LLM Acceleration**

**Definition:**  
Novel inference accelerations for diffusion LLMs must be evaluated not only in isolation but also for their ability to compose orthogonally with existing, complementary optimizations (e.g., parallel decoding, prefix caching) and to generalize across diverse model architectures. A method that achieves speedups only on a specific model family or that destabilizes cache mechanisms when combined with other techniques has limited practical impact. Reviewers must demand disentangled analyses that isolate the method's contribution from confounding factors (e.g., early termination) and verify scaling trends across both native diffusion models and autoregressive-to-diffusion adapted models.

**Core Evaluation Criteria:**
- **Disentangled Contribution Analysis**: When combining the proposed method with existing accelerations (e.g., parallel decoding + prefix caching), does the work isolate and report the marginal contribution of each component? Are confounding optimizations (e.g., early termination) explicitly disentangled from the core mechanism?
- **Cross-Architecture Validation**: Does the evaluation span multiple architectural families (e.g., natively trained diffusion models vs. autoregressive-initialized diffusion models)? Are performance variations across architectures explained through mechanistic hypotheses (e.g., pre-training exposure to bidirectional attention)?
- **Long-Context Scaling Behavior**: Does the work demonstrate that efficiency gains (latency, throughput, memory) scale favorably with sequence length, and does it explain why the method's advantages compound or diminish as context grows? Are throughput metrics interpreted carefully to account for shorter generation lengths caused by higher-quality output?