**Principle 1: Adaptive Trust-Region Control for Quantized Behavior and Full-Precision Proximal Policy Divergence in Decoupled RL**

**Definition:**  
Research in this area must rigorously address the training instability that arises from decoupling the behavior policy—represented by a quantized low-precision actor used for rollout generation—from the proximal policy represented by full-precision parameters used for gradient updates. Because quantization introduces a persistent distributional shift between these two policies, standard importance sampling and clipping mechanisms are prone to exponential divergence over long training horizons. A high-quality contribution must present a principled, dynamically adaptive mechanism (e.g., adaptive clipping ranges responsive to real-time policy ratios) that functions as a negative feedback system, ensuring that trust-region constraints remain valid even as the quantized actor drifts.

**Core Evaluation Criteria:**
- **Mechanistic Clarity of Divergence Control**: Does the work clearly explain how the quantized-to-full-precision policy ratio evolves during training, and does the proposed method explicitly bound this divergence through a theoretically grounded adaptive mechanism rather than static thresholds?
- **Long-Horizon Stability Validation**: Are experiments conducted over extended training regimes (e.g., 1,000+ steps, ideally approaching 2,000 steps or more) demonstrating that the method avoids late-stage collapse, exponential KL growth, or reward degradation that naive decoupled objectives exhibit?
- **Theoretical and Empirical Causality**: Does the work establish a causal link between the adaptive trust-region mechanism and stability, rather than merely showing improved final accuracy? This includes analyzing gradient norms, clipping fractions, and KL trajectories throughout training.

---

**Principle 2: Numerical Fidelity Preservation for Micro-Scale Weight Updates Under Low-Precision Quantization in Trust-Region RL**

**Definition:**  
A critical and often overlooked challenge in this subfield is the granularity mismatch between trust-region-constrained weight updates (typically on the order of $10^{-7}$ to $10^{-6}$) and the quantization error floor (on the order of $10^{-5}$ to $10^{-4}$), which effectively masks learning signals and decouples the quantized model from the training dynamics. High-quality research must explicitly identify this mismatch and propose solutions that amplify effective updates relative to quantization noise without violating RL stability constraints, all while maintaining compatibility with standard inference engines and avoiding per-step computational overhead.

**Core Evaluation Criteria:**
- **Explicit Quantification of the Mismatch**: Does the work clearly articulate the scale disparity between weight updates and quantization errors, backed by empirical measurements across RL steps?
- **Update-to-Noise Improvement**: Does the proposed method demonstrate a super-linear (e.g., quadratic via invariant scaling) improvement in the ratio of weight update magnitude to quantization noise, compared to naive baselines such as learning-rate tuning?
- **Operational Efficiency and Compatibility**: Is the solution implemented as a one-time preprocessing step (or similarly lightweight operation) that introduces no additional computational overhead during training and requires no modifications to inference engine kernels or model architecture?

---

**Principle 3: Comprehensive Multi-Scale Validation Across Quantization Formats and Extended Reasoning Horizons**

**Definition:**  
Because the ultimate goal is practical deployment, research must provide rigorous empirical evidence across a spectrum of model scales (from sub-1B to 7B and beyond), multiple quantization precisions (INT8, FP8, and optionally more aggressive formats such as INT4), and diverse RL algorithms. Furthermore, reasoning-centric LLM training often involves multi-stage curricula with increasing context lengths and long-horizon rollouts. High-quality work must demonstrate that quantization-based acceleration maintains accuracy and stability under these demanding, extended conditions rather than only on small-scale benchmarks or short training runs.

**Core Evaluation Criteria:**
- **Scale and Precision Coverage**: Are end-to-end accuracy results reported on sufficiently large models (e.g., 7B parameters) in addition to smaller ones, and across multiple quantization formats (at least INT8 and FP8)?
- **Long-Context and Extended Training Rigor**: Does the evaluation include multi-stage training with increasing sequence lengths (e.g., 8k to 24k context windows) or agentic tasks, with training extending beyond 1,000 steps to verify that quantization artifacts do not accumulate catastrophically?
- **Hardware Throughput Verification**: Are rollout acceleration claims (e.g., 20–80% faster throughput) validated on modern accelerators (A100, H100) with realistic inference engines, and do larger models show commensurately greater benefits?

---

**Principle 4: Orthogonality Positioning and Minimal-Invasiveness Within Broader Efficiency Paradigms**

**Definition:**  
Quantization-based rollout acceleration exists within a larger ecosystem of efficiency techniques including pruning, distillation, and architectural modifications. High-quality contributions must clearly position quantization as uniquely suitable for the RL rollout phase—emphasizing its architectural neutrality, hardware compatibility, and negligible synchronization cost—while demonstrating minimal invasiveness to existing RL training frameworks. The method should integrate seamlessly with standard asynchronous or hybrid RL systems without requiring per-step calibration, re-distillation, or mask recomputation.

**Core Evaluation Criteria:**
- **Comparative Positioning**: Does the work provide a clear, technically grounded argument for why quantization is preferable to or orthogonal with alternatives (e.g., pruning requires dynamic mask updates, distillation requires expensive re-synchronization) for RL rollout acceleration?
- **Framework Integration Evidence**: Is there explicit evidence that the method integrates into existing RL frameworks (e.g., VeRL, vLLM-based pipelines) with minimal engineering overhead (e.g., single-digit lines of code or one-minute preprocessing)?
- **Algorithmic Generality**: Does the method generalize across multiple RL algorithms (e.g., PPO, GRPO, DAPO) without algorithm-specific re-engineering, indicating broad applicability within the RL-for-reasoning paradigm?

---

**Principle 5: Hyperparameter Robustness and Sensitivity Analysis in Quantized Actor Training Dynamics**

**Definition:**  
Because RL training is already notoriously sensitive to hyperparameters, introducing quantization into the loop must not compound this fragility. Any newly introduced hyperparameters—such as scaling factors for invariant quantization—must be thoroughly ablated and shown to be robust across tasks and algorithms. Ideally, core stabilization mechanisms should be hyperparameter-free, deriving their operating ranges directly from training dynamics (e.g., from observed policy ratios rather than hand-tuned bounds).

**Core Evaluation Criteria:**
- **Ablation and Sensitivity Rigor**: Are systematic ablation studies provided for any new hyperparameters (e.g., invariant scale factors), demonstrating stable performance across a reasonable range and comparing favorably against naive alternatives (e.g., simply increasing the learning rate)?
- **Hyperparameter-Free Design Preference**: For components central to training stability (e.g., clipping bounds), does the method avoid introducing additional tunable knobs by deriving thresholds directly from observable training quantities?
- **Cross-Task and Cross-Algorithm Transferability**: Is evidence provided that hyperparameter choices (or the absence thereof) transfer reliably across different RL algorithms and reasoning tasks, reducing rather than increasing the tuning burden on practitioners?