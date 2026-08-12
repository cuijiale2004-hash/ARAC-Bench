
**Principle 1: Mechanistic Decomposition and Rank Analysis of Attention Weights into Sparse-High-Rank and Dense-Low-Rank Components**

**Definition:**  
The work must provide rigorous empirical and theoretical justification for its central claim that full attention weights can be decomposed into a small fraction of large, high-rank values and a large fraction of small, extremely low-rank values. This decomposition is the foundational premise justifying the hybrid use of sparse and linear attention. Reviewers must assess whether the authors have moved beyond superficial visualization to provide quantitative evidence—such as stable rank measurements, singular value decomposition analysis across multiple layers and heads, and statistical distributions sampled from diverse sequence lengths and model checkpoints—that this structure is inherent to diffusion transformers rather than an artifact of specific initialization or input prompts. The principle demands that the observation be robust, reproducible, and sufficiently characterized to motivate the subsequent architectural design.

**Core Evaluation Criteria:**
- **Quantitative Robustness of Decomposition:** Does the paper provide systematic rank analysis (e.g., stable rank, SVD) across multiple layers, heads, and timesteps to prove the high-rank/low-rank split? Is the threshold (e.g., top 8% vs. bottom 92%) justified by data rather than arbitrarily chosen?
- **Causal Link to Method Design:** Does the paper establish a clear causal chain from the observed decomposition to the design choice of applying sparse attention to the high-rank component and linear attention to the low-rank component? Are alternative decompositions or thresholding strategies considered and rejected based on evidence?
- **Generalization of Observation:** Is the attention weight structure validated on multiple model architectures (e.g., Wan2.1, LightningDiT, MM-DiT) and sequence lengths, ensuring the finding is not limited to a single experimental setup?
- **Distinction from Prior Art:** Does the work clearly differentiate its observation from prior sparsity or low-rank assumptions in attention, explaining why previous methods failed to exploit this specific joint structure?

---

**Principle 2: Principled Integration and Differentiable Routing of Hybrid Sparse-Linear Attention Kernels**

**Definition:**  
The paper proposes a novel hybrid attention mechanism that routes different blocks of attention weights to sparse FlashAttention, linear attention, or skipping based on a compressed prediction mask. Reviewers must evaluate whether this routing mechanism is algorithmically sound, computationally coherent, and implementable as a unified GPU kernel. The integration must not be a naive concatenation of two existing methods; it should address the fundamental incompatibility between softmax-based sparse attention and feature-map-based linear attention, including distribution mismatch, normalization consistency, and gradient flow. The principle emphasizes that the fusion strategy must be mathematically justified and empirically shown to be trainable end-to-end without destabilizing the pre-trained diffusion model.

**Core Evaluation Criteria:**
- **Algorithmic Coherence of Fusion:** Is the combination of sparse and linear attention mathematically principled? Does the paper adequately address how the outputs from two fundamentally different attention formulations (softmax vs. feature-map) are normalized and combined? Is the learnable projection sufficiently motivated?
- **Efficiency of Routing Overhead:** Does the compressed mask prediction (pooling + softmax) introduce negligible overhead compared to the attention computation saved? Is the block-level classification robust to prediction errors, and does the paper analyze misprediction costs?
- **Kernel Fusion and Implementation Soundness:** Is the forward and backward pass truly fused into a single GPU kernel? Does the paper provide evidence (e.g., memory bandwidth analysis, kernel profiling) that the fused implementation achieves the theoretical speedup, or are there hidden synchronization costs?
- **Trainability and Gradient Stability:** Does the hybrid mechanism support stable backpropagation? Are there analyses or safeguards against gradient mismatch between the sparse and linear branches?

---

**Principle 3: Trainability, Backward Pass Efficiency, and Fine-Tuning Convergence of Approximate Attention**

**Definition:**  
A critical differentiator for efficient attention methods in the diffusion transformer era is whether they are trainable end-to-end, not merely inference-time approximations. This principle evaluates whether the proposed method supports full forward and backward passes, how much fine-tuning is required to recover or maintain generation quality, and whether the fine-tuning cost is practical relative to pre-training. Reviewers must scrutinize the convergence dynamics: whether quality improves monotonically with fine-tuning steps, whether zero-shot application fails catastrophically (indicating poor compatibility with pre-trained weights), and whether efficient alternatives like LoRA are viable. The principle acknowledges that inference-only speedups are valuable but demands higher scrutiny for methods claiming trainability.

**Core Evaluation Criteria:**
- **Zero-Shot vs. Fine-Tuning Analysis:** Does the paper clearly report zero-shot performance degradation? Is the convergence trajectory (e.g., 250 steps, 1000 steps, 2000 steps) documented with sufficient granularity to justify the claimed "few steps"?
- **Fine-Tuning Cost Proportionality:** Is the fine-tuning compute budget (e.g., 2000 steps × batch size 64) contextualized as a small fraction of pre-training cost? Are the hardware requirements and wall-clock time reported transparently?
- **Backward Pass Efficiency:** Is the backward pass algorithm derived correctly and implemented efficiently? Does the backward kernel achieve proportional speedup to the forward kernel, or does gradient recomputation become a new bottleneck?
- **Feasibility of Parameter-Efficient Fine-Tuning:** Does the paper explore parameter-efficient alternatives (e.g., LoRA) and honestly report their limitations? If full fine-tuning is required, is the justification based on ablation evidence rather than convenience?

---

**Principle 4: Quality-Preserving Validation at Extreme Sparsity for Long-Sequence Video Diffusion**

**Definition:**  
In video diffusion, attention approximations must be evaluated not merely by FLOP reduction but by rigorous, multi-dimensional quality preservation at extreme sparsity levels (e.g., ≥90%). This principle demands that the experimental validation goes beyond single-metric reporting to encompass perceptual quality, temporal consistency, semantic alignment, and human preference. Reviewers must assess whether the baselines are state-of-the-art and fairly configured, whether ablation studies isolate the contribution of each component (sparse vs. linear vs. fusion), and whether the quality metrics are interpreted with appropriate statistical context. The principle specifically targets the subfield's tendency to report sparsity without verifying that the remaining computation still captures long-range temporal dependencies critical to video coherence.

**Core Evaluation Criteria:**
- **Multi-Dimensional Quality Metrics:** Does the evaluation include diverse metrics covering imaging quality (IQ), temporal consistency (OC/SC), aesthetic quality (AQ), and human preference (VR)? Are these metrics standard in the video generation community?
- **Fair Baseline Configuration:** Are baseline methods (sparse-only, linear-only, VSA, VMoBa, SpargeAttn) implemented or configured at their optimal settings? Are comparisons made at matched sparsity levels or matched computational budgets?
- **Component Ablation Rigor:** Are ablation studies comprehensive? Specifically, do they include: (1) sparse-only at comparable sparsity, (2) linear-only, (3) naive sum of both (S+L), and (4) variations in critical hyperparameters (kh, kl, activation functions)?
- **Visual Evidence and Failure Analysis:** Do qualitative results (video frames) support quantitative metrics? Are failure cases or degradation patterns (e.g., object distortion, temporal flickering) analyzed and attributed to specific components?

---

**Principle 5: Hardware-Efficient Kernel Implementation and End-to-End Deployment Feasibility**

**Definition:**  
The ultimate value of an efficient attention method lies in its real-world deployment impact, which requires hardware-aware kernel optimization rather than just theoretical complexity reduction. This principle evaluates whether the authors have bridged the gap between algorithmic innovation and practical acceleration by implementing custom GPU kernels, profiling on multiple hardware generations, and reporting end-to-end (not just kernel-level) speedups. Reviewers must verify that the reported speedups account for all overheads (mask prediction, memory movement, kernel launch), that the method generalizes across different GPU architectures (consumer vs. datacenter), and that memory bandwidth is not a hidden bottleneck. The principle rejects papers that claim asymptotic improvements without demonstrating wall-clock time reduction in realistic generation pipelines.

**Core Evaluation Criteria:**
- **Realized vs. Theoretical Speedup:** Is the gap between theoretical FLOP reduction (e.g., 20×) and actual kernel speedup (e.g., 13.7×) honestly analyzed? Are overheads such as mask prediction, block classification, and fused kernel scheduling quantified?
- **Cross-Hardware Generalization:** Is the kernel evaluated on multiple GPU types (e.g., RTX 4090/5090, A800, H800, L20)? Do speedups remain consistent, or are they artifacts of a specific memory hierarchy or tensor core generation?
- **End-to-End Pipeline Impact:** Does the attention acceleration translate to meaningful end-to-end generation speedup (e.g., 2.2× on Wan2.1)? Is the attention module still the bottleneck after acceleration, or have other components (e.g., feed-forward networks, data loading) become dominant?
- **Memory Footprint and Bandwidth Analysis:** Are memory usage, L2/DRAM bandwidth utilization, and compute utilization (MFU) reported? Does the method avoid introducing memory bottlenecks that could negate compute savings in batched or long-sequence scenarios?