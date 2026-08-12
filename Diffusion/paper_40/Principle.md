**Principle 1: End-to-End Joint Optimization versus Decoupled Two-Stage Training in Diffusion-Based Speech Tokenizers**

**Definition:**  
In diffusion-based speech tokenization, a critical evaluation axis is whether the proposed method achieves genuine end-to-end joint optimization of the encoder, quantizer, and diffusion decoder, as opposed to decoupled two-stage pipelines where representation learning is separated from generative reconstruction. Reviewers must assess whether the joint training produces qualitatively distinct representations that cannot be recovered by simply fine-tuning a diffusion decoder on top of pre-trained discrete tokens, and whether the quantizer itself benefits from the generative objective. The distinction between "using diffusion for reconstruction" and "jointly optimizing quantization with diffusion" is central to determining methodological novelty and effectiveness.

**Core Evaluation Criteria:**
- **Ablation Rigor**: Does the work include direct ablations comparing end-to-end diffusion training against two-stage alternatives (e.g., regression pre-training followed by diffusion decoder fine-tuning) under identical capacity and data regimes?
- **Representation Quality**: Is there evidence that discrete codes are optimized for both compression and reconstruction simultaneously, rather than forcing the diffusion decoder to adapt to suboptimal fixed latents?
- **Novelty Differentiation**: Does the work clearly distinguish its integrated formulation from prior two-stage diffusion tokenizers (e.g., CosyVoice, Vevo) and text-conditioned diffusion decoders (e.g., TaDiCodec)?
- **Causal Analysis**: Can the authors demonstrate that end-to-end training produces representations that are inherently superior, rather than attributing gains solely to decoder capacity or data scale?

---

**Principle 2: Semantic-Acoustic Disentanglement and Auxiliary Supervision Mechanisms in Ultra-Low-Bitrate Regimes**

**Definition:**  
At extreme compression rates (e.g., below 0.5 kbps or 12.5 Hz), speech tokenizers face a severe information bottleneck that risks collapsing to acoustically ambiguous outputs. The introduction of auxiliary semantic supervision (e.g., CTC loss, ASR distillation) must be evaluated not only for its impact on downstream understanding tasks but also for its mechanistic role in structuring the latent space—specifically whether it acts as a structural regularizer that enables better acoustic reconstruction rather than merely trading acoustic fidelity for semantic alignment. Reviewers should expect a principled explanation of how semantic guidance prevents collapse and improves speaker similarity despite apparent trade-offs.

**Core Evaluation Criteria:**
- **Mechanistic Clarity**: Does the work provide a clear explanation of why semantic supervision improves acoustic metrics (e.g., speaker similarity, WER) rather than degrading them, especially at aggressive compression?
- **Ablation Sensitivity**: Are there systematic ablations varying semantic loss weights to map the optimization landscape and identify catastrophic failure modes (e.g., complete loss of intelligibility without regularization)?
- **Comparative Analysis**: Does the work compare against alternative semantic enrichment strategies (self-supervised feature distillation, LLM-based supervision, external ASR encoders) to justify its chosen approach?
- **Disentanglement Evidence**: Is there empirical or analytical evidence that the latent space separates linguistic and speaker-specific features, or that semantic anchoring frees capacity for acoustic detail?

---

**Principle 3: Comprehensive Tri-Modal Evaluation Across Speech Reconstruction, Understanding, and Generation**

**Definition:**  
A speech tokenizer intended as a universal interface for speech language models must be evaluated on three distinct and non-substitutable capabilities: (1) high-fidelity reconstruction from discrete codes back to audio, (2) effectiveness as representations for downstream understanding tasks (ASR, emotion recognition, speaker verification, keyword spotting), and (3) suitability as a generative target for synthesis tasks like zero-shot text-to-speech. Research that omits any of these three evaluation modes is considered incomplete, as each mode tests different properties of the discrete representation. Furthermore, objective metrics must be complemented by subjective evaluations and comparisons against both reconstruction-optimized and understanding-optimized baselines.

**Core Evaluation Criteria:**
- **Generation Evaluation**: Does the work include direct downstream generation evaluations (e.g., zero-shot TTS) with both objective metrics (WER, SIM, RTF) and subjective metrics (CMOS, MOS) with statistical significance tests?
- **Baseline Diversity**: Are comparisons drawn against tokenizers specialized for each mode (e.g., waveform-based codecs for reconstruction, S3Tokenizer/GLM4-Voice for understanding, CosyVoice for generation)?
- **Metric Completeness**: Beyond WER and SIM, does the evaluation include perceptual quality metrics (UTMOS), signal-level metrics (PESQ, STOI where relevant), and subjective listening tests?
- **Qualitative Evidence**: Are demo pages or qualitative samples provided to substantiate claims about reconstruction fidelity, particularly for expressive or in-the-wild speech?

---

**Principle 4: Real-Time Computational Efficiency and Streaming Feasibility of Iterative Diffusion Decoding**

**Definition:**  
While diffusion-based decoders offer high reconstruction quality, their iterative nature introduces inherent latency challenges that must be rigorously quantified for practical deployment in speech language modeling. Evaluation must extend beyond quality metrics to include wall-clock efficiency measurements (real-time factor, inference latency), acceleration strategies (shortcut fine-tuning, lightweight diffusion heads, few-step sampling), and architectural constraints on streaming inference (causal vs. non-causal decoder design). Claims of efficiency must be backed by actual runtime measurements against single-pass deterministic baselines, not merely theoretical step counts.

**Core Evaluation Criteria:**
- **Wall-Clock Efficiency**: Are RTF and latency measurements provided at varying diffusion step counts (e.g., 2, 4, 8, 16 steps) with explicit hardware specifications and averaging protocols?
- **Acceleration Validation**: Do shortcut fine-tuning or lightweight head designs maintain reconstruction quality at extreme low-step regimes, and are they compared against standard iterative decoding?
- **Streaming Analysis**: Does the work address streaming limitations, including encoder causality, decoder look-ahead requirements, and the feasibility of chunk-wise or autoregressive diffusion for low-latency applications?
- **Practical Cost**: Is the computational and memory cost analyzed relative to sequence length, demonstrating advantages (or disadvantages) of low token rates (e.g., 12.5 Hz vs. 25–50 Hz) in downstream language modeling?

---

**Principle 5: Systematic Scalability Analysis and Optimal Capacity-Dimension Trade-offs in Speech Tokenization**

**Definition:**  
Given claims of "scaling" speech tokenizers, reviewers must evaluate whether the paper provides a systematic analysis of how model capacity, codebook configuration (size, dimension, number of codebooks), and training data volume interact under extreme compression. This includes identifying whether performance scales monotonically with parameters or exhibits nuanced trade-offs where excessive capacity prioritizes fine-grained acoustic detail over abstract semantic features critical for understanding. The analysis should yield actionable insights about optimal operating points rather than simply demonstrating that "bigger models exist."

**Core Evaluation Criteria:**
- **Scaling Curves**: Are there experiments across multiple model sizes showing both reconstruction and understanding metrics, revealing whether scaling benefits all tasks uniformly or creates fidelity-vs-abstraction trade-offs?
- **Non-Monotonic Behavior**: Does the work identify and explain cases where larger models degrade on understanding tasks (e.g., ASR, SV) while improving reconstruction, and does it recommend an optimal configuration?
- **Quantization Ablations**: Are codebook design choices systematically ablated, including single vs. residual VQ, codebook size/dimension sensitivity, and comparisons with alternative quantization methods (FSQ, BSQ)?
- **Data Scaling Evidence**: Is there analysis distinguishing between acoustic learning (which may saturate with moderate data) and semantic representation learning (which may benefit more from large-scale training)?