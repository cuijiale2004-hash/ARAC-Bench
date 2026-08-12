1.  **Research Background and Existing Pain Points**

Video Diffusion Models (VDMs) trained on internet-scale video data are poised to become the next transformative leap in artificial intelligence. While GAN-based approaches previously drove video generation, they suffered from training instability and lacked scalability to larger datasets. Diffusion Models introduced a stable training objective enabling scaling to large datasets, but at the expense of significantly higher inference complexity. Unlike GANs, which generate samples in a single forward pass, diffusion models require numerically integrating a learned probability-flow ODE/SDE through multiple iterative steps, resulting in substantial computational overhead.

The existing pain points are:
1.  **Accessibility and Infrastructure**: Heavy computational demands of frontier open-source VDMs limit accessibility, forcing reliance on cloud infrastructure which introduces latency, privacy, and cost concerns, creating barriers for creators in low-connectivity or resource-constrained settings.
2.  **Mobile Hardware Constraints**: Enabling on-device generation is technically challenging due to tight compute, memory, bandwidth, and thermal/power budgets on handheld devices. Specifically, the Qualcomm Hexagon NPU (target platform) has severe memory constraints.
3.  **Memory Bottleneck of Text Encoders**: The T5XXL Text-Encoder used in baseline models like Pyramidal-Flow has a model footprint of 4.726 billion parameters, which exceeds the mobile device’s memory budget, making on-device execution infeasible.
4.  **Memory Bottleneck of Decoders**: The decoder in Pyramidal-Flow requires storing large 4D feature-map buffers during its forward computation graph, making it impossible to fit even a single forward pass for the target resolution on the NPU.
5.  **Latency Bottleneck of Denoisers**: The spatio-temporally pyramidal and causal latent generation performed by the MMDiT denoiser is computationally expensive. Initial measurements showed approximately 184.2 seconds of latency to produce a 2-second video, far from the goal of practical interactive generation.

2.  **Core Research Motivation and Scientific Questions**

**Core Research Motivation**: To democratize access to high-quality video synthesis by allowing generation directly on mobile hardware. Mobile video generation is critical for privacy-preserving, always-available creative workflows and equitable access in bandwidth or cost-constrained regions. The motivation is to bridge the gap between frontier open-source VDMs and practical on-device execution by systematically reducing model and runtime complexity while maintaining fidelity.

**Scientific Questions**:
1.  Is the full capacity of the large T5XXL Text-Encoder actually necessary for high-quality text-to-video generation? Can a much smaller encoder be substituted without perceptible fidelity loss?
2.  Are the video latent spaces across different models easily transferable through lightweight fine-tuning?
3.  How can we reconcile disparities in latent compression factors among different pre-trained codec-latent-VAEs to replace the native decoder with an efficient one?
4.  How can we remove entire blocks from the MMDiT denoiser based on relative importance and recover original performance through distillation?
5.  How can we reduce the diffusion sampling cost and Neural Functional Evaluations (NFEs) by adapting Distribution Matching Distillation (DMD) to the Pyramidal Flow-Matching objective?

3.  **Overall Core Idea and Design Philosophy**

**Overall Core Idea**: Transform the Pyramidal-Flow pipeline into Neodragon, an efficient on-device text-to-video system, through four systematic optimisations: Text-Encoder Distillation, Asymmetric Decoder Distillation, Block Pruning, and Step Distillation. The system runs with a Peak-Memory usage of approximately 3.5GB, enabling execution on low-power NPUs.

**Design Philosophy**: The approach mirrors the paradox of the "Ship of Theseus": when every part of the system changes, does the system remain the same? The design systematically refines or replaces every component of the pipeline (Text Encoder, Decoder, Denoiser, Scheduler) to ensure mobile compatibility while preserving the essence of the original generative capacity. The philosophy relies on the hypotheses that large encoders are under-utilised for short prompts, compressive video latent spaces are universal and transferable, redundant computation exists in denoiser blocks, and the pyramidal flow-matching objective can be accelerated via distribution matching.

4.  **Core Innovation Points**

(A) **Text-Encoder Distillation Framework**: A novel distillation framework that compresses the 4.762B-parameter T5XXL model by 35x into a lightweight 0.130B-parameter DistilT5 (DT5) using a newly trained ContextAdapter (CA) module. This requires no image or video data, utilizing only a prompt-only distillation setup with a combination of MSE and Cosine Distance losses. It supports multiple modes including Replace Mode, Extend Mode, LoRA Mode, and Trainable-DT5 Mode.

(B) **Asymmetric Decoder Distillation**: An approach allowing the replacement of the native codec-latent-VAE decoder with a more efficient architecture from a different model without disturbing the generative latent-space of the video generation pipeline. It introduces asymmetry by retaining the original encoder while replacing the decoder, minimally adapting the architecture to match compression factors, and fine-tuning end-to-end.

(C) **MMDiT Block Pruning Strategy**: A strategy to remove entire blocks from the MMDiT denoiser based on their relative importance (calculated via Cosine Distance of input/output tokens and visual impact). Performance is recovered through a two-stage distillation process: Stage-1 Finetuning with ground-truth data and Stage-2 Finetuning with feature-matching losses against the Full Teacher model.

(D) **Extended Pyramidal DMD for Step Distillation**: A novel extended version of Distribution Matching Distillation (DMD) adapted for the Pyramidal Flow-Matching objective. This reduces the number of NFEs from 480 to 21 (greater than 95% reduction) without affecting the VBench score, utilizing a Student model, a Teacher Score-Model, and a Fake-Score-Model trained on student-predicted distributions.

5.  **Overview of the Overall Technical Solution**

The overall technical solution integrates the four innovations into an end-to-end mobile VDM system. Starting with the Pyramidal-Flow pipeline:
1.  **Text Encoding**: The large T5XXL is replaced with the distilled DT5 and a 4-layer MLP-based ContextAdapter (CA) operating in Replace Mode [RM]. The text prompt is processed by DT5, and the output is mapped to multi-modal conditioning tokens via the CA. A pre-trained CLIP model also provides embeddings.
2.  **First Frame Generation**: To mitigate semantic artifacts from step distillation in the initial frame, the system uses SSD-1B (a text-to-image model) to generate a high-quality first frame in four steps. This frame is encoded using the VAE Encoder.
3.  **Denoising Backbone**: The 24-block MMDiT is pruned to 18 blocks. The denoiser uses Causal Attention and a Spatio-temporal Pyramid to generate subsequent latent frames autoregressively. The scheduler uses Pyramidal DMD with a 1-1-1 inference schedule (1 step per stage per resolution) to drastically reduce NFEs.
4.  **Decoding**: The generated latent video is decoded using the modified TinyAEHV VAE Decoder (obtained via Asymmetric Decoder Distillation), which replaces the native high-memory decoder.
5.  **Post-Processing**: QuickSRNet performs 2x super-resolution to upscale the output to 640x1024 resolution.

6.  **Detailed Module Design**

**Module 1: Text-Encoder Distillation**
*   **Components**: Original T5XXL (frozen), CE (ContextEmbedder from MMDiT, frozen linear layer), DT5 (DistilT5, smaller encoder), CA (ContextAdapter, 4-layer MLP with skip connections, trainable).
*   **Objective**: Distill only the components of text understanding relevant to video synthesis. Match the ground-truth tokens from CE(T5XXL(prompt)) using CA(DT5(prompt)).
*   **Modes**:
    *   [RM] Replace Mode: CA substitutes CE entirely.
    *   [EM] Extend Mode: CA complements CE; both are cascaded during inference.
    *   [LORA] LoRA Mode: MLP-based CA is replaced by LoRA adapter layers on top of DT5. Only LoRA layers are trainable.
    *   [TDT5] Trainable DT5 Mode: DT5 model weights are updated in addition to the CA.
*   **Training**: Uses ~1.4M text-prompts. Loss weights: wmse = 1.0, wcd = 0.1.

**Module 2: Asymmetric Decoder Distillation**
*   **Components**: Frozen encoder E_enc, New decoder F_dec (from other models like TinyAEHV, Cosmos, LTXVideo, Wan).
*   **Architectural Adaptation**: Minimally adapt F_dec to match the fixed encoder’s compression factor of [8x8x8].
    *   *TinyAEHV*: Modify the first TGrow layer to perform 2x temporal upsampling instead of 1x.
    *   *Cosmos*: Use Continuous Tokens variant [8x8x8], no changes.
    *   *LTXVideo*: Remove unpatchification layer, update conv_out, replace conv_in for 16-dimensional latents.
    *   *Wan*: Modify first upsampling block for 2x spatio-temporal upsampling instead of 2x spatial-only.
*   **Training**: End-to-end using reconstruction objective. Encoder remains frozen to preserve latent space; KL regularizer is omitted. Dataset: ~350K videos.

**Module 3: MMDiT Block Pruning**
*   **Analysis**: Compute Block Importance (BI) scores for visual and textual tokens separately using a calibration set of 100 text prompts (5 samples each). Calculate Cosine Distance between input and output tokens. Select blocks for pruning based on BI scores and visual impact (specifically 6 blocks removed, reducing 24 to 18).
*   **Stage-1 Finetuning**: Train the pruned model with the original Flow-Matching objective. Condition on past frames sampled from ground truth videos, corrupted with Gaussian noise. Converges in ~300 iterations.
*   **Stage-2 Finetuning**: Incorporate feature-matching losses between the Full Teacher model (24 blocks) and the pruned Student model (18 blocks).
    *   *Losses*: MSE on visual tokens, Cosine Distance on textual tokens, Flow-Matching losses (using Teacher outputs and ground-truth flow).
    *   *Block Mapping*: Simple Block Mapping (output of student block k matches output of teacher block k) vs. Next-Block Mapping (output of student block k matches input of next available teacher block). Simple Block Mapping used for final deployment.

**Module 4: Step Distillation (Pyramidal DMD)**
*   **Components**: Student model D_θ, Teacher Score-Model D (frozen), Fake-Score-Model D_φ.
*   **Mechanism**: The Student predicts the clean latent in a single-step Euler solver. The Fake-Score-Model is trained with the Pyramidal Flow-Matching objective on the distribution of student-predicted clean latents. The Student is updated using the DMD loss gradient derived from the difference between the Teacher and Fake-Score-Model.
*   **Training**: Alternate updates: one update of θ per two updates of φ. For Student updates, limit local noise levels τ_local^i to four evenly selected values. For Fake model, sample τ_local^i from U(0,1). Use supervised Cauchy loss L_teacher with weight 0.5.

7.  **All Mathematical Formulas and Symbol Definitions**

**Text-Encoder Distillation**:
Ldistil(t, t̂) := wmse∥∥t− t̂∥∥2_2 + wcd(1− t.t̂/|t|.|t̂|)  (1)
where, t = CE (T5XXL(prompt)) and, t̂ = CA(DT5(prompt))  (2)

**MMDiT Block Importance**:
(zk+1 , t̂k+1) = Dk(zk , t̂k , ĉ)  (3)
BI vk := 1− E[zk.zk+1/∥zk∥2∥zk+1∥2] and BI tk := 1− E[t̂k.t̂k+1/∥t̂k∥2∥t̂k+1∥2]  (4)
BI k := (BI vk, BI t k)  (5)

**Pyramidal Flow-Matching**:
z̃σi start := (1− σi start) Up(Down(z, 2i+1), 2) + σi start ϵ,  (6)
z̃σi end := (1− σi end) Down(z, 2i) + σi end ϵ.  (7)
Global noise-level σ = (1 − σi local)σi end + σi localσi start.

**Pyramidal DMD**:
Student prediction: z̃θ := z̃σ − (σ/(σi start − σi end))Dθ(z̃σ, σ)
ỹσi start := (1− σi start) Up(Down(z̃θ, 2), 2) + σi start ε,  (8)
ỹσi end := (1− σi end) z̃θ + σi end ε,  (9)
z̃τ := (1− τ ilocal)ỹσi end + τ ilocalỹσi start,  (10)
where ε ∼ N (0, I) and τ = (1−τ ilocal)σi end+τ ilocalσi start.
DMD Loss Gradient: ∇θLi DMD ∝ (D(z̃τ , τ)−Dφ(z̃τ , τ)) · ∇θz̃θ.
Sample-specific weight: ∥∥D(z̃τ , τ)−(ỹσi start − ỹσi end)∥∥−1 1.
Supervised Cauchy loss: Lteacher = log(1 + ∥z̃θ −Down(z, 2i)∥2_2)

**Complexity Analysis**:
Bidirectional Attention: Cbi = (hwt)2 (C1)
Causal Attention: Ccausal = ∑t k=1 (h · w) × (h · w · k) = (hw)2 · t(t+ 1)/2 (C2-C5)
Speedupcausal = Cbi/Ccausal ≈ 2× as t → ∞ (C6, C7)
Temporally Pyramidal Causal: Cpyr(t, S) = M2[t + t A(S) − D(S) + u(u+ 1)/2 · 4S−1] (C8)
Speeduppyr(S) → 2 · 4S−1. For S=3, Speedup = 32× (C9, C10)
Spatial Speedup: Speedupspatial(p) = 1/βS(p), where βS(p) := ∑S−1 i=0 pi/16 i.
Combined Speedup: Speedupcombined ≈ 32× 2.81 ≈ 90×.

**Upsampling in Pyramidal Flow**:
z̃σi start = (2− σi start)/2 Up(z̃σi+1 end , 2) + σi start/sqrt(3)/2 ϵ (D5)
ϵ′ ∈ N (0,Σ′) and Σ′ block = [[1, γ, γ, γ], [γ, 1, γ, γ], [γ, γ, 1, γ], [γ, γ, γ, 1]] (D6)
γ = −1/3.

**Step Distillation Objectives (Appendix H)**:
Pyramidal Mean-Flows Loss: LMF = Eσ,β,z,ϵ[∥D(z̃σ, β, σ)− vmean(z̃σ, β, σ)∥2_2].
Mean velocity target: vmean-pyr := v(z̃σ, σ)− (σ − β)(v(z̃σ, σ)∂zD + (σi start − σi end)∂σD)
Pyramidal Progressive Distillation Loss: Lpyr-prog := Eσ,z,ϵ[∥z̃σis+j stud − z̃σis+j teach∥2_2]
Pyramidal Adversarial Loss: Lpyr-adv := wreconLpyr-prog + wadvLGAN(z̃σis+j stud , z̃σis+j teach)

8.  **Algorithm Pseudocode**

The paper does not contain explicit algorithm pseudocode blocks. However, the iterative update rules for Pyramidal DMD (Section 3.4) are described as follows:

**Pyramidal DMD Training Loop:**
1. Sample input z̃σ using local noise level σi_local.
2. Compute Student prediction of clean latent z̃θ.
3. Construct noisy versions ỹσi_start, ỹσi_end, and z̃τ for Teacher and Fake model input.
4. Alternate Updates:
   - Update Fake-Score-Model D_φ twice:
     - Sample τi_local from U(0,1).
     - Train with Pyramidal Flow-Matching objective Lpyr-FM on student-predicted clean latents distribution.
   - Update Student D_θ once:
     - Limit set of local noise levels τi_local to four evenly selected values.
     - Compute gradient ∇θLi_DMD ∝ (D(z̃τ, τ)−Dφ(z̃τ, τ)) · ∇θz̃θ.
     - Apply sample-specific weight ∥D(z̃τ, τ)−(ỹσi_start − ỹσi_end)∥^−1_1.
     - Apply supervised Cauchy loss L_teacher = log(1 + ∥z̃θ − Down(z, 2i)∥2_2) with weight 0.5.

**Stage-2 Block Pruning Finetuning:**
1. For each present block k in Student (1 to 18):
   - Compute MSE loss on visual tokens between Student output and Teacher output (using Simple Block Mapping).
   - Compute Cosine Distance loss on textual tokens.
   - Compute Flow-Matching losses using Teacher model's output.
   - Compute Flow-Matching losses using ground-truth flow from data.
2. Backpropagate aggregate loss to update Student.