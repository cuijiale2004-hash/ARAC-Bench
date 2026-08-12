**Principle 1: Attribution Disentanglement and Teacher-Capacity Bias Control in Reasoning-Augmented Multimodal Embedding**

**Definition:**  
In frameworks that augment embedding models with generative reasoning traces produced by large teacher MLLMs, reviewers must require strict disentanglement of the proposed architectural mechanism from the confounding effect of teacher model scale. The raw performance improvement of a smaller embedder conditioned on teacher-generated traces is insufficient evidence of methodological novelty if the gains simply reflect the capacity of a much larger pretrained model. The evaluation must determine whether the improvement persists when the teacher is replaced by smaller models of comparable size to the embedder, and whether the reasoning traces themselves—rather than the teacher's parametric knowledge—are the active ingredient. This principle ensures that test-time scaling via external reasoners is distinguished from mere model ensemble or knowledge distillation effects. Without such attribution analysis, the community cannot assess whether the advance lies in the reasoning-before-embedding paradigm or in unrestricted access to a stronger pretrained model.

**Core Evaluation Criteria:**
- **Teacher Scale Ablation**: Does the work compare reasoning traces from teachers of varying capacities (e.g., 7B, 32B, 72B, proprietary APIs) while holding the embedder fixed, and does it show diminishing returns or consistent mechanism-driven gains?
- **Zero-Shot vs. Fine-Tuned Reasoner Isolation**: Are ablations provided for zero-shot teacher traces versus a distilled student reasoner of the same scale as the embedder, demonstrating that the embedder itself learns to leverage reasoning rather than relying on teacher parametric knowledge?
- **Mechanism vs. Capacity Control**: Does the paper include controls showing that direct inference with the teacher model (without the proposed embedding mechanism) does not achieve equivalent gains, proving the method adds value beyond raw teacher scale?

---

**Principle 2: Quantitative Robustness Verification of Embedder Performance Under Imperfect Intermediate Reasoning Traces**

**Definition:**  
When a paper claims that an embedder is robust to imperfect or noisy intermediate reasoning traces, the evaluation must demand quantitative experimental validation rather than qualitative assertions. Robustness is a distinct claim from average-case performance; it requires systematically degrading reasoning quality and measuring the embedder's sensitivity across a spectrum of trace fidelity. Reviewers should look for controlled experiments that inject noise, truncate reasoning steps, or use partially incorrect traces to establish performance boundaries and failure thresholds. This is particularly important because the quality of generative traces varies dramatically across tasks and models, and deployable systems must tolerate such variance without catastrophic degradation. Qualitative descriptions or single-point observations are inadequate for establishing the reliability of the reasoning-embedding pipeline. A rigorous study should report correlation metrics between reasoning quality and embedding accuracy, as well as worst-case analyses under adversarially corrupted traces.

**Core Evaluation Criteria:**
- **Noise Injection Experiments**: Does the work quantitatively measure retrieval performance as a function of reasoning trace quality, such as by artificially corrupting a fraction of CoT steps or using lower-capacity reasoners to generate noisy traces?
- **Correlation Analysis**: Is there a reported correlation (e.g., Pearson or rank correlation) between automated metrics of reasoning trace quality and final embedding accuracy across tasks or samples?
- **Stability Boundary Characterization**: Does the paper identify thresholds beyond which embedder performance degrades significantly, demonstrating that robustness is bounded rather than absolute?

---

**Principle 3: Systematic Empirical Justification for Hidden-State Extraction Layers and Pluggable Embedding Head Design**

**Definition:**  
The selection of hidden-state extraction layers and embedding head architecture in generative MLLMs must be empirically justified with systematic layer-wise and design ablations. Unlike encoder-only models, generative MLLMs optimize late-layer representations for next-token prediction, which may not align with the semantic similarity objectives required for retrieval. Simply defaulting to the last token or final layer is insufficient; the field requires rigorous evidence that the chosen extraction point maximizes representational quality for embedding tasks. Furthermore, when pluggable heads such as Q-Former, self-initialized attention, or latent poolers are introduced, their design must be validated against simpler baselines with matched parameter counts to isolate architectural benefits from pure capacity effects. This principle ensures that representational design choices are treated as first-class scientific claims rather than incidental implementation details. Authors must clearly articulate whether gradients from the embedding objective flow into the generative backbone, and how unified models prevent catastrophic interference between contrastive and autoregressive objectives.

**Core Evaluation Criteria:**
- **Layer-Wise Extraction Ablation**: Does the paper compare performance when embeddings are extracted from different depths (e.g., bottom-1, bottom-4, bottom-8, bottom-16 layers) under both frozen and fine-tuned backbone regimes?
- **Head Design Controlled Comparison**: Are embedding heads compared under controlled capacity (e.g., equal trainable parameters), and is the final design shown to outperform not only naive pooling but also alternative multi-layer architectures?
- **Gradient Flow Clarity**: Is it clearly articulated whether gradients from the embedding objective flow into the generative backbone, and how this is prevented or managed in unified models to avoid interference between generative and contrastive objectives?

---

**Principle 4: Pretraining Data Leakage Assessment and Test-Set Contamination Control in MLLM-Generated Reasoning Traces**

**Definition:**  
Any work that leverages off-the-shelf pretrained MLLMs to generate intermediate reasoning traces or embeddings must proactively address and empirically rule out test-set data leakage. State-of-the-art MLLMs are frequently trained on broad web corpora that include standard benchmark tasks, making it plausible that generated reasoning traces regurgitate memorized answers or target structures rather than performing true compositional reasoning. The evaluation must scrutinize whether reported improvements stem from genuine task understanding or from the teacher model's prior exposure to evaluation data. Authors should document the training data composition of the reasoner, cite technical reports confirming proper train-test splits, and provide out-of-distribution or leakage-controlled comparisons where feasible. This principle is essential for maintaining benchmark integrity and ensuring that advances in multimodal representation learning are not illusory artifacts of data contamination. Without such scrutiny, performance gains attributed to reasoning may actually reflect memorization.

**Core Evaluation Criteria:**
- **Training Data Transparency**: Does the paper disclose the pretraining corpus overlap with evaluation benchmarks and cite official technical reports or data sheets that confirm excluded test sets?
- **Leakage-Controlled Validation**: Are results reported on tasks or data splits demonstrably unseen by the teacher model, or are ablations provided comparing in-domain versus out-of-domain tasks to detect memorization effects?
- **Trace Content Audit**: Does the work audit generated reasoning traces for verbatim reproduction of target labels or benchmark-specific phrasing that would indicate memorization rather than reasoning?

---

**Principle 5: Cross-Backbone Generalization and Modality-Specific Benefit Validation Beyond Text-Only Retrieval**

**Definition:**  
Proposed embedding frameworks that leverage generative reasoning must demonstrate cross-backbone generalization and modality-specific necessity. If the method is framed as a multimodal advance, reviewers must require evidence that the gains are tied to alignment challenges unique to multimodal data—such as grounding visual regions or composing cross-modal instructions—and do not merely replicate improvements achievable in text-only retrieval. Additionally, the framework must be validated across multiple MLLM families to ensure that observed gains are not artifacts of a single backbone's idiosyncrasies or pretraining recipe. Generalization across architectures is a hallmark of principled methodological contribution as opposed to overfitted benchmark engineering. Authors should analyze which modalities or task types most benefit from explicit reasoning and explain why certain domains show larger gains. Comparative validation against text-only baselines or clarification of multimodal-specific bottlenecks is necessary to establish the work's scope and contribution.

**Core Evaluation Criteria:**
- **Multimodal vs. Unimodal Disambiguation**: Does the paper compare the proposed method against equivalent text-only retrieval baselines or clarify why multimodal compositional reasoning is the specific bottleneck being addressed, rather than generic query understanding?
- **Cross-Backbone Consistency**: Are experiments conducted on at least two distinct MLLM families (e.g., Qwen-VL, InternVL, LLaVA-OV), and do the relative improvements remain consistent, indicating method robustness rather than backbone-specific tuning?
- **Task-Modality Alignment Analysis**: Does the work analyze which modalities (image, video, visual document) or task types (VQA, grounding, moment retrieval) most benefit from explicit reasoning, and does it explain why certain modalities show larger gains?