**Principle 1: Mechanistic Attribution of Asymmetric Update Dynamics Caused by Cross-Modal Norm Discrepancy in Pre-Norm MLLMs**

**Definition:**  
The work must move beyond surface-level observation of performance differences to formally identify and theoretically characterize how Pre-Norm architectures induce divergent optimization dynamics between visual and textual tokens. This involves proving or rigorously arguing how high-norm visual tokens develop "representational inertia" that slows semantic evolution relative to text tokens, and whether this asymmetry causally degrades cross-modal attention and fusion. The theoretical framework should be grounded in the specific mechanics of Pre-Norm residual updates, norm accumulation in vision encoders, and the resulting geometric divergence in shared embedding spaces. Reviewers must assess whether the theory is internally consistent, whether its simplifying assumptions are explicitly acknowledged and justified, and whether the mechanistic narrative is falsifiable rather than post-hoc. A strong submission in this subfield must establish a clear causal chain from architectural choice to dynamic imbalance to functional failure, rather than merely correlating norm differences with performance gaps.

**Core Evaluation Criteria:**
- **Causal Clarity**: Does the work explicitly derive and empirically validate that norm disparity causes asymmetric update rates (e.g., via inter-layer cosine similarity, gradient norm analysis), rather than assuming or inferring it from final performance?
- **Theoretical Grounding**: Are the premises—such as uniform update magnitude, consistent expected angles, and symmetric orthogonal distributions—justified in the context of Pre-Norm transformers, and are their limitations discussed?
- **Discriminatory Power**: Does the theory distinguish between static norm mismatch and dynamic representational inertia, and does it explain why existing projector designs fail to fully resolve the issue?
- **Empirical Premise Validation**: Does the submission provide empirical distributions (e.g., of update magnitudes, angular velocities, or spectral energy) to substantiate the theoretical assumptions rather than relying solely on abstract proofs?

---

**Principle 2: Generalization and Architectural Coverage of Norm Alignment Interventions Across Diverse MLLM Paradigms**

**Definition:**  
Because the claimed norm discrepancy is posited as a fundamental artifact of Pre-Norm architectures, any proposed intervention must be evaluated across a sufficiently broad spectrum of model families, vision encoders, and fusion paradigms to establish universality. The subfield demands evidence that the phenomenon persists in disparate architectural settings—from simple MLP projectors to complex adapters and discrete visual vocabularies—and that the remedy remains effective and non-destructive across these variants. Reviewers should scrutinize whether the authors have avoided overfitting their analysis to a single model-encoder pair or an outdated backbone, and whether they have addressed the practical infeasibility of replicating proprietary large-scale pre-training by designing controlled, scientifically valid proxy experiments. Robust contributions must demonstrate that the diagnosis and cure are not idiosyncratic to a specific configuration but reflect a generalizable property of multimodal Pre-Norm systems. Furthermore, the evaluation should assess whether boundary conditions are clearly articulated.

**Core Evaluation Criteria:**
- **Backbone and Encoder Diversity**: Is the phenomenon replicated across multiple vision encoders (e.g., CLIP, SigLIP) and LLM backbones (e.g., Llama, Qwen), including contemporary instruction-tuned models?
- **Cross-Paradigm Consistency**: Does the analysis cover distinct architectural paradigms, such as continuous feature projection and discrete visual tokenization, and does it explain why the dynamic may or may not manifest in each?
- **Controlled Comparability**: When proprietary pipelines cannot be replicated, does the work construct controlled, from-scratch training setups that isolate the norm-alignment variable without introducing confounding recipe mismatches?
- **Ablation of Competing Designs**: Are comparisons made against existing projector designs that implicitly mitigate norm disparity, and is the relative efficacy of the minimal intervention properly contextualized?

---

**Principle 3: Disentanglement of Norm Magnitude Equalization from Learnable Rescaling Mechanisms in Minimal Fixes**

**Definition:**  
In evaluating minimal normalization-based remedies—such as inserting a single LayerNorm layer—the review must rigorously separate the contribution of pure norm-scale equalization from the effects of learnable affine parameters, specific initialization strategies, and gradient-flow engineering. The subfield requires clarity on whether performance gains stem primarily from reducing the static L2-norm gap, or from secondary mechanisms such as the re-centering/bias terms, dynamic gain adaptation, or backward gradient compensation. Reviewers should demand ablations that compare the full learnable LayerNorm against fixed scalar rescaling, parameter-free RMSNorm, and naive initialization versus principled initialization, ensuring that the proposed "simple" fix is not surreptitiously relying on non-obvious optimization tricks. Without this disentanglement, the work risks misattributing success to architectural simplicity while obscuring the true operative mechanism. Moreover, the analysis should explicitly examine whether the forward-pass normalization and backward-pass gradient compensation can be independently varied to confirm their respective contributions.

**Core Evaluation Criteria:**
- **Pure Rescaling vs. Learnable Normalization**: Does the work include ablations comparing fixed scalar rescaling (no trainable parameters) against the proposed LayerNorm with learnable gain/bias to isolate the source of improvement?
- **Normalization Variant Ablations**: Are alternative normalization forms tested (e.g., RMSNorm vs. LayerNorm) to verify whether re-centering or mean-subtraction operations independently contribute to the outcome?
- **Gradient Flow Isolation**: When extreme initialization is required for norm alignment, does the work rigorously demonstrate that gradient compensation is necessary and sufficient, and does it disentangle the forward alignment effect from the backward optimization stabilization?
- **Mechanism-Specific Claims**: Do the authors avoid overclaiming that "norm alignment alone" drives results when the empirical evidence suggests that initialization or gradient rescaling may be co-causal?

---

**Principle 4: Statistical and Diagnostic Rigor in Tracking Cross-Modal Representation Evolution and Attention Collapse**

**Definition:**  
Claims about internal dynamics—such as representational inertia, asymmetric angular velocity, and attention signal degradation—must be supported by robust, multidimensional diagnostics that go beyond aggregate benchmark scores. The subfield requires temporal and layer-wise analyses that track how visual and textual representations evolve throughout training and across depth, using metrics like inter-layer cosine similarity, L2-norm trajectories, gradient norms, and attention entropy. Reviewers must evaluate whether the evidence is statistically sound (e.g., multiple random seeds, variance reporting, confidence intervals) and whether visualizations and case studies are representative rather than cherry-picked. Furthermore, the work should directly link these internal diagnostics to downstream performance, demonstrating that changes in update symmetry or attention sharpness precede or predict benchmark improvements, thereby satisfying a mechanistic standard of proof rather than a purely correlational one. It is equally important that diagnostics cover both initial and converged states.

**Core Evaluation Criteria:**
- **Layer-Wise Dynamic Tracking**: Are token norms and update rates (e.g., cosine similarity between consecutive layers) reported across all layers and training stages, with clear pre/post-intervention comparisons?
- **Statistical Robustness**: Are experiments run with multiple seeds, and are standard deviations or confidence intervals reported for key metrics (e.g., norm measurements, benchmark scores)?
- **Attention Mechanism Evidence**: Does the work provide quantitative or qualitative attention analyses (e.g., entropy, cross-modal focus maps) that directly demonstrate reduced attention collapse after intervention?
- **Temporal Evolution**: Is the representational evolution tracked across training checkpoints to confirm that dynamic asymmetry is a persistent bottleneck rather than a transient initialization artifact?

---

**Principle 5: Holistic Capability Impact Beyond Multimodal Gains Including Text-Only and Downstream Transfer**

**Definition:**  
Because norm discrepancy is hypothesized to degrade the shared backbone's overall optimization landscape, effective interventions should ideally yield benefits that transcend the multimodal task domain and improve—or at least preserve—the model's core linguistic capabilities and general reasoning. Reviewers must assess whether the evaluation protocol includes text-only benchmarks and fine-grained sub-task analyses to detect whether norm alignment causes homogenization, information loss, or trade-offs between modalities. A submission that claims architectural improvement must show that correcting the cross-modal imbalance does not come at the cost of textual nuance or subtle visual feature discrimination. The presence of broad, holistic gains serves as stronger evidence that the intervention addresses a foundational optimization bias rather than a narrow modality-specific quirk. Consequently, reviewers should treat isolated multimodal improvements with caution unless they are accompanied by evidence of stable or enhanced unimodal performance and a plausible theoretical justification for such holistic effects.

**Core Evaluation Criteria:**
- **Unimodal Benchmark Preservation**: Does the evaluation include text-only tasks to verify that the intervention does not degrade linguistic capability, and are there statistically significant gains or at least non-inferiority margins?
- **Sub-Task Disaggregation**: Are multimodal benchmark results broken down by sub-capability (e.g., fine-grained perception, reasoning, OCR) to detect whether improvements are uniform or concentrated in specific areas?
- **Cross-Modal Trade-off Analysis**: Does the work explicitly test for potential homogenization of visual and textual features, ensuring that norm alignment enhances rather than flattens nuanced, modality-specific information?
- **Generalization Evidence**: Are there indications that the fix improves downstream transfer or emergent capabilities beyond the immediate training objectives, confirming that the intervention strengthens the foundational model rather than exploiting benchmark-specific loopholes?