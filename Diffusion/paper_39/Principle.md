**Principle 1: Perceptual Alignment and Metric Robustness in Continuous Aesthetic Attribute Quantification**

**Definition:**
For frameworks that map continuous scalar values to aesthetic attributes, the choice of quantification metrics must be rigorously justified against human perceptual standards rather than relying solely on computationally convenient proxies. Reviewers must assess whether the proposed metrics—such as HSV Value for brightness, Shannon entropy for detail, or CLIP similarity for realism and safety—genuinely capture the intended perceptual dimensions or introduce systematic artifacts. These artifacts may include conflating illumination with object color, equating entropy with structural complexity, or producing unrealistic outputs at extreme values. The framework should explicitly acknowledge inherent limitations of proxy metrics, demonstrate awareness of metric-induced trade-offs, and ideally provide comparative validation against alternative quantification strategies. This principle is crucial because the entire control pipeline depends on the foundational assumption that normalized scalar values correspond to meaningful, human-interpretable intensity gradations.

**Core Evaluation Criteria:**
- **Metric Validity:** Are the chosen quantification metrics perceptually grounded? Is there evidence (human studies or comparative analysis) that they outperform alternative metrics such as spectral power or Laplacian variance?
- **Artifact Awareness:** Does the work explicitly discuss and analyze metric-induced artifacts (e.g., semantic shifts under extreme intensity values, unnatural object scaling for low-detail controls)?
- **Terminological Consistency:** Are attribute definitions consistent throughout the paper (e.g., "semantic" versus "aesthetic"), and do they align with established conventions in image processing and aesthetic assessment?
- **Upgrade Path:** Does the framework explicitly allow for substitution of proxy metrics with more perceptually aligned estimators as they become available?

---

**Principle 2: Modular Architecture Design and Cross-Backbone Generalizability for Plug-and-Play Control**

**Definition:**
A core evaluation criterion for lightweight control adapters is whether the proposed method achieves genuine modularity—maintaining the base diffusion model frozen while enabling seamless integration across diverse architectures—without requiring retraining or architectural modification of the backbone. The method must demonstrate that its control mechanism, such as value encoders producing token sequences, is not tightly coupled to a specific model family (e.g., FLUX DiT) but can transfer to other widely adopted backbones (e.g., Stable Diffusion 1.5, SDXL, SD3.5) with minimal adaptation. Reviewers should assess whether the independence of attribute encoders enables compositional multi-attribute control at inference time, and whether the computational overhead remains negligible relative to the base model. This principle distinguishes truly generalizable frameworks from methods that are merely demonstrated on a single architecture with limited transferability.

**Core Evaluation Criteria:**
- **Architectural Agnosticism:** Is the method validated on multiple diffusion backbones with different underlying architectures (DiT, U-Net) and prompt embedding dimensions?
- **Compositional Multi-Attribute Control:** Can independently trained single-attribute encoders be combined at inference time without joint training, shared datasets, or optimization?
- **Computational Efficiency:** Does the adapter introduce negligible overhead (e.g., lightweight encoders with ~18M parameters) compared to the base model, ensuring practical deployment?
- **Integration Compatibility:** Does the framework seamlessly compose with existing conditional generation pipelines (e.g., ControlNet, Eligen) without disrupting their structural or semantic control functionality?

---

**Principle 3: Comprehensive Experimental Validation with Diverse Baselines and Human Perceptual Assessment**

**Definition:**
Claims of superior continuous attribute control must be supported by a multi-faceted experimental regime that extends far beyond single-metric leaderboard rankings. Reviewers must evaluate whether the paper provides strong quantitative baselines including oracle prompt-tuning strategies and interpolation-based methods, rigorous ablation studies isolating key design choices, and user studies verifying that human observers perceive the control as smooth, accurate, and high-quality. The experimental design should cover both single-attribute and multi-attribute scenarios, include safety-related evaluations with appropriate benchmarks, and honestly present failure cases. This principle ensures that reported gains reflect genuine methodological advances rather than evaluation artifacts or cherry-picked examples.

**Core Evaluation Criteria:**
- **Baseline Diversity:** Are comparisons made against both naive strategies (literal prompt engineering with percentage values) and sophisticated alternatives (attention interpolation, CLIP-guided latent interpolation, weighted embeddings)?
- **Ablation Rigor:** Are key architectural and training choices systematically ablated with quantitative impact analysis (e.g., token sequence length, positional embeddings, normalization strategies)?
- **Human Evaluation:** Are user studies conducted under controlled, double-blind conditions with sufficient participants and comparison instances to validate perceptual quality and control smoothness?
- **Failure Analysis:** Does the paper honestly present failure cases, boundary conditions, and limitations rather than selectively showcasing only successful examples?

---

**Principle 4: Semantic Fidelity and Content Preservation Under Continuous Attribute Manipulation**

**Definition:**
A critical distinction between effective attribute control and mere image filtering is whether the method preserves core semantic content—such as object identity, scene composition, and structural integrity—while modulating the target attribute intensity. Reviewers must assess whether the framework maintains semantic consistency across the full [0,1] intensity range, or whether extreme values induce undesirable content shifts (e.g., changing a cow's color when adjusting brightness, or shrinking objects when reducing detail). The evaluation should examine whether the method prioritizes semantic coherence over rigid metric adherence when conflicts arise, and whether this trade-off is explicitly acknowledged and quantified. This principle is essential because users expect attribute control to modify style or quality dimensions without altering the underlying subject matter or scene semantics.

**Core Evaluation Criteria:**
- **Content Consistency:** Do qualitative and quantitative results demonstrate preservation of object identity and scene structure across the entire intensity spectrum?
- **Trade-off Quantification:** Is there explicit analysis of the precision-fidelity trade-off, distinguishing between "attribute match fidelity" and "semantic preservation"?
- **Range Stability:** Does control remain semantically coherent at extreme intensity values, or does it produce artifacts, structural collapses, or semantic inversions?
- **Conflict Handling:** How does the method behave when the text prompt contains strong attribute modifiers that conflict with the scalar control signal, and is this degradation quantified?

---

**Principle 5: Fine-Grained Safety Intensity Control with Robustness to Checker Dependence and Alternative Priors**

**Definition:**
For frameworks incorporating safety as a controllable attribute, evaluation must extend beyond binary concept erasure to assess whether the method enables nuanced, tiered safety levels—such as moderate filtering versus strict suppression—while preserving generation quality on benign concepts. Reviewers must scrutinize whether safety definitions are tied to a single, potentially biased reference model (e.g., Stable Diffusion Safety Checker) or validated against diverse alternative priors (e.g., LAION NSFW classifiers). The evaluation should include per-category removal rates with confidence intervals, user studies confirming perceived safety gradations, and quality metrics on unrelated concepts to detect over-filtering. This principle ensures that safety control is treated as a continuous, user-configurable or administratively tiered dimension rather than a brittle binary switch.

**Core Evaluation Criteria:**
- **Granular Control:** Does the framework support continuous safety intensity adjustment across a spectrum (e.g., from moderate to strict filtering) rather than binary safe/unsafe classification?
- **Prior Robustness:** Is safety control validated against multiple independent safety classifiers or checkers, with sensitivity analysis to prior coverage and bias shifts?
- **Quality Preservation:** Do FID and CLIP Score on unrelated, benign concepts remain competitive compared to safety-specific baselines, indicating no collateral damage to general generation quality?
- **User Perception:** Are human evaluators able to perceive intended safety level changes, and is satisfaction or agreement quantified with statistical confidence?