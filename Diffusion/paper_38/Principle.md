
### Principle 1: Causal Validity Verification of Visual Point Prompting Mechanisms in Counterfactual VQA Tasks

**Definition:**  
In research combining visual point prompting with counterfactual reasoning for Visual Question Answering (VQA), reviewers must rigorously assess whether the introduced point prompts genuinely establish a causal link between specific image regions and the model's counterfactual inference, rather than serving as mere spatial attention biases or spurious correlations. The evaluation requires verifying that the model’s ability to answer "what-if" questions stems from an explicit manipulation of the referenced visual evidence, ensuring that the point prompt acts as a valid intervention variable in the causal graph of multimodal reasoning. This is critical because without proven causality, performance gains may simply reflect dataset shortcuts or enhanced region localization rather than true counterfactual cognitive capabilities.

**Core Review Criteria:**
-   **Intervention Effectiveness:** Does the ablation study demonstrate that removing or perturbing the point prompt significantly degrades counterfactual accuracy while leaving factual QA performance relatively stable, proving the prompt's specific role in counterfactual chains?
-   **Causal vs. Correlational Distinction:** Are there controlled experiments showing the model fails when points are placed on semantically irrelevant but visually salient regions, confirming it relies on semantic content at the point rather than just spatial coordinates?
-   **Granularity of Causal Attribution:** Does the work provide visualization or probing analysis linking specific point locations to intermediate reasoning steps or token generation probabilities in the counterfactual branch?
-   **Robustness to Point Noise:** Is the causal mechanism robust to slight spatial perturbations of the input points, or does the model exhibit brittle behavior indicative of overfitting to exact pixel coordinates?

---

### Principle 2: Disentanglement Assessment of Factual Knowledge Retention and Counterfactual Adaptation in Multimodal Alignment

**Definition:**  
For Multimodal Large Language Models (MLLMs) trained with point-based counterfactual data, it is essential to evaluate whether the counterfactual tuning process catastrophically forgets foundational factual grounding or general vision-language alignment. Reviewers must scrutinize whether the proposed method maintains a balanced trade-off where the model can faithfully adhere to real-world visual facts when prompted normally, yet flexibly deviate from them only when explicitly guided by counterfactual point prompts. This principle addresses the fundamental tension in MLLM safety and utility: ensuring that enhancing imaginative or hypothetical reasoning does not compromise the reliability of the base multimodal assistant.

**Core Evaluation Criteria:**
-   **Factual Grounding Preservation:** Are standard VQA benchmarks (e.g., VQAv2, GQA) included in the evaluation to quantify any regression in factual accuracy post-counterfactual training?
-   **Instruction Following Discriminability:** Can the model reliably distinguish between factual and counterfactual instructions based solely on prompt type, without leaking counterfactual hallucinations into factual responses?
-   **Training Data Balance Analysis:** Is there an analysis of the ratio and mixing strategy between factual and counterfactual samples, and how this ratio impacts the retention-forgetting trade-off?
-   **Safety and Hallucination Metrics:** Beyond accuracy, are metrics like hallucination rate or object existence verification used to ensure counterfactual training hasn't increased general sycophancy or visual hallucination?

---

### Principle 3: Methodological Rigor in Constructing Point-Annotated Counterfactual Datasets for Vision-Language Pretraining

**Definition:**  
Given that high-quality point-annotated counterfactual datasets are scarce, the validity of the entire study hinges on the scientific rigor of the dataset construction pipeline. Reviewers must evaluate whether the synthetic or semi-synthetic generation of counterfactual QA pairs with point annotations avoids tautological reasoning, ensures visual plausibility of the counterfactual premise, and maintains semantic consistency between the pointed region and the hypothetical scenario. Poorly constructed datasets often contain artifacts (e.g., pointing to empty space for "remove object" queries, or logically impossible premises) that allow models to learn superficial text patterns rather than genuine visuo-spatial counterfactual reasoning.

**Core Evaluation Criteria:**
-   **Visual-Semantic Consistency Check:** Is there human validation or automated filtering ensuring that the pointed region actually corresponds to the entity being modified in the counterfactual question?
-   **Diversity of Counterfactual Types:** Does the dataset cover diverse operations (addition, removal, attribute change, relation change) rather than biasing towards a single easy-to-generate transformation type?
-   **Avoidance of Textual Shortcuts:** Have n-gram overlap or language model perplexity checks been performed to ensure answers cannot be predicted from the question text alone without looking at the image/points?
-   **Annotation Quality Reporting:** Are inter-annotator agreement scores or detailed error analyses of the generated dataset provided to establish trust in the ground truth?

---

### Principle 4: Comprehensive Benchmarking Against Non-Point and Text-Only Counterfactual Baselines in Multimodal Reasoning

**Definition:**  
To validate the unique contribution of *point* prompting, the paper must provide exhaustive comparisons against strong baselines that achieve counterfactual reasoning through alternative means, such as text-only descriptions, bounding boxes, segmentation masks, or implicit latent editing. Reviewers should assess whether the added complexity of acquiring and processing point prompts yields statistically significant and practically meaningful improvements over simpler or more readily available modalities. This principle prevents the publication of incremental methods that introduce new input requirements without demonstrating clear superiority over existing, less burdensome approaches.

**Core Evaluation Criteria:**
-   **Modality-Efficiency Trade-off:** Does the point-prompting method outperform text-description baselines sufficiently to justify the extra annotation/inference cost?
-   **Comparison with Other Spatial Primitives:** Are comparisons made against box or mask prompts to determine if points offer unique advantages (e.g., ambiguity resolution) or disadvantages (e.g., lack of extent information)?
-   **SOTA Contextualization:** Is the performance contextualized against current state-of-the-art MLLMs (e.g., GPT-4V, LLaVA-Next) on relevant counterfactual benchmarks, even if those models don't use points?
-   **Failure Mode Differentiation:** Does the analysis show *where* points help specifically (e.g., disambiguating similar objects) versus where they offer no advantage over text?

---

### Principle 5: Interpretability and Faithfulness Evaluation of Point-Guided Attention Dynamics During Counterfactual Inference

**Definition:**  
Since point prompting is inherently an interpretability-oriented intervention, reviewers must demand evidence that the model’s internal attention mechanisms actually align with the intended function of the points during counterfactual reasoning. It is insufficient to report final accuracy; the work must demonstrate that the model attends to the pointed region *when generating the counterfactual delta* and reverts to global/contextual attention when maintaining unchanged aspects of the scene. This verifies that the point prompt is functioning as a designed cognitive scaffold rather than being ignored or misused by the black-box transformer layers.

**Core Evaluation Criteria:**
-   **Attention Map Alignment:** Are attention visualizations provided showing focused activation on pointed regions specifically during counterfactual token generation phases?
-   **Token-Level Attribution:** Do gradient-based or perturbation-based attribution methods confirm that the pointed region’s features contribute disproportionately to counterfactual answer tokens compared to factual ones?
-   **Temporal Dynamics Analysis:** Is there analysis of how attention shifts over the decoding sequence (e.g., attending to point first, then integrating context)?
-   **Faithfulness to User Intent:** In cases of ambiguous or multiple points, does the model’s attention distribution reflect reasonable prioritization or user-intended weighting?