**1. Research Background and Existing Pain Points**

Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for improving the factuality of large language models (LLMs) by conditioning generation on passages retrieved from external corpora, aiming to ground model outputs in verifiable evidence. However, in practice, grounding does not eliminate unfaithfulness. Models may still contradict the retrieved content, introduce unsupported details, or extrapolate beyond what the evidence justifies. These faithfulness failures, commonly referred to as hallucinations in the RAG setting, undermine user trust and limit deployment in domains where faithfulness to source information is critical. Various approaches have been proposed to address this challenge, but they suffer from significant pain points:

- **Fine-tuning specialized detectors:** One direction is to fine-tune specialized detectors to distinguish faithful from unfaithful generations. While this method provides direct supervision, its effectiveness is often constrained by the need for large amounts of high-quality annotated training data.
- **Querying external LLM judges:** Another line of work employs LLMs as judges, where an auxiliary LLM is prompted to assess faithfulness given the retrieved passages and generated answers. However, these approaches struggle to detect hallucinations produced by the same model and introduce significant computational overhead when relying on large-scale external LLMs. Furthermore, they exhibit sensitivity to prompt design and produce explanations that may be plausible but do not faithfully reflect the underlying decision process.
- **Leveraging LLM internal representations:** More recently, researchers have explored the use of the LLM’s internal representations, such as hidden states or attention scores, to capture hallucinations directly. While these methods show promise, the extraction of reliable hallucination-related signals remains challenging due to the polysemanticity of neurons and the opacity of hidden states, resulting in insufficient detection performance.

**2. Core Research Motivation and Scientific Questions**

Recent advances in mechanistic interpretability have shown that sparse autoencoders (SAEs) can disentangle specific, semantically meaningful features from the hidden states of LLMs. By enforcing sparsity, SAEs identify features that correspond to concrete concepts, as evidenced by their consistent activation across similar cases. This property, known as monosemanticity, provides a transparent link between internal activations and model behaviors. While recent work has explored the use of SAEs to detect signals associated with generic LLM hallucinations, hallucinations in RAG settings pose unique challenges due to the complex interplay between retrieved evidence and generated content. The core research motivation is to investigate whether SAEs can identify interpretable features that are predictive of hallucinations in RAG. The central scientific question is: Can SAE features effectively capture the dynamics of RAG hallucinations, enabling both accurate detection and deeper insight into failure cases?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is to employ sparse autoencoders (SAEs) to disentangle internal activations of LLMs, successfully identifying features that are specifically triggered during RAG hallucinations. Building on a systematic pipeline of information-based feature selection and additive feature modeling, the design philosophy centers on creating a lightweight hallucination detector that accurately flags unfaithful RAG outputs using LLM internal representations. Crucially, the design prioritizes interpretability, ensuring that the detector provides interpretable rationales for its decisions, which in turn enables effective post-hoc mitigation of unfaithful RAG. The architecture is designed to be lightweight, requiring the encoding of only a small subset of SAE features, while leveraging the monosemanticity of SAE features and the transparency of additive models to bridge internal model states with human-understandable concepts.

**4. Core Innovation Points**

- Demonstrate that SAEs capture nuanced features specifically activated during RAG hallucinations, establishing a strong foundation for detecting RAG unfaithfulness from LLM internal representations.
- Introduce RAGLens, a lightweight hallucination detector that outperforms existing methods in detection accuracy while providing transparent and interpretable feedback to aid in hallucination mitigation.
- Through detailed analyses, justify the key design choices in RAGLens and offer new insights into the distribution of hallucination-related signals within LLMs, including findings that mid-layer SAE features with high mutual information are most informative, and GAMs are particularly well-suited for mapping SAE features to hallucination predictions.
- Provide a systematic pipeline that supports interpretability at both local (instance-specific) and global (instance-invariant) levels, demonstrating how these interpretations facilitate effective post-hoc mitigation of unfaithfulness through multi-level feedback.

**5. Overview of the Overall Technical Solution**

RAGLens is a lightweight SAE-based detector that flags unfaithful RAG outputs by leveraging LLM internal activations. The overall technical solution operates through a systematic pipeline:

1. Given a generated answer sequence conditioned on a query and retrieved context, extract hidden states from a specific layer of a frozen LLM.
2. Transform these hidden states via a pre-trained SAE encoder into sparse, token-level features.
3. Summarize token-level activations into an instance representation via channel-wise max pooling to align with instance-level target labels.
4. Apply information-based feature selection, quantifying the mutual information of each pooled feature with the hallucination label, and selecting the top $K'$ dimensions.
5. Model the instance label from the selected pooled representation using a Generalized Additive Model (GAM) for transparent and lightweight prediction.
6. Utilize the detection results and the interpretable structure of SAE features and GAMs for hallucination explanation, providing both local (token-level feedback) and global (feature-level auditing) interpretations.
7. Leverage the explanation signals directly for mitigation at inference time by providing multi-level feedback (instance-level warnings and token-level highlights) to LLMs to reconsider and revise potentially hallucinated content.

**6. Detailed Module Design**

- **Problem Setting**: Following prior work, "RAG" denotes any context-conditioned generation process where an LLM produces an answer based on a user query/instruction and provided context. The faithfulness detection task determines whether the generated answer is consistent with the context. Each instance consists of: (1) a user query or instruction $q$; (2) a set of retrieved passages $C$; (3) an answer sequence $y_{1:T}$ generated by an LLM, where $T$ is the sequence length and $y_{1:t}$ denotes the prefix up to position $t$; and (4) a binary label $\ell \in \{0, 1\}$ indicating whether the answer contains hallucination relative to $C$. The method assumes access to a frozen LLM $\Phi$ and a corresponding SAE with encoder $E$ trained on the hidden states in the L-th layer of $\Phi$.

- **Instance-level Feature Summary**: Because target labels are instance-level, the method summarizes token-level activations into an instance representation via channel-wise max pooling. This takes the maximum activation value across all tokens for each feature dimension.

- **Information-based Feature Selection**: To identify the most relevant features for hallucination detection, the method quantifies the information of each pooled feature about the hallucination label using mutual information (MI). Features are ranked by MI, and the top $K'$ dimensions are selected. Although the hidden states of retrieved passages $C$ are not explicitly utilized, the encoding of the model output is conditioned on $C$, allowing the SAE features to implicitly capture interactions between the generated answer and retrieved content. Empirical results show that the selected SAE features encode knowledge relevant to retrieved passages, and their activations are dynamically influenced by counterfactual interventions on $C$.

- **Transparent Prediction with Generalized Additive Models**: After selecting informative SAE features, the instance label is modeled from the pooled representation using a generalized additive model (GAM). Each univariate shape function in the GAM is learned using bagged gradient boosting. By selecting only $K'$ features ($K' \ll K$), the fitted GAM serves as a lightweight detector, requiring the encoding of only a small subset of SAE features. Analysis validates that GAM is well-suited for modeling hallucination signals from SAE features, outperforming more complex predictors such as MLP and XGBoost because while the effect of individual features is often nonlinear, the overall contribution can be effectively captured additively.

- **Justification of Max Pooling on Sparse Activations**: Beyond storage efficiency, there is a theoretical justification for using max pooling: in the sparse activation regime, it can help distinguish hallucination-related features from random noise by amplifying signals associated with relevant targets. The token-level activation is modeled as conditionally independent across tokens given the label, with a rare activation mechanism where the activation is either 0 or a positive value $V_t$. Theorem 1 proves that if the expected number of activations is small ($T \times \bar{p} \ll 1$), the mutual information between the max-pooled feature and the label is strictly positive if the activation probabilities differ between hallucinated and non-hallucinated instances, with a leading dependence linear in sequence length $T$ and quadratic in the probability difference $\Delta p$.

- **Hallucination Explanation and Mitigation**:
    - **Local Explanations via Sparse Feature Attribution**: Because the GAM operates additively on a small set of selected SAE features, each prediction can be decomposed into a sum of feature contributions. For any given example, the hallucination prediction is attributed to the specific sparse features that are most strongly activated. By aligning these activations with token positions, token-level feedback is obtained that highlights which parts of the generation are likely ungrounded relative to the retrieved passages. This fine-grained attribution enables users to directly pinpoint fabricated factual inserts such as numbers, dates, or named entities.
    - **Global Explanations via Intrinsic Model Interpretability**: Beyond instance-specific attributions, RAGLens provides global, instance-invariant explanations. With the dictionary learning property of SAEs, each SAE feature corresponds to a semantically coherent concept that can be summarized as human-understandable knowledge. Furthermore, the use of GAMs enables visualization of the learned shape function for each feature, offering a stable explanation of the mapping from feature magnitude to predicted hallucination risk. Practitioners can therefore inspect how changes in a given feature systematically increase or decrease the prediction, enabling consistent feature-level auditing.
    - **Mitigation through Multi-level Feedback**: The interpretability of the framework enables explanation signals to be directly incorporated into mitigation strategies at inference time. Detection results can be provided to LLMs as instance-level warnings, prompting the model to reconsider and revise potentially hallucinated content. By aligning sparse activations with text spans identified by local explanations, problematic tokens that may require editing can be further highlighted, thereby guiding the model to refine its output.

- **Design Choice Analyses**:
    - **LLM Layer Selection**: Analysis of varying the layer from which SAE features are extracted shows that the performance trend is consistent among LLMs but varies by task. In Summary and QA tasks, performance peaks around the middle layers, whereas the Data2txt task exhibits a comparatively flat performance pattern across layers.
    - **Feature Extractor Comparison**: Comparing SAE and Transcoder feature extractors, pre-activation features consistently outperform post-activation features for both extractors. Transcoder and SAE achieve similar accuracy, indicating no clear advantage for either architecture, suggesting that the choice of activation point is more critical.
    - **Analysis of Feature Count**: As the number of selected features $K'$ decreases, performance drops, but the decline is much more gradual with MI-based selection compared to random selection, demonstrating that MI effectively prioritizes informative features.
    - **Predictor Ablation**: GAM consistently outperforms logistic regression and surpasses more complex models such as MLP and XGBoost, confirming that while the effect of individual features on the output is often nonlinear, the overall contribution of SAE features can be effectively captured in an additive manner.

**7. All Mathematical Formulas and Symbol Definitions**

- $q$: User query or instruction
- $C$: Set of retrieved passages
- $y_{1:T}$: Answer sequence generated by an LLM, where $T$ is the sequence length and $y_{1:t}$ denotes the prefix up to position $t$
- $\ell \in \{0, 1\}$: Binary label indicating whether the answer contains hallucination relative to $C$
- $\Phi$: Frozen LLM
- $E$: SAE encoder
- $L$: Layer index
- $\Phi^L(\cdot)$: Mapping that returns layer-L hidden states
- $h_t = \Phi^L(y_{1:t}, q, C), t = 1, \ldots, T$ (Equation 1)
- $z_t = E(h_t), z_t \in \mathbb{R}^K$ (Equation 2), where $K$ is the size of the dictionary
- $\bar{z}_k = \max_{1 \le t \le T} z_{t,k}, k = 1, \ldots, K$ (Equation 3), where $z_{t,k}$ is the $k$-th element of $z_t$
- $\bar{z} = (\bar{z}_1, \ldots, \bar{z}_K) \in \mathbb{R}^K$
- $I(\bar{z}_k; \ell) = \int_{\mathbb{R}} \sum_{\ell \in \{0,1\}} p(\bar{z}_k, \ell) \log_2 \frac{p(\bar{z}_k, \ell)}{p(\bar{z}_k) p(\ell)} d\bar{z}_k$ (Equation 4)
- $S = \arg\max_{|S|=K'} \sum_{k \in S} I(\bar{z}_k; \ell)$ (Equation 5)
- $\tilde{\bar{z}} \in \mathbb{R}^{K'}$: Subvector of $\bar{z}$ restricted to indices $S$
- $g(E[\ell | \tilde{\bar{z}}]) = \beta_0 + \sum_{j=1}^{K'} f_j(\tilde{\bar{z}}_j)$ (Equation 6), where $g$ is the link function (e.g., logit for binary classification) and each univariate shape function $f_j$ is learned using bagged gradient boosting
- For Theorem 1:
    - $z_t \ge 0$: Token-level activation
    - $z_t = \begin{cases} 0, & \text{with probability } 1 - p_\ell \\ V_t, & \text{with probability } p_\ell \end{cases}$ for $t = 1, \ldots, T$ (Equation 7)
    - $V_t$: "active-value" random variable with distribution $F$ supported on $(0,\infty)$ that is independent of $\ell$ and i.i.d. across tokens
    - $\bar{z} = \max_{1 \le t \le T} z_t$
    - $\pi = \Pr(\ell = 1)$
    - $p_\ell$: Activation probability given label $\ell$
    - $\bar{p} = \frac{1}{2} (p_1 + p_0)$
    - $\Delta p = p_1 - p_0$
    - Theorem 1 (Max pooling in the sparse-activation regime): If $T \times \bar{p} \ll 1$, then $I(\bar{z}; \ell) = \frac{\pi(1-\pi)}{2 \ln 2} \frac{T (\Delta p)^2}{\bar{p}} + O((T \bar{p})^2)$ (Equation 8), where $I(\bar{z}; \ell) > 0$ iff $p_1 \neq p_0$. The leading dependence is linear in $T$ and quadratic in $\Delta p$.
    - Proof sketch: Let $A = 1\{\bar{z} > 0\}$. Independence across tokens implies $\Pr(A=1 | \ell) = q_\ell = 1 - (1-p_\ell)^T$, so $I(A; \ell) = h(\pi q_1 + (1-\pi)q_0) - [\pi h(q_1) + (1-\pi)h(q_0)]$. For $p_\ell \ll 1$, $q_\ell \approx Tp_\ell$ and a second-order expansion of $h$ gives $I(A; \ell) = \frac{\pi(1-\pi)}{2 \ln 2} \frac{T (\Delta p)^2}{\bar{p}} + O((T \bar{p})^2)$ (Equation 9). Since $A$ is a deterministic function of $\bar{z}$, $I(A; \ell) \le I(\bar{z}; \ell)$. In the single-hit regime ($T \bar{p} \ll 1$), the extra information in $\bar{z}$ beyond $A$ occurs only on rare multi-activation events, contributing $O((T \bar{p})^2)$. Combining yields the claim.

**8. Algorithm Pseudocode**

No explicit algorithm pseudocode is provided in the original text.