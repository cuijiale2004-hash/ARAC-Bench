1. **Research Background and Existing Pain Points**
Multi-modal Large Language Models (MLLMs) have shown remarkable generative capabilities across multi-modal tasks by integrating visual information with Large Language Models (LLMs). However, MLLMs exhibit a notable hallucination problem, where the generated textual descriptions are semantically inconsistent with the input visual content. This phenomenon includes describing non-existent objects, incorrect object attributes, or misinterpreting relationships. The hallucination issue causes a disconnect between outputs and visual facts, severely degrading user experience and undermining the reliability of downstream applications, thereby limiting their deployment in real-world scenarios. 

To alleviate this issue, strategies incorporating preference learning, such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), have been widely investigated. Existing work demonstrates that DPO mitigates hallucinations in MLLMs and improves alignment with human preference. However, current human preference alignment processes suffer from two primary limitations and pain points:
(1) They overlook the varying degrees of association between different tokens in the response and the visual content, as well as their differing contributions to high-quality outputs, treating all tokens uniformly and thus lacking fine-grained correction. Different response tokens processed by the model have varying degrees of relevance to the visual content, and their contributions to reducing hallucinations and generating high-quality answers also differ.
(2) They rely on external visual detection models, additional noise injection techniques, expensive closed-source LLM API, or even external tools, thereby neglecting the intrinsic capabilities of MLLMs and leading to a waste of existing resources and increased costs.

2. **Core Research Motivation and Scientific Questions**
The core research motivation stems from the observation that different response tokens exhibit varying degrees of association with visual content, and consequently, their contributions to reducing hallucinations and generating high-quality responses differ. As shown by statistical experiments, when applying DPO only to the top 50% rewarded tokens in chosen responses, significant improvements in hallucination metrics are observed compared to the original DPO. Applying DPO with all rewarded tokens yields even more outstanding results. Therefore, how to deeply exploit token-level fine-grained alignment signals, construct a more refined DPO feedback mechanism, and fully leverage the inherent multimodal capabilities of MLLMs to reduce additional costs and overhead remains a critical issue.

The scientific questions addressed are: How to deeply mine token-level fine-grained alignment signals between text and images using the intrinsic multimodal capabilities of MLLMs without external tools? How to construct a more refined DPO feedback mechanism using token-rewards for Preference Optimization to more effectively mitigate hallucinations?

3. **Overall Core Idea and Design Philosophy**
The overall core idea is to propose a Cross-modal Adaptive Token-rewarded Preference Optimization (Cat-PO). This framework fully leverages the multimodal capabilities and advantages of MLLMs to deeply mine token-level fine-grained alignment signals between text and images, using token-rewards for Cat-PO, with the aim of more effectively mitigating hallucinations.

The design philosophy involves calculating hierarchical visual relevance rewards for each response token at global, local, and semantic levels without any external tools or techniques. It organically integrates these three rewards to construct a smooth reward mechanism and designs an innovative KL-based customized loss for rewarded tokens, thereby enabling fine-grained correction of hallucinatory outputs.

4. **Core Innovation Points**
1. We propose a Cross-modal Adaptive Token-rewards for Preference Optimization (Cat-PO) in MLLMs. By assigning token-rewards to highlight visually critical tokens and incorporating a fine-grained KL regularization, Cat-PO effectively reduces multimodal hallucinations.
2. We introduce a hierarchical token-rewards calculation method that relies solely on the model’s inherent multimodal capabilities, without introducing any external tools or technologies. Specifically, it first computes global relevance based on cross-modal attention between text and image, then calculates local relevance based on patch entropy, and finally uses cross-modal semantic similarity for further refinement.
3. We design a novel Cat-PO loss based on token-level rewards and KL divergence for further optimization, enabling fine-grained correction of hallucinatory outputs.
4. Extensive experiments across multiple datasets and benchmarks evaluate the effectiveness of Cat-PO. Notably, on the AMBER-Generation and MM-Hal benchmarks, the proposed Cat-PO outperforms existing state-of-the-art methods by 7% – 15% metrics.

5. **Overview of the Overall Technical Solution**
Within the MLLMs, before the image features (projected by CLIP and ViT) are fed into the LLM’s transformer layers, the cross-modal semantic similarity between response tokens and the image is first calculated, representing the semantic relevance of tokens to visual content. Subsequently, within the transformer layers, based on the cross-modal attention of response tokens to the image, the global and local relevance of each token to the visual content is computed. Furthermore, the three hierarchical relevance scores are normalized and aggregated, the result is mapped through an activation function, and the final reward for each token is computed. Finally, a novel Cat-PO loss based on token-level rewards and KL divergence is designed for further optimization.

6. **Detailed Module Design**
**6.1 Hierarchical Visual Relevance of Tokens**
Without any external tools or techniques, this module leverages the intrinsic multimodal capabilities of MLLMs, to hierarchically compute each token’s global, local, and semantic relevance to the visual input.

*6.1.1 Cross-modal Attention Based Global Relevance*
When MLLMs process DPO training data within the Transformer architectures, the feature representation of each token in a response interacts with visual features via a cross-modal attention mechanism. The activation distribution of these attention scores intuitively reflects the focus of specific text tokens on different image regions. For a given image I and its corresponding preferred response yw or rejected response yl (collectively denoted y) from the dataset, the representation of the t-th token yt in y serves as the query. The set of Np visual token features, {v1, v2, . . . , vNp}, derived from image I via a visual encoder, acts as the keys and values. The sequence of cross-modal attention scores from token yt to all Np visual tokens is denoted by {at,1, at,2, . . . , at,Np}. The global relevance Sglobal(yt) is defined as the sum of the attention scores for token yt. A higher Sglobal(yt) indicates that the model attends more intensively to the entire image when processing token yt, implying a stronger global alignment between the token and the visual content.

*6.1.2 Patch Entropy Based Local Relevance*
Although the global relevance score Sglobal(yt) captures the overall association between response tokens and visual content, it does not reveal whether attention is concentrated on key regions or dispersed across the image. To accurately characterize the focusing pattern within this attention distribution, the concept of information entropy is leveraged to compute the patch entropy scores for each token yt based on its image attention distribution. First, extract the cross-modal attention vector at = [at,1, at,2, . . . , at,Np] for token yt with respect to all Np image patches, where at,j represents the attention strength of yt towards the j-th image patch. Next, normalize the attention strengths in at to form a probability distribution Pt = [Pt,1, Pt,2, . . . , Pt,Np], and Pt,j = at,j / sum(at,k for k=1 to Np). Compute the patch entropy H(Pt) of this probability distribution. To ensure numerical stability in the logarithm, a small epsilon value ϵ (e.g., 10−12) is introduced. This entropy value H(Pt) measures the uncertainty or dispersion of the attention distribution. Subsequently, for Np > 1, normalize the computed entropy H(Pt), and the Patch Entropy Score, Sentropy(yt), is then defined as 1 minus this normalized entropy: Slocal(yt) = 1− H(Pt)/logNp (for Np > 1). A higher Slocal(yt) score indicates lower entropy in the attention distribution, implying that attention is more sharply focused on a few image patches, signifying a stronger degree of association between the token yt and specific local regions of the image.

*6.1.3 Cross-modal Similarity Based Semantic Relevance*
Beyond the global and local relevance, a prior semantic signal obtained from a pretrained cross–modal encoder is exploited to quantify the semantic alignment between response tokens and visual content. Given a sample (I, y), let the embedding of the t-th token be e(yt). The image I is divided into Np patches, each encoded as a visual feature {v1, . . . ,vNp}. With the cross–modal attention weights αt,j (normalized over patches), obtain a context–aware visual vector: vc(yt) = sum(αt,j * vj for j=1 to Np). The semantic relevance score is then defined as the cosine similarity between e(yt) and vc(yt). This score captures the semantic relevance between the token representation and the visual content of its most attended region, complementing the attention-based global and local relevance.

**6.2 Token Weighting Scheme**
*6.2.1 Unified Visual Relevance Score:* After obtaining hierarchical relevance scores for every response token yi, normalize all scores to [0, 1] and fuse them into a unified visual relevance score. Here, α balances the attention branch (global & local) against the semantic branch.

*6.2.2 Smooth Mapping to Token Weights:* Directly injecting si into the loss may yield unstable gradients. Therefore, apply a smooth non-linearity: Ti = tanh(si) ∈ (0, 1), and introduce a base weight λref > 0 to maintain a controlled dynamic range. This design (i) rewards tokens in the preferred response that strongly align with the image (Ti ↑), and (ii) penalises hallucinated or weakly aligned tokens in the dispreferred response ((1− Ti)↑).

**6.3 Weighted Integration and KL-regularised Cat-PO Loss**
Incorporating token weights {w+t, w−t} and token-level KL into the DPO loss yields the Cat-PO loss.
*6.3.1 Weighted DPO:* For a preference pair (y+, y−), weight the log-likelihood ratio of the policy πθ and the reference πref. The weighted policy π(w)θ and the weighted reference π(w)ref are defined by summing the weighted log probabilities of the tokens.
*6.3.2 Token-weighted KL regulariser:* To keep the policy close to the reference and to stabilise training, with a regularisation strength λ > 0, add a token-level, weight-modulated KL penalty. The final Cat-PO Loss objective minimises the sum of the Weighted DPO loss and the Token-weighted KL regulariser, enabling the policy model to encode human preferences and fine-grained token–vision alignments, suppressing hallucinations while preserving generation quality.

7. **All Mathematical Formulas and Symbol Definitions**
- DPO loss function: 
LDPO = − log σ(β(log(πθ(y+ | x)/πref (y+ | x)) − log(πθ(y− | x)/πref (y− | x))))
- Global relevance score: 
Sglobal(yt) = sum(at,j for j=1 to Np)
- Patch entropy: 
H(Pt) = − sum(Pt,j * log(Pt,j + ϵ) for j=1 to Np)
- Local relevance score: 
Slocal(yt) = 1− H(Pt)/logNp (for Np > 1)
- Context-aware visual vector: 
vc(yt) = sum(αt,j * vj for j=1 to Np)
- Semantic relevance score: 
Ssemantic(yt) = cos(e(yt), vc(yt)) = e(yt) · vc(yt) / (∥e(yt)∥ ∥vc(yt)∥)
- Unified visual relevance score: 
si = α[0.5 ∗ Sglobal,i + 0.5 ∗ Slocal,i] + (1− α)Ssemantic,i, α ∈ [0, 1]
- Smooth mapping to token weights: 
Ti = tanh(si) ∈ (0, 1)
- Token weights definition: 
wi = {λref + Ti, yi ∈ y+, λref + (1− Ti), yi ∈ y−}
- Weighted policy: 
π(w)θ = sum(w+t log πθ(y+t | h+t) − w−t log πθ(y−t | h−t))
- Weighted reference: 
π(w)ref = sum(w+t log πref(y+t | h+t) − w−t log πref(y−t | h−t))
- Weighted DPO loss: 
LwDPO = − log σ[β(π(w)θ − π(w)ref)]
- Token-weighted KL regulariser: 
LKL = λ(sum(w+t KL[πθ(· | h+t) ∥πref(· | h+t)]) + sum(w−t KL[πθ(· | h−t) ∥πref(· | h−t)]))
- Final Cat-PO Loss: 
LCat-PO = LwDPO + LKL

8. **Algorithm Pseudocode**
No algorithm pseudocode is provided in the original paper.