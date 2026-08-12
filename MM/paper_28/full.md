## ABSTRACT

Multi-modal Large Language Models (MLLMs) have shown remarkable generative capabilities across multi-modal tasks, yet remain plagued by hallucinations where generated textual contents are semantically inconsistent with the input images. This work reveals that existing multi-modal preference optimization methods exhibit shortcomings at the preference data decoding stage. Specifically, different response tokens exhibit varying degrees of association with visual content, and consequently, their contributions to reducing hallucinations and generating high-quality responses differ. Nevertheless, most existing methods do not distinguish in their treatment, often handling them uniformly. To address this challenge, we propose a novel preference alignment method: Cross-modal Adaptive Token-rewarded Preference Optimization (Cat-PO). Building upon direct preference optimization, Cat-PO calculates hierarchical visual relevance rewards for each response token at global, local, and semantic levels. It then organically integrates these three rewards to construct a smooth reward mechanism and designs an innovative KL-based customized loss for rewarded tokens, thereby enabling fine-grained correction of hallucinatory outputs. Extensive experiments on various base models and evaluation benchmarks demonstrate that our Cat-PO can significantly reduce hallucinations and align with human preferences to enhance the truthfulness of MLLMs. <sup>1</sup>

## 1 INTRODUCTION

The success of Multimodal Large Language Models (MLLMs) marks a significant advancement in artificial intelligence research Liu et al. (2024b); Amirloo et al. (2024). By integrating visual information with Large Language Models (LLMs), MLLMs have demonstrated unprecedented capabilities in multimodal understanding, reasoning, and interaction Xiao et al. (2024); Pi et al. (2024); Zhang et al. (2024). However, MLLMs exhibit a notable hallucination problem, where the generated textual descriptions are inconsistent with the input visual content. This phenomenon includes describing non-existent objects, incorrect object attributes, or relationships Bai et al. (2024); Gunjal et al. (2024). The hallucination issue causes a disconnect between outputs and visual facts, severely degrading user experience and undermining the reliability of downstream applications, thereby limiting their deployment in real-world scenarios Liu et al. (2024a); Liang et al. (2024).

To alleviate this issue, strategies incorporating preference learning, such as Reinforcement Learning from Human Feedback (RLHF) Christiano et al. (2017), have been widely investigated as a form of fine-tuning. The core idea is to leverage preference feedback to align model outputs with desired expectations. Recently, Direct Preference Optimization (DPO) Rafailov et al. (2023) has gained prominence for achieving excellent results without a separate reward model by simplifying complex reinforcement learning steps. Existing work demonstrates that DPO, by efficiently incorporating preference data, mitigates hallucinations in MLLMs, and improves the alignment with human

(b)

![](images/2b54e0f4f7c7570e9c59ba219bac7b09b0ac0a4d99295e810e911f1bc186a948.jpg)  
Figure 1: The motivation of our framework. (a) A visual question answering example where the model identifies "a laptop and a cup" on a table, with cross-modal attention heatmaps and cross-modal similarity scores indicating the model’s visual focus and word importance in the response. (b) A performance comparison of token-rewarded DPO, showing AMBER F1 (↑) improving and MM-Hal Hallucination Rate (↓) declining as the percentage of rewarded tokens increases. (c) A comparison of standard DPO versus our Cross-modal Adaptive Token-rewarded Preference Optimization (Cat-PO). The former uses a flat gradient distribution for maximal likelihood optimization. And the latter employs a targeted gradient distribution, suggesting potentially superior performance for the latter in refining a pre-trained MLLM.

preference Li et al. (2023b); Zhu et al. (2024); Chen et al. (2024). However, in current human preference alignment processes, different response tokens processed by the model have varying degrees of relevance to the visual content, and their contributions to reducing hallucinations and generating high-quality answers also differ. As shown in Fig. 1 (a), When MLLMs process different tokens, the cross-modal attention they allocate to the image varies, and the token–image similarity also differs, indicating that different tokens exhibit distinct degrees of association with the visual content. Nevertheless, most existing works suffer from two primary limitations: (1) They overlook the varying degrees of association between different tokens in the response and the visual content, as well as their differing contributions to high-quality outputs, treating all tokens uniformly and thus lacking fine-grained correction, as depicted in the upper part of Fig. 1 (c). (2) They rely on external visual detection models, additional noise injection techniques, expensive closed-source LLM API, or even external tools, thereby neglecting the intrinsic capabilities of MLLMs and leading to a waste of existing resources and increased costs.

Therefore, how to deeply exploit token-level fine-grained alignment signals, construct a more refined DPO feedback mechanism, and fully leverage the inherent multimodal capabilities of MLLMs to reduce additional costs and overhead remains a critical issue. Motivated by this, we conducted a series of explorations. As shown by the statistical experiment in Fig. 1 (b), when we applied DPO only to the top 50% rewarded tokens in chosen responses, we observed significant improvements in hallucination metrics AMBER-F1 and MM-Hal compared to the original DPO. Furthermore, applying DPO with all rewarded tokens yielded even more outstanding results.

Building upon these explorations, we propose a Cross-modal Adaptive Token-rewarded Preference Optimization (Cat-PO). This framework fully leverages the multimodal capabilities and advantages of MLLMs to deeply mine token-level fine-grained alignment signals between text and images, using token-rewards for Cat-PO, with the aim of more effectively mitigating hallucinations. A simplified pipeline is shown in the lower part of Fig. 1 (c). Specifically, within the MLLMs, before the image features (projected by CLIP Radford et al. (2021) and ViT) are fed into the LLM’s transformer layers, we first calculate the cross-modal semantic similarity between response tokens and the image, representing the semantic relevance of tokens to visual content. Subsequently, within the transformer layers, based on the cross-modal attention of response tokens to the image, we compute the global and local relevance of each token to the visual content. Furthermore, we normalize and aggregate the three hierarchical relevance scores, map the result through an activation function, and compute the final reward for each token. Finally, we design a novel Cat-PO loss based on token-level rewards and KL divergence for further optimization. Experiments on open-source datasets and benchmarks demonstrate that our Cat-PO achieves excellent performance, significantly reducing hallucinations and improving response accuracy, thereby enhancing model truthfulness. Concurrently, this work offers a new perspective on mitigating hallucinations by fully exploiting the inherent multimodal capabilities of MLLMs without introducing external technologies or tools.

Our main contributions are summarized as follows:

1. We propose a Cross-modal Adaptive Token-rewards for Preference Optimization (Cat-PO) in MLLMs. By assigning token-rewards to highlight visually critical tokens and incorporating a fine-grained KL regularization, Cat-PO effectively reduces multimodal hallucinations.

2. We introduce a hierarchical token-rewards calculation method that relies solely on the model’s inherent multimodal capabilities, without introducing any external tools or technologies. Specifically, it first computes global relevance based on cross-modal attention between text and image, then calculates local relevance based on patch entropy, and finally uses cross-modal semantic similarity for further refinement.

3. We conducted extensive experiments across multiple datasets and benchmarks to evaluate the effectiveness of Cat-PO. Notably, on the AMBER-Generation and MM-Hal benchmarks, our proposed Cat-PO outperforms existing state-of-the-art methods by 7% – 15% metrics.

## 2 RELATED WORKS

## 2.1 MLLMS HALLUCINATION

MLLMs hallucination refers to outputs that are factually inconsistent with the input image, such as identifying non-existent objects, misdescribing attributes, or misinterpreting relationships. For example, mentioning a “dog” in a landscape image that contains no animals Bai et al. (2024).

To address hallucinations in MLLMs, researchers have proposed a variety of strategies, which broadly classified as training-free or training-based Xiao et al. (2025). Training-free methods, including decoding strategies like Opera Huang et al. (2024) and VCD Leng et al. (2024). Training-based approaches reduce hallucinations through further training. Among these, preference learning such as RLHF Christiano et al. (2017) are prominent for their efficiency and effectiveness.

## 2.2 PREFERENCE LEARNING FOR HALLUCINATION

Preference learning was initially applied to LLM alignment via methods such as RLHF with PPO. These approaches typically necessitate an explicit reward model and involve complex reinforcement learning. Recently, DPO has gained widespread adoption as a simpler and more stable alternative to traditional alignment techniques. HA-DPO Zhao et al. (2023) constructs high-quality sample pairs for preference learning. POVID Zhou et al. (2024a) creates a fine-grained DPO dataset by injecting hallucinated text and adding noise to images. MDPO Wang et al. (2024) addresses “unconditional preference,” where the model may ignore image. CSR Zhou et al. (2024b) iteratively constructs a preference dataset by self-generating responses, integrating visual constraints into reward modeling.

Recently, RLHF-V Yu et al. (2024) collects segment-level human preference data and performs dense DPO training. TPO Gu et al. (2024) explores token-level information in DPO for LVLMs. V-DPO Xie et al. (2024) pairs response preferences with image-contrast preferences and employs vision-guided DPO to reinforce visual context learning.

![](images/09acf7806e814d2e72c068486f52af0d1aaa620e494b805301a481542c42a56e.jpg)  
Figure 2: Overview of our proposed Cat-PO framework: (1) The visual images are first projected into the feature space via CLIP+ViT, and the textual question/response tokens are embedded by LLM tokenizer. (2) Cross-modal attention and semantic similarity are extracted in the multi-modal transformer to hierarchically form the global, local, and semantic relevance scores. (3) Token weights are computed by normalizing these scores with positive/negative sample formulas. (4) The weights are integrated into the standard DPO loss to enhance alignment and mitigate hallucinations.

## 3 METHODOLOGY

The overview of our proposed Cat-PO is illustrated in Fig. 2. We first introduce traditional DPO in Sec. 3.1. Then, we introduce the Hierarchical Visual Relevance of Tokens in Sec. 3.2 and Token-rewards in Sec. 3.3. Lastly, we describe our novel Cat-PO Loss in Sec. 3.4.

## 3.1 PRELIMINARIES: DIRECT PREFERENCE OPTIMIZATION (DPO)

DPO directly optimizes the model through a contrastive learning objective, making it more inclined to generate human-preferred responses while reducing the probability of generating dispreferred responses. DPO learns from preference data $( x , y ^ { + } , y ^ { - } ) \sim \mathcal { D }$ , where x is the input prompt, $y ^ { + }$ is the human-preferred /chosen response, $y ^ { - }$ is the dispreferred /rejected response, and $\mathcal { D }$ is the dataset.

The DPO objective function originates from Bradley-Terry model, which assume that the human preference probability $p ^ { * } ( y ^ { + } \succ y ^ { - } \mid x )$ can be modeled via a latent reward function $r ^ { * } ( x , y ) \colon$ $\begin{array} { r } { \dot { p } ^ { * } ( y ^ { + } \succ \dot { y ^ { - } } \mid x ) = \dot { \sigma } ( r ^ { * } ( x , y ^ { + } ) - \dot { r } ^ { * } ( x , y ^ { - } ) ) } \end{array}$ ). DPO further relates the reward function to the model’s policy $\pi _ { \theta }$ and a reference policy $\pi _ { \mathrm { r e f } } \colon r \ast ( x , y ) = \beta ( \log ( \pi _ { \theta } ( y \mid x ) ) - \log ( \pi _ { \mathrm { r e f } } ( y \mid x ) ) )$

where $\beta$ is a hyperparameter controlling the ratio between reward function and policy deviation. DPO’s loss can directly optimize the model to maximize the probability of generating $y ^ { + }$ and minimize generating $y ^ { - }$ <sup>−</sup>. For a given preference pair $( x , y ^ { + } , y ^ { - } )$ , DPO loss function is defined as:

$$
\mathcal {L} _ {\mathrm{DPO}} = - \log \sigma \left(\beta \left(\log \frac {\pi_ {\theta} (y ^ {+} \mid x)}{\pi_ {r e f} (y ^ {+} \mid x)} - \log \frac {\pi_ {\theta} (y ^ {-} \mid x)}{\pi_ {r e f} (y ^ {-} \mid x)}\right)\right)\tag{1}
$$

By minimizing this loss function, the model $\pi _ { \theta }$ is trained to increase the difference between the log-probabilities of preferred and dispreferred responses, relative to the reference model $\pi _ { r e f } .$ This direct method makes DPO simpler and demonstrates comparable or superior performance to RLHF.

## 3.2 HIERARCHICAL VISUAL RELEVANCE OF TOKENS

Without any external tools or techniques, we leverage the intrinsic multimodal capabilities of MLLMs, to hierarchically compute each token’s global, local, and semantic relevance to the visual input.

## 3.2.1 CROSS-MODAL ATTENTION BASED GLOBAL RELEVANCE

When MLLMs process DPO training data within the Transformer architectures, the feature representation of each token in a response interacts with visual features via a cross-modal attention mechanism. The activation distribution of these attention scores intuitively reflects the focus of specific text tokens on different image regions. Leveraging this, we define and compute a global relevance score for each token concerning the visual content, thereby quantifying its overall association with the image.

Specifically, for a given image I and its corresponding preferred response $y _ { w }$ or rejected response $y _ { l }$ (collectively denoted $y )$ from the dataset, the representation of the t-th token $y _ { t }$ in y serves as the query. The set of $N _ { p }$ visual token features, $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N _ { p } } \}$ , derived from image I via a visual encoder, acts as the keys and values. The sequence of cross-modal attention scores from token $y _ { t }$ to all $N _ { p }$ visual tokens is denoted by $\{ a _ { t , 1 } , a _ { t , 2 } , \ldots , a _ { t , N _ { p } } \}$

The global relevance $S _ { \mathrm { g l o b a l } } ( y _ { t } )$ is defined as the sum of the attention scores for token $y _ { t }$ :

$$
S _ {\mathrm{global}} (y _ {t}) = \sum_ {j = 1} ^ {N _ {p}} a _ {t, j}\tag{2}
$$

A higher $S _ { \mathrm { g l o b a l } } ( y _ { t } )$ indicates that the model attends more intensively to the entire image when processing token $y _ { t }$ , implying a stronger global alignment between the token and the visual content.

## 3.2.2 PATCH ENTROPY BASED LOCAL RELEVANCE

Although the global relevance score $S _ { \mathrm { g l o b a l } } ( y _ { t } )$ ) captures the overall association between response tokens and visual content, it does not reveal whether attention is concentrated on key regions or dispersed across the image. Typically, focused attention indicates a strong link to specific local information, while dispersed attention suggests higher uncertainty or weaker visual grounding.

To accurately characterize the focusing pattern within this attention distribution, we leverage the concept of information entropy to compute the patch entropy scores for each token $y _ { t }$ based on its image attention distribution. First, we extract the cross-modal attention vector $a _ { t } = [ a _ { t , 1 } , a _ { t , 2 } , \ldots , a _ { t , N _ { p } } ]$ for token $y _ { t }$ with respect to all $N _ { p }$ image patches, where $a _ { t , j }$ represents the attention strength of $y _ { t }$ towards the j-th image patch. Next, we normalize the attention strengths in $a _ { t }$ to form a probability distribution $P _ { t } = [ P _ { t , 1 } , P _ { t , 2 } , . . . , P _ { t , N _ { v } } ]$ , and $\begin{array} { r } { P _ { t , j } = a _ { t , j } \ / \ \sum _ { k = 1 } ^ { N _ { p } } a _ { t , k } } \end{array}$ . We then compute the patch entropy $H ( P _ { t } )$ of this probability distribution. To ensure numerical stability in the logarithm, a small epsilon value $\bf { \dot { \epsilon } } ( e . g . , 1 \bar { 0 } ^ { - 1 2 } )$ is introduced:

$$
H (P _ {t}) = - \sum_ {j = 1} ^ {N _ {p}} P _ {t, j} \log (P _ {t, j} + \epsilon)\tag{3}
$$

This entropy value $H ( P _ { t } )$ measures the uncertainty or dispersion of the attention distribution. Subsequently, for $N _ { p } > 1$ , we normalize the computed entropy $H ( P _ { t } )$ , and the Patch Entropy Score, $S _ { \mathrm { { e n t r o p y } } } ( y _ { t } )$ ), is then defined as 1 minus this normalized entropy:

$$
S _ {\mathrm{local}} (y _ {t}) = 1 - \frac {H (P _ {t})}{\log N _ {p}} \quad (\text { for } N _ {p} > 1)\tag{4}
$$

A higher $S _ { \mathrm { l o c a l } } ( y _ { t } )$ score indicates lower entropy in the attention distribution, implying that attention is more sharply focused on a few image patches. This generally signifies a stronger degree of association between the token $y _ { t }$ and specific local regions of the image.

## 3.2.3 CROSS-MODAL SIMILARITY BASED SEMANTIC RELEVANCE

Beyond the global and local relevance, we exploit a prior semantic signal obtained from a pretrained cross–modal encoder to quantify the semantic alignment between response tokens and visual content. Given a sample $( I , y )$ , let the embedding of the t-th token be $\mathbf { e } ( y _ { t } )$ . The image I is divided into $N _ { p }$ patches, each encoded as a visual feature $\{ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { N _ { p } } \}$ . With the cross–modal attention weights $\alpha _ { t , j }$ (normalized over patches), we obtain a context–aware visual vector: $\begin{array} { r } { \mathbf { v } _ { c } ( y _ { t } ) = \sum _ { j = 1 } ^ { N _ { p } } \alpha _ { t , j } \mathbf { v } _ { j } } \end{array}$ The semantic relevance score is then defined as

$$
S _ {\mathrm{semantic}} (y _ {t}) = \cos \bigl (\mathbf {e} (y _ {t}), \mathbf {v} _ {c} (y _ {t}) \bigr) = \frac {\mathbf {e} (y _ {t}) \cdot \mathbf {v} _ {c} (y _ {t})}{\| \mathbf {e} (y _ {t}) \| \| \mathbf {v} _ {c} (y _ {t}) \|}.\tag{5}
$$

This score captures the semantic relevance between the token representation and the visual content of its most attended region, complementing the attention-based global and local relevance.

## 3.3 TOKEN WEIGHTING SCHEME

Unified Visual Relevance Score: After obtaining hierarchical relevance scores for every response token $y _ { i } .$ , we normalize all scores to [0, 1] and fuse them into a unified visual relevance score.

$$
s _ {i} = \alpha \left[ 0. 5 * S _ {\text { global }, i} + 0. 5 * S _ {\text { local }, i} \right] + (1 - \alpha)   S _ {\text { semantic }, i}, \quad \alpha \in [ 0, 1 ].\tag{6}
$$

Here, α balances the attention branch (global & local) against the semantic branch.

Smooth Mapping to Token Weights: Directly injecting $s _ { i }$ into the loss may yield unstable gradients. We therefore apply a smooth non-linearity: $\dot { T } _ { i } \stackrel { \cdot } { = } \operatorname { t a n h } ( s _ { i } ) \in ( 0 , 1 )$ , and introduce a base weight $\lambda _ { \mathrm { { r e f } } } > 0$ to maintain a controlled dynamic range:

$$
w _ {i} = \left\{ \begin{array}{l l} \lambda_ {\text { ref }} + T _ {i}, & y _ {i} \in y ^ {+}, \\ \lambda_ {\text { ref }} + (1 - T _ {i}), & y _ {i} \in y ^ {-}. \end{array} \right.\tag{7}
$$

This design (i) rewards tokens in the preferred response that strongly align with the image $( T _ { i } \uparrow )$ , and (ii) penalises hallucinated or weakly aligned tokens in the dispreferred response $( ( 1 - \bar { T } _ { i } ) \uparrow )$ .

## 3.4 WEIGHTED INTEGRATION AND KL-REGULARISED CAT-PO LOSS

Incorporating token weights $\{ w _ { t } ^ { + } , w _ { t } ^ { - } \}$ and token-level KL into the DPO loss yields the Cat-PO loss.

Weighted DPO. For a preference pair $( y ^ { + } , y ^ { - } )$ , we weight the log-likelihood ratio of the policy π and the reference $\pi _ { r e f }$ . The weighted policy $\pi _ { \boldsymbol { \theta } } ^ { ( w ) }$ is defined as $\begin{array} { r } { \pi _ { \theta } ^ { ( w ) } = \sum _ { t } ( w _ { t } ^ { + } } \end{array}$ log $\pi _ { \theta } ( y _ { t } ^ { + } \mid h _ { t } ^ { + } ) -$ $w _ { t } ^ { - } \log \pi _ { \theta } ( y _ { t } ^ { - } \mid h _ { t } ^ { - } ) )$ , and the weighted reference $\pi _ { r e f } ^ { ( w ) }$ is defined as $\begin{array} { r } { \pi _ { r e f } ^ { ( w ) } = \sum _ { t } ( w _ { t } ^ { + } } \end{array}$ log $\pi _ { r e f } ( y _ { t } ^ { + } \mid$ $h _ { t } ^ { + } ) - w _ { t } ^ { - }$ log $\pi _ { r e f } ( y _ { t } ^ { - } \mid h _ { t } ^ { - } ) \Big )$ . Thus, the weighted DPO loss is defined as:

$$
\mathcal {L} _ {\mathrm{wDPO}} = - \log \sigma \big [ \beta \big (\pi_ {\theta} ^ {(w)} - \pi_ {r e f} ^ {(w)} \big) \big ]\tag{8}
$$

Token-weighted KL regulariser. To keep the policy close to the reference and to stabilise training, with a regularisation strength $\lambda > 0 .$ , we add a token-level, weight-modulated KL penalty:

$$
\mathcal {L} _ {\mathrm{KL}} = \lambda \left(\sum_ {t} w _ {t} ^ {+} \operatorname{KL} \left[ \pi_ {\theta} (\cdot | h _ {t} ^ {+}) \| \pi_ {r e f} (\cdot | h _ {t} ^ {+}) \right] + \sum_ {t} w _ {t} ^ {-} \operatorname{KL} \left[ \pi_ {\theta} (\cdot | h _ {t} ^ {-}) \| \pi_ {r e f} (\cdot | h _ {t} ^ {-}) \right]\right)\tag{9}
$$

The final Cat-PO Loss objective is:

$$
\mathcal {L} _ {\mathrm{Cat-PO}} = \mathcal {L} _ {\mathrm{wDPO}} + \mathcal {L} _ {\mathrm{KL}}\tag{10}
$$

Minimising equation 10 enables the policy model encode human preferences and fine-grained token–vision alignments, suppressing hallucinations while preserving generation quality.

## 4 EXPERIMENTS

## 4.1 DATASETS AND METRICS

Training Data: Our experiments primarily employ the widely used RLHF-V dataset Yu et al. (2024). It comprises 5,733 samples, each including an image, a question, a high-quality response, and a relatively low-quality response. We use these data to compute token-weights and train our model.

Evaluation Benchmarks: To comprehensively evaluate the model’s performance in reducing hallucinations and enhancing general capabilities, we employ several widely used benchmarks:

For hallucination evaluation, AMBER Wang et al. (2023) is a LLM-free benchmark which consists of two main sub-tasks: (a) Discrimination: Determining whether a given statement about an image is correct or not. (b) Generation: Generating a descriptive text based on the image and question. MM-Hal Sun et al. (2023) evaluates response-level hallucination rate and informativeness. It requires GPT-4 to compare model outputs with human responses and object labels for evaluation.

Table 1: Performance comparison on the Discrimination and Generative of AMBER Wang et al. (2023), MM-Hal Sun et al. (2023), LLaVA-Bench Liu et al. (2023b) and SEED Li et al. (2023a) benchmarks. All methods are based on LLaVA-v1.5-7B and -13B Liu et al. (2023b) models with the RLHF-V Yu et al. (2024) dataset, with the best results highlighted in bold.

<table><tr><td rowspan="2">Method</td><td colspan="2">AMBER-Disc</td><td colspan="3">AMBER-Gene</td><td colspan="2">MM-Hal</td><td rowspan="2">LLaVA ↑</td><td rowspan="2">SEED ↑</td></tr><tr><td>Acc ↑</td><td>F1 ↑</td><td>CHAIR ↓</td><td>Hal ↓</td><td>Cog ↓</td><td>Score ↑</td><td>Rate ↓</td></tr><tr><td>LLaVA-v1.5-7B</td><td>71.7</td><td>74.3</td><td>7.8</td><td>36.4</td><td>4.2</td><td>2.01</td><td>61.4</td><td>65.6</td><td>66.1</td></tr><tr><td>+ DPO Rafailov et al. (2023)</td><td>77.5</td><td>82.1</td><td>5.7</td><td>27.3</td><td>2.6</td><td>2.14</td><td>58.3</td><td>69.1</td><td>66.4</td></tr><tr><td>+ CSR Zhou et al. (2024b)</td><td>73.2</td><td>76.1</td><td>5.4</td><td>25.5</td><td>2.6</td><td>2.05</td><td>60.4</td><td>68.9</td><td>65.9</td></tr><tr><td>+ POVID Zhou et al. (2024a)</td><td>71.9</td><td>74.7</td><td>5.7</td><td>26.9</td><td>3.0</td><td>2.26</td><td>55.2</td><td>68.2</td><td>66.1</td></tr><tr><td>+ RLHF-V Yu et al. (2024)</td><td>74.8</td><td>78.5</td><td>5.5</td><td>26.3</td><td>2.5</td><td>2.02</td><td>60.4</td><td>68.0</td><td>66.1</td></tr><tr><td>+ V-DPO Xie et al. (2024)</td><td>-</td><td>81.6</td><td>5.6</td><td>27.3</td><td>2.7</td><td>2.16</td><td>56.0</td><td>-</td><td>-</td></tr><tr><td>+ TPO Gu et al. (2024)</td><td>79.3</td><td>85.0</td><td>-</td><td>-</td><td>-</td><td>2.47</td><td>51.0</td><td>70.2</td><td>66.6</td></tr><tr><td>+ Cat-PO (Ours)</td><td>78.0</td><td>85.3</td><td>4.8</td><td>23.7</td><td>2.1</td><td>2.74</td><td>42.0</td><td>70.3</td><td>67.0</td></tr><tr><td>LLaVA-v1.5-13B</td><td>71.3</td><td>73.1</td><td>7.0</td><td>33.1</td><td>3.3</td><td>2.38</td><td>53.1</td><td>73.1</td><td>68.2</td></tr><tr><td>+ DPO (Rafailov et al., 2023)</td><td>83.2</td><td>86.9</td><td>6.1</td><td>26.3</td><td>2.7</td><td>2.47</td><td>51.0</td><td>72.8</td><td>68.6</td></tr><tr><td>+ TPO (Gu et al., 2024)</td><td>83.9</td><td>88.0</td><td>-</td><td>-</td><td>-</td><td>2.72</td><td>45.8</td><td>72.8</td><td>68.7</td></tr><tr><td>+ Cat-PO (Ours)</td><td>82.9</td><td>88.0</td><td>4.3</td><td>22.0</td><td>1.6</td><td>2.85</td><td>42.0</td><td>74.3</td><td>69.2</td></tr></table>

For general capability evaluation, LLaVA-Bench Liu et al. (2023b) is a comprehensive benchmark that uses GPT-4 scoring to evaluate model generalization. SEED-Bench Li et al. (2023a) is a large-scale multimodal benchmark assessing visual understanding and text/image generation.

## 4.2 IMPLEMENTATION DETAILS

In our experiments, we leverage the widely adopted LLaVA-v1.5 Liu et al. (2023a) and Qwen2.5- VL Bai et al. (2025) series models to evaluate the scalability and effectiveness. The training of main experiment was performed over 6 epochs with an effective batch size of 32, implemented through gradient accumulation. And the DPO hyperparameter β<sub>DPO</sub> set to 0.1.

## 4.3 MAIN RESULTS

We compare Cat-PO with advanced preference alignment methods, which include: DPO Rafailov et al. (2023). CSR, Zhou et al. (2024b) A calibrated self-rewarding method. POVID, Zhou et al. (2024a) A GPT-4V based alignment method. RLHF-V, Yu et al. (2024) A method that segments human preference collection. TPO, Gu et al. (2024) A DPO variant employing self-calibrated, visually anchored rewards. V-DPO, Xie et al. (2024) A vision-guided DPO variant.

In table 1, we evaluate existing methods and Cat-PO across multiple benchmarks. On the AMBER-Generation and MM-Hal, Cat-PO significantly improves response quality while effectively reducing hallucinations. On the AMBER-Discrimination, it achieves competitive performance, highlighting its ability to evaluate image-related statements. Furthermore, on general benchmarks such as LLaVA-Bench and SEED-Bench, Cat-PO also remains outstanding.

To verify the cross-model generalization of Cat-PO, we also conduct experiments on Qwen-2.5VL-3B Bai et al. (2025). As shown in Fig 3, Cat-PO achieves improvements over Qwen and Qwen+DPO on MM-Hal and AMBER benchmarks, demonstrating its strong generalization ability.

![](images/c0addee48ba6d5764af6522011ffe40dca9726031ffbf04bca4a5c8a5e7a08f5.jpg)  
Figure 3: Performance comparison of different Qwen2.5-VL models in terms of AMBER and MM-Hal Benchmarks.

![](images/1ffa06d1828ab7e6dc5faef5f33cdc8a75f80d06adb12f9b9b7da0151a9a0e24.jpg)  
(a)Weighted Token Ratio

![](images/616abd806904d0b8008c63659e1f4d861e83d11e18cda0b317a8deb6e60b566d.jpg)  
(b)Alpha ( ) Value

![](images/e79a7a8045b2ffad2a632c78cc17e7138bba67fc3d6c8362a96effcf2b8f22cd.jpg)  
(c) (KL) Value

![](images/f0f8ed61d6d3224dda8f738f346158c4f8a5a03527b620b588858dc689d5745d.jpg)  
(d)Data Category  
Figure 4: The performance comparison of (a) different weighting proportions, (b)(c) important hyper-parameters α $/ \lambda _ { K L } ,$ , and (d) weighting positions in our proposed Cat-PO framework.

## 4.4 ABLATION STUDY

Modules’ Contribution. To further validate Cat-PO’s effectiveness, we conducted a comprehensive ablation study in Table 2. The results show that cross-modal attention relevance and semantic relevance play critical roles: weighting either alone improves performance, but their combination yields even greater gains. Moreover, removing the KL-based loss optimization causes a performance drop, confirming the necessity of the token-level KL term.

Impact of Weighted-Tokens Proportion. We further investigate how varying the proportion of weighted tokens in the chosen responses affects Cat-PO performance. As shown in Figure 4 (a), performance steadily improves with increasing weight proportion. However, applying weights to only the top 50% of tokens yields a smaller gain than to the top 30%, indicating that weighting the remaining 50% also provides a notable contribution to overall performance.

Hyperparameter Analysis. We examine two key hyperparameters in Cat-PO: (1) Balance Coefficient α. Figure 4 (b) shows that both excessively large and small values of α impair performance, underscoring the need to balance cross-modal attention and semantic relevance; (2) KLdivergence Coefficient λ<sub>KL</sub>. Figure 4 (c) demonstrates that $\lambda _ { \mathrm { K L } } ~ = ~ 0 . 0 3$

Table 2: Performance of individual Cat-PO modules.

<table><tr><td rowspan="2">Modules</td><td colspan="2">MM-Hal</td><td colspan="2">AMBER-Gene</td></tr><tr><td>Score ↑</td><td>Rate ↓</td><td>CHAIR ↓</td><td>Hal ↓</td></tr><tr><td>DPO-only</td><td>2.14</td><td>58.3</td><td>5.7</td><td>27.3</td></tr><tr><td>Attention-only</td><td>2.34</td><td>55.0</td><td>5.3</td><td>25.9</td></tr><tr><td>Similarity-only</td><td>2.51</td><td>47.0</td><td>5.1</td><td>27.7</td></tr><tr><td>Cat-PO without KL</td><td>2.36</td><td>53.0</td><td>5.1</td><td>26.9</td></tr><tr><td>Cat-PO (Ours)</td><td>2.74</td><td>42.0</td><td>4.8</td><td>23.7</td></tr></table>

achieves the optimal trade-off between maintaining model flexibility and constraining deviation from the reference distribution.

![](images/00b6f97578c8ce498c3ef64fb83af98b6d204a1a7d31631c2524b7b4c4a7e415.jpg)  
Please describe the image

![](images/ba9217d9b0d8d9cdcd700c4c9228556b98eb01b8a673c1a4165db4053ce85e86.jpg)

![](images/7461a5526487bc6ec8225d6a549bf81dad928b8aede7bf3a42b28582909efaff.jpg)  
Figure 5: Token–level attention, similarity, and aggregated scores for a single example. The aggregated calculation lifts truly visual tokens (blue) while down-weighting hallucination tokens.

## 4.5 ANALYSIS EXPERIMENTS

The Effectiveness of Token-Weights Calculation. To further validate the effectiveness of our computed token weights, we applied weighting to either the top 30% or the bottom 30% of tokens. As shown in Figure 4 (d), weighting the top 30% of tokens yields a significant improvement compared to weighting the bottom 30%. This result confirms the accuracy of our weight computation, and demonstrates that key tokens play a decisive role in enhancing the alignment capability.

Table 3: CatPO performance on overall and adversarial subsets of POPE Li et al. (2023c).  
Table 4: Comparison of Cat-PO (general) and Cat-PO (with learnable fusion).

<table><tr><td>CatPO’s Score</td><td>Acc.</td><td>Precision</td><td>F1</td></tr><tr><td>Average</td><td>85.6</td><td>95.2</td><td>84.0</td></tr><tr><td>Adversarial (most difficult)</td><td>84.0 (-2%)</td><td>91.3 (-4%)</td><td>82.5 (-2%)</td></tr></table>

<table><tr><td>Model</td><td>MM-Hal (↑)</td><td>Hal-Rate (↓)</td></tr><tr><td>Cat-PO (general)</td><td>2.74</td><td>42%</td></tr><tr><td>Cat-PO(w/ learnable fusion)</td><td>2.55</td><td>50%</td></tr></table>

The Analysis of Training logits. Training logs and visualizations further demonstrate its stability. We track the evolution of the training loss and the reward margin throughout optimization. As shown in Fig. 6, the loss curve decreases smoothly, while the reward margin increases steadily without noticeable oscillations. The monitored gradient norms also remain stable. These observations indicate that Cat-PO maintains good optimization stability under token-level reward modulation.

![](images/fdab0f2e53e219d51e97e0d2dcbdbf297baee1f674762975d109b2789109445c.jpg)

![](images/2c777527ea9881b803d5d41822ea80d4d440a0ada880016d63c60b445544a15e.jpg)  
Figure 6: Training dynamics of Cat-PO, showing a smoothly decreasing loss and a steadily increasing reward margin with relatively stable gradient norms, indicating relatively stable optimization.

Evaluation on Edge-Case Diagnostics We additionally include POPE benchmark Li et al. (2023c) (with Object, Relation, and Adversarial subset). The Adversarial subset is explicitly designed to elicit visual hallucinations and is widely regarded as the hardest visual alignment benchmark.

Table 3 shows that Cat-PO’s scores on the adversarial subset are slightly lower than average, indicating that our rewards remain relatively robust even under highly biased and adversarial edge-case settings.

Learnable Fusion Weights. We modified the fusion Eq. 6 to be learnable parameters as follows:

$$
s _ {i} = \gamma * S _ {\text { global }, i} + \delta * S _ {\text { local }, i} + (1 - \gamma - \delta) * S _ {\text { semantic }, i}, \quad \gamma , \delta \in [ 0, 1 ].\tag{11}
$$

with $\gamma$ and δ are the learnable parameters. Then, we jointly optimized them within training loss. Table 4 show that introducing learnable coefficients yields the performance below the original one. Learnable fusion showed no benefit, possibly because (1) DPO does not directly supervise weight allocation, causing learnable coefficients to overfit noise, and (2) Cat-PO mainly gains from the complementary cross-modal signals, making the fixed uniform weighting a more stable design.

Robustness Analysis of Token-reward Calculation. To further assess robustness in edge scenarios, we visualize rare cases where attention and similarity disagree (Fig. 7). For “horse”, misaligned attention is corrected by high semantic similarity, while for “train”, an abnormally low similarity score is compensated by sharply focused attention. These examples show that Cat-PO’s fused multi-signal scoring mutually compensates single-branch failures instead of amplifying isolated alignment errors.

From a theoretical perspective, Cat-PO computes each token-level reward by fusing three complementary cross-modal signals: global attention, local patch entropy, and cross-modal semantic similarity. A token receives a high reward only when it simultaneously exhibits strong attention strength, a stable focus pattern, and high semantic consistency with visual content. This complementarity reduces reliance on any single noisy branch and structurally limits the misalignment.

Training Overhead Analysis. (1) Only one-time pre-computation: Pre-computing token-level rewards for all positive and negative samples takes 2h16m18s. This cost is incurred only once and the results can be reused indefinitely. (2) We compared the training overhead of Cat-PO with DPO. Table 5 shows that Cat-PO matches DPO across all metrics, introducing only marginal overhead.

Similarity (semantic) = 0.7594918817754842 ( )

Similarity (semantic) = 0.04862285485427614 ( )

Attention (global) = 0.26805039819690357 ( ）

Data idx = 1520, RLHF-V Dataset

![](images/959de8eab86cbc77d13af75cd009721d2101b24d020029a7abf3be3cd18682de.jpg)  
Edge Case of Attention Misalignment, but Similarity Corrects the Rewards  
Token = "horse" —— Attention Map ( )  
Data idx = 2290, RLHF-V Dataset

![](images/4e10a3528dec2c636f81b36f9c5c0c33c59cd6550c74cc02a419c10382b6b2b4.jpg)  
Edge Case of Similarity Misalignment, but Attention Corrects the Rewards  
Token = "train" —— Attention Map ( )  
Attention (global) = 0.08606535786179087 ( )  
Token-Rewards = 0.9820886664604802 ( )  
Token-Rewards = 0.7167500575241625 ( )

Figure 7: Robustness Analysis in Corner Cases of Token-Reward Computation. Blue highlights indicate regions of concentrated attention. All the data are from RLHF-V Dataset Yu et al. (2024). Left Part: When visual attention for the token "horse" is misaligned (rarely distributed in horses), the high semantic similarity effectively rectifies the final reward. Right Part: Conversely, when the semantic similarity for "train" is anomalously low, the focused attention distribution (correctly distributed in train) ensures a reasonable reward. This demonstrates that the fused metrics in Cat-PO mutually compensate for single-metric misalignments, ensuring reliable preference optimization.

## 4.6 CASE STUDY

Attention–Similarity Fusion for Token Scoring. Figure 5 illustrates attention scores (global and local), similarity scores, and their weighted sum for a sample. Attention or similarity alone can distinguish visually critical from fact-violating tokens, validating each module’s utility. However, biases exist: in the chosen sample, “floor” has a low attention score, while “on” shows low similarity. But fusing the two signals ranks critical tokens higher, enabling precise token weighting in Cat-PO and further supporting our weighting strategy.

Comparison of Cat-PO and DPO Generations. The comparison is presented in Appendix A.6.

## 5 CONCLUSION

Table 5: Training comparison of DPO vs. Cat-PO: average processing time of per sample and the peak memory usage.

<table><tr><td>Model</td><td>Avg. time (s)</td><td>Peak Memory Usage (GB)</td></tr><tr><td>DPO</td><td>2.1s</td><td>40.420</td></tr><tr><td>Cat-PO (Ours)</td><td>2.9s (+38%)</td><td>40.450 (+0.07%)</td></tr></table>

In this paper, we propose Cross-modal Adaptive Token-rewarded Preference Optimization (Cat-PO) for mitigating

hallucinations and improving MLLM truthfulness. Each token’s global, local, and semantic relevance is computed from cross-modal attention and similarity, fused and incorporated into the DPO loss for fine-grained optimization. Experiments on public benchmarks show that Cat-PO effectively reduces hallucinations, improves response accuracy, enhances MLLMs truthfulness.

## ACKNOWLEDGMENTS

This work was supported in part by the Artificial Intelligence-National Science and Technology Major Project under Grant 2023ZD0121200.

This work was supported by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM103).