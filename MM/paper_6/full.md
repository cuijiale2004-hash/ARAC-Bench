## ABSTRACT

Event-based multimodal large language models (MLLMs) enable robust perception in high-speed and low-light scenarios, addressing key limitations of frame-based MLLMs. However, current event-based MLLMs often rely on dense image-like processing paradigms, overlooking the spatiotemporal sparsity of event streams and resulting in high computational cost. In this paper, we propose EventFlash, a novel, efficient MLLM to explore spatiotemporal token sparsification for reducing data redundancy and accelerating inference. Technically, we built EventMind, a largescale and scene-diverse dataset with over 500k instruction sets, providing both short and long event stream sequences to support our curriculum training strategy. Then, we present the adaptive temporal window aggregation module for efficient temporal sampling, which adaptively compresses temporal tokens while retaining key temporal cues. Finally, the sparse density-guided attention module is designed to improve spatial token efficiency by selecting informative regions and suppressing empty or sparse areas. Experimental results show that EventFlash achieves a 12.4× throughput improvement over the baseline (EventFlash-Zero) while maintaining comparable performance. It supports long-range event stream processing with up to 1,000 bins, significantly outperforming EventGPT’s 5-bin limit. We believe EventFlash serves as an efficient foundation model for event-based vision. Our codes will be released in https://github.com/XduSyL/EventFlash

## 1 INTRODUCTION

Event cameras (Gallego et al., 2020; Posch et al., 2014; Li & Tian, 2021), bio-inspired vision sensors, operate differently from frame-based cameras. Each pixel responds to intensity changes by generating asynchronous events (Li et al., 2022; Kudithipudi et al., 2025). Due to their high temporal resolution and high dynamic range, event cameras have been applied to various vision tasks (e.g., scene understanding (Zhu et al., 2018; Kong et al., 2024; Yao et al., 2024; Zhou et al., 2024; Li et al., 2025a; Liu et al., 2025b; Li et al., 2025b; Zhou & Lee, 2025)) in high-speed or low-light scenarios.

Recent multimodal large language models (MLLMs) (Xiang et al., 2025; Li et al., 2024a; Tang et al., 2025; Huang et al., 2024a; Qian et al., 2024; Fang et al., 2026a;b; 2024a; Xu et al., 2024; Yan et al., 2025b) have achieved remarkable breakthroughs in processing conventional frames and language, showing strong capabilities in scene understanding and visual question answering. However, these models are primarily designed for frame-based inputs and cannot directly handle the unique spatiotemporal properties of event streams. A straightforward approach to extending MLLMs to eventbased vision involves converting event streams into dense, image-like representations before feeding them into existing MLLMs (e.g., LLaVA (Liu et al., 2023), GPT-4 (Bubeck et al., 2023), or Qwen (Bai et al., 2023)). However, this transformation often overlooks the inherent spatiotemporal sparsity of event data and introduces substantial redundancy (Gehrig & Scaramuzza, 2024; Messikommer et al., 2020; Wu et al., 2024a). In other words, applying dense image-like processing paradigms to event streams not only incurs significant computational overhead but also substantially limits the effective length and efficiency of event stream understanding. Thus, developing efficient MLLMs that fully exploit the unique spatiotemporal properties of event data remains a critical and unresolved challenge.

Despite recent progress, most existing event-based MLLMs (Liu et al., 2025b; Li et al., 2025b; Zhou & Lee, 2025; Liu et al., 2025a) still rely on dense image-like representations, which hinders computational efficiency and scalability to long event sequences. For example, EventGPT (Liu et al., 2025b) converts event streams into dense token sequences for language modeling. EventVL (Li et al., 2025b) integrates RGB frames with event data to enhance multimodal reasoning. LLaFEA (Zhou & Lee, 2025) employs frame-event fusion for region-level spatiotemporal grounding. Although these models perform well in challenging scenarios such as high-speed motion and low-light conditions, their dense processing of sparse event data leads to significant overhead and limits real-time or longrange understanding. Meanwhile, the scene diversity of their datasets is relatively limited, and the event streams are short, making it difficult to support general-purpose models for long event-stream understanding.

In this paper, we propose EventFlash, a novel efficient MLLM that leverages spatiotemporal token sparsification to reduce data redundancy and accelerate inference. Unlike prior works that focus on maximizing reasoning accuracy, our goal is to address three key challenges in efficient MLLMs: (i) Temporal inefficiency: The microsecond resolution of event streams results in prohibitively large token volumes when processed over long temporal durations; (ii) Spatial inefficiency: The inherent sparsity of event data leads to numerous empty or low-information tokens that incur computational overhead due to uniform attention allocation; (iii) Dataset limitations: Existing instruction-augmented datasets are not publicly available and often lack diversity, contain low-quality annotations, and cover short temporal sequences, making them inadequate for training generalizable models.

To address these challenges, our EventFlash presents a density-aware spatiotemporal token sparsification strategy that exploits the inherent sparsity and high temporal resolution of event streams. Specifically, we propose an adaptive temporal window aggregation module for efficient temporal sampling, which dynamically compresses temporal tokens while preserving essential temporal cues. Then, a sparse density-guided attention module is presented to enhance spatial token efficiency by selecting informative regions and suppressing empty or low-density areas. Moreover, we design a progressive curriculum learning strategy following a short-to-long paradigm to improve EventFlash’s generalization and generative capabilities. To support this, we built a large-scale scene-diverse dataset over 500k instruction sets, including both short and long event stream sequences. Experimental results show that EventFlash achieves a 12.4× improvement in throughput over our baseline (EventFlash-Zero) while maintaining comparable performance. Notably, EventFlash enables long-range event stream processing of up to 1,000 bins compared to only 5 bins in the competing EventGPT.

In summary, the main contributions of this work are:

• We propose EventFlash, an efficient event-based vision MLLM, which explores a spatiotemporal token sparsification strategy for raw event streams to reduce redundancy, accelerate inference (12.4× throughput), and enable long-range event stream understanding (up to 1,000 bins).

• We present a density-aware spatiotemporal token sparsification strategy for event-based MLLMs, which effectively reduces redundancy while maintaining comparable reasoning accuracy by leveraging the fine-grained temporal resolution and inherent sparsity of raw event streams.

• We build a large-scale scene-diverse dataset for long-range event stream understanding. We believe this standardized dataset will accelerate future research in event-based MLLMs.

## 2 RELATED WORK

Event-based Vision with MLLMs. Early works (Wu et al., 2023; Zhou et al., 2023) have explored the alignment between event data and textual information. Event-CLIP (Wu et al., 2023) builds on pre-trained vision-language models (Radford et al., 2021; Yang et al., 2023; Klenk et al., 2024; Huang et al., 2024b; Tong et al., 2025) for event-based recognition, and EventBind (Zhou et al., 2023) incorporates an event encoder to unify images, events, and texts. Yet both overlook the world knowledge embedded in LLMs, constraining nuanced scene understanding. More recently, emerging event-based MLLMs (Liu et al., 2025b; Li et al., 2025b; Zhou & Lee, 2025) have demonstrated strong reasoning capabilities in challenging conditions. For example, EventGPT (Liu et al., 2025b) is the first to design an event-based MLLM for accurate generation. EventVL (Li et al., 2025b) enhances robustness by fusing complementary modalities from event streams and RGB frames. LLaFEA (Zhou & Lee, 2025) achieves region-level spatiotemporal grounding through the complementary fusion of frame and event modalities. However, these event-based LLMs rely on dense image-like processing of inherently sparse events (Peng et al., 2024; Perot et al., 2020; Vemprala et al., 2021; Zhu et al., 2022; Tulyakov et al., 2019; Qu et al., 2024; Lin et al., 2023; Shrestha & Orchard, 2018; Wu et al., 2024b; Engelken, 2023; Cho et al., 2024; Wan et al., 2022; Mei et al., 2023; Li et al., 2025c; 2023; Chen et al., 2024), leading to excessive computation and hindering long-sequence inference.

![](images/f075f88fb6713c998b6b7677c6edee71f8812cc3d0a115b9a6ed9fc72aae63b9.jpg)  
Figure 1: Instructions and data statistics of our EventMind. (a) Seven tasks instructions for event stream understanding. (b) Data distributions of each task. (c) Data distributions of the three stages.

Efficient Token Sparsification in MLLMs. Recent MLLMs (Weng et al., 2024; Jiang et al., 2025; Qian et al., 2024; Yan et al., 2025a) have revealed that visual tokens extracted from foundation models like CLIP contain substantial redundancy, leading to significant computational overhead. Consequently, several token sparsification strategies (Yehezkel et al., 2024; He et al., 2024; Zhang et al., 2024) have been attempted to reduce token counts while preserving essential semantics in video tasks. However, asynchronous events differfundamentallyfrom structuredframes: while video redundancy mainly stemsfrom spatial repetition within a regular patch grid, event streams consist of sparse spatiotemporal points with redundancy arising from uneven temporal sampling. Their tokens are distributed irregularly and vary in density, makingframe-based sparsification not only computationally costly but also ineffective for long event stream understanding. Thus, this work presents a novel spatiotemporal token sparsification strategy specifically tailored for event streams.

## 3 EVENTMIND DATASET

Data Collection. To support the curriculum learning strategy in our EventFlash, we construct a large scale multimodal dataset named EventMind for event stream understanding. EventMind provides long temporal sequences, diverse scenes, multiple tasks, and high-quality instructions. The raw event data is sourced from both real-world and synthetic domains. Real-world data includes short-duration event sequences from DSEC (Gehrig et al., 2021) and N-ImageNet (Kim et al., 2021), as well as longer-duration streams from HARDVS (Wang et al., 2024b) and E2VID (Rebecq et al., 2019). Synthetic data are generated by converting large-scale video datasets (i.e., Kinetics-700 (Carreira et al., 2019), UCF-101 (Soomro et al., 2012), Wevid-10 M (Bain et al., 2021), PLM-Data (Cho et al., 2025), and MotionBench (Hong et al., 2025)) into event streams using the V2E simulator (Hu et al., 2021). To ensure high-quality simulated events, we use GPT-4o to automatically filter videos using their captions before simulation. To align with our curriculum stages, we categorize them into three groups: short (0–50 ms), medium (50–5,000 ms), and long (5,000–20,000 ms).

Instruction Generation. To evaluate the modeling capacity and generalization ability of our EventFlash, we define seven distinct task types for event stream understanding. As shown in Fig. 1(a), these tasks include motion captioning, event question answering (Event QA), human action QA, multiple-choice QA (MCQA), simple captioning, fine-grained QA (FGQA), and scene captioning.

![](images/ebd6fabc40161c0c872558fcc7b1adaeaf14946bcd7437845a8a9a9e2161231f.jpg)  
Figure 2: The pipeline of efficient MLLMs (EventFlash). The adaptive temporal window aggregation module is presented for efficient temporal sampling, which adaptively compresses temporal tokens while retaining key temporal cues. Besides, the sparse density-guided attention module is designed to improve spatial token by selecting informative regions and suppressing empty or sparse areas.

Text instructions are constructed via two pathways: (i) For samples with existing textual annotations, we use GPT-4o to refine the descriptions by removing static attributes and irrelevant visual details (e.g., texture, color), ensuring better alignment with event streams. (ii) For samples lacking groundtruth text, we leverage Qwen-VL-Max to automatically generate annotations from corresponding video inputs, enabling a scalable and consistent data synthesis pipeline. In addition, we organize a multi-person team to manually inspect and filter the generated instruction sets for quality assurance.

Dataset Statistics. We analyze the composition of the EventMind dataset from a curriculum learning perspective (see Fig. 1(b)). It is structured into three stages based on event sequence length and task complexity. In Stage 1, short sequences are used for the simple captioning task, contributing 200k instruction samples. Stage 2 utilizes medium-length sequences for scene captioning and human action understanding, with a total of 110k instructions. Stage 3 focuses on long sequences for more complex tasks such as motion captioning, EventQA, FGQA, and MCQA, comprising 190k instructions. Overall, our EventMind comprises 500k instruction samples spanning seven task types (see Fig. 1(c)): 200k for simple captioning, 90k for scene captioning, 30k for motion captioning, 90k for EventQA, 60k for FGQA, 10k for MCQA, and 20k for human action QA.

All in all, the novel event-text modality and labor-intensive design make EventMind a highly competitive dataset with several key strengths: (i) High temporal sampling resolution at the microsecond level from event streams; (ii) Coverage of temporal sequences of various lengths; (iii) Diverse scene types supporting 7 distinct tasks; (iv) A large-scale high-quality instruction set with 500k samples.

## 4 METHOD

## 4.1 EVENTFLASH OVERVIEW

This work aims at designing an efficient MLLM for event stream understanding, termed EventFlash, which presents a spatiotemporal token sparsification strategy to reduce redundancy and accelerate inference. As illustrated in Fig. 2, our framework consists of five modules: adaptive temporal window aggregation module, sparse density-guided attention module, event encoder, event-language projector, and large language model (LLM) decoder. More precisely, the adaptive temporal window aggregation module first segments the continuous event stream into uniform short bins and adaptively merges adjacent bins based on token similarity or event density. These processed bins are then passed by an event encoder (e.g., CLIP) to extract semantic embeddings. In parallel, the sparse densityguided attention module improves spatial token efficiency by emphasizing informative regions and suppressing empty or low-density areas. The event-language projector aligns the event tokens with text tokens to enable coherent multimodal fusion. Finally, the compact event tokens are fused with text tokens and processed by an LLM decoder (e.g., Qwen-2.5) for multimodal generation tasks.

## 4.2 TEMPORAL SPARSE

The microsecond-level resolution of raw event streams generates an excessive number of temporal tokens, resulting in high computational overhead. To address this, we introduce a two-stage densityguided adaptive temporal window aggregation (ATWA) module that compresses event streams while preserving key motion dynamics. The event stream is first divided into fine-grained bins, which are iteratively merged based on an asynchronous spatiotemporal spike metric (Li et al., 2022). Each bin is treated as a polarity-aware spatiotemporal point process with an intensity function $\lambda _ { B } \colon$

$$
\lambda_ {B} (x, y, t, p) = \sum_ {e _ {n} \in B} f (p _ {n}) \cdot \exp \left(- \frac {(x - x _ {n}) ^ {2}}{2 \sigma_ {x} ^ {2}} - \frac {(y - y _ {n}) ^ {2}}{2 \sigma_ {y} ^ {2}} - \frac {(t - t _ {n}) ^ {2}}{2 \sigma_ {t} ^ {2}}\right),\tag{1}
$$

where $f ( p _ { n } )$ encodes the polarity for an event $( x _ { n } , y _ { n } , t _ { n } , p _ { n } ) . \ \sigma _ { x } , \sigma _ { y } ,$ and $\sigma _ { z }$ are the parameters of the Gaussian kernel. The similarity distance between two bins $B _ { i }$ and $B _ { i + 1 }$ can be computed as:

$$
D (B _ {i}, B _ {i + 1}) = \left\| \lambda_ {B _ {i}} - \lambda_ {B _ {i + 1}} \right\| _ {2},\tag{2}
$$

where a lower D indicates higher temporal correlation between two bins. We iteratively merge adjacent bins when the distance is below a threshold τ, forming meta event windows $\{ M _ { 1 } , \bar { M _ { 2 } } , \ldots , \bar { M _ { K } } \}$

In the second stage, we perform semantic-aware aggregation of meta bins. Each window $M _ { i }$ is passed through an event encoder (e.g., ViT (Arnab et al., 2021)) to obtain a CLS token representation $z _ { i } ,$ and the similarity $S _ { i }$ between adjacent windows is defined as cosine similarity as follows:

$$
S _ {i} = \frac {z _ {i} ^ {\top} z _ {i + 1}}{\| z _ {i} \| \cdot \| z _ {i + 1} \|}.\tag{3}
$$

To incorporate event sparsity, we define a normalized event density factor $\begin{array} { r } { r _ { i } = \frac { 1 } { | M _ { i } | } \sum _ { e _ { n } \in M _ { i } } \mathbf { 1 } _ { e _ { n } } , } \end{array}$ and compute a density-aware weight. The final adaptive merging score can be formulated by:

$$
A _ {i} = S _ {i} \cdot \exp (- \alpha \cdot r _ {i}),\tag{4}
$$

where α controls the decay sensitivity. which jointly considers semantic similarity and event sparsity. We iteratively merge windows with high $A _ { i }$ to obtain a compressed yet semantically meaningful temporal sequence that preserves key temporal cues with reduced computational cost.

## 4.3 SPATIAL SPARSE

While temporal aggregation reduces sequence length, spatial redundancy still persists due to the inherent sparsity and uneven event distribution across the sensor plane. To tackle this, we propose the sparse density-guided attention (SDGA) module (see Fig. 3), which adaptively prunes uninformative tokens based on both visual semantics and event density. For each aggregated event bin, we use an encoder (i.e., ViT) to extract patch-level features $\{ x _ { j } \} _ { j = 1 } ^ { n } ,$ which are fed into a multi-head self-attention mechanism as:

![](images/58b90a74d52a75ca3161ac1245259e6fd8a879ea31076400397341736eb21c62.jpg)  
Figure 3: The architecture of the sparse density-guided attention module. It enhances spatial token efficiency by selecting informative regions and suppressing empty or low-density areas.

$$
\text { Attention } (Q, K, V) = \text { softmax } \left(\frac {Q K ^ {\top}}{\sqrt {d _ {k}}}\right) V,\tag{5}
$$

where $Q , K$ , and $V$ are the projected queries, keys, and values from $\{ x _ { j } \}$ , and $d _ { k }$ is the key dimension. In parallel, we compute the event density $D _ { j }$ of each token region based on the number of events falling within its receptive field. This scalar value is then passed through a density encoding unit consisting of a linear transformation followed by GELU activation:

$$
f (D _ {j}) = \operatorname{GELU} (\operatorname{Linear} (D _ {j})),\tag{6}
$$

where $f ( D _ { j } )$ is a soft modulation signal that reflects the importance of each spatial token. The encoded density is added to the attention scores to focus on denser and more important areas as:

$$
\tilde {A} _ {i j} = \frac {Q _ {i} K _ {j} ^ {\top}}{\sqrt {d _ {k}}} + f (D _ {j}).\tag{7}
$$

Finally, we apply a Token Selector operation that ranks the aggregated attention responses and discards low-importance tokens, which can be formulated as follows:

$$
\hat {x} _ {i} = \text { TokenSelector } \left(\sum_ {j} \operatorname{softmax} (\tilde {A} _ {i j}) \cdot V _ {j}\right).\tag{8}
$$

In summary, this density-guided token pruning strategy enables EventFlash to keep important spatial details while greatly cutting down on redundant computations. By combining semantic relevance with event density, SDGA produces more compact tokens for the efficient MLLM.

## 4.4 SHORT-TO-LONG CURRICULUM LEARNING

To support scalable training across different event durations and enhance generalization, we propose a progressive short-to-long curriculum learning strategy. Unlike prior event-based MLLMs such as EventVL (Li et al., 2025b) and EventGPT (Liu et al., 2025b), which train different modules in separate stages, our curriculum emphasizes a gradual progression from short to long event streams. This design facilitates smoother training dynamics, enabling EventFlash to evolve from mastering simple alignments to handling complex reasoning and long-range event understanding.

To be specific, Stage 1 focuses on event-language alignment by training on 200k short sequences (0-50 ms) paired with simple scene descriptions to establish basic cross-modal understanding. Stage 2 expands to 110k medium sequences (50-5,000 ms) featuring complex motions like human actions, enhancing the model’s reasoning and ability to handle instruction-following and event-based QA over longer inputs. Stage 3 fine-tunes the model on 190k long sequences (5,000–20,000 ms) with rich scene descriptions, enabling holistic scene understanding and open-ended language generation.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Implements Details. We initialize the event encoder with CLIP-ViT-Large-Patch14 (Radford et al., 2021) and use Qwen2.5 (Bai et al., 2023) as the LLM backbone. A two-layer MLP serves as the Event-Language Projector to align the event and semantic spaces. EventFlash is implemented in both 3B and 7B variants and trained on 8 A100 GPUs. For throughput evaluation, the inference is conducted on an A100 GPU using Hugging Face deployment. Our three-stage curriculum learning strategy proceeds as follows: only the Event-Language alignment module is trained in Stage 1, using a learning rate of $2 \times 1 0 ^ { - 3 }$ and a batch size of 64. For Stage 2 and Stage 3, all model parameters are unfrozen and trained with a learning rate of $2 \times 1 0 ^ { - 5 }$ , a batch size of 8, and a gradient accumulation step of 4. A cosine learning rate decay schedule is applied throughout training. We set the temporal aggregation interval to 10 ms and use a density attenuation factor α of 0.1 for spatial sparsification.

Evaluation Metrics. To thoroughly evaluate the generalization and reasoning capabilities of our EventFlash, we adopt four metrics aligned with protocols established in LLaVA (Liu et al., 2023) and other widely used benchmarks (Fang et al., 2024b). More precisely, we use the following evaluation metrics: (i) Global detailed captioning (GDC) to assess scene-level summarization, (ii) Fine-grained question answering (FGQA) to evaluate the model’s understanding of localized event details, (iii) Human action question answering (HAQA) to measure temporal reasoning at the action level, and (iv) Multiple choice question answering (MCQA) to assess instruction-following and discriminative reasoning. For open-ended tasks (GDC and FGQA), we employ LLM-based evaluation using GPT-4o (i.e., LLM-Judge) consistent with prior benchmarks. For HAQA and MCQA, we report the accuracy based on exact matches with ground-truth answers. In addition, throughput and maximum event bin capacity are used to evaluate the efficiency of all MLLMs. Throughput is typically defined as the number of tokens generated per second during inference, while maximum event bin capacity refers to the largest number of event bins the model can process in a single input.

![](images/0b4df82dc9c059b9b4455ebab29bb11bde398b49aa98ae44af957f3cd387adbb.jpg)  
Figure 4: Representative visualization tests on motion captioning and multiple-choice question answering (MCQA) are conducted in high-speed scenarios. Our EventFlash demonstrates superior accuracy in recognizing fast-moving objects, such as a sudden bullet being fired at a doll.

## 5.2 QUALITATIVE RESULTS

Table 1: Comparison of video-based MLLMs and event-based MLLMs on our EventMind dataset and EventChat-Sub dataset (Liu et al., 2025b). Notably, it can process significantly longer event bins than the event-based competitor EventGPT.

<table><tr><td rowspan="2">Models</td><td rowspan="2">Params</td><td rowspan="2">LLM Backbone</td><td rowspan="2">Max Bins</td><td rowspan="2">Throughput (Token/s)</td><td colspan="4">EventMind</td><td colspan="2">EventChat-Sub</td></tr><tr><td>GDC</td><td>FGQA</td><td>HAQA</td><td>MCQA</td><td>GDC</td><td>FGQA</td></tr><tr><td colspan="11">Video-Base ~3B Scale MLLMs</td></tr><tr><td>Qwen2.5 VL (Bai et al., 2023)</td><td>3B</td><td>Qwen2.5</td><td>768</td><td>-</td><td>20.6</td><td>41.7</td><td>23.8</td><td>34.6</td><td>34.5</td><td>51.2</td></tr><tr><td>VideoChat2-Flash (Li et al., 2024b)</td><td>2B</td><td>Qwen2.5</td><td>1,000</td><td>-</td><td>31.6</td><td>38.9</td><td>16.2</td><td>43.6</td><td>36.9</td><td>43.8</td></tr><tr><td>InternVL2.5 (Lu et al., 2025)</td><td>4B</td><td>Qwen2.5</td><td>-</td><td>-</td><td>17.9</td><td>37.0</td><td>21.3</td><td>27.3</td><td>28.9</td><td>44.6</td></tr><tr><td colspan="11">Video-Base ~7B Scale MLLMs</td></tr><tr><td>VideoChat2-Flash (Li et al., 2024b)</td><td>7B</td><td>Qwen2.5</td><td>1,000</td><td>-</td><td>36.2</td><td>41.9</td><td>18.9</td><td>48.2</td><td>53.1</td><td>53.6</td></tr><tr><td>LLaVA-Next-Video (Liu et al., 2023)</td><td>7B</td><td>Qwen2.5</td><td>56</td><td>-</td><td>31.2</td><td>44.6</td><td>22.8</td><td>42.7</td><td>46.3</td><td>54.8</td></tr><tr><td>Qwen2.5 VL (Bai et al., 2023)</td><td>7B</td><td>Qwen2.5</td><td>768</td><td>-</td><td>22.1</td><td>43.9</td><td>28.6</td><td>41.8</td><td>41.6</td><td>53.2</td></tr><tr><td>InternVL2.5 (Lu et al., 2025)</td><td>8B</td><td>InternLM2.5</td><td>-</td><td>-</td><td>19.7</td><td>40.0</td><td>25.3</td><td>38.2</td><td>42.5</td><td>55.6</td></tr><tr><td colspan="11">Event-Base MLLMs</td></tr><tr><td>EventGPT-7B (Liu et al., 2025b)</td><td>7B</td><td>Vicuna-v1.5</td><td>5</td><td>42.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>71.2</td><td>78.2</td></tr><tr><td>EventFlash-Zero</td><td>3B</td><td>Qwen-2.5</td><td>1,000</td><td>2.3</td><td>45.3</td><td>60.4</td><td>85.0</td><td>58.2</td><td>70.4</td><td>77.1</td></tr><tr><td>EventFlash-3B (Ours*)</td><td>3B</td><td>Qwen-2.5</td><td>1,000</td><td>28.5</td><td>46.8</td><td>61.1</td><td>84.9</td><td>60.0</td><td>71.5</td><td>78.6</td></tr><tr><td>EventFlash-7B (Ours*)</td><td>7B</td><td>Qwen-2.5</td><td>1,000</td><td>24.0</td><td>52.3</td><td>64.2</td><td>87.6</td><td>63.1</td><td>74.1</td><td>79.5</td></tr></table>

Comparison with State-of-the-Art MLLMs. To evaluate the effectiveness and efficiency of EventFlash, we compare it against four state-of-the-art video-based MLLMs and the only opensourced event-based MLLM (i.e., EventGPT (Liu et al., 2025b)). We select strong video-based models at both the 3B and 7B scales, including Qwen2.5-VL (Bai et al., 2023), VideoChat2-Flash (Li et al., 2024b), LLaVA-Next-Video (Liu et al., 2023), and InternVL 2.5 (Lu et al., 2025). EventGPT uses fixed bin encoding for event stream understanding. We also construct a baseline, EventFlash-Zero, by removing spatiotemporal sparsification from EventFlash.

Qualitative Evaluation. As illustrated in Table 1, EventFlash outperforms four video-based MLLMs and the event-based EventGPT on all four tasks (i.e., GDC, FGQA, HAQA, and MCQA). This demonstrates that EventFlash excels at understanding and describing dynamic event scenes. While EventGPT implements a fixed configuration of 5 event bins, EventFlash can process up to 1,000 event bins, achieving a 200× increase in processing capacity. In other words, our EventFlash is enabled by our efficient sparsification strategy for longer-term understanding. In addition, EventFlash reaches a speed of 28.5 tokens per second during inference. This is 12.4× faster than our baseline EventFlash-Zero (2.3 tokens per second), and it still maintains comparable performance on all tasks.

![](images/08f96ff967f8a74d58f509518f34baeb1a85b7c050fa4d6b4f82e5538d9370c8.jpg)  
Figure 5: Representative visualization tests on event questioning answering (QA) and scene caption are conducted in low-light scenarios. EventFlash showcases strong scene description and reasoning capabilities, such as identifying a car in a nighttime scene where it is barely visible on RGB images.

Visualization Evaluation. We further evaluate EventFlash under challenging scenarios, such as highspeed motion and low illumination. As shown in Fig. 4 and Fig. 5, our model demonstrates strong descriptive and reasoning capabilities in both cases. In high-speed case: The scene depicts a goblin being struck by a high-velocity projectile, resulting in a mid-air explosion with scattered fragments. EventFlash generates an accurate and fine-grained description of this dynamic event and correctly answers a multiple-choice question. In low-light case: The scenario involves a vehicle driving through darkness. Despite the absence of frame-based visual cues, EventFlash generates a coherent and precise description, along with an accurate response to the corresponding QA prompt. These results validate EventFlash’s ability to understand complex dynamics in edge-case environments where traditional frame-based models often fail.

To further demonstrate the advantages of EventFlash on longduration event streams, we compare it with EventGPT on a 10,000 ms sequence. As shown in Fig. 6, EventGPT operates on a fixed number of bins (e.g., 0–50 ms), limiting its understanding to moment-level segments. In contrast, EventFlash leverages its high maximum event bin capacity to process extended sequences, enabling coherent reasoning across the full temporal window and capturing sequence-level motion dynamics. As a result, EventFlash generates more contextually accurate de-

![](images/64318e3d2aef0a59753d49cd054cd9a690d18a5c792e0f5a7f8d0674d642ff92.jpg)  
Figure 6: Comparison of EventFlash and EventGPT on longduration event streams from our EventMind dataset.

scriptions, highlighting its potential for real-world applications that require long-range understanding, such as surveillance analysis and autonomous driving.

This gap stems from their different temporal modeling strategies. EventGPT relies on short, fixedduration bins, which fragment long-term motion and hinder the capture of cross-bin dependencies. In contrast, EventFlash maintains a unified representation over extended event streams, preserving temporal continuity and enabling consistent reasoning across long time horizons. As a result, EventFlash produces descriptions that better reflect the overall motion evolution of the scene, rather than isolated moment-level observations.

## 5.3 ABLATION STUDY

Contribution of Each Component. To explore the impact of each component on overall performance, we conduct an ablation study by comparing our full model against three variants: a baseline without any sparsification (EventFlash-Zero), a model with only temporal sparsification, and a model with only spatial sparsification. As shown in Table 2, our full model

Table 2: The contribution of each component to our EventMind dataset. The baseline uses our EventFlash without the spatiotemporal token sparsification strategy.

<table><tr><td rowspan="2">Model</td><td rowspan="2">S</td><td rowspan="2">T</td><td rowspan="2">Token/s</td><td colspan="4">EventMind</td></tr><tr><td>GDC</td><td>FGQA</td><td>HAQA</td><td>MCQA</td></tr><tr><td>Baseline</td><td>X</td><td>X</td><td>2.3</td><td>45.3</td><td>60.4</td><td>85.0</td><td>58.2</td></tr><tr><td>A</td><td>√</td><td>X</td><td> $5.3_{+2.3 \times}$ </td><td>46.3</td><td>61.2</td><td>85.1</td><td>59.6</td></tr><tr><td>B</td><td>X</td><td>√</td><td> $14.0_{+6.1 \times}$ </td><td>47.1</td><td>60.6</td><td>83.8</td><td>60.3</td></tr><tr><td>Ours*</td><td>√</td><td>√</td><td> $28.5_{+12.4 \times}$ </td><td>46.8</td><td>61.1</td><td>84.9</td><td>60.0</td></tr></table>

achieves a 12.4× increase in throughput (28.5 tokens/s vs. 2.3 tokens/s) while maintaining comparable performance across four evaluation metrics (i.e., GDC, FGQA, HAQA, and MCQA). With temporal sparsification alone, the model achieves 14.0 tokens/s, representing a 6.1× speedup over the baseline. In contrast, spatial sparsification alone yields a 2.3× improvement, reaching 5.3 tokens/s. The results show that both temporal and spatial sparsification contribute to efficiency gains.

Influence of the Aggregation Interval Length. To explore how the initial temporal bin duration affects performance and efficiency, we evaluate model throughput and accuracy across different initial event bin durations. As shown in Table 3, we compare four settings with bin lengths of 5 ms, 10 ms, 20 ms, and 30 ms. We observe that shorter bin dura-

Table 3: The influence of aggregation interval length on our EventMind dataset.

<table><tr><td rowspan="2">Aggregation interval</td><td rowspan="2">Throughput (Token/s)</td><td colspan="4">EventMind</td></tr><tr><td>GDC</td><td>FGQA</td><td>MCQA</td><td>HAQA</td></tr><tr><td>5ms</td><td>15.8</td><td>47.1</td><td>61.8</td><td>84.6</td><td>58.2</td></tr><tr><td>10ms</td><td>28.5</td><td>46.8</td><td>61.1</td><td>84.9</td><td>60.0</td></tr><tr><td>20ms</td><td>52.6</td><td>43.2</td><td>56.3</td><td>72.6</td><td>48.4</td></tr><tr><td>30ms</td><td>63.3</td><td>36.8</td><td>48.2</td><td>61.8</td><td>46.2</td></tr></table>

tions (e.g., 5 ms) provide finer temporal resolution but significantly increase the number of windows, resulting in lower throughput (15.8 tokens/s) compared to our default setting of 28.5 tokens/s at 10 ms. Despite the increased computational load, the model maintains strong performance across all tasks. Conversely, increasing the bin size to 20 ms and 30 ms improves throughput to 52.6 and 63.3 tokens/s, respectively, indicating greater efficiency. However, this comes at the cost of performance degradation on GDC, FGQA, MCQA, and HAQA. In this work, a bin duration of 10 ms offers a trade-off between accuracy and efficiency, and is therefore adopted as our default setting.

Impact of Density Attenuation Factor α. We investigate how the density attenuation factor α affects model throughput and task performance (see Table 4). To explore the trade-off between density-guided and similarity-guided token merging, we evaluate four values of α to identify the optimal balance between accuracy and ef-

Table 4: The influence of density factor α on throughput and performance on our EventMind dataset.

<table><tr><td>Density Factor α</td><td>Throughput (Token/s)</td><td>GDC</td><td>FGQA</td><td>MCQA</td><td>HAQA</td></tr><tr><td>0.1</td><td>28.5</td><td>46.8</td><td>61.1</td><td>84.9</td><td>60.0</td></tr><tr><td>0.2</td><td>27.6</td><td>45.6</td><td>61.4</td><td>85.2</td><td>58.4</td></tr><tr><td>0.4</td><td>28.8</td><td>45.3</td><td>61.6</td><td>85.2</td><td>58.4</td></tr><tr><td>0.6</td><td>26.8</td><td>47.2</td><td>60.8</td><td>83.2</td><td>60.1</td></tr></table>

ficiency. The results show that increasing α leads to higher throughput, indicating that stronger density suppression accelerates the token aggregation process. For example, FGQA and MCQA stay mostly stable when α is between 0.2 and 0.4. However, GDC and HAQA rely more on detailed timing information. Because of this, their performance drops when α gets higher. The results confirm the effectiveness of our density-aware weighting mechanism. Notably, $\alpha = 0 . 1$ and α = 0.4 achieve a favorable trade-off, providing substantial speed gains while preserving strong task performance.

## 5.4 EXTENSIVE APPLICATION

We further investigate additional downstream applications enabled by our EventFlash. For instance, EventFlash can be readily fine-tuned to support action recognition tasks. As shown in 5, we evaluate its performance on the DailyDVS-200 (Wang et al., 2024a) dataset, where EventFlash predicts ac-

Table 5: Action recognition results on processed DailyDVS-200 (Wang et al., 2024a) dataset.

<table><tr><td>Methods</td><td>Venue</td><td>Input Type</td><td>Backbone</td><td>top-1 acc. (%)</td></tr><tr><td>Swin-T (Liu et al., 2022)</td><td>CVPR&#x27;22</td><td>Frame</td><td>Transformer</td><td>48.06</td></tr><tr><td>GET (Peng et al., 2023)</td><td>ICCV&#x27;23</td><td>Event</td><td>Transformer</td><td>37.28</td></tr><tr><td>SDT (Yao et al.)</td><td>NeurIPS&#x27;24</td><td>Event</td><td>Transformer</td><td>35.43</td></tr><tr><td>ESTF (Wang et al., 2024b)</td><td>AAAI&#x27;24</td><td>Event</td><td>ResNet50</td><td>24.68</td></tr><tr><td>EventFlash</td><td>Ours*</td><td>Event</td><td>Qwen2.5</td><td>48.36</td></tr></table>

tion categories in an open-ended QA setting. Our EventFlash achieves outstanding performance and strong generalization capability.

## 6 CONCLUSION

This paper presents EventFlash, a novel efficient MLLM that leverages spatiotemporal token sparsification to reduce data redundancy and accelerate inference. We also built a large-scale dataset for event stream understanding. The results show that EventFlash achieves a 12.4× improvement in throughput over our baseline (EventFlash-Zero) while maintaining comparable performance. Notably, EventFlash enables long-range event stream processing of up to 1,000 bins compared to only 5 bins in the EventGPT. Our EventFlash serves as an efficient foundational model for event-based vision.

## 7 ACKNOWLEDGMENTS

This work is supported by the National Natural Science Foundation of China (Grant No. 62502317).