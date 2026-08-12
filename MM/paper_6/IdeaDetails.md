## 1. Research Background and Existing Pain Points

**Research Background:** 
Event cameras (bio-inspired vision sensors) operate differently from frame-based cameras. Each pixel responds to intensity changes by generating asynchronous events. Due to their high temporal resolution and high dynamic range, event cameras have been applied to various vision tasks (e.g., scene understanding) in high-speed or low-light scenarios. Recent multimodal large language models (MLLMs) have achieved remarkable breakthroughs in processing conventional frames and language, showing strong capabilities in scene understanding and visual question answering. 

**Existing Pain Points:**
1. **Dense Image-like Processing Paradigm:** Current event-based MLLMs often rely on dense image-like processing paradigms, overlooking the spatiotemporal sparsity of event streams and resulting in high computational cost. Applying dense image-like processing paradigms to event streams not only incurs significant computational overhead but also substantially limits the effective length and efficiency of event stream understanding.
2. **Inefficiency in Existing Event-based MLLMs:** Most existing event-based MLLMs still rely on dense image-like representations, which hinders computational efficiency and scalability to long event sequences. For example, EventGPT converts event streams into dense token sequences for language modeling; EventVL integrates RGB frames with event data to enhance multimodal reasoning; LLaFEA employs frame-event fusion for region-level spatiotemporal grounding. Their dense processing of sparse event data leads to significant overhead and limits real-time or long-range understanding.
3. **Dataset Limitations:** Existing instruction-augmented datasets are not publicly available and often lack diversity, contain low-quality annotations, and cover short temporal sequences, making them inadequate for training generalizable models. The scene diversity of their datasets is relatively limited, and the event streams are short, making it difficult to support general-purpose models for long event-stream understanding.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** 
To develop efficient MLLMs that fully exploit the unique spatiotemporal properties of event data to address high computational overhead and scalability limitations, while maintaining comparable reasoning accuracy. Unlike prior works that focus on maximizing reasoning accuracy, the goal is to address three key challenges in efficient MLLMs: (i) Temporal inefficiency, (ii) Spatial inefficiency, and (iii) Dataset limitations.

**Scientific Questions:**
1. **Temporal Inefficiency:** The microsecond resolution of event streams results in prohibitively large token volumes when processed over long temporal durations. How to effectively compress temporal tokens while retaining key temporal cues?
2. **Spatial Inefficiency:** The inherent sparsity of event data leads to numerous empty or low-information tokens that incur computational overhead due to uniform attention allocation. How to improve spatial token efficiency by selecting informative regions and suppressing empty or sparse areas?
3. **Dataset Limitations:** How to build a large-scale, scene-diverse dataset with long temporal sequences and high-quality annotations to support general-purpose models for long event-stream understanding and curriculum training strategies?

## 3. Overall Core Idea and Design Philosophy

The core idea is to propose EventFlash, a novel, efficient MLLM that explores spatiotemporal token sparsification for reducing data redundancy and accelerating inference. The design philosophy leverages a density-aware spatiotemporal token sparsification strategy that exploits the inherent sparsity and high temporal resolution of event streams. Specifically, it involves:
1. **Adaptive Temporal Compression:** Designing an adaptive temporal window aggregation module for efficient temporal sampling, which dynamically compresses temporal tokens while preserving essential temporal cues.
2. **Density-Guided Spatial Pruning:** Presenting a sparse density-guided attention module to enhance spatial token efficiency by selecting informative regions and suppressing empty or low-density areas.
3. **Curriculum Training:** Designing a progressive curriculum learning strategy following a short-to-long paradigm to improve EventFlash’s generalization and generative capabilities, supported by a custom-built large-scale scene-diverse dataset (EventMind).

## 4. Core Innovation Points

1. **Efficient Event-based Vision MLLM (EventFlash):** Proposed EventFlash, an efficient event-based vision MLLM, which explores a spatiotemporal token sparsification strategy for raw event streams to reduce redundancy, accelerate inference (12.4× throughput improvement over the baseline), and enable long-range event stream understanding (up to 1,000 bins compared to only 5 bins in the competing EventGPT).
2. **Density-aware Spatiotemporal Token Sparsification Strategy:** Presented a density-aware spatiotemporal token sparsification strategy for event-based MLLMs, which effectively reduces redundancy while maintaining comparable reasoning accuracy by leveraging the fine-grained temporal resolution and inherent sparsity of raw event streams.
3. **Adaptive Temporal Window Aggregation Module:** Proposed the adaptive temporal window aggregation module for efficient temporal sampling, which adaptively compresses temporal tokens while retaining key temporal cues using a two-stage density-guided mechanism (spatiotemporal spike metric and semantic-aware aggregation).
4. **Sparse Density-guided Attention Module:** Designed the sparse density-guided attention module to improve spatial token efficiency by selecting informative regions and suppressing empty or sparse areas using a density encoding unit and Token Selector operation.
5. **Large-scale Scene-diverse Dataset (EventMind):** Built a large-scale scene-diverse dataset (EventMind) for long-range event stream understanding, comprising over 500k instruction samples spanning seven task types and diverse temporal lengths, supporting the progressive curriculum learning strategy.

## 5. Overview of the Overall Technical Solution

The EventFlash framework consists of five modules: adaptive temporal window aggregation module, sparse density-guided attention module, event encoder, event-language projector, and large language model (LLM) decoder. 
The continuous event stream is first segmented into uniform short bins by the adaptive temporal window aggregation module, which adaptively merges adjacent bins based on token similarity or event density. These processed bins are then passed by an event encoder (e.g., CLIP) to extract semantic embeddings. In parallel, the sparse density-guided attention module improves spatial token efficiency by emphasizing informative regions and suppressing empty or low-density areas. The event-language projector aligns the event tokens with text tokens to enable coherent multimodal fusion. Finally, the compact event tokens are fused with text tokens and processed by an LLM decoder (e.g., Qwen-2.5) for multimodal generation tasks. The training utilizes a progressive short-to-long curriculum learning strategy supported by the EventMind dataset.

## 6. Detailed Module Design

**6.1 Adaptive Temporal Window Aggregation Module (ATWA)**
The ATWA module addresses temporal inefficiency by compressing event streams while preserving key motion dynamics. It operates in two stages:
- **Stage 1: Density-guided Adaptive Temporal Aggregation:** The event stream is first divided into fine-grained bins, which are iteratively merged based on an asynchronous spatiotemporal spike metric. Each bin is treated as a polarity-aware spatiotemporal point process with an intensity function λB. The similarity distance between two bins Bi and Bi+1 is computed. We iteratively merge adjacent bins when the distance is below a threshold τ, forming meta event windows {M1, M2, . . . , MK}.
- **Stage 2: Semantic-aware Aggregation of Meta Bins:** Each window Mi is passed through an event encoder (e.g., ViT) to obtain a CLS token representation zi, and the similarity Si between adjacent windows is defined as cosine similarity. To incorporate event sparsity, a normalized event density factor ri is defined, and a density-aware weight is computed. The final adaptive merging score Ai jointly considers semantic similarity and event sparsity. We iteratively merge windows with high Ai to obtain a compressed yet semantically meaningful temporal sequence that preserves key temporal cues with reduced computational cost.

**6.2 Sparse Density-guided Attention Module (SDGA)**
The SDGA module addresses spatial inefficiency by adaptively pruning uninformative tokens based on both visual semantics and event density:
- **Attention Mechanism:** For each aggregated event bin, an encoder (i.e., ViT) extracts patch-level features {xj}, which are fed into a multi-head self-attention mechanism.
- **Density Encoding:** In parallel, the event density Dj of each token region is computed based on the number of events falling within its receptive field. This scalar value is passed through a density encoding unit consisting of a linear transformation followed by GELU activation to produce f(Dj), a soft modulation signal that reflects the importance of each spatial token.
- **Density-guided Attention:** The encoded density is added to the attention scores to focus on denser and more important areas, resulting in modified attention scores.
- **Token Selection:** A Token Selector operation is applied that ranks the aggregated attention responses and discards low-importance tokens, generating compact tokens.

**6.3 Short-to-Long Curriculum Learning Strategy**
To support scalable training across different event durations and enhance generalization, a progressive short-to-long curriculum learning strategy is designed, emphasizing a gradual progression from short to long event streams:
- **Stage 1:** Focuses on event-language alignment by training on 200k short sequences (0-50 ms) paired with simple scene descriptions to establish basic cross-modal understanding.
- **Stage 2:** Expands to 110k medium sequences (50-5,000 ms) featuring complex motions like human actions, enhancing the model’s reasoning and ability to handle instruction-following and event-based QA over longer inputs.
- **Stage 3:** Fine-tunes the model on 190k long sequences (5,000–20,000 ms) with rich scene descriptions, enabling holistic scene understanding and open-ended language generation.

**6.4 EventMind Dataset**
- **Data Collection:** Raw event data is sourced from real-world (DSEC, N-ImageNet, HARDVS, E2VID) and synthetic domains. Synthetic data are generated by converting large-scale video datasets (Kinetics-700, UCF-101, Wevid-10 M, PLM-Data, MotionBench) into event streams using the V2E simulator. GPT-4o is used to automatically filter videos before simulation. Data is categorized into short (0–50 ms), medium (50–5,000 ms), and long (5,000–20,000 ms).
- **Instruction Generation:** Seven distinct task types are defined: motion captioning, event question answering (Event QA), human action QA, multiple-choice QA (MCQA), simple captioning, fine-grained QA (FGQA), and scene captioning. Text instructions are constructed via two pathways: (i) For samples with existing textual annotations, GPT-4o refines the descriptions; (ii) For samples lacking ground-truth text, Qwen-VL-Max automatically generates annotations. Manual inspection and filtering are applied.
- **Dataset Statistics:** EventMind comprises 500k instruction samples: 200k for simple captioning, 90k for scene captioning, 30k for motion captioning, 90k for EventQA, 60k for FGQA, 10k for MCQA, and 20k for human action QA.

## 7. All Mathematical Formulas and Symbol Definitions

**Formula 1: Intensity function of polarity-aware spatiotemporal point process**
λB(x, y, t, p) = ∑ en∈B f(pn) · exp( − (x− xn) 2 2σ2 x − (y − yn) 2 2σ2 y − (t− tn) 2 2σ2 t )
*Symbol Definitions:*
- f(pn): encodes the polarity for an event (xn, yn, tn, pn).
- σx, σy , and σz: parameters of the Gaussian kernel.

**Formula 2: Similarity distance between two bins**
D(Bi, Bi+1) = ∥λBi − λBi+1 ∥ 2
*Symbol Definitions:*
- D: similarity distance between two bins Bi and Bi+1. A lower D indicates higher temporal correlation between two bins.

**Formula 3: Cosine similarity between adjacent windows**
Si = z⊤i zi+1 ∥zi∥ · ∥zi+1∥
*Symbol Definitions:*
- zi: CLS token representation of window Mi passed through an event encoder.
- Si: similarity between adjacent windows.

**Formula 4: Final adaptive merging score (density-aware weight)**
Ai = Si · exp(−α · ri)
*Symbol Definitions:*
- Si: cosine similarity between adjacent windows.
- α: controls the decay sensitivity.
- ri = 1 |Mi| ∑ en∈Mi 1en: normalized event density factor incorporating event sparsity.
- Ai: final adaptive merging score jointly considering semantic similarity and event sparsity.

**Formula 5: Multi-head self-attention mechanism**
Attention(Q,K, V ) = softmax( QK⊤ √ dk )V
*Symbol Definitions:*
- Q, K, and V: projected queries, keys, and values from patch-level features {xj}.
- dk: key dimension.

**Formula 6: Density encoding unit**
f(Dj) = GELU(Linear(Dj))
*Symbol Definitions:*
- Dj: event density of each token region based on the number of events falling within its receptive field.
- f(Dj): soft modulation signal reflecting the importance of each spatial token.

**Formula 7: Modified attention scores with density**
Ãij = QiK⊤j √ dk + f(Dj)
*Symbol Definitions:*
- Ãij: attention scores focused on denser and more important areas.

**Formula 8: Token Selector operation**
ˆxi = TokenSelector( ∑ j softmax(Ãij) · Vj )
*Symbol Definitions:*
- TokenSelector: operation that ranks the aggregated attention responses and discards low-importance tokens.

## 8. Algorithm Pseudocode

The original text does not provide a formal algorithm pseudocode block. However, the iterative processes and flowchart logic are strictly extracted as follows:

**Iterative Process 1: Adaptive Temporal Window Aggregation (ATWA) - Stage 1**
1. Divide the event stream into fine-grained bins B.
2. Treat each bin as a polarity-aware spatiotemporal point process and compute intensity function λB(x, y, t, p).
3. Compute similarity distance D(Bi, Bi+1) between adjacent bins.
4. IF D(Bi, Bi+1) < threshold τ, THEN iteratively merge adjacent bins.
5. Output meta event windows {M1, M2, . . . , MK}.

**Iterative Process 2: Adaptive Temporal Window Aggregation (ATWA) - Stage 2**
1. Pass each window Mi through an event encoder to obtain CLS token representation zi.
2. Compute cosine similarity Si between adjacent windows.
3. Compute normalized event density factor ri.
4. Compute final adaptive merging score Ai = Si · exp(−α · ri).
5. Iteratively merge windows with high Ai.
6. Output compressed temporal sequence.

**Flowchart Logic: Sparse Density-guided Attention Module (SDGA)**
1. Input: Aggregated event bin.
2. Extract patch-level features {xj}n j=1 using encoder (ViT).
3. Compute Q, K, V from {xj}.
4. Compute standard attention scores QK⊤/√dk.
5. Compute event density Dj of each token region.
6. Pass Dj through Density Encoding Unit (Linear + GELU) to get f(Dj).
7. Add f(Dj) to attention scores: Ãij = QK⊤/√dk + f(Dj).
8. Compute softmax(Ãij) · V.
9. Apply Token Selector operation to rank responses and discard low-importance tokens.
10. Output: Compact spatial tokens x̂i.