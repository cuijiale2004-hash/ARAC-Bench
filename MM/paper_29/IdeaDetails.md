1. Research Background and Existing Pain Points
Multimodal information retrieval has emerged as a pivotal research direction at the intersection of multimodal understanding and information retrieval. It serves as a cornerstone for a wide range of modern AI applications, including retrieval-augmented generation (RAG), text-image retrieval, and visual question answering (VQA). While existing methods such as CLIP, BLIP, and CoCa have demonstrated impressive performance in cross-modal retrieval by leveraging large-scale contrastive learning to align image-text pairs, they often struggle to address the diverse and complex requirements encountered in real-world scenarios, ranging from fine-grained instruction following to multi-turn interleaved interactions. To tackle these challenges, the research community has increasingly focused on Universal Multimodal Retrieval (UMR), which aims to develop unified, instruction-guided retrievers capable of handling diverse retrieval tasks across multiple modalities. Recent advancements in this area, such as LamRA, MM-Embed, GME, UniME, and VLM2VEC, have shown promising results on UMR powered by Multimodal Large Language Models (MLLMs). However, most of the existing approaches often directly adapt MLLMs to embedding tasks without systematically examining training schemes tailored for MLLM-based embedding models. Consequently, the mechanisms underlying their retrieval capabilities remain largely unexplored. The impact of design decisions on retrieval performance remains inadequately understood, potentially resulting in suboptimal performance and limited generalization ability. Unlike LLMs that generate outputs via autoregressive decoding, embedding models encode features of the entire input sequence, necessitating specialized strategies for feature extraction and tailored training methodologies. This gap in systematic investigation of the design space for MLLM-based embedding models highlights the need for a more comprehensive study, which motivates this research.

2. Core Research Motivation and Scientific Questions
The core research motivation is to address the existing gaps by systematically exploring the design principles for building high-performing universal multimodal retrievers with pre-trained MLLMs. The study aims to uncover the key factors that drive effective embedding learning for UMR through MLLMs and to understand the often-overlooked factors that can have a substantial impact on model performance. To this end, the paper investigates three primary scientific questions:
(1) How to adapt decoder-only MLLMs into instruction-aware embedders?
(2) How to train MLLM-based embedders within the framework of contrastive learning?
(3) How to distill the recall-then-rerank paradigm into a single model to further enhance retrieval performance while reducing computational inefficiencies?

3. Overall Core Idea and Design Philosophy
The overall core idea is to implement a general MLLM-based embedding learning pipeline and systematically analyze the primary contributors to high-performing universal retrieval systems. Based on empirical findings across various aspects of embedding generation and training strategies, the study unveils key design principles that are often overlooked. Building on these discoveries, the paper introduces a unified framework termed U-MARVEL (Universal MultimodAl Retrieval via Embedding Learning). The design philosophy centers on progressively adapting the decoder-only MLLM to the embedding task through stepwise training, optimizing the contrastive learning process via parameter interactions and hard negative filtering, and finally enhancing the single-model retrieval capability by efficiently distilling knowledge from a recall-then-rerank pipeline. The framework is designed to outperform state-of-the-art competitors by a large margin in supervised settings and exhibit strong zero-shot performance across various embedding-based retrieval tasks.

4. Core Innovation Points
The paper completely lists the following core innovation points derived from systematic exploration:
(1) Innovation in Embedding Extraction: Generating embeddings with bidirectional attention and mean pooling outperforms the common approach of using compression prompts with the last token mechanism. The experimental results reveal that the effectiveness of the compression prompt exhibits a dual dependency—it establishes a strong coupling with the last token mechanism, and causal attention enhances its impact. Bidirectional attention partially decouples this dependency. When the compression prompt is removed from bidirectional attention with the mean token mechanism, it achieves superior performance compared to the commonly employed approach. This difference stems from the last token embedding being affected by recency bias.
(2) Innovation in Instruction Integration: Masking instruction tokens during the mean pooling process enhances embedding performance. Since the query inherently incorporates instruction information during forward propagation through the bidirectional self-attention mechanism, filtering out instruction tokens and focusing on feature comparisons between the query and candidate mitigates calculation bias without compromising performance.
(3) Innovation in Progressive Transition: Progressive transition effectively adapts decoder-only MLLMs to embedding models through stepwise training. Since LLMs were originally trained with causal attention, switching to bidirectional attention degrades text-visual encoder alignment. A stepwise training strategy ensures a smooth transition through three key steps: Text Retrieval Adaptation, Cross-modal Alignment, and Instruction-tuned Multimodal Retrieval.
(4) Innovation in InfoNCE Training Parameters: Increasing batch size yields performance gains, but these improvements plateau without appropriate learning rate scaling. Simply increasing the batch size without adjusting the learning rate results in marginal improvements; applying a scaling rule significantly boosts performance. Additionally, learnable temperature parameters play a pivotal role in enhancing the effectiveness of contrastive learning by effectively optimizing the sharpness of the probability distribution during training.
(5) Innovation in Hard Negative Mining: Hard negatives may hinder convergence during training. Directly selecting the top-k hard negatives proved to be excessively challenging, resulting in complete failure, as some hard negatives are false negatives that adversely affect model training. Filtering false negatives (eliminating hard negatives with scores exceeding a predefined threshold) and mixing random in-batch negatives help balance difficulty and improve performance.
(6) Innovation in Reranker Distillation: An improved distillation approach reduces computational overhead while increasing feature diversity, thereby enabling resource-efficient and effective distillation. Constructing samples in the form of (query, positive, top-k hard negatives) and confining distillation to the scope of positive and hard negative samples significantly reduces computational overhead while substantially increasing the diversity of features encountered during model training.

5. Overview of the Overall Technical Solution
The U-MARVEL framework is a unified framework for learning universal multimodal embeddings tailored to UMR tasks. It utilizes a pre-trained Qwen2-VL-7B-Instruct model where the visual side is frozen, and the LLM component is fine-tuned using LoRA. The overall technical solution consists of three main stages:
Stage 1: Progressive Transition. The model is progressively fine-tuned on increasing complexity levels of the retrieval data. It starts with Text Retrieval Adaptation using the NLI dataset with unidirectional InfoNCE, advances to Cross-modal Alignment using the CC3M dataset with bidirectional InfoNCE, and finally masters Instruction-tuned Multimodal Retrieval using the M-BEIR dataset.
Stage 2: Hard Negative Mining and Fusion Reranker Model. Building on Progressive transition, the model first extracts features to identify top-k challenging negatives per query, filters out false negatives exceeding a similarity threshold of 0.7, and selects top-5 hard negatives mixed with in-batch negatives for continual finetuning (Recall model). Subsequently, a generative reranker is trained on data triplets (query, positive, hard negative) to output "YES" or "NO". The recall and reranker models are linearly combined to obtain a recall–reranker model.
Stage 3: Distillation. An improved knowledge distillation is performed on the recall–reranker model. Samples are constructed as (query, positive, top-k hard negatives), and distillation is confined to the scope of positive and hard negative samples. The combined knowledge is compressed into one unified student model by using KL divergence, transferring the discriminative power of the reranker into the embedding space efficiently.

6. Detailed Module Design
6.1 Embedding Extraction Module
This module extracts embeddings of the token sequence of multimodal queries using MLLMs. The instruction is directly concatenated with the query. Two mechanisms are compared:
- Last token: Employs a compression prompt (e.g., "Summarize above image and sentence in one word: emb"). The feature of the last token (<emb>) is used as the output embedding.
- Mean token: Transforms the unidirectional attention mechanism into a bidirectional one and applies mean pooling to the features from the last hidden states to derive the embedding for the entire sequence. The optimal design uses bidirectional attention with mean token mechanism without the compression prompt.

6.2 Instruction Integration Module
During the mean pooling process, instruction tokens are masked out. The query inherently incorporates instruction information during forward propagation through the bidirectional self-attention mechanism. By filtering out instruction tokens and focusing on feature comparisons between the query and candidate, the approach mitigates calculation bias.

6.3 Progressive Transition Module
This module adapts the decoder-only MLLM to the embedding task stepwise:
- Step 1 (Text Retrieval Adaptation): Learning retrieval tasks using text-only datasets (NLI) to establish foundational capabilities using unidirectional InfoNCE. This strengthens the text encoder’s semantic representation.
- Step 2 (Cross-modal Alignment): Advancing to cross-modal retrieval tasks through text-image paired data (CC3M) using bidirectional InfoNCE. This aligns the text-visual encoders.
- Step 3 (Instruction-tuned Multimodal Retrieval): Mastering instruction-guided multimodal retrieval tasks with comprehensive training on multimodal datasets (M-BEIR).

6.4 InfoNCE Training Module
This module trains the MLLM-based embedders under the contrastive learning framework. It requires careful interaction among large batch size, learning rate, and temperature. Increasing batch size requires scaling the learning rate accordingly to avoid plateauing gains. A learnable temperature parameter is utilized to dynamically optimize the sharpness of the probability distribution, superior to fixed temperature settings.

6.5 Hard Negative Mining Module
This module addresses challenging false positives. The mining strategy is:
- Feature Extraction: Using the model from the previous training stage, extract features for all queries and candidates, compute similarity scores (excluding positive matches), and rank candidates to identify the most challenging negatives per query.
- Filtered Hard Negative: For each query, filter out negative samples that exceed a predefined threshold (0.7, considered as false negatives) and then select the top-k (k=5) negative samples as hard negatives.
- Balanced Training: To prevent model convergence difficulties, select a suitable k-value and mix these hard negatives with other in-batch negatives. Perform continual finetuning by InfoNCE.

6.6 Recall-Rerank Module
A generative reranker based on the decoder-only MLLM architecture is trained. Training data consists of pairs for each query: a positive match and the top-50 most challenging negatives. The model outputs "YES" for positives and "NO" for negatives, optimized with next-token prediction loss. During inference, the recall model retrieves top-k candidates, the reranker judges each candidate, and the probability of "YES" serves as the reranker’s score. Scores are combined via linear interpolation: Smulti = 0.5 * Srecall + 0.5 * Srerank.

6.7 Improved Distillation Module
This module distills the Recall-Rerank pipeline into a single student model. Samples are constructed in the form of (query, positive, top-k hard negatives), and distillation is confined to the scope of positive and hard negative samples. This tight alignment with the recall-then-rerank inference pipeline (retrieving with hard negatives, then reranking top-k) significantly reduces computational overhead while substantially increasing the diversity of features encountered during model training.

7. All Mathematical Formulas and Symbol Definitions
Formula 1: InfoNCE Loss
LInfoNCE = − log \frac{exp(sim(eq, e+c )/τ)}{∑i exp(sim(eq, eci)/τ)}
Symbol Definitions:
eq: Embeddings of the query.
e+c: Embeddings of the positive candidate.
eci: Embeddings of any sample from the candidate set.
τ: Temperature parameter.
sim(·, ·): Cosine similarity.

Formula 2: Reranker Loss
Lrerank(θ) = − logPθ(y | query + candidate)
Symbol Definitions:
θ: Model parameters.
y: Target output, where y = yes if the candidate belongs to the set of positive candidates, and y = no otherwise.

Formula 3: Reranker Score
score = Pθ(YES | query + candidate)
Symbol Definitions:
score: The relevance score derived from the reranker’s predicted probability of the token “YES”.

Formula 4: Combined Score for Recall-Rerank
Smulti = α · Srecall + (1− α) · Srerank
Symbol Definitions:
Smulti: Final relevance score.
Srecall: Retrieval scores from the recall model.
Srerank: Scores from the reranker model.
α: Weight hyperparameter (fixed at 0.5).

Formula 5: Distillation Loss
Ldistill = DKL(Smulti ∥ Ssingle) = ∑i Smulti(i) log \frac{Smulti(i)}{Ssingle(i)}
Symbol Definitions:
Ldistill: Distillation loss using KL divergence.
Smulti: Multi-model ensemble scores (softmax-normalized).
Ssingle: Single model’s predictions (softmax-normalized).

8. Algorithm Pseudocode
Algorithm 1: U-MARVEL Framework Training Pipeline
Input: Pre-trained MLLM, NLI dataset D_NLI, CC3M dataset D_CC3M, M-BEIR dataset D_MBEIR, Threshold T=0.7, Hard negative count k_hnm=5, Reranker hard negative count k_rr=50, Interpolation weight α=0.5
Output: Unified U-MARVEL Model

// Stage 1: Progressive Transition
1: Train MLLM on D_NLI using unidirectional InfoNCE loss // Text Retrieval Adaptation
2: Train MLLM on D_CC3M using bidirectional InfoNCE loss // Cross-modal Alignment
3: Train MLLM on D_MBEIR using bidirectional InfoNCE loss with learnable temperature τ and scaled learning rate η // Instruction-tuned Multimodal Retrieval
4: Model_M <- Trained MLLM from Step 3

// Stage 2: Hard Negative Mining and Fusion Reranker Model
5: Extract features for all queries and candidates in D_MBEIR using Model_M
6: Compute similarity scores and rank candidates
7: Filter out negative samples with similarity score > T (false negatives)
8: Select top-k_hnm hard negatives per query
9: Mix top-k_hnm hard negatives with in-batch negatives
10: Continual finetune Model_M using InfoNCE loss on mixed negatives
11: Model_Recall <- Trained Recall Model from Step 10
12: Extract top-k_rr hard negatives using Model_Recall to construct triplets (query, positive, hard negative)
13: Train a generative reranker (Model_Rerank) using next-token prediction loss to output "YES"/"NO"
14: For each query, compute Smulti = α · Srecall + (1 - α) · Srerank

// Stage 3: Improved Distillation
15: Construct distillation samples in the form of (query, positive, top-k_rr hard negatives)
16: Compute softmax-normalized Smulti using Model_Recall and Model_Rerank
17: Initialize Student Model with Model_Recall parameters
18: Train Student Model using Distillation Loss: Ldistill = DKL(Smulti ∥ Ssingle)
19: Return Student Model as U-MARVEL