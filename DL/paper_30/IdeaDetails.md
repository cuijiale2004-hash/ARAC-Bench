### 1. Research Background and Existing Pain Points

**Research Background:** Speech brain-computer interfaces (BCIs) aim to restore communication for people with paralysis by translating neural activity into text. Recent revolutions in speech neuroprosthetics have enabled the decoding of attempted and imagined speech.

**Existing Pain Points:**
1.  **Cascaded Framework Limitations:** Most existing systems rely on cascaded frameworks that decode phonemes before assembling sentences with an n-gram language model (LM). This prevents joint optimization of all stages simultaneously.
2.  **Performance Dissociation:** Separate optimization of each component in cascaded systems results in a dissociation between the performance of the recurrent neural network (RNN) and the system as a whole. For instance, lower phoneme error rates (PER) from the RNN do not always translate to lower word error rates (WER) when decoding with the n-gram model.
3.  **Architectural Limitations in End-to-End Models:** The previous end-to-end approach (Feng et al., 2024) connects RNNs with LLMs but does not explore modern architectures, such as transformers, which may better capture complex neural representations.
4.  **Lack of Large-Scale Pretraining:** Existing transformer-based approaches (Feghhi et al., 2025) do not employ end-to-end training with LLMs or leverage large-scale pretraining. Since transformers typically require large datasets to reach their full potential, the absence of pretraining limits decoding performance and stable representation generation to effectively guide LLMs.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** To overcome the limitations of RNN-only encoders and small-scale datasets, there is a need for a fully end-to-end optimized speech decoding framework. Combining transformer-based neural encoders with large-scale pretraining could improve decoding performance and provide stable representations to effectively guide LLMs.

**Scientific Questions:**
1.  Can a cross-task, cross-species pretrained neural encoder transfer to both attempted and imagined speech, yielding generalizable neural representations?
2.  Can an end-to-end framework integrating a pretrained transformer encoder with an audio-LLM decoder translate neural activity directly into coherent sentences, narrowing the performance gap with cascaded systems?
3.  Can contrastive learning effectively align neural embeddings with the language structure of LLMs to enable cross-task generalization?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:** The paper introduces an end-to-end BraIn-to-Text (BIT) framework that translates neural activity into coherent sentences using a single differentiable neural network. The approach combines a transformer neural encoder pretrained with self-supervised masked modeling on human and monkey Utah array recordings with an audio-LLM decoder.

**Design Philosophy:** Analogous to LLAVA, which augments a pretrained LLM with an image encoder as its eyes, BIT equips the LLM with a "brain" to interpret neural activity. The design leverages transformer-based pretraining and end-to-end LLM integration to overcome the limitations of RNN-only encoders, yielding neural representations that generalize across tasks and align speech-related neural activity with the language structure of LLMs.

### 4. Core Innovation Points

1.  **End-to-End BraIn-to-Text (BIT) Framework:** Introduction of a framework that translates neural activity directly into coherent sentences using a single differentiable neural network, moving away from cascaded phoneme-to-text pipelines.
2.  **Cross-Task, Cross-Species Pretrained Neural Encoder:** A transformer neural encoder pretrained with self-supervised masked modeling on 367 hours of Utah array recordings from humans and monkeys across speech and arm-related motor tasks.
3.  **Integration with Audio-LLMs:** Discovery that small-scale audio-LLMs markedly improve end-to-end decoding compared to text-based LLMs, leveraging speech knowledge acquired during LLM pretraining.
4.  **Contrastive Learning for Cross-Modal Alignment:** A modality aligner trained with contrastive learning projects mean-pooled neural and text embeddings into a shared latent space, aligning neural activity with the language structure in LLMs.
5.  **Cross-Task Generalization:** BIT aligns attempted and imagined speech embeddings to enable cross-task generalization, demonstrating that SSL pretraining yields larger transfer gains than same-subject, cross-task supervised pretraining.

### 5. Overview of the Overall Technical Solution

The technical solution consists of a three-stage training pipeline:
1.  **Self-Supervised Pretraining:** The transformer neural encoder is pretrained using masked modeling on 367 hours of human and monkey Utah array data. Neural activity is tokenized into temporal patches, randomly masked, and the model reconstructs the original neural activity using MSE loss.
2.  **Phoneme-Level Fine-tuning:** The pretrained encoder is fine-tuned to predict phoneme sequences from neural activity using a Connectionist Temporal Classification (CTC) loss. This injects phonetic information into the neural representations.
3.  **Sentence-Level Fine-tuning:** The phoneme-aware encoder is integrated with an audio-LLM decoder. Neural representations are projected into the text embedding space via a shallow MLP. The system is fine-tuned end-to-end to predict the next token using cross-entropy loss against the ground-truth sentence, reinforced by a contrastive learning objective for modality alignment. LoRA is applied to the LLM for efficient fine-tuning.

### 6. Detailed Module Design

**6.1 Transformer Neural Encoder**
*   **Input Processing:** The original data shape is $(T, C)$, where $T$ is the number of time bins and $C$ is the number of electrodes. Every $T_{patch}$ time bins are grouped into a patch, yielding data of shape $(T/T_{patch}, C \times T_{patch})$.
*   **Patch Embedding Module:** Patches are passed through a module (LayerNorm, Linear, LayerNorm) to produce tokens.
*   **Transformer Blocks:** Each block includes a self-attention layer followed by a feed-forward network. Relative positional embeddings (RoPE) encode temporal information, and a bidirectional attention mask allows each time patch to attend to all others.
*   **SSL Output:** Transformer outputs are reconstructed to the original neural data through a reversed patch embedding module.
*   **Decoding Output:** Transformer outputs are passed through a linear layer to generate logits over phonemes, the blank token, and the silence token.

**6.2 LLM Decoder**
*   **Projector:** Encoder outputs are mapped into the text embedding space via a shallow MLP projector (Linear, ReLU, Linear).
*   **Prompt Insertion:** A text prompt (e.g., "decode the above neural activity into an English sentence:") is inserted between neural and text embeddings.
*   **Audio-LLM Mapping:** For audio-LLMs, neural encoder outputs pass through the MLP projector and are then mapped into the audio embedding space via the multimodal projector originally used for the audio encoder.
*   **LoRA Adaptation:** Low-rank adaptation (LoRA) is applied to a subset of LLM parameters (attention projections and feed-forward layers). For audio-based LLMs, LoRA is also applied to the multimodal projector.

**6.3 Cross-Modal Alignment (Modality Aligner)**
*   **Pooling:** Mean-pools neural and text embedding tokens into single "modality tokens".
*   **Projection:** Projects them into a shared latent space using separate linear layers.
*   **Normalization:** L2-normalizes the resulting vectors.
*   **Optimization:** Optimized with a contrastive loss to increase similarity between embeddings of matching sentences and push non-matching embeddings apart.

### 7. All Mathematical Formulas and Symbol Definitions

**7.1 Contrastive Loss**
Let $z^s_i \in \mathbb{R}^{T \times D}$ and $z^t_i \in \mathbb{R}^{L \times D}$ denote the neural and text embeddings for the $i$-th sample in a batch of $B$ trials. These are mean-pooled to a single vector and projected into a shared latent space of dimension $P$ via modality-specific linear layers, followed by $\ell_2$ normalization, resulting in modality-specific vectors $\tilde{z}^s_i \in \mathbb{R}^P$ and $\tilde{z}^t_i \in \mathbb{R}^P$.

The symmetric InfoNCE loss is defined as:
$$ \mathcal{L}_{contrastive} = \frac{1}{2B} \sum_{i=1}^{B} \left[ - \log \frac{\exp(\tilde{z}^s_i \cdot \tilde{z}^t_i / \tau)}{\sum_{j=1}^{B} \exp(\tilde{z}^s_i \cdot \tilde{z}^t_j / \tau)} - \log \frac{\exp(\tilde{z}^t_i \cdot \tilde{z}^s_i / \tau)}{\sum_{j=1}^{B} \exp(\tilde{z}^t_i \cdot \tilde{z}^s_j / \tau)} \right] $$
Where $\tau$ is a learnable temperature parameter.

**7.2 Cross-Entropy Loss**
Let $y_i = (y_{i,1}, \dots, y_{i,L_i})$ denote the ground-truth token sequence. The predicted probability distribution over a vocabulary of size $V$ at the $t$-th token is $\hat{p}_{i,t} = f(\text{concat}(z^s_i, z^t_i)) \in \mathbb{R}^V$, where $f$ represents the LLM decoder.
$$ \mathcal{L}_{CE} = - \frac{1}{B} \sum_{i=1}^{B} \sum_{t=1}^{L_i} \log \hat{p}_{i,t}(y_{i,t}) $$

**7.3 Total Training Objective**
$$ \mathcal{L}_{BIT} = \mathcal{L}_{CE} + \mathcal{L}_{contrastive} $$

**7.4 Cross-Attention Projector (Ablation Variant)**
$$ z^*, A = \text{CrossAttentionProjector}(z^s, z^t) $$
Where $z^*$ are the aligned neural-text representations and $A$ are the attention weights.

### 8. Algorithm Pseudocode

**Stage 1: Self-Supervised Pretraining**
1.  **Input:** Neural activity $X \in \mathbb{R}^{T \times C}$.
2.  **Tokenization:** Group $X$ into temporal patches $X_{patch} \in \mathbb{R}^{(T/T_{patch}) \times (C \times T_{patch})}$.
3.  **Embedding:** Apply Patch Embedding (LayerNorm $\rightarrow$ Linear $\rightarrow$ LayerNorm) to get tokens $H$.
4.  **Masking:** Randomly replace a subset of tokens in $H$ with a learnable mask token to get $\tilde{H}$.
5.  **Encoding:** Pass $\tilde{H}$ through Transformer Encoder (with RoPE and bidirectional mask) to get latent $Z$.
6.  **Reconstruction:** Pass $Z$ through Reversed Patch Embed to reconstruct $\hat{X}_{patch}$.
7.  **Update:** Minimize MSE Loss between $X_{patch}$ and $\hat{X}_{patch}$.

**Stage 2: Phoneme-Level Fine-tuning**
1.  **Input:** Neural activity $X$.
2.  **Tokenization & Embedding:** Generate tokens $H$ without masking.
3.  **Encoding:** Pass $H$ through Transformer Encoder to get $Z$.
4.  **Classification:** Pass $Z$ through Linear Layer to generate phoneme logits $Y_{phoneme}$.
5.  **Update:** Minimize CTC Loss between $Y_{phoneme}$ and ground-truth phoneme sequences.

**Stage 3: Sentence-Level End-to-End Fine-tuning**
1.  **Input:** Neural activity $X$, Target text $Y$.
2.  **Neural Encoding:** Pass $X$ through Transformer Encoder (from Stage 2) to get $Z$.
3.  **Projection:** Pass $Z$ through MLP Projector to get Neural Embeddings $E_{neural}$.
4.  **Alignment:** Compute Modality Tokens $\tilde{z}^s, \tilde{z}^t$ via Mean Pooling $\rightarrow$ Linear $\rightarrow$ L2-Norm. Compute $\mathcal{L}_{contrastive}$.
5.  **LLM Decoding:** Concatenate $E_{neural}$, Prompt, and Text Embeddings $E_{text}$. Pass through Audio-LLM (with LoRA) to get predicted logits.
6.  **Update:** Minimize $\mathcal{L}_{BIT} = \mathcal{L}_{CE} + \mathcal{L}_{contrastive}$.