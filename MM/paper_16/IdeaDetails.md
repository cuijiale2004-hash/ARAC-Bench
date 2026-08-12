**1. Research Background and Existing Pain Points**

**Research Background:**
Multimodal representation learning models, such as CLIP, SigLIP, and CLAP, aim to learn a shared latent representation for various types of input data, including images, text, and audio. The goal of these multimodal encoders is to learn embeddings that are "similar" to one another (e.g., in terms of cosine similarity) when the input data contains shared semantic information, regardless of the data's modality. The ability of such models to learn expressive multimodal data representations has been demonstrated by their performance on cross-modal tasks like zero-shot classification, retrieval, and generation. However, the precise way in which they encode semantic information is not well-understood. 

The Linear Representation Hypothesis asserts that the embeddings learned by neural networks can be understood as linear combinations of features corresponding to high-level concepts. Based on this ansatz, sparse autoencoders (SAEs) have recently become a popular method for decomposing embeddings into a sparse combination of linear directions, which have been shown empirically to often correspond to human-interpretable semantics. Building on classical ideas from dictionary learning and sparse coding, SAEs are unsupervised models that aim to decompose neural network embeddings as sparse linear combinations of a large (overcomplete) set of dictionary vectors.

**Existing Pain Points:**
Recent attempts to apply SAEs to the aligned embedding space of multimodal models like CLIP have obtained mixed results. While the neurons learned by SAEs trained on CLIP demonstrate improved monosemanticity and often correspond to human-interpretable concepts, many works identify a "split-dictionary" phenomenon. In this phenomenon, many of the sparse features are only ever active for inputs coming from a single modality. In other words, SAEs learn to map embeddings which are aligned between modalities to sparse features which may be more semantic, but which have very poor modality alignment. 

Specifically, in a modality-split dictionary, for any two embeddings corresponding to data from different modalities, their sparse codes have no overlapping support, even though the original embedding space is aligned between modalities. This phenomenon limits the potential of SAEs in cross-modal tasks like retrieval or generation, where, for example, manipulation of text embeddings based on a certain "concept" should ideally be able to influence a visual or audio output. The poor modality alignment in SAEs is not exclusively due to limitations of the Linear Representation Hypothesis, but is also a product of the implicit bias of SAEs trained exclusively on reconstruction loss.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
The core research motivation is to mitigate the tradeoff between monosemanticity and modality alignment in SAEs trained on multimodal embeddings. There is a need to investigate whether it is possible to learn multimodal linear concept dictionaries and to design a model which mitigates the implicit bias towards split-dictionaries observed in regular SAEs. Improving modality alignment in the sparse decomposition would enable improvements in the interpretability and control of cross-modal tasks.

**Scientific Questions:**
1. How to effectively adapt SAEs for the setting of multimodal embeddings while ensuring multimodal alignment?
2. Is it possible to learn multimodal linear concept dictionaries where paired data samples of different modalities have aligned sparse codes?
3. How to induce a more favorable implicit bias in SAEs training on multimodal embeddings to overcome the modality-split dictionary phenomenon?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The core idea is establishing a theoretical and practical link between the modality-split phenomenon and the implicit bias of standard SAE training. Theoretically, it is argued that if a modality-split dictionary exists on an aligned embedding space, there must exist a non-split dictionary with strictly improved modality alignment and similar reconstruction loss. This implies that the poor modality alignment is a product of the SAE's optimization bias rather than a fundamental limitation of the data representation. 

**Design Philosophy:**
The design philosophy centers on enforcing group-sparsity on the sparse codes of paired multimodal samples during training. By applying a group-sparse loss (L2,1 norm), the model encourages coordinates of sparse codes for paired samples from different modalities to be jointly sparse (i.e., have the same support). Additionally, a cross-modal random masking mechanism is introduced, applying a shared random mask prior to the TopK sparsification operation. This forces the TopK operation to choose from the same subset of coordinates for both modalities at each training step, further encouraging shared sparsity structure and mitigating the occurrence of dead neurons.

**4. Core Innovation Points**

1. **New metrics for multimodal SAEs:** A formally defined paired-modality monosemanticity score (Multimodal Monosemanticity Score, MMS) quantifies the semanticity of a neuron with respect to any two modalities. It takes a value of 0 if the neuron never co-activates for inputs of both modalities and a higher value if the co-activating inputs contain similar semantic information. This generalizes existing methods for measuring semanticity at the neuron-specific level to the setting where SAE neurons are expected to co-activate for data from diverse modalities.
2. **Argument for the existence of multimodal dictionaries:** Under realistic technical assumptions, it is proven (Theorem 1) that the existence of a split dictionary on an aligned embedding space implies the existence of a non-split dictionary with strictly improved modality alignment. This suggests that the poor modality alignment in SAEs is not exclusively due to limitations of the Linear Representation Hypothesis, but rather is also a product of the implicit bias of SAEs trained exclusively on reconstruction loss.
3. **Group-sparse autoencoder architecture with improved multimodality:** A new SAE-based approach to multimodal embedding decomposition using cross-modal random masking and group-sparse regularization. The group-sparse loss function acts on paired samples during the training of SAEs, encouraging an SAE to learn multimodal dictionary vectors.
4. **Cross-modal random masking mechanism:** The introduction of a shared random mask applied with probability $p$ during the computation of sparse codes prior to the TopK operation. This forces the TopK operation to choose from the same subset of coordinates at each step prior to decoding, mitigating the occurrence of dead neurons and encouraging multimodality of concepts.
5. **Application to Audio/Text Joint Embedding Spaces:** To the authors' knowledge, this is the first work to apply SAEs to the joint embedding space of audio/text samples (specifically CLAP embeddings) and measure semanticity of the learned dictionary.

**5. Overview of the Overall Technical Solution**

The overall technical solution involves training a Masked Group-Sparse Autoencoder (MGSAE) on paired multimodal embeddings. Given paired embeddings $x, y \in \mathbb{R}^d$ corresponding to semantically similar data from different modalities, the model encodes them into a higher-dimensional vector using a shared encoder matrix and ReLU activation. During training, a shared random mask (with probability $p$) is applied to the pre-sparsified latent vectors for both modalities. The TopK function is then applied to keep only the $K$ largest elements, setting the rest to 0. The sparse codes are decoded using a shared linear decoder matrix to reconstruct the input embeddings. 

The training objective minimizes the sum of the reconstruction errors for both modalities and a group-sparse regularization term scaled by a hyperparameter $\lambda$. The group-sparse loss, defined as the L2,1 norm of the concatenated sparse codes, encourages paired samples to have jointly sparse representations (same support). This approach is evaluated on CLIP (image/text) and CLAP (audio/text) data, demonstrating improvements in the number of multimodal concepts, a reduction in dead neurons, improvements in zero-shot cross-modal tasks, and improved capabilities for interpreting multimodal models.

**6. Detailed Module Design**

**Sparse Autoencoder (SAE) Base Module:**
Given an input $x \in \mathbb{R}^d$, SAEs take the following form:
$$z = \Pi(W_{enc}(x - b_0) + b)$$
$$\hat{x} = W_{dec}z + b_0$$
Here, $b_0 \in \mathbb{R}^d$ is a bias applied prior to encoding/decoding, $W_{enc} \in \mathbb{R}^{p \times d}$ is the encoder matrix, $b$ is the encoder bias parameter, and $W_{dec} \in \mathbb{R}^{d \times p}$ is the decoder matrix. The projection function $\Pi: \mathbb{R}^p \rightarrow \mathbb{R}^p$ is chosen to promote sparsity in the latent vector $z$. In this work, $\Pi$ is the TopK function, which keeps only the $K$ largest elements of the input and sets the remaining elements to 0.

**Multimodal Encoding and Decoding Module:**
Given paired embeddings $x, y \in \mathbb{R}^d$ (corresponding to semantically similar data from different modalities), the model takes the form:
$$z_x = TopK(ReLU(W_{enc}(x - b_0) + b)), \quad z_y = TopK(ReLU(W_{enc}(y - b_1) + b))$$
$$\hat{x} = W_{dec}z_x + b_0, \quad \hat{y} = W_{dec}z_y + b_1$$
Here, $b_0$ and $b_1$ are learnable pre-coding bias terms and the remaining parameters are shared across modalities.

**Cross-Modal Random Masking Module:**
To mitigate the occurrence of dead neurons (dictionary elements which never activate for any inputs) and to further encourage multimodality of concepts, a shared random mask with probability $p$ is applied during the computation of $z_x$ and $z_y$ prior to the TopK operation. This forces the TopK operation to choose from the same subset of coordinates at each step prior to decoding.

**Multimodal Monosemanticity Score (MMS) Module:**
For each neuron and any pair of modalities $(m,n)$, the MMS is defined as follows:
1. On an unseen validation dataset, find all samples from each modality with non-zero activations for the given neuron. Store these activations as vectors $a^{(m)} \in \mathbb{R}^M$ and $a^{(n)} \in \mathbb{R}^N$.
2. Let $S \in \mathbb{R}^{M \times N}$ be the matrix of cosine similarities of these samples under a separate multimodal encoder (i.e., a different encoder than the one used in learning the dictionary).
3. Compute the co-activation matrix $A \in \mathbb{R}^{M \times N}$, where $A_{ij} = |a^{(m)}_i a^{(n)}_j|(1 - \delta_{mn})$, where $\delta_{mn} = 1$ if $m = n$, else 0.
4. Normalize co-activations to sum to 1: $\tilde{A}_{ij} = \frac{A_{ij}}{\sum_{i,j} A_{ij}}$.
5. Compute MMS(m,n) as the weighted sum of cosine similarities, with weights given by normalized co-activations:
$$MMS(m,n) = \sum_{i,j} \tilde{A}_{ij}S_{ij}$$

**7. All Mathematical Formulas and Symbol Definitions**

**Modality-Split Dictionary Definition:**
Let $W \in \mathbb{R}^{d \times p}$ with $p \gg d$ be a dictionary matrix that admits a K-sparse decomposition of any $x \in X$. Then, $W$ is said to be modality-split with respect to $X$ if, for any two embeddings $x^{(1)}, x^{(2)} \in X$ corresponding to data from different modalities,
$$\text{supp}(z^{(1)}) \cap \text{supp}(z^{(2)}) = \emptyset,$$
where $x^{(1)} = Wz^{(1)}$ and $x^{(2)} = Wz^{(2)}$ are any K-sparse decompositions of $x^{(1)}$ and $x^{(2)}$ in the dictionary.

**Theorem 1:**
Consider a set of $n$ paired unit-vector embeddings corresponding to data from different modalities $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$. Suppose the following conditions are met:
1. Paired embeddings are aligned: There is a $c > 0$ such that $\langle x^{(i)}, y^{(i)} \rangle > c$ for all $i \in [n]$.
2. Sparse decomposition in a split dictionary: There exists a modality-split dictionary $W \in \mathbb{R}^{d \times p}$ satisfying the following: for all $v \in \{x_1, y_1, \ldots, x_n, y_n\}$ there is a K-sparse vector $z_v$ with non-negative entries such that $Wz_v = v$.
Then, there exists a dictionary $\tilde{W}$ of size $p+n$ admitting a $(K+1)$-sparse decomposition of all $2n$ embeddings, and where sparse codes of all pairs have strictly positive inner product.

**Proof of Theorem 1:**
Assume that the columns of $W$ are unit-norm. Take any pair of embeddings $x$ and $y$. By the two conditions, $\langle x, y \rangle > c$ and there are K sparse vectors $z_x$ and $z_y$ with non-negative entries such that $x = Wz_x$ and $y = Wz_y$. Since $W$ is modality-split, $z_x$ and $z_y$ have disjoint support, so their inner product is 0 and
$$L_{gs}(z_x, z_y) = \|z_x\|_1 + \|z_y\|_1.$$
Assume the support of these two sparse vectors is $\{1, \ldots, K\}$ and $\{K+1, \ldots, 2K\}$. Then:
$$c < \langle z_x, z_y \rangle = \sum_{i=1}^K \sum_{j=K+1}^{2K} z_{x,i} z_{y,j} w_i^\top w_j.$$
Assume for the sake of contradiction that $w_i^\top w_j < \frac{c}{\|z_x\|_1 \|z_y\|_1}$ for all $(i, j)$ pairs in the double summation. Then:
$$\sum_{i=1}^K \sum_{j=K+1}^{2K} z_{x,i} z_{y,j} w_i^\top w_j < \sum_{i=1}^K \sum_{j=K+1}^{2K} z_{x,i} z_{y,j} \frac{c}{\|z_x\|_1 \|z_y\|_1} = c,$$
which is a contradiction. Hence, there exists some $(i, j)$ with $i \in \{1, \ldots, K\}$ and $j \in \{K+1, \ldots, 2K\}$ satisfying $w_i^\top w_j \ge \frac{c}{\|z_x\|_1 \|z_y\|_1}$. Assume WLOG these are $w_1$ and $w_{K+1}$.
Define the matrix $\tilde{W} = [w_1, \ldots, w_K, w_{K+1}, w_{K+2}, \ldots, w_p, \tilde{w}_{K+1}]$, where
$$\tilde{w}_{K+1} := \frac{w_{K+1} - (w_1^\top w_{K+1})w_1}{\|w_{K+1} - (w_1^\top w_{K+1})w_1\|_2}$$
A decomposition of $y$ in the dictionary $\tilde{W}$ can be computed as:
$$y = z_{y,K+1}w_{K+1} + \sum_{j=K+2}^{2K} z_{y,j}w_j$$
$$= z_{y,K+1} \left( \tilde{w}_{K+1}\|w_{K+1} - (w_1^\top w_{K+1})w_1\|_2 + (w_1^\top w_{K+1})w_1 \right) + \sum_{j=K+2}^{2K} z_{y,j}w_j$$
$$= z_{y,K+1}w_1^\top w_{K+1}w_1 + z_{y,K+1}\|w_{K+1} - (w_1^\top w_{K+1})w_1\|_2\tilde{w}_{K+1} + \sum_{j=K+2}^{2K} z_{y,j}w_j$$
Letting the sparse codes of $x$ and $y$ under $\tilde{W}$ be denoted as $\tilde{z}_x$ and $\tilde{z}_y$, respectively, we conclude:
$$\langle \tilde{z}_x, \tilde{z}_y \rangle = \langle z_x, \tilde{z}_y \rangle = z_{x,1}z_{y,K+1}w_1^\top w_{K+1} \ge z_{x,1}z_{y,K+1} \frac{c}{\|z_x\|_1 \|z_y\|_1} > 0.$$

**Group-Sparse Loss:**
The group-sparse loss corresponding to two sparse codes $z$ and $w$ is given by:
$$L_{gs}(z,w) = \left\| \begin{bmatrix} z^\top \\ w^\top \end{bmatrix} \right\|_{2,1} = \sum_{i=1}^p \sqrt{z_i^2 + w_i^2}$$

**Total Training Loss:**
$$L = \|x - \hat{x}\|_2^2 + \|y - \hat{y}\|_2^2 + \lambda L_{gs}(z_x, z_y)$$

**Concept Naming Formula:**
$$Concept(i) = \arg\max_{v \in V} \left\langle \frac{w_i}{\|w_i\|_2}, f(v) \right\rangle$$
where $V$ is a vocabulary of words and $f$ is the CLIP text encoder.

**8. Algorithm Pseudocode**

Based on the training approach summarized in Section 4 and Figure 2, the algorithm steps are as follows:

**Algorithm 1: Masked Group-Sparse Autoencoder (MGSAE) Training**
**Input:** Paired multimodal embeddings $(x, y) \in \mathbb{R}^d$, Encoder $W_{enc}$, Decoder $W_{dec}$, Biases $b_0, b_1, b$, Mask probability $p$, Sparsity level $K$, Regularization parameter $\lambda$.
1: **while** not converged **do**
2:     Sample paired batch $(x, y)$
3:     Compute pre-activations: 
        $h_x \leftarrow ReLU(W_{enc}(x - b_0) + b)$
        $h_y \leftarrow ReLU(W_{enc}(y - b_1) + b)$
4:     Generate shared random mask $M \sim Bernoulli(1-p) \in \{0,1\}^p$
5:     Apply shared random mask: 
        $h'_x \leftarrow h_x \odot M$
        $h'_y \leftarrow h_y \odot M$
6:     Sparsify using TopK: 
        $z_x \leftarrow TopK(h'_x)$
        $z_y \leftarrow TopK(h'_y)$
7:     Reconstruct embeddings: 
        $\hat{x} \leftarrow W_{dec}z_x + b_0$
        $\hat{y} \leftarrow W_{dec}z_y + b_1$
8:     Compute Group-Sparse Loss: 
        $L_{gs}(z_x, z_y) \leftarrow \sum_{i=1}^p \sqrt{z_{x,i}^2 + z_{y,i}^2}$
9:     Compute Total Loss: 
        $L \leftarrow \|x - \hat{x}\|_2^2 + \|y - \hat{y}\|_2^2 + \lambda L_{gs}(z_x, z_y)$
10:    Update parameters $(W_{enc}, W_{dec}, b_0, b_1, b)$ via optimizer using gradients of $L$
11: **end while**