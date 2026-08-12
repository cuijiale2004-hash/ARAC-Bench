## 1. Research Background and Existing Pain Points

**Research Background:**
Large pretrained vision transformers (ViTs) and convolutional neural networks (CNNs) have transformed computer vision, powering applications from self-driving cars to security systems and social media platforms. Trained on vast datasets and distributed via repositories such as HuggingFace, PyTorchHub, and TensorFlow Model Zoo, these models achieve strong performance across tasks including image classification, object detection, semantic segmentation, and face recognition. Yet, deployment environments vary dramatically from powerful cloud servers to resource-constrained edge devices. This poses a fundamental challenge: how can we efficiently deploy pretrained models under diverse computational budgets without sacrificing accuracy? Many-to-many NAS with model stitching offers a promising solution by combining blocks from different pretrained “anchor” models to form stitched networks that interpolate accuracy-efficiency tradeoffs. For example, SN-Net finetunes lightweight stitching layers (simple linear transformations) between anchors to drastically reduce cost.

**Existing Pain Points:**
1. **Prohibitive Search Cost:** Even with only two anchor models of depth n, there are O(n^2) possible stitch configurations, making exhaustive search computationally infeasible.
2. **Suboptimal Heuristics:** Existing stitching approaches like SN-Net employ naive heuristic-based selection (e.g., stitching adjacent anchors and blocks, termed "nearest stitching" or "paired/unpaired stitching"), which limits both the accuracy-efficiency tradeoff and generalizability across model families. These heuristics ignore actual compatibility between blocks.
3. **Failure of Existing Similarity Metrics:** Commonly used similarity metrics fail to automatically recover effective stitch configurations. CKA recovers only 5.5% of configurations, while MSE, CE, and DM recover between 22.2% and 33.3%. Prior work has found no correlation between common similarity metrics (e.g., CKA) and stitched model accuracy, and direct similarity measures often fail to reflect task performance.
4. **Lack of Generalizability:** Current many-to-many NAS approaches are confined to a single model design space or rely on weak heuristics that prevent them from generalizing across diverse model architectures (e.g., stitching across hierarchical Swin Transformers and residual CNNs).

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Constructing improved accuracy-efficiency tradeoffs requires explicitly capturing and leveraging the similarity between pretrained models being stitched. Stitching success hinges on activation similarity, which determines whether two networks can be integrated through a linear transformation layer. The stitching layer must map the source activations to the target activations while keeping finetuning costs low and preserving task performance. The first requirement corresponds to representational similarity, and the second to functional similarity. Existing metrics capture only one or neither, necessitating a principled similarity-driven approach that satisfies both objectives.

**Scientific Questions:**
1. Can network B produce similar outputs when driven by transformed internal representations from network A?
2. Can such transformations be effective with minimal finetuning?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Replace heuristic-based stitching with a principled similarity-driven approach that automates and generalizes stitch selection across model families by leveraging KL divergence between intermediate representations. 

**Design Philosophy:**
KL divergence satisfies the dual objectives of measuring both representational and functional similarity. As a representational similarity metric, KL divergence captures distributional differences between intermediate activations, indicating whether two blocks generate patterns that can be mapped via lightweight transformations. At the same time, it serves as a functional similarity metric by assessing how well one block’s outputs preserve the target model’s decision boundaries: a low KL value implies that the source block provides task-relevant information that the target can leverage with minimal finetuning. By minimizing KL divergence between intermediate activations, the stitching layer requires only minor adjustments to align representations while preserving the target’s original performance.

## 4. Core Innovation Points

1. **Identifying Limitations of Heuristics:** Identifying the limitations of heuristic-based many-to-many NAS approaches such as SN-Net, demonstrating that naive heuristics (nearest stitching) yield suboptimal accuracy-efficiency tradeoffs and lack generalizability.
2. **Introducing KL Divergence for Stitch Selection:** Introducing KL divergence as a principled similarity metric for selecting stitch configurations without instantiating or training a stitch layer, effectively capturing both representational and functional alignment.
3. **The KLAS Framework:** Presenting the KLAS (KL divergence based Anchor Stitching) framework for efficient, generalizable, and low-cost stitching, which automates the selection of anchors and block pairs using ProbeNet and stitch scores.
4. **Broad Empirical Validation:** Demonstrating consistent improvements over SN-Net across model families (Swin, DeiT, LeViT, ResNet, cross-architecture ResNet-Swin, LLMs), datasets (ImageNet-1K, CIFAR-100, ADE20K, TruthfulQA), and tasks (classification, semantic segmentation, language generation).

## 5. Overview of the Overall Technical Solution

The KLAS framework operates through the following pipeline:
1. **Probe Training:** Unified architecture ProbeNet is introduced to jointly train linear classifier probes placed after different blocks of anchor models to estimate the softmax probability distributions of activations with negligible one-time cost.
2. **Anchor Selection:** Compute the KL divergence between the softmax probability distributions produced by the last block of each anchor. Low KL divergence indicates compatible anchors.
3. **Block Selection:** Compute a stitch score Γ for all source-target block pairs. The stitch score is the ratio of cross-anchor activation distance (Ω) to intra-anchor block capacity (Σ).
4. **Candidate Configuration Selection:** Partition the computational space between anchors into FLOPs buckets. Within each bucket, select stitch configurations where the stitch score is below a tunable threshold τ, alongside the configuration with the absolute minimum stitch score, ensuring bucket-level coverage and threshold-based filtering.

## 6. Detailed Module Design

**1. ProbeNet Module:**
To estimate the KL divergence between the softmax probability distributions of activations, each block in an anchor model is equipped with a simple 1×1 convolutional probe. Instead of training all probes independently, ProbeNet jointly trains all probes by activating one at a time within a single forward-backward pass. This design adds only a negligible one-time cost, as probes converge rapidly (e.g., by epoch 4). The source anchor always has lower complexity than the target anchor to ensure stable training.

**2. Anchor Selection Module:**
Identifies suitable anchors by computing the KL divergence between the softmax probability distributions produced by the last block of each anchor. Since the softmax distribution is already computed in the final blocks, additional probes and training are not required at this stage. A low KL divergence indicates that two models have similar decision boundaries and class-confidence distributions, making them more compatible for stitching.

**3. Block Selection Module (Stitch Score):**
Evaluates the compatibility of block pairs without instantiating or training a stitch layer.
- **Cross-anchor activation distance (Ω):** Measures how closely the activations of source block i from f align with those of target block j in g. It quantifies the transformation a hypothetical stitch layer (T) would need to apply to the source activations to match the target activations. A smaller Ω implies that the activations after layer i in f are already close to the activations after layer j in g, making the block pair (i, j) more favorable for stitching.
- **Intra-anchor block capacity (Σ):** Captures how much a block in a pretrained anchor transforms its input representations, measured by comparing the outputs of consecutive blocks within the same anchor. A large Σ indicates a substantial shift in activations from block j to block j+1 in model g, meaning block j+1 has high learning capacity. High-capacity blocks are valuable for stitching as they can absorb distributional differences from the source while maintaining task-relevant transformations.

**4. Candidate Set Construction Module:**
During the stitch finetuning phase, a set of candidate stitch configurations (S) is selected using a tunable threshold (τ). Candidates are drawn from a set of FLOPs buckets (B), which partition the computational space between f and g. The bucket granularity controls the density of the resulting accuracy-efficiency tradeoff curve. To ensure coverage, B is chosen such that each block in the target anchor g has at least one stitch configuration represented in S. The formulation balances computational efficiency with stitch quality.

## 7. All Mathematical Formulas and Symbol Definitions

**Notations for Model Stitching:**
- f : X → Y : A feedforward neural network with m blocks.
- f = f_m ◦ · · · ◦ f_1 : Composition of m blocks.
- f_i : A^f_{i-1} → A^f_i : Mapping from activation space of block i-1 to block i.
- A^f_0 = X : The input samples.
- f_{≤i} = f_i ◦ · · · ◦ f_1 : Blocks from 1 to i.
- f_{>i} = f_m ◦ · · · ◦ f_{i+1} : Blocks from i+1 to m.
- g_{>j} : The target (later layers of network g).
- f_{≤i} : The source (earlier layers of network f).
- T : A^f_i → A^g_j : The stitching layer.
- (i, j) : A stitch configuration (i-th layer of f stitched to j-th layer of g).
- f_{≤i}(x) : Source representation.
- g_{≤j}(x) : Target representation.

**KL Divergence Similarity Score:**
- P^f_i(x) : Softmax probability distribution produced by the probe inserted after block i of anchor model f on input x ∈ D_v.
- P^g_j(x) : Distribution from the probe after block j of anchor model g.
- D_v : Validation split.
- D_t : Training split.
- D_KL(p || q) : KL divergence from distribution p to target q.
- Θ(P^f_i, P^g_j) = \frac{\sum_{x \in D_v} D_{KL}(P^f_i(x) || P^g_j(x))}{|D_v|}  (Equation 1)

**Stitch Score:**
- Γ(i, j) = \frac{\overbrace{\Theta(P^f_i, P^g_j)}^{\text{cross-anchor activation distance } (\Omega)}}{\underbrace{\Theta(P^g_j, P^g_{j+1})}_{\text{intra-anchor block capacity } (\Sigma)}}  (Equation 2)

**Candidate Set Selection:**
- B : Set of FLOPs buckets partitioning the computational space.
- τ : Tunable threshold.
- S = \bigcup_{b \in B} R^*_b  (Equation 3)
- R^*_b = \left( \{ (i^*, j^*) \mid \Gamma(i, j) \le \tau, \forall (i, j) \in b \} \bigcup \{ (i^*, j^*) = \text{argmin}_{(i,j) \in b} \Gamma(i, j) \} \right)

## 8. Algorithm Pseudocode

**Algorithm 1 ProbeNet Pseudocode**
1: Class ProbeNet
2: Init(model, stage_block_info, output_dim = 1000):
3: Store model as anchor (frozen)
4: for each stage in stage_block_info:
5: for each block_dim in stage:
6: Create probe: Linear(block_dim→ output_dim) + Softmax
7: Store probe in self.layers[stage][block]
8: Set current_features← None
9: Freeze anchor parameters; keep probes trainable
10:
11: Method extract_block_features(x):
12: Run anchor.extract_block_features(x) with no_grad
13: Store result in current_features
14:
15: Method clear_block_features():
16: Set current_features← None
17:
18: Method iter_blocks():
19: yield (stage_id, block_id) for each stage/block
20:
21: Method forward(x, stage_id, block_id, get_all_probe_losses, criterion, target):
22: if get_all_probe_losses:
23: return anchor.get_probe_losses(x, self.layers, criterion, target, dims)
24: if current_features is None:
25: extract_block_features(x)
26: Get block feature→ reshape→ pass through probe
27: return output

**Algorithm 2 Get Similarity Scores**
Inputs: Set of all anchor models M
Output: Set of all similarity scores K
1: K ← {}
2: for each model m in M do
3: for each block i in m do
4: Add a linear classifier probe P^m_i
5: Train P^m_i on train split (D_t)
6: end for
7: end for
8: for each model pair (m,n) in M , with m having lower complexity than n do
9: for each block i in anchor m do
10: for each block j in anchor n do
11: K ← K ∪ {Θ(P^m_i, P^n_j)}
12: end for
13: end for
14: end for
15: Return K

**Algorithm 3 KLAS: KL divergence based Anchor Stitching**
Inputs: Anchor models M , similarity scores K from Alg. 2, threshold τ , buckets B
Output: Set of selected stitch configurations S
1: S ← {}
2: Pick anchors f, g ∈M based on their final layer KL divergence
3: Sort all the stitches based on FLOPs
4: for each bucket b ∈ B do
5: R_b ← {}
6: for each stitch configuration (i, j) ∈ b do
7: Compute Γ(i, j) (see Eq. 2)
8: R_b ← R_b ∪ Γ(i, j)
9: end for
10: Compute set of stitch configs R^*_b using R_b (see Eq. 3)
11: S ← S ∪ R^*_b
12: end for
13: Return S