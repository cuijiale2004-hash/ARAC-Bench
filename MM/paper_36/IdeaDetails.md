**1. Research Background and Existing Pain Points**

**Research Background:**
In real-world scenarios, deep learning models are often required to generalize reliably to unseen distributions. However, due to domain shift, their performance often degrades substantially when deployed in new environments. To address this issue, Domain Generalization (DG) has emerged as a key research area, aiming to learn representations and prediction functions that transfer well to unseen domains using only source-domain data. Existing DG approaches generally fall into four categories: invariant representation learning, data augmentation, regularization, and meta-learning. Recently, multi-modal large language models (MLLMs) have made significant progress and demonstrate strong reasoning capabilities. This capability offers a new opportunity for DG: rather than relying solely on invariant feature representations, one could exploit reasoning chains to bridge domain gaps and achieve more robust generalization.

**Existing Pain Points:**
1. Existing DG methods primarily operate at the representation level and focus on feature-level invariance. They often fail to capture higher-level cross-domain commonalities, limiting their ability to generalize in complex scenarios. They overlook the reasoning processes that underlie generalizable decision-making.
2. Existing improvements in MLLMs for DG remain rooted in label-level supervision and fail to capture the explicit reasoning processes required for robust generalization under distribution shift.
3. Simply applying reasoning-chain supervision to classification tasks remains challenging. There is an optimization gap: fine-tuning MLLMs with reasoning chains for classification is more challenging than direct label supervision, since the model must optimize complex reasoning sequences before label prediction.
4. There are mismatches in reasoning patterns between supervision signals and fine-tuned MLLMs. Commercial-model chains usually contain detailed contextual information (semantically rich but difficult to optimize), while self-generated chains from the fine-tuned MLLM tend to be simpler in logic and more label-focused (easier to fit but less informative). This creates a trade-off between semantic richness and optimization efficiency.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
To leverage the reasoning capability of multimodal large language models (MLLMs) and explore the potential of constructing reasoning chains that derives image categories to achieve more robust predictions under domain shift. By producing structured, class-relevant reasoning chains, we can leverage MLLMs to explicitly decompose a classification task into interpretable and domain invariant steps with strong transferability across domains. We aim to encourage process-level invariance through class-relevant reasoning chains, which serve as complementary signals to feature-level invariance for improving out-of-domain generalization.

**Scientific Questions:**
1. Does directly distilling class-relevant reasoning chains into MLLMs improve domain generalization in classification? (Investigation reveals that zero-shot reasoning can enhance out-of-domain generalization, but directly distilling reasoning chains during supervised fine-tuning is less effective due to optimization difficulty.)
2. How do mismatches in reasoning patterns across different reasoning sources influence the supervision signals and optimization behavior of models? (Investigation reveals a trade-off: large-model reasoning provides richer but harder-to-optimize signals, while self-generated reasoning provides simpler cues that focus more directly on the category label, reducing the effectiveness of direct reasoning-chain supervision.)

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
To propose RD-MLDG (Reasoning-Driven Multimodal LLM for Domain Generalization), the first DG framework that explicitly incorporates class-relevant reasoning chains to enhance out-of-domain performance. It addresses the optimization difficulty and reasoning-pattern mismatches by integrating reasoning into DG through Multi-Task Cross-Training (MTCT) and Self-Aligned Reasoning Regularization (SARR).

**Design Philosophy:**
1. **Process-level Invariance:** Instead of relying solely on feature-level invariance, use class-relevant reasoning chains as complementary signals to capture higher-level cross-domain commonalities.
2. **Guided Optimization:** Address the optimization gap of reasoning-chain supervision by introducing a direct classification pathway that serves as a simple and stable anchor objective to guide reasoning optimization.
3. **Self-Alignment:** Address reasoning-pattern mismatches by retaining informative reasoning chains while producing signals that the model can fit more effectively through iterative self-labeling, thereby mitigating the trade-off between semantic richness and optimization efficiency.

**4. Core Innovation Points**

1. **Construction of DomainBed-Reasoning:** An extension of the DomainBed dataset where each sample is paired with class-relevant reasoning chains, enabling systematic evaluation of both classification accuracy and reasoning consistency under domain shift.
2. **Identification of Two Key Challenges:** Systematic analysis revealing that (i) fine-tuning MLLMs with reasoning chains for classification is more challenging than direct label supervision due to the optimization gap, and (ii) mismatches in reasoning patterns between supervision signals and fine-tuned MLLMs lead to a trade-off between semantic richness and optimization efficiency.
3. **Multi-Task Cross-Training (MTCT):** A module that introduces an additional direct classification pathway to guide reasoning supervision. The classification objective provides a stable signal that helps reasoning supervision contribute more effectively to the classification task.
4. **Self-Aligned Reasoning Regularization (SARR):** A module that preserves the semantic richness of reasoning chains while mitigating reasoning-pattern mismatches via iterative self-labeling. It shifts supervision toward signals the model can fit more readily while retaining class-relevant information.

**5. Overview of the Overall Technical Solution**

The RD-MLDG framework integrates reasoning into DG through a two-stage process using a base MLLM (with LoRA adapters inserted into both the vision encoder and the language decoder).
1. **Data Preparation:** Extend the DomainBed dataset to DomainBed-Reasoning using GPT-4o to generate multi-stage reasoning chains (<SUMMARY>, <CAPTION>, <REASONING>, <REFLECTION>, <CONCLUSION>) without access to ground-truth labels, filtered by rejection sampling.
2. **Stage 1 - Multi-Task Cross-Training (MTCT):** Fine-tune the model by jointly optimizing a direct classification pathway (using a classification prompt) and a reasoning-augmented pathway (using a reasoning prompt with the GPT-4o generated chains). The classification loss stabilizes the optimization of the reasoning chain.
3. **Stage 2 - Self-Aligned Reasoning Regularization (SARR):** Perform iterative self-labeling for N rounds. The model generates its own reasoning chains using the reasoning prompt. Extract the predicted conclusion and compare it with the ground-truth label. Only retain reasoning chains with correct final conclusions. Fine-tune the model again using these self-generated, filtered reasoning chains combined with the classification prompt. This procedure progressively strengthens the semantic alignment between the reasoning chain and the model’s final prediction.

**6. Detailed Module Design**

**Module 1: DomainBed-Reasoning Construction**
*   **Input:** Image-label pairs from standard DG datasets (PACS, VLCS, OfficeHome, TerraInc).
*   **Mechanism:** Employ GPT-4o to generate multi-stage reasoning chains consisting of <SUMMARY>, <CAPTION>, <REASONING>, <REFLECTION>, and <CONCLUSION>. The ground-truth label is withheld during generation, encouraging reasoning grounded in visual evidence. An additional reflection stage prompts the model to self-check intermediate reasoning, empirically reducing invalid generations.
*   **Filtering:** Adopt a rejection sampling strategy: for each instance, multiple candidate chains are generated, and only those that contain all required components and produce a coherent conclusion are retained.

**Module 2: Multi-Task Cross-Training (MTCT)**
*   **Input:** A batch of training data $\{(x_i, y_i, r_i)\}_{i=1}^B$, where $r_i$ is the class-relevant reasoning chain for image $x_i$.
*   **Prompt Construction:** Construct two prompts: (i) a reasoning prompt $q_{reason}$ from DomainBed-Reasoning with its corresponding class-relevant reasoning chain $r_i$, and (ii) a classification prompt $q_{cls}$ — “What type of object is in this photo? Choose from the following options:”.
*   **Fine-tuning Strategy:** Insert LoRA adapters into both the vision encoder and the language decoder, enabling parameter-efficient supervised fine-tuning (SFT) without updating all parameters.
*   **Mechanism:** Jointly optimize reasoning-chain supervision with a direct classification pathway. The classification supervision serves as a simple and stable anchor objective, which guides reasoning optimization and helps alleviate the difficulty of fitting intermediate reasoning steps before predicting the final label. The loss function combines the negative log probability of the ground-truth label given the classification prompt and the normalized negative log probability of the reasoning chain tokens given the reasoning prompt.

**Module 3: Self-Aligned Reasoning Regularization (SARR)**
*   **Input:** Model from MTCT stage, source domain training data.
*   **Mechanism:** A soft self-labeling procedure. The model generates its own reasoning under the instruction $q_{reason}$. From each output, extract the predicted conclusion enclosed between <CONCLUSION> and </CONCLUSION> and compare it with the ground-truth label $y_i$. Only reasoning chains with correct final conclusions are retained and used as refined supervision signals.
*   **Iterative Refinement:** These retained reasoning chains are then combined with the classification prompt $q_{cls}$ as part of the MTCT fine-tuning process. In each round of SARR, the reasoning chains, initially sourced from GPT-4o in the first stage, are replaced by self-generated reasoning chains in later stages. The training objective for each round remains the same as MTCT. This self-labeling procedure can be repeated for N rounds.

**7. All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
*   $X$: Input space.
*   $Y$: Label space.
*   $N_s$: Number of labeled source domains.
*   $\hat{D}_S = \{\hat{D}_S^m\}_{m=1}^{N_s}$: Set of labeled source domains.
*   $\hat{D}_S^m = \{(x_i, y_i)\}_{i=1}^{N_s^m}$: The $m$-th source domain with $N_s^m$ labeled samples.
*   $x_i \in X$: Input image.
*   $y_i \in Y$: Corresponding label.
*   $\hat{D}_T = \{\hat{D}_T^m\}_{m=1}^{N_t}$: Unseen target set.
*   $B$: Batch size.
*   $r_i = (r_{i,1}, \dots, r_{i,T_i})$: The class-relevant reasoning chain for $x_i$.
*   $r_{i,<t} = (r_{i,1}, \dots, r_{i,t-1})$: The prefix sequence before predicting token $r_{i,t}$.
*   $T_i$: The length (number of tokens) of the reasoning chain $r_i$.
*   $q_{reason}$: Reasoning prompt instruction template.
*   $q_{cls}$: Classification prompt instruction template.
*   $p_\theta$: Probability distribution parameterized by $\theta$.
*   $\hat{r}_i$: A reasoning chain retained because its final conclusion matches the ground-truth label.
*   $B'$: Batch size for self-generated reasoning chains.
*   $f(x)$: Embedding extracted from CLIP’s vision encoder for visual features, or from CLIP’s text encoder for reasoning-chain embeddings.

**Mathematical Formulas:**

*   **MTCT Loss (Equation 1):**
    $$ \mathcal{L}_{MTCT} = \frac{1}{B} \sum_{i=1}^B \left[ -\log p_\theta(y_i | x_i, q_{cls}) - \frac{1}{T_i} \sum_{t=1}^{T_i} \log p_\theta(r_{i,t} | r_{i,<t}, x_i, q_{reason}) \right] $$

*   **SARR Loss (Equation 2):**
    $$ \mathcal{L}_{SARR} = \frac{1}{B'} \sum_{i=1}^{B'} \left[ -\log p_\theta(y_i | x_i, q_{cls}) - \frac{1}{T_i} \sum_{t=1}^{T_i} \log p_\theta(\hat{r}_{i,t} | \hat{r}_{i,<t}, x_i, q_{reason}) \right] $$

*   **Maximum Mean Discrepancy (MMD) for Domain Divergence Validation:**
    $$ \text{MMD}^2(\hat{D}_S^m, \hat{D}_T^{m'}) = \left\| \frac{1}{N_s^m} \sum_{i=1}^{N_s^m} f(x_i) - \frac{1}{N_t^{m'}} \sum_{j=1}^{N_t^{m'}} f(x_j) \right\|_2^2 $$

**8. Algorithm Pseudocode**

**Algorithm 1 RD-MLDG with MTCT and SARR**
**Require:** Source domains $\{\hat{D}_S^m\}_{m=1}^{N_S}$, base model $f_\theta$ , DomainBed-Reasoning dataset, self-labeling rounds $N$
**Ensure:** Fine-tuned model $f_\theta$

▷ Stage 1: Multi-Task Cross-Training (MTCT)
1: for each minibatch $\{(x_i, y_i, r_i)\}_{i=1}^B$ do
2:     Construct classification prompt $q_{cls}$ and reasoning prompt $q_{reason}$
3:     Compute classification loss $\mathcal{L}_{cls}$
4:     Compute reasoning loss $\mathcal{L}_{reason}$
5:     Update $\theta$ with $\mathcal{L}_{MTCT} = \mathcal{L}_{cls} + \mathcal{L}_{reason}$ (Eq. 1)
6: end for

▷ Stage 2: Self-Labeled Reasoning Regularization (SARR)
7: for $k = 1$ to $N$ do
8:     for each sample $(x_i, y_i)$ do
9:         Generate reasoning $\hat{r}_i$ with $q_{reason}$
10:        Extract predicted conclusion between <CONCLUSION> and </CONCLUSION> tags
11:        if conclusion $= y_i$ then
12:            Retain $(x_i, y_i, \hat{r}_i)$ as supervision
13:        end if
14:    end for
15:    Fine-tune on retained pairs with $\mathcal{L}_{SARR} = \mathcal{L}_{cls} + \mathcal{L}_{reason}$ (Eq. 2)
16: end for