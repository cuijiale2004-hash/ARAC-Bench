### 1. Research Background and Existing Pain Points
Large Language Models (LLMs) have seen remarkable advancements, achieving state-of-the-art results in diverse applications. Fine-tuning is an important step for adapting LLMs to specific downstream tasks, typically involving further training on corresponding datasets. However, a fundamental discrepancy exists between current fine-tuning datasets and the token-level optimization mechanism of LLMs: most datasets are designed at the sentence-level, which introduces token-level noise, causing negative influence to final performance. While LLM fine-tuning involves computing a loss at each token and updating model parameters accordingly, most fine-tuning datasets provide label sentences as the target output. Since not all tokens in the label sentence are valuable for performance improvement, training on the entire label sentence can introduce token-level noise and misguide the direction of convergence, ultimately reducing performance of fine-tuned LLMs in the target downstream task.

Current research lacks the capability to optimize datasets at the token level for LLM fine-tuning tasks. Mainstream data optimization methods fall into two categories: data filtering and data augmentation, all of which operate at the sample level and thus fail to further eliminate token-level noise. Some existing studies have explored the differences between token-level and sentence-level data from various perspectives, such as pretraining, human preference optimization, and knowledge distillation. However, these works are often limited to specific scenarios (e.g., pretraining with lower data quality requirements or direct preference optimization (DPO) that relies on prior knowledge of labeled text pairs) or do not sufficiently investigate the value differences between tokens (e.g., as in some knowledge distillation approaches), rendering them unsuitable for fine-tuning dataset optimization. Achieving token-level dataset optimization for fine-tuning necessitates filtering out tokens in output labels that do not contribute to final performance, which is a non-trivial task.

### 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:** To address the fundamental discrepancy between sentence-level fine-tuning datasets and token-level optimization mechanisms of LLMs. There is a critical need to assess the value of token-level data within fine-tuning datasets and filter out noisy tokens by considering the specific characteristics of LLM fine-tuning. 

**Scientific Questions:**
1. No existing research clearly elucidates the relationship between individual tokens within these labels and fine-tuning effectiveness. Although some explainability studies can identify connections between tokens in the input content and the correct generation of label sentences during the reasoning process, they cannot explain the value of specific tokens in label sentences for fine-tuning tasks.
2. Fine-tuning performance depends on both the base model’s pre-existing knowledge and the specifics of the target task. When filtering noisy tokens, it is essential to consider both the base model’s understanding of the data and that data’s relevance to the downstream task. Therefore, filtering noisy tokens from fine-tuning datasets requires a comprehensive consideration of fine-tuning task requirements, rather than reliance on a single assessment criterion.

### 3. Overall Core Idea and Design Philosophy
The core idea is to propose an explainable token-level data filtering method, XTF. This method decomposes the complex and subtle contributions of token-level data to the fine-tuning process into three distinct and explicit attributes (reasoning importance, knowledge novelty, and task relevance), which can be assessed using scoring methods, and then masks the gradients of selected noisy tokens accordingly to optimize the performance of fine-tuned LLMs. 

**Design Philosophy:**
A fine-tuning process can be conceptualized as an alignment between a high-performance base model and a task-specific dataset. Consequently, the performance of the fine-tuned model is influenced by three factors: the cognition of the base model, the knowledge in the task dataset, and the contradiction between the base model and the task dataset. When masking a token from the label sentence, its potential impact can be assessed from these three perspectives. It is not feasible to consider all three properties simultaneously and assign a composite score, as there is no clear basis to determine their interrelationship or any hierarchical order. However, if a token completely lacks any of the three attributes, it can be considered as noise. This transforms a complex problem into several simpler problems, using multiple perspectives to provide complementary effects.

### 4. Core Innovation Points
1. **Revealing the Research Gap:** The paper reveals the research gap of token-level data optimization for LLM fine-tuning, highlighting the fundamental discrepancy between sentence-level datasets and token-level optimization mechanisms.
2. **Attribute Decomposition Strategy:** The paper explores solutions for filtering token-level noise in fine-tuning datasets via three decomposed attributes: reasoning importance (RI), knowledge novelty (KN), and task relevance (TR). This decomposition reduces the complexity of token value analysis and provides complementary effects for noise identification.
3. **Parameterized Scoring Mechanisms:** The paper designs scoring mechanisms for the three attributes with controllable computational costs, jointly considering the base model and the task dataset: using attention scores for RI, probability of correct token prediction (PCP) for KN, and embedding semantic distance for TR.
4. **Conservative Filtering and Gradient Masking:** The paper adopts a conservative strategy with adaptive statistical thresholds (Quantile method, probability threshold, Multi-Ostu) and a union operation to identify noisy tokens, subsequently masking their gradients during training.
5. **Rigorous Theoretical Justification:** The paper provides complete, assumption-transparent, and parameterization-invariant theoretical foundations proving that token filtering strictly improves M-alignment and guarantees a one-step decrease in ideal risk.

### 5. Overview of the Overall Technical Solution
The XTF pipeline comprises three steps. In the first step, the dataset is preprocessed based on a regular format function. In the second phase, the method gets the sentence-level data item and assesses three scores, i.e., attention score, PCP score and relevance score, for the tokens of output label, suggesting the selection of noisy tokens. In the third phase, noisy tokens are masked and the target LLM is fine-tuned. Specifically, the token-level noise in the fine-tuning tasks is represented as the union of data lacking reasoning importance, knowledge novelty, and task relevance. Tokens are scored based on the base model, and noisy tokens are filtered using adaptive thresholds. During training, noisy tokens are assigned a default value (often [-100]) and thereby excluded from gradient computation.

### 6. Detailed Module Design
**Attribute Decomposition Module:** 
Defines three attributes that positively influence the fine-tuning process:
- **Reasoning Importance (RI):** whether the presence or absence of this token significantly affects the base model’s inference results.
- **Knowledge Novelty (KN):** whether the presence of this token is novel to the base model.
- **Task Relevance (TR):** whether the presence of this token is related to the objective of the task dataset.

Formally, the token-level noise is defined as:
$D_{noise} = (D_{RI\downarrow}) \cup (D_{KN\downarrow}) \cup (D_{TR\downarrow})$
where $(D_{RI\downarrow})$, $(D_{KN\downarrow})$ and $(D_{TR\downarrow})$ respectively represent data lacking reasoning importance, knowledge novelty, and task relevance.

**Scoring Mechanism Module:**
1. **Attention Score for RI:** Concatenate the input and label output, then compute attention scores for each token using the base model. A lower attention score indicates a lower reasoning importance. The reasoning importance score $S_{RI}$ for the $k$-th token in the output label sentence is formulated as:
$S_{RI}(O_k) = A (\theta, I +O) [l_I + k]$
where $\theta$ denotes the parameters of base model, $A$ is the function to compute attention value, $I$, $O$ represents the input tokens and output label respectively, and $l_I$ is the length of input tokens.

2. **PCP Score for KN:** Introduce the probability of correct token prediction (PCP) to quantify the novelty of knowledge learned from the fine-tuning dataset. A higher PCP indicates lower knowledge novelty. The knowledge novelty score $S_{KN}$ is formulated as:
$S_{KN}(O_k) = 1- P (O_k | I + [O_0, O_1, . . . , O_{k-1}])$

3. **Distance Score for TR:** Assess task relevance using embedding vectors generated by the base model for context-free inputs. The task domain is approximated by the average embedding of data samples, and token relevance scores are determined by their distance from the domain center. A larger distance implies lower task relevance. The specific formulas for the task relevance scores are:
$V(Domain) =\frac{\sum (E(\theta, exp_w))}{n_w}$
$S_{TR}(O_k) = 1 -Normalize (D (E(O_k),V(Domain)))$
where $E$ denotes the function to get the embedding vector of inputs, $exp_w$ and $n_w$ represents the expert words and its number, respectively. $D$ is the function to compute the distance of two vector.

**Token Filtering Module:**
Employs adaptive mechanisms for each attribute based on score distribution features:
- **RI Filtering (Quantile method):** The reasoning importance scores exhibit an extreme distribution where many scores share identical values. Apply the quantile method (Interquartile Range) to filter out tokens with extremely low scores:
$Q_1, Q_3 = Quantile(S_{RI}(O), [25, 75])$
$IQR = Q_1 - (Q_3 -Q_1)$
$O_k \in (D_{RI\downarrow}) \text{ if } S_{RI}(O_k) < Q_1 - IQR$
where $S_{RI}(O)$ means all the reasoning importance score of tokens in output label $O$.

- **KN Filtering (Heuristic threshold):** The knowledge novelty scores display a uniform distribution. Adopt a heuristic threshold. Consider tokens with a PCP higher than 95% as only containing knowledge without novelty and treat them as noise:
$O_k \in (D_{KN\downarrow}) \text{ if } S_{KN} (O_k) < 0.05$

- **TR Filtering (Multi-Otsu method):** The task relevance scores exhibit cluster characteristics. Employ the Multi-Otsu method to partition the scores. Filter out the tokens in the cluster with the second smallest mean value:
$O_k \in (D_{TR\downarrow}) \text{ if } S_{TR}(O_k) \in M(S_{TR})_{2nd}$
where $M$ is the Multi-Otsu method and $M(S_{TR})_{2nd}$ denotes the cluster with the second smallest mean value.

**Training Module:**
Mask the input tokens by assigning them a default value (often [-100]) and thereby exclude them from gradient computation. After identifying noisy tokens in the output labels, mark them with this default value and use the resulting data for fine-tuning. Given the noisy token list $N$, the loss function $L_F$ for learning a data item can be expressed as:
$L_F = -\sum_{O_k \notin N} \log P(O_k | I + [O_0, O_1, . . . , O_{k-1}])$

### 7. All Mathematical Formulas and Symbol Definitions
**Theoretical Foundations (Appendix A):**
A token–level context is $c := (x, t_{<i})$ and the associated label is $t$. Let $p_\theta(t | c)$ denote the model’s conditional distribution at parameters $\theta$, and let $\phi_\theta(c, t) := \nabla_\theta \log p_\theta(t | c)$ be the score of the pair $(c, t)$.

Damped Fisher information at $\theta$:
$F_\lambda(\theta) := E_{c\sim Q} E_{t\sim p_\theta(\cdot|c)} [\phi_\theta(c, t)\phi_\theta(c, t)^\top] + \lambda I, \lambda \geq 0$

For any SPD matrix $M(\theta) \succ 0$:
$\langle u, v \rangle_M := u^\top M(\theta)v, \quad \|u\|_{M^{-1}} := \sqrt{u^\top M(\theta)^{-1}u}$

Ideal per–token KL risk:
$L^\star(\theta) := E_{(c,t)\sim p^\star} [- \log p_\theta(t | c)] = KL(p^\star \| p_\theta) + const.$
$\nabla_\theta L^\star(\theta) = -E_{(c,t)\sim p^\star} [\phi_\theta(c, t)]$

Preconditioned gradient direction:
$\nablã_M L^\star(\theta) := M(\theta)^{-1}\nabla_\theta L^\star(\theta)$

**Assumptions:**
Assumption 1 (Damped Fisher and teacher forcing): (1) $Q(c)$ is fixed by teacher forcing and independent of filtering. (2) There exists $\lambda \geq 0$ such that $F_\lambda(\theta) \succ 0$ for all $\theta$ in a neighborhood of interest.

Assumption 2 (Mixture model of labels): There exists $\varepsilon \in [0, 1)$ and distributions $p_{core}, p_{noise}$ on $(c, t)$ such that $p_{train}(c, t) = (1- \varepsilon) p_{core}(c, t) + \varepsilon p_{noise}(c, t)$, with $p_{core}(c, t) \equiv p^\star(c, t)$.

Assumption 3 (Selector quality & independence within components): Let $Z(c, t) \in \{0, 1\}$ be the indicator that the token is kept by the filter. Define the error rates $\alpha := Pr[Z = 0 | (c, t) \sim p_{core}]$, $\beta := Pr[Z = 1 | (c, t) \sim p_{noise}]$. Assume $\alpha + \beta < 1$.
Strong (MAR within components): $Z \perp (c, t) | G$.
Weak (bounded selection bias): There exist $\rho_c, \rho_n \geq 0$ such that
$\|E[\phi_\theta(c, t) | G=core, Z=1] - E[\phi_\theta(c, t) | G=core]\|_{M^{-1}} \leq \rho_c \|g_{core}(\theta)\|_{M^{-1}}$
$\|E[\phi_\theta(c, t) | G=noise, Z=1] - E[\phi_\theta(c, t) | G=noise]\|_{M^{-1}} \leq \rho_n \|g_{core}(\theta)\|_{M^{-1}}$

Assumption 4 (Teacher forcing invariance): Filtering removes token losses but does not change the context law $Q(c)$ in the forward pass.

Assumption 5 (Local smoothness of $L^\star$): For each $\theta$ there exist $r > 0$ and $L = L(\theta, r) > 0$ such that, for all $u$ with $\|u\|_2 \leq r$, $L^\star(\theta + u) \leq L^\star(\theta) + \nabla L^\star(\theta)^\top u + \frac{L}{2} \|u\|_2^2$.

Assumption 6 (Score–norm control for high–confidence tokens): Assume the logits $z_\theta(c) \in \mathbb{R}^K$ are coordinatewise $L_z$–Lipschitz in $\theta$. If $F_\lambda(\theta) \succeq \mu I$ for some $\mu > 0$, then for all $(c, t)$ $\|\phi_\theta(c, t)\|_{F_\lambda^{-1}} \leq \frac{2L_z}{\sqrt{\mu}} (1- p_\theta(t | c))$.

Assumption 7 (Weak incoherence of noise): Let $g_{core}(\theta) := E_{(c,t)\sim p_{core}} [\phi_\theta(c, t)]$, $g_{noise}(\theta) := E_{(c,t)\sim p_{noise}} [\phi_\theta(c, t)]$. There exists $\zeta_M \in [0, 1)$ such that for all $\theta$ $\langle g_{core}(\theta), g_{noise}(\theta) \rangle_{M^{-1}} \leq \zeta_M \|g_{core}(\theta)\|_{M^{-1}}^2$.

**Gradient Identities:**
Let $a := 1- \varepsilon, b := \varepsilon$ and $Z_{fil} := a(1- \alpha) + b \beta$.
$g(\theta; p_{train}) = a g_{core}(\theta) + b g_{noise}(\theta)$
(Strong MAR): $g(\theta; p_{fil}) = \frac{a(1- \alpha) g_{core}(\theta) + b\beta g_{noise}(\theta)}{Z_{fil}}$
(Weak bias): $g(\theta; p_{fil}) = \frac{a(1- \alpha) g_{sel}^{core}(\theta) + b\beta g_{sel}^{noise}(\theta)}{Z_{fil}}$
$\nabla_\theta L^\star(\theta) = - g_{core}(\theta)$

**Alignment Definitions:**
$A_M (\theta; \tilde{g}) := \langle \nablã_M L^\star(\theta), \tilde{g} \rangle_M$
$A_M (\theta; \tilde{g}(\cdot;D)) = g_{core}^\top M^{-1} g(\theta;D)$
$A^M_{train}(\theta) := a \|g_{core}\|_{M^{-1}}^2 + b \langle g_{core}, g_{noise} \rangle_{M^{-1}}$
(Strong MAR): $A^M_{fil}(\theta) := \frac{a(1-\alpha)}{Z_{fil}} \|g_{core}\|_{M^{-1}}^2 + \frac{b\beta}{Z_{fil}} \langle g_{core}, g_{noise} \rangle_{M^{-1}}$

**Theorem 1 (Filtering strictly improves M–alignment):** Under Assumptions 2–4 and 7,
$A^M_{fil}(\theta) - A^M_{train}(\theta) = \frac{ab (1- \alpha- \beta)}{Z_{fil}} (\|g_{core}(\theta)\|_{M^{-1}}^2 - \langle g_{core}(\theta), g_{noise}(\theta) \rangle_{M^{-1}})$
If $\alpha+ \beta < 1$ and $\zeta_M < 1$,
$A^M_{fil}(\theta) - A^M_{train}(\theta) \geq \frac{ab (1- \alpha- \beta) (1- \zeta_M)}{Z_{fil}} \|g_{core}(\theta)\|_{M^{-1}}^2 > 0$

**Theorem 2 (Robust alignment gain under bounded selection bias):**
$A^M_{fil}(\theta) - A^M_{train}(\theta) \geq \frac{ab (1- \alpha- \beta) (1- \zeta_M)}{Z_{fil}} \|g_{core}\|_{M^{-1}}^2 - \frac{a(1-\alpha)\rho_c + b\beta\rho_n}{Z_{fil}} \|g_{core}\|_{M^{-1}}^2$
Net gain remains positive whenever $ab(1- \alpha- \beta)(1- \zeta_M) > a(1- \alpha)\rho_c + b\beta\rho_n$.

**One–step Decrease:**
$L^\star(\theta^+) \leq L^\star(\theta) - \eta A_M (\theta; \tilde{g}) + \frac{L}{2} \eta^2 \|\tilde{g}\|_2^2$
$L^\star(\theta^+_{fil}) - L^\star(\theta^+_{train}) \leq - \eta (A^M_{fil}(\theta) - A^M_{train}(\theta)) + \frac{L}{2} \eta^2 (\|\tilde{g}(\cdot; p_{fil})\|_2^2 - \|\tilde{g}(\cdot; p_{train})\|_2^2)$

**KN Tokens Negligible Contribution:**
Let $S_{KN} \subseteq Supp(p_{train})$ such that $p_\theta(t | c) \geq 1- \delta$ for all $(c, t) \in S_{KN}$.
$\|E_{(c,t)\sim p_{train}} [\phi_\theta(c, t) I\{(c, t) \in S_{KN}\}]\|_{F_\lambda^{-1}} \leq \frac{2L_z}{\sqrt{\mu}} \delta p_{train}(S_{KN})$
$|A_{F_\lambda} (\theta; \tilde{g}(\cdot; p_{train})) - A_{F_\lambda} (\theta; \tilde{g}(\cdot; p_{train} \setminus S_{KN}))| \leq \frac{2L_z}{\sqrt{\mu}} \delta p_{train}(S_{KN}) \|g_{core}(\theta)\|_{F_\lambda^{-1}}$

### 8. Algorithm Pseudocode
The paper does not provide explicit algorithm pseudocode in a standard block format. The detailed pipeline logic is fully extracted and described in the "Overview of the Overall Technical Solution" and "Detailed Module Design" sections above, strictly following the original text's steps.