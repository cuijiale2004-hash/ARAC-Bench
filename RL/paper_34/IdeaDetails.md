### 1. Research Background and Existing Pain Points

**Research Background:**
Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training. A wide array of post-training strategies, ranging from SFT to RL, has been developed to enhance model performance. Particularly, RL-based fine-tuning has witnessed rapid advancements, encompassing the development of reward models from Outcome Reward Models (ORM) to Process Reward Models (PRM), alongside training algorithms like Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO). Emerging empirical evidence indicates that RL-based fine-tuning can enhance the capability of LLMs beyond what is achieved by SFT alone, improving performance across a range of downstream tasks, including writing, reasoning, and coding. Seeking to understand the role of different components within LLMs and the origins of their powerful capabilities, a growing body of research has focused on probing their internal structures. Initial studies revealed the working mechanisms of LLMs when solving mathematical problems by analyzing and statistically examining their internal weights. Subsequently, some research has analyzed patterns in LLM weights by training external neural probes, which are lightweight auxiliary models. Recently, researchers have investigated the internal residual pathways of LLMs from a graph-theoretic perspective. They have developed methods such as Automated Circuit Discovery (ACDC) and Edge Attribution Patching (EAP), which assign importance scores to edges or sub-modules and reveal internal functional circuits that determine the capabilities of LLMs. In the field of explainable RL, research can be generally categorized into pre-hoc techniques (creating inherently interpretable agents like neuro-symbolic systems) and post-hoc techniques (feature attribution, policy distillation, counterfactual methods). Research into the interpretability of LLMs has largely progressed along two complementary paradigms: mechanistic interpretability (reverse-engineering patterns by analyzing fundamental components using causal tracing) and representation interpretability (investigating what information is encoded in internal activation states via external probing models).

**Existing Pain Points:**
The underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored. Existing studies on RL-based post-training have predominantly focused on the external behavioral changes of LLMs, while the underlying internal mechanisms remain under-explored. Conversely, works that do investigate the internal mechanisms concentrate on given LLMs, but do not correlate the internal mechanisms to the RL-based post-training methodology with which the LLMs are commonly obtained. As a result, the two lines of research, external evaluation of RL effects and internal mechanistic analysis, have largely progressed in parallel. This gap is partly due to the primary goal of RL post-training, namely enhancing the ability of LLMs to solve complex reasoning tasks, which makes it nontrivial to directly transfer analytical strategies developed on toy problems to the study of RL-induced improvements in real-world problem-solving capabilities. Furthermore, explainable RL research mainly focuses on lightweight RL agents for conventional decision-making tasks, while it remains unexplored how RL works in the emerging post-training applications, where LLMs are trained as the agent. While interpretability studies of LLMs offer valuable perspectives, they predominantly focus on analyzing given LLMs without integrating the training methodology with which the LLMs are obtained into the investigation. In particular, it remains unclear how RL, the widely adopted technique in post-training, is able to broadly enhance the capabilities of diverse LLMs with distinct architectural and functional characteristics.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To address the gap between external evaluation of RL effects and internal mechanistic analysis, the study aims to construct a framework for systematically analyzing the mechanisms through which RL fine-tuning affects LLMs. The motivation is to bridge empirical performance gains with interpretable shifts in internal information pathways. By providing new insights into how RL post-training reshapes the internal circuitry of LLMs, the work seeks to offer guidance for the future development of both LLMs and post-training methodologies.

**Scientific Questions:**
1. What are the internal differences of LLMs before and after RL fine-tuning?
2. How does RL fine-tuning systematically alter the internal circuitry of LLMs?
3. Why is RL fine-tuning able to enhance the capability of various LLMs with distinct intrinsic characteristics?

### 3. Overall Core Idea and Design Philosophy

The overall core idea is to draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning. The design philosophy adopts an efficient Edge Attribution Patching (EAP) framework, leveraging the cross-entropy computed from partially truncated generations on mathematical problem-solving tasks to estimate the contribution weights of internal edges. Based on these estimated importance weights, the approach analyzes their distributions before and after RL fine-tuning to interpret changes in internal neuron activations and derive general conclusions regarding the structural effects of RL in the context of mathematical problem solving. The core idea is to provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches, offering a mechanistic perspective connecting sampling distributions and gradient coefficients to internal circuit changes.

### 4. Core Innovation Points

1. **Identification of Two Robust Effects of Online RL Post-training:** Across multiple model families and mathematical datasets, the study identifies two robust effects: (i) an overall increase in average activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions.
2. **Explanation for RL's Advantage in Mathematical Generalization:** The study demonstrates that RL reshapes information flow to be both more redundant and more flexible, which provides a mechanistic explanation for its advantage in mathematical generalization.
3. **Highlighting Methodological Distinctions Between Online RL and Preference-Based Approaches:** The study shows that models fine-tuned with Direct Preference Optimization (DPO) deviate from the trends observed in online RL, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training, thereby highlighting the fundamental differences between static preference optimization and dynamic online RL.
4. **Unified Graph-Theoretic View of LLM Internal Circuitry:** The study provides a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs from a graph-theoretic perspective, representing the LLM as a directed acyclic graph (DAG) where nodes correspond to individual sub-modules and edges encode the residual information pathways.
5. **Systematic Filtering and Truncation Procedure for Fair Analysis:** The study constructs a systematic filtering and truncation procedure on model-generated token sequences for fair and tractable edge attribution analysis, including question filtering based on correctness and length balance, and token truncation with self-entropy computation.

### 5. Overview of the Overall Technical Solution

The overall technical solution is based on the Edge Attribution Patching (EAP) framework, adopting a graph-theoretic view of LLMs via their residual pathways. The LLM is represented as a directed acyclic graph (DAG) where nodes correspond to individual sub-modules (attention blocks and FFN blocks) and edges encode the residual information pathways. To quantify edge importance efficiently, a gradient-based linearization method (IEAP) is used to approximate the loss perturbation caused by edge ablation. To ensure fair and tractable edge attribution analysis, a systematic filtering and truncation procedure is implemented on model-generated token sequences. This includes question filtering (selecting questions correctly answered by both models, filtering by length constraints and balance coefficient) and token truncation (defining a truncation length and computing self-entropy). Differences in LLM behavior before and after RL fine-tuning are quantified by analyzing the internal edge-weight matrices using three complementary metrics: Activation Intensity, Information Complexity, and Distribution Kurtosis. Experiments are conducted across multiple LLM pairs on diverse mathematical datasets. Finally, a theoretical interpretation based on the unified view of LLM post-training is provided to explain the observed phenomena by analyzing the fundamental differences in the support of the sampling distribution and the properties of the gradient coefficient across SFT, Online RL, and DPO.

### 6. Detailed Module Design

**Graph View of Transformer Residual Computation:**
Owing to the residual connections in Transformer layers, the input to any sub-module corresponds to the sum of all preceding sub-module outputs, including the original embedding input. The attention branch transformation is denoted as $O^{\text{attn}}_{\ell} = A_{\ell}(H^{(2\ell)})$ and the FFN transformation as $O^{\text{ffn}}_{\ell} = F_{\ell}(H^{(2\ell+1)})$. The attention block is treated as a single composite transformation. Each sub-module can be interpreted as a node in a directed graph. The set of nodes is defined as $V = \{A1, F1, A2, F2, . . . , AL, FL\}$, where $H0$ corresponds to the original embedding input. The directed edges, representing the flow of information from sub-module outputs to subsequent inputs, are formalized as $E = \{(H^{(0)}, H^{(j)}) | 1 \leq j \leq 2L\} \cup \{(O^{\text{attn}}_i, H^{(2\ell-1)}), (O^{\text{ffn}}_i, H^{(2\ell)}) | 1 \leq i \leq \ell \leq L\}$. The LLM is represented as a directed acyclic graph (DAG) $G = (V, E)$.

**Edge-Level Attribution:**
To quantify the importance of individual residual edges, the ACDC method evaluates the change in loss when a given edge is removed. Let $(O,H) \in E$ denote a directed edge. ACDC defines the edge importance by the loss perturbation: $I_{\text{ACDC}}(O,H) = L(y; f^{\setminus(O,H)}(x)) - L (y; f(x))$, where $f^{\setminus(O,H)}$ represents the model with edge $(O,H)$ ablated. This requires two forward passes per edge, rendering it computationally infeasible. By contrast, the EAP framework proposes a gradient-based linearization that estimates the same loss perturbation more efficiently. For a given edge $(O,H)$, consider the ablation $H \mapsto H - O$. A first-order Taylor expansion around $H$ yields: $\Delta L(O,H) \approx -\langle \nabla_H L(y; f(x)), O \rangle \equiv I_{\text{EAP}}(O,H)$. $I_{\text{EAP}}$ can be computed for all edges simultaneously with a single forward and backward pass under the zeroing perturbation, as both the forward activations $O$ and the backward gradients $\nabla_H L$ are available.

**Sample Selection and Token-Level Truncation:**
To ensure fair and tractable edge attribution analysis, a systematic filtering and truncation procedure is implemented:
1. **Question Filtering:** Select only questions that are correctly answered by both models, denoted as $Q$. Compute the mean token length $\bar{T}$. Define minimum and maximum allowable lengths $T_{\min} = \beta \bar{T}$, $T_{\max} = \gamma \bar{T}$, and retain only questions satisfying $T_{\min} \leq T^{q}_{\text{base}}$ and $T^{q}_{\text{RL}} \leq T_{\max}$. To control for comparable sequence lengths, require $\frac{|T^{q}_{\text{base}} - T^{q}_{\text{RL}}|}{(T^{q}_{\text{base}} + T^{q}_{\text{RL}})/2} < \delta$, where $\delta \in (0, 1)$ is a balance coefficient.
2. **Token Truncation and Self-Entropy Computation:** For the filtered set of questions, define a truncation length $T_{\text{cut}} = \alpha \bar{T}$, where $\alpha > 0$ is a scaling coefficient. Only the first $T_{\text{cut}}$ tokens are used. Compute the self-entropy (cross-entropy of the model with respect to its own output) as $L_{\text{trunc}}$.

**Metrics Definition:**
Differences in LLM behavior are quantified by analyzing the internal edge-weight matrices $W \in \mathbb{R}^{n \times n_o \times n_i}$:
1. **Activation Intensity (Act.Intens.):** Quantifies the average magnitude of all edge weights: $\text{Act.Intens.} = \frac{1}{n n_o n_i} \sum_{k=1}^n \sum_{o=1}^{n_o} \sum_{i=1}^{n_i} |W^{(k)}_{oi}|$.
2. **Information Complexity (Info.Complex.):** Computes a Shannon entropy over the absolute values of all edges: $\text{Info.Complex.} = -\sum_{b=1}^B p_b \log(p_b + \epsilon)$, where $p_b$ is the normalized probability of bin $b$ in a histogram of all $|W^{(k)}_{oi}|$ values.
3. **Distribution Kurtosis (Dist.Kurt.):** Computes the kurtosis of each sample’s edge-weight matrix and averages: $\text{Dist.Kurt.} = \frac{1}{n} \sum_{k=1}^n \left[ \frac{\frac{1}{n_o n_i} \sum_{o,i} (W^{(k)}_{oi} - \mu^{(k)})^4}{(\frac{1}{n_o n_i} \sum_{o,i} (W^{(k)}_{oi} - \mu^{(k)})^2)^2} - 3 \right]$.

**Theoretical Interpretation Module:**
Based on the unified post-training framework $\nabla_{\theta} J_A(\theta)$, the observed phenomena are interpreted by analyzing the sampling distribution $\mathcal{D}$ and gradient coefficient $G_{CA}$:
- **SFT:** $\mathcal{D}_{\text{SFT}} = \{(q, o) \sim P_{\text{data}}\}$, $G_{C}^{\text{SFT}} = 1$. Optimizes internal representations to minimize cross-entropy on a narrow manifold, resulting in activations concentrated on a small number of outlier edges (high Distribution Kurtosis) and limited engagement of redundant pathways (low Activation Intensity).
- **Online RL:** $\mathcal{D}_{\text{RL}} = \{(q, \{o_i\}_{i=1}^G) | q \sim P_{\text{data}}, o_i \sim \pi_{\theta}(\cdot|q)\}$. Introduces on-policy sampling, expanding the stochastic support set. Gradient coefficient varies dynamically, e.g., $G_{C}^{\text{GRPO}} = \hat{A}_{i,t}(q, o, t, \pi_{\text{rd}}) + \beta (\frac{\pi_{\text{ref}}(o_{i,t}|o_{i,<t})}{\pi_{\theta}(o_{i,t}|o_{i,<t})} - 1)$. Drives the model to activate latent circuits for harder problems, increasing Activation Intensity and decreasing Distribution Kurtosis.
- **DPO:** $\mathcal{D}_{\text{DPO}} = \{(q, o^+, o^-) \sim P_{\text{data}}\}$. Operates as an offline algorithm with a static distribution. Gradient coefficient $G_{C}^{\text{DPO}} = \sigma(\beta \log \frac{\pi_{\theta}(o^-|q)}{\pi_{\text{ref}}(o^-|q)} - \beta \log \frac{\pi_{\theta}(o^+|q)}{\pi_{\text{ref}}(o^+|q)})$. The soft margin mechanism relaxes strict token-matching constraints, reducing kurtosis, but lacks the on-policy exploration dynamics essential for enhancing average internal activation intensity and diversity.

### 7. All Mathematical Formulas and Symbol Definitions

**Transformer Layer Computation:**
$$ \text{Attention}(X^{\text{attn}}_\ell) = O^{\text{attn}}_\ell = \text{softmax}\left(\frac{(X^{\text{attn}}_\ell W^q_\ell) (X^{\text{attn}}_\ell W^k_\ell)^T}{\sqrt{d_k}}\right) (X^{\text{attn}}_\ell W^v_\ell) W^o_\ell $$
where $W^q_\ell, W^k_\ell \in \mathbb{R}^{d_{\text{model}} \times d_{\text{query}}}$, $W^v_\ell \in \mathbb{R}^{d_{\text{model}} \times d_{\text{attn}}}$, $W^o_\ell \in \mathbb{R}^{d_{\text{attn}} \times d_{\text{model}}}$ are the query, key, value and output projection matrices. $d_{\text{query}}$ is the dimensionality of the query and key vectors, and $d_{\text{attn}}$ represents the dimensionality of the value vectors.

$$ \text{FFN}(X^{\text{ffn}}_\ell) = O^{\text{ffn}}_\ell = \left(\text{Activation}(X^{\text{ffn}}_\ell W^{\text{gate}}_\ell) \odot (X^{\text{ffn}}_\ell W^{\text{up}}_\ell)\right) W^{\text{down}}_\ell $$
where $W^{\text{gate}}_\ell \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W^{\text{up}}_\ell \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, and $W^{\text{down}}_\ell \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ are learned weight matrices, $\odot$ denotes element-wise multiplication, and $d_{\text{ff}}$ is the expanded inner dimension of the FFN.

$$ \mathcal{L} = P(H^{(2L)}) = \text{LayerNorm}(H^{(2L)}) W^T_{\text{emb}} $$
where $W_{\text{emb}} \in \mathbb{R}^{V \times d_{\text{model}}}$ is the output embedding matrix and $V$ is the vocabulary size. The resulting tensor $\mathcal{L} \in \mathbb{R}^{B \times P \times V}$ contains the unnormalized logits.

**Unified View of LLM Post-Training:**
$$ \nabla_{\theta} J_A(\theta) = \mathbb{E}_{(q,o)\sim\mathcal{D}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} G_{CA}(q, o, t, \pi_{\text{rd}}, \pi_{\text{ref}}, \pi_{\theta}) \nabla_{\theta} \log \pi_{\theta}(o_t | q, o_{<t}) \right] $$
where $\pi_{\theta}$ denotes the current policy parameterized by $\theta$, $(q, o)$ represents a query–response pair, $\mathcal{D}$ specifies the sampling distribution, $\pi_{\text{rd}}$ denotes the reward model or evaluation rule, $\pi_{\text{ref}}$ is the reference policy, and $G_{CA}$ represents the token-level weighting factor derived from these signals in algorithm $A$.

**Graph View of Transformer Residual Computation:**
$$ H^{(2\ell)} = H^{(0)} + \sum_{i=1}^\ell O^{\text{attn}}_i + \sum_{j=1}^\ell O^{\text{ffn}}_j, \quad H^{(2\ell+1)} = H^{(0)} + \sum_{i=1}^{\ell+1} O^{\text{attn}}_i + \sum_{j=1}^\ell O^{\text{ffn}}_j $$
$$ V = \{A1, F1, A2, F2, . . . , AL, FL\} $$
$$ E = \{(H^{(0)}, H^{(j)}) | 1 \leq j \leq 2L\} \cup \{(O^{\text{attn}}_i, H^{(2\ell-1)}), (O^{\text{ffn}}_i, H^{(2\ell)}) | 1 \leq i \leq \ell \leq L\} $$

**Edge-Level Attribution:**
$$ I_{\text{ACDC}}(O,H) = L(y; f^{\setminus(O,H)}(x)) - L (y; f(x)) $$
where $f(x)$ is the model output under input $x$, $L(y; \cdot)$ denotes the supervised loss relative to target $y$, and $f^{\setminus(O,H)}$ represents the model with edge $(O,H)$ ablated.
$$ \Delta L(O,H) \approx -\langle \nabla_H L(y; f(x)), O \rangle \equiv I_{\text{EAP}}(O,H) $$
where $\nabla_H L(y; f(x)) \in \mathbb{R}^{B \times P \times d_{\text{model}}}$ is the loss gradient with respect to the hidden state $H$, and $\langle \cdot, \cdot \rangle$ denotes the Euclidean inner product.

**Sample Selection and Token-Level Truncation:**
$$ \bar{T} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{T^{q}_{\text{base}} + T^{q}_{\text{RL}}}{2} $$
where $T_{\min} = \beta \bar{T}$, $T_{\max} = \gamma \bar{T}$.
$$ \frac{|T^{q}_{\text{base}} - T^{q}_{\text{RL}}|}{(T^{q}_{\text{base}} + T^{q}_{\text{RL}})/2} < \delta $$
where $\delta \in (0, 1)$ is a balance coefficient. $T_{\text{cut}} = \alpha \bar{T}$, where $\alpha > 0$ is a scaling coefficient.
$$ L_{\text{trunc}} = - \frac{1}{T_{\text{cut}}} \sum_{t=1}^{T_{\text{cut}}} \log \frac{\exp(\mathcal{L}_t[s_t])}{\sum_{v=1}^V \exp(\mathcal{L}_t[v])} $$
where $\mathcal{L}_t \in \mathbb{R}^V$ denotes the model’s logit output at token position $t$, and $s_{1:T_{\text{cut}}}$ is the sequence of generated tokens truncated to $T_{\text{cut}}$.

**Metrics:**
$$ \text{Act.Intens.} = \frac{1}{n n_o n_i} \sum_{k=1}^n \sum_{o=1}^{n_o} \sum_{i=1}^{n_i} |W^{(k)}_{oi}| $$
where $W^{(k)} \in \mathbb{R}^{n_o \times n_i}$ denotes the edge-weight matrix for sample $k$, $n$ is the number of samples, $n_o$ denotes the number of output nodes, and $n_i$ denotes the number of input nodes.
$$ \text{Info.Complex.} = -\sum_{b=1}^B p_b \log(p_b + \epsilon) $$
where $p_b$ denotes the normalized probability of bin $b$ in a histogram of all $|W^{(k)}_{oi}|$ values, with $B$ bins and a small constant $\epsilon$.
$$ \text{Dist.Kurt.} = \frac{1}{n} \sum_{k=1}^n \left[ \frac{\frac{1}{n_o n_i} \sum_{o,i} (W^{(k)}_{oi} - \mu^{(k)})^4}{(\frac{1}{n_o n_i} \sum_{o,i} (W^{(k)}_{oi} - \mu^{(k)})^2)^2} - 3 \right] $$
where $\mu^{(k)}$ is the mean edge weight of sample $k$.

**Theoretical Interpretation Specific Coefficients:**
$$ G_{C}^{\text{GRPO}} = \hat{A}_{i,t}(q, o, t, \pi_{\text{rd}}) + \beta \left(\frac{\pi_{\text{ref}}(o_{i,t}|o_{i,<t})}{\pi_{\theta}(o_{i,t}|o_{i,<t})} - 1\right) $$
$$ G_{C}^{\text{DPO}} = \sigma\left(\beta \log \frac{\pi_{\theta}(o^-|q)}{\pi_{\text{ref}}(o^-|q)} - \beta \log \frac{\pi_{\theta}(o^+|q)}{\pi_{\text{ref}}(o^+|q)}\right) $$

### 8. Algorithm Pseudocode

No explicit algorithm pseudocode is provided in the original text.