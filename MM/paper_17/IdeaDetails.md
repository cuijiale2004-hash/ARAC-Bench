### 1. Research Background and Existing Pain Points

**Research Background:**
Large Language Models (LLMs) have demonstrated outstanding capabilities across a diverse range of tasks and domains. Their multimodal expansions, Multimodal Large Language Models (MLLMs), further unlock capabilities spanning images, videos, and other modalities beyond text. MLLMs process not only text but also images, videos, and other modalities (such as molecules), achieving strong performance on a broad range of tasks, from foundational ones such as classification and captioning, to domain-specific, high-stakes applications such as medical image question answering and pharmacological property prediction. A central factor in unlocking their full potential lies in the design of prompts, which directly influence model performance.

**Existing Pain Points:**
1. **Labor-intensive Manual Prompt Crafting:** Crafting high-quality prompts is often a labor-intensive and iterative process that requires substantial human intervention.
2. **Text-only Limitation of Automatic Prompt Optimization (APO):** Despite the advances of MLLMs and their wide-ranging applications, existing prompt optimization methods remain restricted to the textual modality and overlook the richer expressive capacity afforded by multimodal inputs that text alone cannot capture. For example, describing the distinct characteristics of a specific bird may require long and potentially ambiguous text, while a single image can convey the same information far more directly. By limiting optimization to text, existing methods are prone to generating less effective, suboptimal prompts that fail to fully exploit the multimodal space that MLLMs are inherently capable of leveraging.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Motivated by the limitation of text-only prompt optimization, the paper aims to expand the prompt optimization space beyond text to incorporate multiple modalities. The core motivation is to establish multimodal prompt optimization as a crucial step to realizing the full potential of MLLMs, moving beyond text-only prompting paradigms. The objective is to automate the discovery of effective prompts that leverage both textual and non-textual inputs, thereby maximizing the performance of MLLMs across diverse modal domains.

**Scientific Questions / Fundamental Challenges:**
1. **Cross-modal Consistency in Exploration:** Exploring the larger, combinatorial space of multimodal prompts requires a prompt update strategy that can efficiently navigate candidate prompts while maintaining cross-modal consistency. Multimodal prompts must maintain cross-modal consistency: textual and non-textual components should provide complementary, not conflicting signals; however, expanding to the combinatorial space greatly increases the risk of semantic misalignment. A naive approach that independently updates textual and non-textual components risks producing misaligned prompts, where one modality contradicts the other.
2. **Sparse Optimal Prompts in Candidate Selection:** The enlarged search space amplifies the difficulty of candidate selection. High-quality prompts become sparse, and low-quality prompts dominate, making it harder to efficiently identify promising candidates given the need to account for both the effectiveness within each modality and the alignment across modalities. Existing selection approaches suffer from an inefficient cold-start problem: newly generated prompts are treated as independent arms with no prior information, leading to unproductive evaluations in the early rounds.

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper first defines the novel problem of multimodal prompt optimization, which expands the prompt optimization space beyond text to incorporate multiple modalities, formally defining a multimodal prompt as a pair of a discrete textual and non-textual prompts. To tackle the challenges of this expanded space, the paper proposes the Multimodal Prompt Optimizer (MPO), a unified framework for optimizing prompts across both the textual and non-textual modalities.

**Design Philosophy:**
MPO is designed around two key components to address the two fundamental challenges:
1. **Alignment-Preserving Exploration:** To ensure cross-modal consistency during exploration, MPO jointly updates the textual prompt and its associated non-textual counterparts. This is achieved through cohesive backpropagation (generating a single semantic gradient/feedback for both modalities) and a joint multimodal update mechanism (using the feedback to derive modality-specific conditions for non-textual revisions). The updates are further diversified through complementary operations (generation, editing, and mixing) to ensure broad and expressive exploration.
2. **Prior-Inheritance-Based Selection:** To efficiently identify high-performing candidates in the sparse multimodal space, MPO leverages a prior-inherited Bayesian-UCB strategy. Instead of treating candidates independently (cold start), it utilizes the performance score of parent prompts as informative priors to warm-start the evaluation process, biasing the selection toward more promising regions of the multimodal space.

### 4. Core Innovation Points

1. **Novel Problem Definition of Multimodal Prompt Optimization:** The paper introduces the new problem of multimodal prompt optimization, which expands the prior definition of prompt optimization from the textual space to the multimodal space defined by the pairs of textual and non-textual prompts, formally defining a multimodal prompt as \( p = (t,m) \in T \times M \).
2. **Alignment-Preserving Exploration with Cohesive Backpropagation and Joint Update:** MPO jointly updates the textual and non-textual prompts to ensure they evolve coherently. It identifies a failure set and generates a unified feedback (single supervisory signal) that guides both modalities simultaneously, mitigating the risk of overfitting updates to one modality. It further derives modality-specific conditions from this unified feedback to direct non-textual revisions, guaranteeing updates remain consistent with the revised textual prompt.
3. **Diverse Exploration Operators (Generation, Edit, Mix):** To effectively explore the multimodal prompt space, MPO designs three systematic operators. The Generation operator explores entirely new non-textual prompts from scratch. The Edit operator performs fine-grained refinements of non-textual prompts while retaining useful structures. The Mix operator blends the complementary strengths of multiple multimodal prompts. These operators jointly enable a more comprehensive exploration of the multimodal prompt space.
4. **Prior-Inherited Bayesian UCB Selection Strategy:** To address the cold-start problem in candidate selection over the enlarged multimodal search space, MPO proposes a prior-inherited Bayesian UCB strategy. It models the expected score of each prompt as a Beta distribution and initializes the prior of a child prompt proportionally to the posterior mean performance of its parent prompt. This provides pseudo-observations to warm-start the evaluation, rapidly eliminating unpromising candidates and accelerating the discovery of high-quality prompts.

### 5. Overview of the Overall Technical Solution

The overall technical solution is the Multimodal Prompt Optimizer (MPO), an iterative beam search-based framework. At each iteration, MPO selects the top-b best-performing multimodal prompts. It then applies alignment-preserving exploration and prior-inherited Bayesian UCB selection to generate and select the next generation of prompts.

1. **Alignment-Preserving Exploration:** For a given parent prompt, MPO first identifies a failure set \( F \) of incorrect predictions. It then uses the MLLM to analyze these failures and generate a unified feedback \( \nabla p = (\nabla t, \nabla m) \) (Cohesive Backpropagation). Using this feedback, the MLLM produces an updated textual prompt \( t' \) and a modality-specific condition \( c \) (Joint Multimodal Update). The condition \( c \) is then passed to a modality-specific generator \( g \) along with one of the three operators (Generation, Edit, Mix) to yield the updated non-textual prompt \( m' \). This generates a pool of child prompts.
2. **Prior-Inherited Bayesian UCB Selection:** To select the top-b prompts from the candidates for the next iteration, MPO models the expected score of each prompt \( p_i \) as a Beta distribution \( Beta(\alpha_i, \beta_i) \). For a child prompt, its prior is initialized by inheriting the posterior mean of its parent prompt. It then proceeds iteratively: at each round, the prompt with the highest UCB score is selected, evaluated on a small batch of data, and its posterior parameters are updated. Once the budget is exhausted, the candidate with the highest expected score is selected as the new parent.

### 6. Detailed Module Design

**Module 1: Multimodal Prompt Optimization Problem Definition**
*   **MLLM Definition:** An MLLM can be represented as a parametric function MLLM : (T ∪ M)∗ → T, where T denotes the textual input space, M denotes the non-textual input space, and ∗ denotes the Kleene Star. Given a multimodal query q and a prompt p, the model generates a textual output y = MLLM(p, q).
*   **Multimodal Prompt Definition:** A multimodal prompt is defined as a pair p = (t,m) ∈ T × M, where t is the textual prompt and m is the non-textual prompt.
*   **Objective:** Given a task dataset D consisting of query–answer pairs (q,a), the objective of multimodal prompt optimization is to discover the optimal prompt (t∗,m∗) that maximizes performance:
    (t∗,m∗) = argmax(t,m)∈T ×M E(q,a)∼D [f(MLLM(t,m, q),a)], where f is a function for a task-specific evaluation metric.

**Module 2: Alignment-Preserving Exploration of Multimodal Prompt Space**
*   **Cohesive Backpropagation:** Begin by identifying a failure set F = {(q,a,y) | y ̸= a} for a multimodal prompt p = (t,m). Instead of treating errors separately for text and non-textual inputs, generate a unified feedback ∇p = (∇t,∇m) = MLLM(t,m;F), which encodes cross-modal weaknesses in textual form. This obtains the single supervisory signal that guides both modalities simultaneously.
*   **Joint Multimodal Update:** Using the feedback, MPO jointly refines the textual prompt while deriving modality-specific conditions (in textual form) that direct non-textual revisions. Specifically, the MLLM produces an updated textual prompt t′ and a modality-specific condition c describing how the non-textual prompt should adapt: (t′, c) = MLLM(t,m;F ,∇p). The condition c is then passed to modality-specific generators g (such as text-to-image or text-to-molecule modules), which yield updated non-textual prompts m′ = g(c). This guarantees that updates to m remain consistent with the revised textual prompt t′.
*   **Beam Search Process:** At each iteration, select top-b best-performing multimodal prompts {pi = (ti,mi)}bi=1, and then apply the cohesive backpropagation and the joint multimodal update to generate new b2 multimodal prompts {p′j = (t′j ,m′j)}b2j=1 from the top-b multimodal prompts p.
*   **Exploration Operators:**
    *   **Generation operator:** Explores entirely new non-textual prompts, e.g., novel spatial arrangements or unique substructures. Conditioned only on the generation signal cgen, it creates a prompt from scratch without referencing prior candidates: m′ = g(cgen,∅), where (cgen, t′) = MLLM(t,m;∇p,F). It explores unexplored regions and avoids local optima.
    *   **Edit operator:** Performs fine-grained refinements of non-textual prompts while retaining useful structures. Given the edit condition cedit, the update is performed by conditioning on the prior non-textual prompt: m′ = g(cedit, {m}), where (cedit, t′) = MLLM(t,m;∇p,F).
    *   **Mix operator:** Blends the complementary strengths of multiple multimodal prompts. It leverages feedback from multiple prompts to generate a mixing condition cmix, which is used by the generator to combine non-textual prompts: m′ = g(cmix, {mi}Ki=1), where (cmix, t′) = MLLM({ti,mi;∇pi,Fi}Ki=1).
    *   **Operator Selection:** For the generation and edit operators, randomly select one parent prompt from top-b to generate a child prompt (p → p′), while K parent prompts are selected for the mix operator ({p1, ...,pK} → p′).

**Module 3: Effective Prompt Selection by Prior-Inherited Bayesian UCB**
*   **Parent-Child Correlation Hypothesis:** The performance of a parent prompt is positively correlated with that of its children. (Empirically supported by Pearson’s r = 0.88).
*   **Prior-Inherited Bayesian UCB:** Model the expected score of each multimodal prompt pi as a Beta distribution, Beta(αi, βi). For a child prompt pi originated from a parent prompt ppar(i), initialize its prior proportionally to the posterior mean performance of the parent µ̂par(i), scaled by a prior strength hyperparameter S > 0.
*   **Theoretical Guarantee (Proposition 3.1):** Fewer Pulls via Prior-Inherited Bayesian UCB. With the prior, and if the prior is more informative than uniform (Ei[d(µi, µ̂par(i))− d(µi, 1/2)] ≤ 0), the best-arm identification cost of Bayesian UCB is nonincreasing, where d(p, q) is the Bernoulli KL divergence.

### 7. All Mathematical Formulas and Symbol Definitions

*   **MLLM Function:** MLLM : (T ∪ M)∗ → T
*   **Model Output:** y = MLLM(p, q)
*   **Multimodal Prompt Definition:** p = (t,m) ∈ T × M
*   **Optimization Objective:** (t∗,m∗) = argmax(t,m)∈T ×M E(q,a)∼D [f(MLLM(t,m, q),a)]
*   **Failure Set:** F = {(q,a,y) | y ̸= a}
*   **Unified Feedback (Cohesive Backpropagation):** ∇p = (∇t,∇m) = MLLM(t,m;F)
*   **Joint Multimodal Update:** (t′, c) = MLLM(t,m;F ,∇p)
*   **Generator Output:** m′ = g(c)
*   **Generation Operator:** m′ = g(cgen,∅), where (cgen, t′) = MLLM(t,m;∇p,F)
*   **Edit Operator:** m′ = g(cedit, {m}), where (cedit, t′) = MLLM(t,m;∇p,F)
*   **Mix Operator:** m′ = g(cmix, {mi}Ki=1), where (cmix, t′) = MLLM({ti,mi;∇pi,Fi}Ki=1)
*   **Prior Initialization (Prior-Inherited Bayesian UCB):** αi = µ̂par(i) · S + 1, βi = (1− µ̂par(i)) · S + 1
*   **Parent Posterior Mean:** µ̂par(i) = αpar(i)/(αpar(i) + βpar(i))
*   **Theoretical Condition:** Ei[d(µi, µ̂par(i))− d(µi, 1/2)] ≤ 0 (where d(p, q) is Bernoulli KL divergence)

### 8. Algorithm Pseudocode

**Algorithm 1 MPO: Multimodal Prompt Optimizer**
Require: Initial prompt (t0,∅), Number of iterations T , Beam size b
Train dataset Dtr, Metric function f
1: p← (t0,∅), P ← {p}, C ← ∅, µ̂← E(q,a)∼Dtr [f(MLLM(t0,∅, q),a)]
2: for i = 1..b2 do
3: Fp ← {(q,a,y) | (q,a)∼Dtr, y = MLLM(p, q), y ̸= a}
4: ∇p ← MLLM.Feedback(t0,∅;Fp)
5: (t′, cgen)← MLLM.Generation(t0,∅;∇p,Fp); m′ ← g(cgen,∅)
6: C ← C ∪ {(t′,m′)}
7: end for
8: P ← BayesianUCBSelect(P, C, b) ▷ Select b prompts for next step
9: for iter = 1..T do
10: C ← ∅
11: for all p = (t,m) ∈ P do
12: for i = 1..b do
13: Fp ← {(q,a,y) | (q,a)∼Dtr, y = MLLM(p, q), y ̸= a}
14: ∇p ← MLLM.Feedback(t,m;Fp) ▷ Cohesive backpropagation
15: op← RandomSample({generation, edit,mix}) ▷ Joint multimodal update
16: if op = generation then
17: (t′, cgen)← MLLM.Generation(t,m;∇p,Fp); m′ ← g(cgen,∅)
18: else if op = edit then
19: (t′, cedit)← MLLM.Edit(t,m;∇p,Fp); m′ ← g(cedit, {m})
20: else if op = mix then
21: p̃← RandomSample(P \ {p})
22: (t′, cmix)← MLLM.Mix((t,m;∇p,Fp), (t̃, m̃;∇p̃,Fp̃)); m′ ← g(cmix, {m, m̃})
23: end if
24: C ← C ∪ {(t′,m′)}
25: end for
26: end for
27: P ← BayesianUCBSelect(P, C, b) ▷ Select b prompts for next step
28: end for
29: return p∗ ≡ (t∗,m∗) where p∗ = argmaxp∈P µ̂p, µ̂p = αp/(αp+βp)

**Algorithm 2 Prior-Inherited Bayesian UCB Selection**
Require: Parent prompts P , A set of k child prompts C = {pi}ki=1, Beam size b
Parent’s performance {µ̂par(i)}ki=1, Train dataset Dtr, Metric function f , Batch size B
Total evaluation budget N , Prior strength S, Exploration parameter c
1: Initialize Beta priors for each child prompt pi ∈ P:
2: for i = 1, . . . , k do
3: αi ← µ̂par(i) · S + 1 , βi ← (1− µ̂par(i)) · S + 1 ▷ Inherit prior from parent
4: end for
5: for t = 1, 2, . . . , (N/B) do
6: qt ← 1− 1/t(logN)c
7: j ← argmaxi∈{1,..,k} BetaQuantile(qt;αi, βi) ▷ Choose prompt with highest UCB
8: Dmini ← Sample(Dtr, B)
9: st ← E(q,a)∼Dmini [f(MLLM(t,m, q),a)] ▷ Evaluate on small data batch
10: αj ← αj + st ·B , βj ← βj + (1− st) ·B ▷ Update posterior
11: end for
12: Return top-b prompts from P ∪ C sorted by posterior mean µ̂i = αi/(αi+βi)