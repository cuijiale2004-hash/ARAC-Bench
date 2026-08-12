1. Research Background and Existing Pain Points
Semantic Textual Similarity (STS) is a core research area in Computational Linguistics with extensive applications across topic modeling, dialogue systems, text summarization, and agent memory. However, traditional STS tasks exhibit inherent ambiguity because similarity definitions are often susceptible to subjective bias. To address this limitation, the Conditional Semantic Textual Similarity (C-STS) task was developed. By incorporating an explicit natural language condition, C-STS enables more precise and objective similarity judgments, yet simultaneously imposes higher demands on a model’s reasoning capabilities. Research on this nascent task has yielded three primary paradigms: Bi-encoder, Tri-encoder, and Cross-encoder. The Cross-encoder architecture, which processes the text pair and the guiding condition simultaneously, is inherently compatible with modern generative pre-trained models. However, existing methods are largely confined to discriminative models, failing to fully leverage recent breakthroughs in the NLP community involving Large Language Models (LLMs) and Reinforcement Learning (RL). The integration of C-STS with LLMs remains in its early stages, limited to direct inference via few-shot prompting or utilizing them as feature extractors for generating text embeddings, leaving a significant research gap. Furthermore, naively applying listwise Reinforcement Learning fails to produce meaningful improvements. As indicated by preliminary experiments, directly applying a listwise ranking reward across an entire batch of completions does not show any advantages compared to the few-shot method. This approach suffers from two fundamental problems: First, the ranking objective is too complex for a model that has not yet grasped the task’s fundamental scoring rules, often leading to training collapse. Second, a single reward computed across the entire batch is too coarse to provide precise credit assignment, as a few poor completions can unfairly penalize other good ones.

2. Core Research Motivation and Scientific Questions
The core research motivation stems from the natural fit between incorporating RL into an LLM-based cross-encoder paradigm and the C-STS task, reflected in two key aspects. First, C-STS requires sophisticated, scenario-based reasoning that demands strong abstraction and inference to transcend surface-level semantics. RL, through its explicit reward signals, can more effectively guide the reasoning process of LLMs. Second, from an optimization standpoint, RL aligns closely with the task’s evaluation criteria. The Spearman correlation coefficient, a core evaluation metric of C-STS, is a non-differentiable measure of ranking quality. Traditional SFT methods can only indirectly and approximately optimize this objective through loss functions like Mean Squared Error. In contrast, RL allows for the direct optimization of ranking-based reward functions designed to correlate strongly with the final Spearman metric. The scientific question addressed is: How to successfully apply Reinforcement Learning to the C-STS task to overcome the optimization difficulties of direct rank-based learning and the coarse-grained credit assignment problem inherent in naive listwise ranking objectives?

3. Overall Core Idea and Design Philosophy
The overall core idea is PoLi-RL, a Point-to-List Reinforcement Learning framework. The design philosophy centers on a two-stage curriculum to manage the complexity of the learning task and a novel Parallel Slice Ranking Reward mechanism to resolve coarse-grained reward signals. In the first stage, the framework uses a simple pointwise reward to ground the model in the basic scoring rules of the task. Building on this foundation, the second stage introduces a richer, hybrid reward signal that combines a robust pointwise anchor with more nuanced pairwise and listwise ranking rewards, enabling the model to discern subtle semantic differences. Crucially, to resolve the problem of coarse-grained reward signals that arises from ranking all completions in a batch together, the framework introduces an innovative Parallel Slice Ranking Reward (PSRR) mechanism that employs a two-level decomposition to provide a precise, differentiated learning signal for each individual completion, enabling granular credit assignment and effective optimization.

4. Core Innovation Points
- First end-to-end LLM-based cross-encoder for C-STS with RL: This is the first work to propose an end-to-end, LLM-based cross-encoder for the C-STS task, and also represents the first application of reinforcement learning in this domain.
- Two-stage training curriculum: Designed and implemented PoLi-RL, a novel two-stage training curriculum that overcomes the optimization difficulties of direct rank-based learning by progressing from a basic pointwise objective to a comprehensive hybrid reward.
- Parallel Slice Ranking Reward (PSRR) mechanism: Proposed the PSRR mechanism, which delivers precise and differentiated learning signals by computing ranking rewards within independent ‘parallel slices.’ This mechanism offers a generalizable strategy for other ranking tasks involving multiple candidate generations.
- Superior performance surpassing proprietary models: On the official C-STS benchmark, PoLi-RL achieves a Spearman’s correlation coefficient of 48.18, establishing a new SOTA for the cross-encoder architecture and surpassing strong proprietary models, including GPT-4o and Deepseek-R1, while also maintaining SOTA status on the re-annotated C-STS dataset.

5. Overview of the Overall Technical Solution
The technical solution formulates the C-STS task within an end-to-end, LLM-based cross-encoder paradigm. Each C-STS data sample is defined as a pair (x, y), where the input x = (t1, t2, c) consists of two text segments and a natural language condition, and the label y ∈ [1, 5] is the human-annotated similarity judgment. The task trains a scoring model πθ, parameterized by θ, which takes a unified prompt p = [I, E , x] to generate an output sequence o = πθ(p), from which the final predicted score ỹ = Parse(o) is parsed. This task is mapped onto the mathematical framework of Reinforcement Learning as a Markov Decision Process (MDP), defined by a tuple M = (S,A, T ,R, γ), using Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) for optimization. The overall pipeline employs a two-stage progressive reward curriculum. Stage I (Foundational Skill Acquisition) uses a weighted sum of Pointwise Accuracy Reward, Binary Judgment Reward, and Format Reward. Stage II (Fine-Grained Semantic Distinction) introduces a hybrid reward signal incorporating pairwise and listwise ranking objectives enabled by the PSRR mechanism. The PSRR restructures the generated outputs into G “parallel slices” for granular reward computation. Within each slice, pairwise and listwise ranking rewards are computed vertically to provide precise, differentiated learning signals.

6. Detailed Module Design
- MDP Formulation Module: The agent is the LLM policy πθ. The generation process is a sequential decision-making process. A state st ∈ S at timestep t is the sequence of tokens generated so far, conditioned on the initial prompt, st = (p, o<t). An action at ∈ A is the selection of the next token ot from the model’s vocabulary, governed by the policy πθ(at|st). The transition function T is deterministic, where the next state st+1 is formed by appending the selected token at to st. A terminal reward RT = R(x, y, o) is given only after the entire sequence o has been generated. The discount factor γ is set to 1.
- DAPO Optimization Module: For each sample x, the policy model generates a set of G completions {oi}Gi=1. A scalar reward ri = R(x, y, oi) is computed for each completion. The advantage Âi for each completion is calculated by normalizing its reward against the statistics of the group rewards via Z-score normalization. Since an on-policy training approach is adopted, completions are sampled directly from the current policy πθ being optimized.
- Stage I: Foundational Skill Acquisition Module: The goal is to ground the model in the fundamental scoring rules of the C-STS task. The LLM policy generates G completions, from which a set of predicted scores {ỹj} is parsed. The total reward for Stage I is a weighted sum of three components: Pointwise Accuracy Reward, Binary Judgment Reward, and Format Reward. The Pointwise Accuracy Reward measures the normalized distance between the predicted score and the ground-truth score. The Binary Judgment Reward mitigates the model’s tendency to converge to safe, intermediate scores by encouraging it to first master the basic binary classification (scores ≥ 3 indicate similarity, while scores ≤ 2 indicate dissimilarity). The Format Reward ensures the output adheres to the required structure (a binary judgment followed by the numerical score in parentheses).
- Stage II: Fine-Grained Semantic Distinction Module: Building on the foundational scoring abilities from Stage I, Stage II refines the model’s capacity for nuanced semantic distinctions through a hybrid reward signal incorporating pairwise and listwise ranking objectives, both enabled by the PSRR mechanism.
- Parallel Slice Ranking Reward (PSRR) Mechanism Module: For a batch of N samples, the policy generates G completions {oi,1, . . . , oi,G} for each sample. From each completion oi,j, the predicted score ỹi,j is parsed. Instead of treating the completions as a single flat list, these N × G predicted scores are organized into G “parallel slices”. Each slice, denoted as Y j pred, is defined as the collection of the j-th predicted score from all N samples in the batch: Y j pred = {ỹ1,j , ỹ2,j , . . . , ỹN,j}, where j ∈ {1, . . . , G}. This slicing architecture forms the foundation of the ranking rewards, ensuring each completion receives a specific learning signal based on its relative performance within its independent slice. To make a sufficiently large batch size N computationally feasible with limited GPU memory, a gradient accumulation strategy is employed: first generate the full set of N ×G completions and organize them into parallel slices to compute rewards and advantages globally; subsequently, process smaller sub-batches sequentially to compute losses and accumulate gradients over multiple backward passes before executing a single optimizer step.
- Pairwise Ranking Reward Module: Computed within each parallel slice Y j pred, this reward leverages the paired structure of the C-STS dataset to provide a local ranking signal. It is applied only to adjacent input samples that form a valid pair. The predicted difference is defined as ∆pred = ỹi,j − ỹi+1,j and the true difference as ∆true = yi − yi+1. To accommodate cases where paired samples share identical ground-truth scores (∆true = 0), they are distinguished from strictly ordered pairs.
- Listwise Ranking Reward Module: While the pairwise reward focuses on local comparisons, the listwise reward provides a more global ranking perspective within each slice. It is calculated as the normalized difference between a completion’s predicted rank and its ground-truth rank.

7. All Mathematical Formulas and Symbol Definitions
- MDP Definition: M = (S,A, T ,R, γ)
  - S: State space, st = (p, o<t)
  - A: Action space, selection of next token ot
  - T: Transition function, deterministic
  - R: Reward function, RT = R(x, y, o)
  - γ: Discount factor, set to 1
- Optimization Objective: θ∗ = argmaxθ E(x,y)∼D,o∼πθ(p)[R(x, y, o)]
- Advantage Calculation: Âi = (ri − mean({ri}Gi=1)) / (std({ri}Gi=1) + ϵ)
  - ri: Scalar reward for completion i
  - Âi: Advantage for completion i
  - G: Number of completions per sample
  - ϵ: Small constant for numerical stability
- DAPO Objective Function: JDAPO(θ) = E(x,y)∼D,{oi}Gi=1∼πθ(·|p) [ (1 / ∑Gi=1 |oi|) ∑Gi=1 ∑|oi|t=1 ( πθ(oi,t|p, oi,<t) / [πθ(oi,t|p, oi,<t)]nograd Âi ) ]
  - |oi|: Length of completion i
  - [·]nograd: Stops gradient flow through the denominator
- Stage I Total Reward: RS1 = λ1Rpointwise + λ2Rbinary + λ3Rformat
  - λ1, λ2, λ3: Reward weights for Stage I
- Pointwise Accuracy Reward: Rpointwise = 1− |ỹj − yj| / (max(Y )−min(Y ))
  - ỹj: Predicted score
  - yj: Ground-truth score
  - max(Y ) = 5, min(Y ) = 1: Bounds of the label space
- Binary Judgment Reward: Rbinary = { 1 if (ỹj ≥ 3 ∧ yj ≥ 3) ∨ (ỹj < 3 ∧ yj < 3); 0 otherwise }
- Parallel Slice Definition: Y j pred = {ỹ1,j , ỹ2,j , . . . , ỹN,j}, where j ∈ {1, . . . , G}
  - N: Number of samples in a batch
- Pairwise Ranking Reward: Rpairwise i,j = { 1− |∆pred|/Dmax if ∆true = 0; Rbase + (1−Rbase) · (1− |∆pred−∆true|/Dmax) if sgn(∆pred) = sgn(∆true); 0 otherwise }
  - ∆pred = ỹi,j − ỹi+1,j: Predicted difference
  - ∆true = yi − yi+1: True difference
  - Dmax: Maximum possible score difference (4 for the 1-5 scale)
  - Rbase: Base reward for correct ordinality (default 0.5)
- Listwise Ranking Reward: Rlistwise i,j = 1− |Rank(ỹi,j , Y j pred)− Rank(yi, Ytrue)| / (N − 1)
  - Rank(v, S): Returns the rank of element v within the set S (from 1 to N)
  - Ytrue = {y1, . . . , yN}: Set of ground-truth labels for the current batch
- Stage II Total Reward: RS2 = µ1Rpointwise + µ2Rpairwise + µ3Rlistwise
  - µ1, µ2, µ3: Reward weights for Stage II

8. Algorithm Pseudocode
The paper does not provide explicit algorithm pseudocode. Based on the methodological description and logical sequence in the original text, the training algorithm flow is strictly structured as follows:

Input: Training dataset D, Policy model πθ, Number of completions G, Batch size N, Reward weights {λ1, λ2, λ3} and {µ1, µ2, µ3}, Base reward Rbase.

Stage I: Foundational Skill Acquisition
1: While not converged do:
2:   Sample a batch of N samples {(xi, yi)} from distribution D.
3:   For each sample xi, generate G completions {oi,1, ..., oi,G} using policy πθ.
4:   Parse predicted scores {ỹi,j} from completions.
5:   Compute Rpointwise for each completion using 1− |ỹj − yj| / (max(Y )−min(Y )).
6:   Compute Rbinary for each completion using the rule: 1 if (ỹj ≥ 3 ∧ yj ≥ 3) ∨ (ỹj < 3 ∧ yj < 3), else 0.
7:   Compute Rformat for each completion based on output structure adherence.
8:   Compute Stage I reward: RS1 = λ1Rpointwise + λ2Rbinary + λ3Rformat.
9:   Compute advantage Âi for each completion via Z-score normalization of rewards.
10:  Update policy πθ using DAPO objective JDAPO(θ).

Stage II: Fine-Grained Semantic Distinction
11: While not converged do:
12:  Sample a batch of N samples {(xi, yi)} from distribution D. (Ensure paired adjacent samples for pairwise reward)
13:  Generate N × G completions and parse predicted scores {ỹi,j}.
14:  Organize scores into G parallel slices: Y j pred = {ỹ1,j , ỹ2,j , . . . , ỹN,j} for j ∈ {1, . . . , G}.
15:  For each parallel slice j:
16:    Compute Rpointwise for each completion.
17:    Compute Rpairwise i,j for adjacent pairs within the slice:
        If ∆true = 0: 1− |∆pred|/Dmax
        Else if sgn(∆pred) = sgn(∆true): Rbase + (1−Rbase) · (1− |∆pred−∆true|/Dmax)
        Else: 0
18:    Compute Rlistwise i,j for each completion within the slice:
        1− |Rank(ỹi,j , Y j pred)− Rank(yi, Ytrue)| / (N − 1)
19:  Compute Stage II reward: RS2 = µ1Rpointwise + µ2Rpairwise + µ3Rlistwise.
20:  Compute advantage Âi for each completion globally using Z-score normalization.
21:  Process smaller sub-batches sequentially, compute losses, accumulate gradients, and update policy πθ using DAPO objective.
22: Return Optimized policy πθ∗.