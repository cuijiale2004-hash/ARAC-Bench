# Standard Research Idea and Full Set of Implementation Plans

## 1. Research Background and Existing Pain Points
**Research Background:**
Multimodal Large Language Models (MLLMs) have recently emerged as general architectures capable of reasoning over diverse modalities. A prominent subclass, Visual–Language Models (VLMs), jointly process images and text and are expected to support downstream tasks that require cross-modal reasoning, such as medical image diagnosis and industrial inspection. Consequently, rigorous and trustworthy multimodal benchmarks are essential for practitioners to choose appropriate models. Item Response Theory (IRT) is a principled framework for assessing subject ability and item difficulty. Without knowing the questions and answers, IRT estimates the ability and difficulty as parameters to predict the records of success or failure of a subject on an item. These parameters allow the construction of a compact subset of items tailored to each subject using Computerized Adaptive Testing (CAT).

**Existing Pain Points:**
1. Benchmarks for MLLMs should measure their ability for cross-modal integration. However, current benchmarks are often filled with shortcut questions that can be solved using only a single modality (e.g., answerable from text alone or image alone). For example, in vision-language cases, the correct answer can be found without either the image or the text.
2. These low-quality questions unnecessarily increase the size and computational requirements of a benchmark and yield unreliable rankings. As the pool of candidate models grows, evaluating thousands of mixed-quality questions per model becomes increasingly costly, while single-modality shortcuts further obstruct evaluating the cross-modal reasoning ability.
3. Classical IRT is agnostic to the modality of inputs and thus contains only a single latent ability or difficulty parameter. IRT cannot determine whether success on a multimodal item reflects true cross-modal reasoning or others.

## 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:**
To address the limitations of current multimodal benchmarks and classical IRT, this research aims to introduce a multi-modal and multidimensional item response theory framework that explicitly models modality-specific and cross-modal components. The goal is to estimate the cross-modal ability of MLLMs and each question’s cross-modal difficulty, enabling the construction of compact, high-quality subsets that better reflect multimodal reasoning while reducing evaluation cost and improving reliability.

**Scientific Questions:**
1. How to decompose both model ability and item difficulty into image-only, text-only, and cross-modal components within an IRT framework?
2. How to formulate a probabilistic model that captures the modality-aware behavior of subjects on items?
3. How to adaptively select informative subsets of items using Computerized Adaptive Testing (CAT) based on the decomposed parameters to minimize estimation uncertainty?

## 3. Overall Core Idea and Design Philosophy
**Overall Core Idea:**
The core idea is to extend classical IRT by decomposing both model ability and item difficulty into latent components: image-only, text-only, and cross-modal integration. This is achieved by introducing MultiModal Item Response Theory (M2IRT) and Multidimensional MultiModal Item Response Theory (M3IRT). 

**Design Philosophy:**
The design assumes that an MLLM has modality-specific abilities as well as an ability to integrate information across modalities. Likewise, each multimodal question exhibits modality-specific and cross-modal characteristics that can determine whether a subject can provide the correct answer. By defining binary indicators for the modalities present in a question and incorporating them into the parameterization of ability, difficulty, and discrimination, the framework can explicitly capture the contribution of cross-modal integration. This decomposition allows for the identification of genuinely cross-modal items and the construction of compact, high-quality benchmark subsets via Computerized Adaptive Testing (CAT) that preserves ranking fidelity while reducing computational cost.

## 4. Core Innovation Points
1. **Explicit Modeling of Modality-Specific and Cross-Modal Components:** Proposed M3IRT, which explicitly models modality-specific (image-only, text-only) and cross-modal components of both item difficulty and model ability for multimodal evaluation, overcoming the limitation of classical IRT being agnostic to input modality.
2. **Compact and High-Quality Subset Generation:** M3IRT yields compact, high-quality subsets that emphasize cross-modal reasoning and maintain reliable model rankings at substantially reduced computational cost.
3. **Robustness to Low-Quality Contamination:** Through experiments with 24 VLMs across three benchmarks, M3IRT is demonstrated to be robust to large fractions of low-quality items (up to 50%) and provides interpretable characterizations of both benchmarks and models.
4. **Modality-Based Decomposition Mechanism:** Introduction of a decomposition mechanism for IRT parameters (ability, difficulty, discrimination) that uses binary modality indicators and interaction terms to reflect the subtractive role of modality difficulties and the additive role of modality abilities.
5. **Adaptive Testing with D-Optimality for M3IRT:** Integration of the decomposed parameters with CAT using Fisher information matrices and the D-optimality criterion to maximize the determinant of cumulative information for efficient and informative subset selection.

## 5. Overview of the Overall Technical Solution
The overall technical solution involves treating MLLMs as subjects and benchmark questions as items. For each subject–question pair, a response indicating success or failure is recorded. The solution introduces modality-based decomposition of IRT parameters:
1. **Modality Indicators:** Define binary indicators $s_{image}, s_{text} \in \{0, 1\}$ to represent the modalities present in a question.
2. **Parameter Decomposition:** Decompose the ability parameter $\theta$, difficulty parameter $b$, and discrimination parameter $a$ into base, image, text, and cross-modal components, conditioned on the modality indicator vector $s$.
3. **Model Formulation:** Formulate M2IRT as a logistic IRT model using the decomposed parameters, and M3IRT as a multidimensional logistic IRT model where the decomposed components are represented as vectors interacting with the modality indicator vector via diagonal matrices.
4. **Parameter Estimation:** Learn the parameters using Stochastic Gradient Descent (SGD) by minimizing the negative log-likelihood of the observed response tensor.
5. **Adaptive Testing:** Apply Computerized Adaptive Testing (CAT) to select an informative subset of items. For M2IRT, selection is based on Fisher information; for M3IRT, selection is based on the D-optimality criterion (maximizing the determinant of the cumulative Fisher information matrix).

## 6. Detailed Module Design
**Module 1: Modality-Based Decomposition of IRT Parameters**
This module defines how modality presence affects ability, difficulty, and discrimination.
- **Modality Indicator:** Let $s = (s_{image}, s_{text}) \in S = \{(0,0), (0,1), (1,0), (1,1)\}$ denote the format representing a question. $s_{image}=1$ if image is provided, $s_{text}=1$ if text is provided.
- **Ability Decomposition:** For subject $i$, denote base, image, text, and cross-modal abilities by $\theta_i^{base}, \theta_i^{image}, \theta_i^{text}, \theta_i^{cross} \in [0, q]$. The ability given format $s$ is defined as the sum of the base ability and the contributions of the present modalities, including the cross-modal interaction term.
- **Difficulty Decomposition:** For item $j$, let $b_j^{base}, b_j^{image}, b_j^{text}, b_j^{cross} \in [0,q]$ be the base, image, text, and cross-modal difficulties. The difficulty given format $s$ is defined as the base difficulty minus the contributions of the provided hints (modalities). 
- **Discrimination Decomposition:** Let $a_j^{base} \in [0,q]$ be the base discrimination, and $a_j^{image}, a_j^{text}, a_j^{cross} \in [0,q]$ capture contributions from image, text, and cross-modal integration. The discrimination given format $s$ is additive.

**Module 2: Multimodal Item Response Theory (M2IRT)**
This module extends the logistic IRT model using the decomposed parameters.
- **Response Tensor:** For each subject–question-format combination $(i, j, s)$, let $r_{i,j,s} \in \{0, 1\}$ indicate correctness. The full response set is the tensor $R' = \{r_{i,j,s}\}_{(i,j,s) \in M \times N \times S}$.
- **Logistic Model:** M2IRT defines the probability of a correct answer using a sigmoid function applied to the product of the decomposed discrimination and the difference between decomposed ability and difficulty.

**Module 3: Multimodal Multi-Dimensional Item Response Theory (M3IRT)**
This module extends the logistic MIRT model with the modality-based decomposition by modifying the decomposed components into vectors.
- **Vector Parameters:** For subject $i$, define the ability vector $\theta_i = [\theta_i^{base}, \theta_i^{image}, \theta_i^{text}, \theta_i^{cross}]^\top$. For item $j$, define the discrimination and difficulty vectors $a_j = [a_j^{base}, a_j^{image}, a_j^{text}, a_j^{cross}]^\top$ and $b_j = [b_j^{base}, b_j^{image}, b_j^{text}, b_j^{cross}]^\top$.
- **Format Indicator Vector:** Introduce a format indicator vector $s = [1, -s_{image}, -s_{text}, -s_{image}s_{text}]^\top$. The negative signs align with the subtractive role of the modality terms in the difficulty definition and the decomposition in the discrimination definition.
- **Probability Calculation:** The probability is defined using a sigmoid function applied to the term $z'_{i,j,s}$, which involves the inner product of the discrimination vector with the element-wise product of the indicator vector and the ability vector, minus the inner product of the indicator vector and the difficulty vector.

**Module 4: Learning M3IRT using Stochastic Gradient Descent**
This module details the parameter estimation process.
- **Loss Function:** The negative log-likelihood function is defined over a training dataset $R'' \subset R'$.
- **Optimization:** The parameters set $\Theta = \{\{a_j\}_{j \in N}, \{b_j\}_{j \in N}, \{\theta_i\}_{i \in M}\}$ are estimated by minimizing $L(\Theta)$ using minibatch SGD. M2IRT is estimated in a similar manner. This approach does not require a dense response matrix, allowing learning from partially observed data like tensor completion.

**Module 5: Computer Adaptive Test with M2IRT and M3IRT**
This module integrates the proposed models with CAT to adaptively select an informative subset of items.
- **Fisher Information for M2IRT:** The Fisher information of item $j$ for subject $i$ under format $s$ is calculated using the product of the predicted probabilities and the square of the decomposed discrimination.
- **Fisher Information for M3IRT:** The Fisher information matrix for item $j$ at ability $\theta$ is calculated using the predicted probabilities and the outer product of the vector $(\text{diag}(s)a_j)$ with itself.
- **Item Selection (D-Optimality):** The D-optimality criterion is adopted to minimize estimation uncertainty by maximizing the determinant of the cumulative information. Let $U_i \subseteq N$ be the set of items not yet answered. At stage $t$, given the cumulative information matrix $I_i^{(t-1)}$, the next item is selected and the information matrix is updated.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $M = \{1, \dots, m\}$: Collection of MLLMs (subjects).
- $N = \{1, \dots, n\}$: Multimodal benchmark questions (items).
- $r_{i,j} \in \{0, 1\}$: Indicates whether subject $i$ answers question $j$ correctly.
- $R = \{r_{i,j}\}_{(i,j) \in M \times N}$: Response matrix.
- $\sigma(x) = 1/(1 + \exp(-x))$: Sigmoid function.
- $\theta_i \in \mathbb{R}$: Ability parameter for subject $i$ in classical IRT.
- $a_j > 0$: Discrimination parameter for item $j$ in classical IRT.
- $b_j \in \mathbb{R}$: Difficulty parameter for item $j$ in classical IRT.
- $s_{image}, s_{text} \in \{0, 1\}$: Binary indicators for the presence of image and text modalities.
- $s = (s_{image}, s_{text}) \in S$: Format of representing a question.
- $q \ge 0$: Shared upper bound for ability, difficulty, and discrimination parameters to balance their scales.
- $\theta_i^{base}, \theta_i^{image}, \theta_i^{text}, \theta_i^{cross} \in [0, q]$: Base, image, text, and cross-modal abilities for subject $i$.
- $b_j^{base}, b_j^{image}, b_j^{text}, b_j^{cross} \in [0, q]$: Base, image, text, and cross-modal difficulties for item $j$.
- $a_j^{base} \in [0, q], a_j^{image}, a_j^{text}, a_j^{cross} \in [0, q]$: Base, image, text, and cross-modal discrimination contributions for item $j$.
- $r_{i,j,s} \in \{0, 1\}$: Indicates whether subject $i$ answers question $j$ given format indicator $s$ correctly.
- $R' = \{r_{i,j,s}\}_{(i,j,s) \in M \times N \times S}$: Full response tensor.
- $z_{i,j,s}$: Intermediate variable for M2IRT.
- $\theta_i \in \mathbb{R}^d$: Ability vector for subject $i$ in MIRT.
- $a_j, b_j \in \mathbb{R}^d$: Discrimination and difficulty vectors for question $j$ in MIRT.
- $\theta_i = [\theta_i^{base}, \theta_i^{image}, \theta_i^{text}, \theta_i^{cross}]^\top$: Ability vector for subject $i$ in M3IRT.
- $a_j = [a_j^{base}, a_j^{image}, a_j^{text}, a_j^{cross}]^\top$: Discrimination vector for item $j$ in M3IRT.
- $b_j = [b_j^{base}, b_j^{image}, b_j^{text}, b_j^{cross}]^\top$: Difficulty vector for item $j$ in M3IRT.
- $s = [1, -s_{image}, -s_{text}, -s_{image}s_{text}]^\top$: Format indicator vector for M3IRT.
- $\text{diag}(s)$: Diagonal matrix whose diagonal elements are the vector $s$.
- $z'_{i,j,s}$: Intermediate variable for M3IRT.
- $R'' \subset R'$: Training dataset.
- $\Theta = \{\{a_j\}_{j \in N}, \{b_j\}_{j \in N}, \{\theta_i\}_{i \in M}\}$: Parameters set.
- $I_{i,j}$: Fisher information of item $j$ for subject $i$.
- $U_i \subseteq N$: Set of items not yet answered by subject $i$.
- $I_i^{(t)}$: Cumulative information matrix for subject $i$ at stage $t$.

**Mathematical Formulas:**
1. Classical 2PL IRT Model:
$$ \Pr(r_{i,j} = 1 | \theta_i, a_j, b_j) = \sigma(a_j(\theta_i - b_j)) $$

2. Multidimensional IRT (MIRT) Model:
$$ \hat{P}(r_{i,j} = 1) = \sigma(a_j^\top \theta_i - b_j) $$
$$ \hat{P}(r_{i,j} = 0) = 1 - \hat{P}(r_{i,j} = 1) $$

3. Ability Decomposition given indicator $s$:
$$ \theta_i(s) = \theta_i^{base} + s_{image}\theta_i^{image} + s_{text}\theta_i^{text} + s_{image}s_{text}\theta_i^{cross} $$

4. Difficulty Decomposition given indicator $s$:
$$ b_j(s) = b_j^{base} - s_{image}b_j^{image} - s_{text}b_j^{text} - s_{image}s_{text}b_j^{cross} $$

5. Discrimination Decomposition given indicator $s$:
$$ a_j(s) = a_j^{base} + s_{image}a_j^{image} + s_{text}a_j^{text} + s_{image}s_{text}a_j^{cross} $$

6. Multimodal Item Response Theory (M2IRT) Model:
$$ z_{i,j,s} = a_j(s)(\theta_i(s) - b_j(s)) $$
$$ \hat{P}(r_{i,j,s} = 1) = \sigma(z_{i,j,s}) $$
$$ \hat{P}(r_{i,j,s} = 0) = 1 - \hat{P}(r_{i,j,s} = 1) $$

7. Multimodal Multi-Dimensional Item Response Theory (M3IRT) Model:
$$ z'_{i,j,s} = a_j^\top \text{diag}(s)\theta_i - s^\top b_j $$
$$ \hat{P}(r_{i,j,s} = 1) = \sigma(z'_{i,j,s}) $$
$$ \hat{P}(r_{i,j,s} = 0) = 1 - \hat{P}(r_{i,j,s} = 1) $$

8. Negative Log-Likelihood for SGD Learning:
$$ L(\Theta) = -\sum_{(i,j,s) \in R''} \left( r_{i,j,s} \log \hat{P}(r_{i,j,s} = 1) + (1 - r_{i,j,s}) \log \hat{P}(r_{i,j,s} = 0) \right) $$
Optimization objective: $\hat{\Theta} = \arg\min_\Theta L(\Theta)$

9. Fisher Information for M2IRT (for CAT):
$$ I_{i,j} = \hat{P}(r_{i,j,s} = 1)\hat{P}(r_{i,j,s} = 0)(a_j(s))^2 $$

10. Fisher Information Matrix for M3IRT (for CAT):
$$ I_{i,j} = \hat{P}(r_{i,j,s} = 1)\hat{P}(r_{i,j,s} = 0)(\text{diag}(s)a_j)(\text{diag}(s)a_j)^\top $$

11. CAT Item Selection and Update Rule (D-Optimality):
$$ j^* = \arg\max_{j \in U_i} \det(I_i^{(t-1)} + I_{ij}) $$
$$ I_i^{(t)} = I_i^{(t-1)} + I_{ij^*} $$

## 8. Algorithm Pseudocode
Based on the paper's description of the Stochastic Gradient Descent learning and Computerized Adaptive Testing, the iterative processes are formulated as follows:

**Algorithm 1: Parameter Estimation via SGD**
Input: Response tensor $R'' \subset R'$, Learning rate $\eta$, Parameter upper bound $q$.
Output: Estimated parameters $\hat{\Theta} = \{\{a_j\}, \{b_j\}, \{\theta_i\}\}$.
1: Initialize parameters $\Theta$ randomly.
2: while not converged do
3:     Sample a minibatch from $R''$.
4:     For each $(i,j,s)$ in the minibatch:
5:         Compute $\hat{P}(r_{i,j,s} = 1)$ using M2IRT (Eq 6) or M3IRT (Eq 7).
6:         Compute loss $L(\Theta)$ based on the negative log-likelihood (Eq 8).
7:     Compute gradient $\nabla_\Theta L(\Theta)$.
8:     Update parameters: $\Theta \leftarrow \Theta - \eta \nabla_\Theta L(\Theta)$.
9:     Clamp parameters such that $\theta, a, b \in [0, q]$.
10: end while
11: Return $\hat{\Theta}$.

**Algorithm 2: Computer Adaptive Testing with M3IRT**
Input: Estimated parameters $\Theta$, New subject $i'$, Modality format $s$, Item pool $N$.
Output: Estimated ability vector $\theta_{i'}$, Selected subset $\hat{N}$.
1: Initialize subject ability $\theta_{i'}^{(0)}$ randomly.
2: Initialize cumulative information matrix $I_{i'}^{(0)} = \mathbf{0}$.
3: Initialize unasked items $U_{i'} = N$, Selected subset $\hat{N} = \emptyset$.
4: for $t = 1, 2, \dots$ do
5:     For all $j \in U_{i'}$, compute Fisher information matrix $I_{i'j}$ using Eq 10.
6:     Select next item:
        $j^* = \arg\max_{j \in U_{i'}} \det(I_{i'}^{(t-1)} + I_{i'j})$
7:     Present question $j^*$ with format $s$ to subject $i'$ and record response $r_{i',j^*,s}$.
8:     Update cumulative information matrix:
        $I_{i'}^{(t)} = I_{i'}^{(t-1)} + I_{i'j^*}$
9:     Update estimated ability $\theta_{i'}^{(t)}$ based on the new response (e.g., using MLE or EAP).
10:    Update sets: $\hat{N} \leftarrow \hat{N} \cup \{j^*\}$, $U_{i'} \leftarrow U_{i'} \setminus \{j^*\}$.
11:   if stopping criterion met then
12:       break
13:   end if
14: end for
15: Return $\theta_{i'}$, $\hat{N}$.