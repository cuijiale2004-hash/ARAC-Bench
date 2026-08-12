### 1. Research Background and Existing Pain Points

**Research Background:**
The widespread adoption of Large Language Models (LLMs) across various domains has brought increasing attention to their critical limitation: their tendency to generate incorrect or misleading information—commonly referred to as “hallucinations.” This issue supports the idea that LLMs are just stochastic parrots answering in a way that is statistically plausible with respect to the input prompt despite not having a real understanding of it. On the other side, recent reasoning capabilities proper to ChatGPT 4o or Deepseek offer counter evidence to actually support this. Ongoing research seeks to characterize and categorize hallucinations, setting them apart from other error types. At the same time, recent discussions have introduced terms such as confabulations and fabrications, sometimes attributing a form of “intention” to LLMs—though the very idea of LLM “intentionality” and other human-like qualities remains contested. Research on LLM hallucinations can be categorized into two main branches: the first one is the extrinsic branch, where the hallucinations are measured with respect to the interpretation that humans give to those errors. The second branch was started by Kadavath et al. (2022b), proposing to study the hallucinations within the model itself. Following Kadavath et al. (2022b), the work in Li et al. (2024) proposes Inference-Time Intervention (ITI) as a way to improve the “truthfulness” of LLMs at inference time. ITI functions by altering model activations at inference time, steering them along specific directions within a restricted set of attention heads. Furthermore, recent foundations suggest hallucination and imagination are mathematically identical phenomena emerging from a necessary violation of information conservation, and formal learning-theoretic proofs show that hallucinations are unavoidable. It has also been shown that hallucinations come from the statistical problem of the pretraining methodology: minimizing the cross entropy naturally causes errors because it does not train the model to express uncertainty and say “I do not know.”

**Existing Pain Points:**
Dominant training-free baselines such as logits or “p(true)” remain weak for hallucination detection. While Orgad et al. (2025) train classifiers on the internal representations of the LLMs to predict correctness, probing classifiers do not generalize across different tasks. Given that LLMs are foundational models, user interactions typically occur in the wild, making it difficult to predict which probe classifier is best suited for detecting hallucinations in real-world scenarios. Furthermore, in this setting, classifier weights should not only be updated dynamically for each task, but the optimal token–layer combination is also dataset-dependent, which conflicts with the broad LLM applicability. There is a critical need for a training-free method that generalizes better across different tasks and is mathematically principled.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To solve the problem of poor generalization and training overhead of existing hallucination detection methods (such as probing classifiers), this work seeks a principled, mathematical framework using Energy-Based Models (EBMs) that allows detecting errors directly from output logits without requiring trained probe classifiers or activation ablations. The motivation is to exploit the mathematical property that energies across consecutive generation steps should match according to the chain rule of probability. Since this constraint is not explicitly optimized for in LLM training, the discrepancy—termed "spilled energy"—can be exploited as a training-free signal for hallucination detection.

**Scientific Questions:**
Can the final Large Language Model (LLM) softmax classifier be reinterpreted as an Energy-Based Model (EBM) to decompose the sequence-to-sequence probability chain into multiple interacting EBMs at inference? Can the discrepancy between energy values that should theoretically match across consecutive generation steps (i.e., spilled energy) serve as a robust, training-free metric that correlates strongly with factual errors, biases, and failures?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
This work reinterprets the final Large Language Model (LLM) softmax classifier as an Energy-Based Model (EBM), decomposing the sequence-to-sequence probability chain into multiple interacting EBMs at inference. This principled approach allows tracking “energy spills” during decoding, which empirically correlate with factual errors, biases, and failures. Similar to Orgad et al. (2025), the method localizes the exact answer token and subsequently tests for hallucinations. Crucially, however, this is achieved without requiring trained probe classifiers or activation ablations. Instead, two completely training-free metrics derived directly from output logits are introduced: spilled energy, which captures the discrepancy between energy values across consecutive generation steps that should theoretically match, and marginalized energy, which is measurable at a single step.

**Design Philosophy:**
The design philosophy is rooted in the mathematics of EBMs and the chain rule of probability. By writing the conditional probability as the ratio between the joint distribution in the numerator and the marginal distribution in the denominator, and recognizing this ratio is implemented in LLMs as a softmax classifier, one can apply the trick from Grathwohl et al. (2020). The methodology relies solely on reading values inside the LLM (logits and the log-sum-exp of logits), enabling natural generalization across tasks and performing better than logit-based detection, striking a good generalization across tasks and LLMs without training or tuning the detector.

### 4. Core Innovation Points

1.  **Training-free, LLM hallucination detection generalizing across tasks using the EBM framework:** A method for detecting hallucinations is introduced that requires no additional training, in contrast to prior work that relies on trained classifiers and ablations of model activations. The approach directly reads values inside the LLM, enabling natural generalization across tasks and performing better than logit-based detection.
2.  **Two energy-based metrics:** Two complementary measures of energy spills are defined: (i) delta energy $\Delta E_{\theta}(x_{i:1})$, which captures discrepancies between energy values across two time steps that should be mathematically equivalent, and (ii) marginal energy $E_{\theta}^{m}(x_{i:1})$, which can be evaluated at a single time step.
3.  **Scaled spilled energy:** A combined metric $\Delta E^{s}(x_{i:1})$ where the spilled energy is multiplied by the absolute value of the marginal energy.
4.  **Scalable and generalizable analysis:** The framework is mathematically principled, training-free, and exhibits strong cross-dataset generalization. The analysis scales to state-of-the-art LLMs (including LLaMA, Mistral, and Gemma) and demonstrates competitive performance across benchmarks, showing robustness across datasets and architectures.
5.  **Exact answer token localization and pooling strategy:** Following the finding that the truthfulness signal is concentrated in the “exact answer tokens,” the method focuses detection on this specific span $[u,w] \subseteq [i+1, N]$ using heuristic matching or LLM-based extraction, and applies pooling strategies (Min, Max, Mean, Last Token, After Last Token) to aggregate energy values.

### 5. Overview of the Overall Technical Solution

1.  **Reinterpretation of Softmax as EBM:** The conditional probability modeled by the LLM is decomposed using the probabilities of the sequences. The marginal term from step $i$ cancels out with the sequence probability from the decomposition at the previous step $i-1$. The conditional is written as the ratio between the joint distribution (numerator) and the marginal distribution (denominator), which corresponds to the softmax classifier implementation in LLMs.
2.  **Energy Decomposition:** The numerator of the softmax (specific logit) is reinterpreted as the energy of the sampled tokens $E_{\theta}^{\ell}(x_{i:1})$, and the denominator (log-sum-exp over vocabulary) is reinterpreted as the marginal energy $E_{\theta}^{m}(x_{i-1:1})$.
3.  **Spilled Energy Definition:** The discrepancy between $E_{\theta}^{m}(x_{i:1})$ measured at timestep $i+1$ and $E_{\theta}^{\ell}(x_{i:1})$ measured at timestep $i$ is defined as spilled energy $\Delta E_{\theta}(x_{i:1})$. According to the chain rule, these two quantities should be identical; however, they differ in the LLM implementation.
4.  **Exact Answer Localization:** Given a prompt and a completion, the method identifies the "exact answer" tokens—those that contain the precise answer—using heuristic matching or an auxiliary instruction-tuned LLM.
5.  **Hallucination Detection:** The spilled energy and marginal energy are computed specifically over the localized exact answer interval $[u,w]$. A pooling strategy (e.g., min pooling) is applied to obtain a final score per sentence. High spilled energy correlates with incorrect outputs.

### 6. Detailed Module Design

**Module 1: Autoregressive Probability Decomposition**
The language modeling objective models the joint probability of tokens in the sequence $X$, through a conditional probability parameterized by $\theta$. The factorization uses discriminative classifiers recursively to model each conditional probability:
$$p(x_{i:1}) = \prod_i p_{\theta}(x_i|x_{i-1:1}) = \prod_i \frac{p_{\theta}(x_{i:1})}{p_{\theta}(x_{i-1:1})}$$
This confirms that the marginal term from step $i$ cancels out with the sequence probability from the decomposition at the previous step $i-1$.

**Module 2: EBM Reinterpretation of Softmax**
The conditional probability is written as the ratio of two EBMs. Following Zhu et al. (2021), the partition functions simplify ($\log \tilde{Z}(\theta) = \log Z(\theta)$). The conditional probability of the softmax is decomposed into:
$$-\log p_{\theta}(x_i | x_{i-1:1}) = -\log \left( \frac{\exp(\theta(x_{i-1:1})[id(x_i)])}{\sum_k \exp(\theta(x_{i-1:1})[k])} \right)$$
This is split into two energy terms:
1.  **Logit Energy:** $E_{\theta}^{\ell}(x_{i:1}) = -\theta(x_{i-1:1})[id(x_i)]$
2.  **Marginal Energy:** $E_{\theta}^{m}(x_{i-1:1}) = -\log \sum_{k=1}^V \exp\theta(x_{i-1:1})[k]$

**Module 3: Spilled Energy Computation**
The total energy of a sequence of length $N$ is $E_{\theta}(x_{N:1})$. Expanding the negative log-likelihood:
$$E_{\theta}(x_{N:1}) = \sum_{i=1}^{N-1} E_{\theta}^{\ell}(x_{i+1:1}) - E_{\theta}^{m}(x_{i:1})$$
The two energies, not interacting at the same step but at steps $i$ and $i-1$, should be equal, but are measured in the LLM at different generation steps and from different components. The discrepancy is defined as spilled energy:
$$\Delta E_{\theta}(x_{i:1}) \triangleq -E_{\theta}^{m}(x_{i:1}) + E_{\theta}^{\ell}(x_{i:1})$$
Since both terms should be equal to $E_{\theta}(x_{i:1})$, delta values should always be zero when correctly modeling the energy at timestep $i$.

**Module 4: Exact Answer Localization and Pooling**
The method focuses on the “exact answer” tokens $[u,w] \subseteq [i+1, N]$.
*   **Heuristic Matching:** For tasks with a closed set of possible labels, string matching is performed.
*   **LLM-based Extraction:** For open-ended generation tasks, an instruction-tuned model (e.g., Mistral-7B-Instruct) is prompted with: "Extract from the following long answer the short answer, only the relevant tokens. If the long answer does not answer the question, output NO ANSWER."
*   **Pooling Strategies:** When the answer spans multiple tokens, a pooling strategy is applied over the interval $[u,w]$:
    *   **Min:** minimum energy value in the pooling window.
    *   **Max:** maximum energy value in the pooling window.
    *   **Mean:** mean of all the energies in the pooling window.
    *   **Last Token:** energy of the last token in the pooling window.
    *   **After Last Token:** energy of the first token after the pooling window.

### 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $V$: Vocabulary of an LLM, the set of all tokens that can be processed as input and generated at each decoding step, with size $|V| = V$.
*   $X = \{x_{N:1}\}$: Sequence of tokens $\{x_N, \dots, x_1\}$.
*   $x_i \in V$: Token in the $i$-th position along the sequence.
*   $\theta : \mathbb{R}^{N \times V} \to \mathbb{R}^V$: LLM function implemented by a transformer.
*   $\theta(x_{i:1})[k]$: Predicted logit assigned to the $k$-th token class in $V$ for the $i+1$ token in the sequence.
*   $id : \{0, 1\}^V \to [1, \dots, V]$: Map that takes as input a one-hot encoding vector $x_i$ for a word token at position $i$ and outputs its index in the vocabulary.
*   $E_{\theta}(x)$: Energy function parameterized by a neural network $\theta$, assigning a scalar energy to each configuration of $x$.
*   $Z_{\theta}$: Partition function (normalizing constant), defined as $Z_{\theta} = \sum_x \exp(-E_{\theta}(x))$ for discrete $x$.
*   $E_{\theta}^{\ell}(x_{i:1})$: Energy of the sampled tokens $\{x_{i:1}\}$ (Logit Energy).
*   $E_{\theta}^{m}(x_{i-1:1})$: Energy $E_{\theta}(x_{i:1})$, marginalized over all possible $x_i$ (Marginal Energy).

**Formulas:**
$$p_{\theta}(x) = \frac{\exp(-E_{\theta}(x))}{Z_{\theta}}$$
$$p(x_{i:1}) = p(x_i | x_{i-1:1}) \dots p(x_2 | x_1) p(x_1) = \prod_i p_{\theta}(x_i | x_{i-1:1}) p_{\theta}(x_1) \quad \text{(Eq. 1)}$$
$$p(x_{i:1}) = \prod_i p_{\theta}(x_i|x_{i-1:1}) = \prod_i \frac{p_{\theta}(x_{i:1})}{p_{\theta}(x_{i-1:1})} \implies \dots \frac{p_{\theta}(x_{i:1})}{p_{\theta}(x_{i-1:1})} \underbrace{\frac{p_{\theta}(x_{i-1:1})}{p_{\theta}(x_{i-2:1})}}_{\text{step } i-1} \cdots = p(x_{i:1}) \quad \text{(Eq. 2)}$$
$$p_{\theta}(x_i|x_{i-1:1}) = \frac{p_{\theta}(x_{i:1})}{p_{\theta}(x_{i-1:1})} = \frac{\exp\theta(x_{i-1:1}) [id(x_i)]}{\sum_{k=1}^V \exp\theta(x_{i-1:1})[k]} \text{ where } id : \{0, 1\}^V \to [1, \dots, V] \quad \text{(Eq. 3)}$$
$$\log p_{\theta}(x_i|x_{i-1:1}) = \log \frac{\exp(-E_{\theta}^{\ell}(x_{i:1}))}{\exp(-E_{\theta}^{m}(x_{i-1:1}))} \frac{\tilde{Z}(\theta)}{Z(\theta)} = -E_{\theta}^{\ell}(x_{i:1}) + E_{\theta}^{m}(x_{i-1:1}) \quad \text{(Eq. 4)}$$
$$-\log p_{\theta}(x_i | x_{i-1:1}) = -\log \left( \frac{\exp(\theta(x_{i-1:1})[id(x_i)])}{\sum_k \exp(\theta(x_{i-1:1})[k])} \right) = \quad \text{(Eq. 5)}$$
$$= \underbrace{-\theta(x_{i-1:1}) [id(x_i)]}_{E_{\theta}^{\ell}(x_{i:1})} + \underbrace{\log \sum_{k=1}^V \exp\theta(x_{i-1:1})[k]}_{-E_{\theta}^{m}(x_{i-1:1})} \quad \text{(Eq. 6)}$$
$$E_{\theta}^{\ell}(x_{i:1}) = -\theta(x_{i-1:1})[id(x_i)], \quad E_{\theta}^{m}(x_{i-1:1}) = -\log \sum_{k=1}^V \exp\theta(x_{i-1:1})[k] \quad \text{(Eq. 7)}$$
$$-\log p(x_{N:1}) = -\log \prod_i p_{\theta}(x_i|x_{i-1:1}) = \sum_i E_{\theta}^{\ell}(x_{i:1}) - E_{\theta}^{m}(x_{i-1:1})$$
$$E_{\theta}(x_{N:1}) = \sum_{i=1}^{N-1} E_{\theta}^{\ell}(x_{i+1:1}) - E_{\theta}^{m}(x_{i:1}) = \dots \underbrace{E_{\theta}^{\ell}(x_{i+1:1})}_{\text{timestep } i+1} \underbrace{\Delta E_{\theta}(x_{i:1})}_{- E_{\theta}^{m}(x_{i:1})} + \underbrace{E_{\theta}^{\ell}(x_{i:1}) - E_{\theta}^{m}(x_{i-1:1})}_{\text{timestep } i} \dots$$
$$\Delta E_{\theta}(x_{i:1}) \triangleq -E_{\theta}^{m}(x_{i:1}) + E_{\theta}^{\ell}(x_{i:1}) = \underbrace{-\log \sum_k \exp(\theta(x_{i:1})[k])}_{\text{timestep } i+1} + \underbrace{\theta(x_{i-1:1})[id(x_i)]}_{\text{timestep } i} \quad \text{(Eq. 8)}$$
$$\Delta E^{s}(x_{i:1}) = |E_{\theta}^{m}(x_{i:1})|\Delta E_{\theta}(x_{i:1})$$

**Appendix A.1 Proof:**
$$\begin{cases} E_{\theta}^{\ell}(x_{i:1}) = -\log(\exp(\theta(x_{i-1:1})[id(x_i)])) \\ E_{\theta}^{m}(x_{i-1:1}) = -\log(\sum_{k=1}^V \exp(\theta(x_{i-1:1})[k])) \end{cases} \quad \text{(Eq. 9)}$$
$$p_{\theta}(x_{i:1}) = \frac{\exp(-E_{\theta}^{\ell}(x_{i:1}))}{Z_{\theta}} \quad \text{(Eq. 10)}$$
$$Z_{\theta} = \sum_{x_{i-1:1}} \sum_{x_i} \exp(\theta(x_{i-1:1})[id(x_i)]) = \sum_{x_{i-1:1}} \sum_{k=1}^V \exp(\theta(x_{i-1:1})[k]) \quad \text{(Eq. 11)}$$
$$p_{\theta}(x_{i-1:1}) = \frac{\exp(-E_{\theta}^{m}(x_{i-1:1}))}{\tilde{Z}_{\theta}} \quad \text{(Eq. 12)}$$
$$\tilde{Z}_{\theta} = \sum_{x_{i-1:1}} \exp(-E_{\theta}^{m}(x_{i-1:1})) = \sum_{x_{i-1:1}} \exp(\log \sum_{k=1}^V \exp(\theta(x_{i-1:1})[k])) \quad \text{(Eq. 13)}$$
$$\tilde{Z}_{\theta} = \sum_{x_{i-1:1}} \sum_{k=1}^V \exp(\theta(x_{i-1:1})[k]) \quad \text{(Eq. 14)}$$
$$Z_{\theta} = \tilde{Z}_{\theta} \quad \text{(Eq. 15)}$$

**Appendix A.2 Temperature Role:**
$$\log p_{\theta}(x_i | x_{i-1:1}) = \log \frac{\exp(\frac{1}{\tau} \theta(x_{i-1:1})[Id(x_i)])}{\sum_k \exp(\frac{1}{\tau} \theta(x_{i-1:1})[k])} \quad \text{(Eq. 16)}$$
$$= \frac{1}{\tau} \theta(x_{i-1:1})[Id(x_i)] - \log \sum_k \exp(\frac{1}{\tau} \theta(x_{i-1:1})[k]) \quad \text{(Eq. 17)}$$
$$\Delta E_{\theta}(x_{i:1}) = \frac{1}{\tau} \theta(x_{i-1:1})[Id(x_i)] - \log \sum_{k=1}^{|V|} \exp(\frac{1}{\tau} \theta(x_{i}, \dots, x_1)[k]) \quad \text{(Eq. 18)}$$
$$\lim_{\tau \to +\infty} \Delta E_{\theta}(x_{i:1}) = \lim_{\tau \to \infty} \frac{1}{\tau} \theta(x_{i-1:1})[Id(x_i)] - \log \sum_{k=1}^{|V|} \exp(\frac{1}{\tau} \theta(x_{i-1:1})[k]) \quad \text{(Eq. 19)}$$
$$= 0 - \log \sum_{k=1}^{|V|} \exp(0) \quad \text{(Eq. 20)}$$
$$= -\log |V| \quad \text{(Eq. 21)}$$
$$\log p_{\theta}(x_{i-1:1}) = \frac{1}{\tau} \theta(x_{i-1:1})[Id(x_i)] - \log \sum_k \exp(\frac{1}{\tau} \theta(x_{i-1:1})[k]) + \sum_{j=1}^i \Delta E_{\theta}(x_{j:1}) \quad \text{(Eq. 22)}$$

### 8. Algorithm Pseudocode

The paper does not provide explicit algorithm pseudocode blocks. The operational flow is derived from the mathematical derivations and experimental sections as follows:

1.  **Input:** Prompt $\{x_{i-1}, \dots, x_1\}$, LLM $\theta$.
2.  **Generation:** Feed the prompt to LLM $\theta$ and obtain the completion $\{x_N, \dots, x_i\}$.
3.  **Localization:** Identify the "exact answer" tokens span $[u,w] \subseteq [i+1, N]$ using:
    *   Heuristic Matching (for closed-set labels).
    *   LLM-based Extraction (for open-ended generation).
4.  **Energy Computation:** For each token in $[u,w]$:
    *   Compute Logit Energy: $E_{\theta}^{\ell}(x_{t:1}) = -\theta(x_{t-1:1})[id(x_t)]$.
    *   Compute Marginal Energy: $E_{\theta}^{m}(x_{t-1:1}) = -\log \sum_{k=1}^V \exp\theta(x_{t-1:1})[k]$.
    *   Compute Spilled Energy: $\Delta E_{\theta}(x_{t:1}) = -E_{\theta}^{m}(x_{t:1}) + E_{\theta}^{\ell}(x_{t:1})$.
5.  **Pooling:** Apply a pooling strategy (e.g., Min, Max) across the interval $[u,w]$ to obtain a final hallucination score.
6.  **Detection:** If the pooled score exceeds a threshold (or based on distribution separability), classify as hallucination/incorrect.