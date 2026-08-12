1. **Research Background and Existing Pain Points**

**Research Background:** Large language models (LLMs) are increasingly adopted in real-world applications, but their safety is compromised by jailbreak attacks. By feeding jailbreak prompts, which are specially constructed harmful instructions, one can "jailbreak" safety-aligned targeted LLMs to induce harmful behaviors. To ensure the robustness of LLMs against jailbreak attacks, adversarial training (AT) is one of the most effective defenses, which trains LLMs on synthesized jailbreak prompts to help them better recognize and refuse harmful inputs. To improve the efficiency of AT for LLMs, recent studies introduce continuous AT (CAT), which performs AT on LLMs with adversarial inputs synthesized in the LLMs' continuous token embedding space. Compared with vanilla AT, CAT can use projected gradient descent (PGD) to search for adversarial examples in the embedding space, which is significantly faster than vanilla AT where a heuristic search is required in the input token space.

**Existing Pain Points:** 
1. The computational cost of vanilla LLM AT is high. The synthesis of jailbreak prompts during AT usually requires solving discrete optimization problems and is thus computation-consuming, which restricts the use of AT for LLMs in practice.
2. The theoretical mechanism behind CAT remains unknown. Despite the empirical success of CAT, the reason behind it is still unknown. The training data in CAT and vanilla LLM AT are very different: data in CAT are sequences of embedding vectors, while data in vanilla AT are sequences of token indices. It is unclear why adversarial perturbations in the embedding space can help LLMs learn to defend against jailbreak prompts from the original input token space.

2. **Core Research Motivation and Scientific Questions**

**Core Research Motivation:** In light of recent advances in understanding LLM jailbreak robustness via in-context learning (ICL) theory, this paper aims to present the first theoretical analysis of LLM CAT based on ICL theory. By rigorously studying how AT helps improve the robustness of linear transformers trained on linear regression tasks against in-context suffix adversarial attacks, the paper seeks to explain the empirical success of CAT and further propose an improved method based on theoretical insights.

**Scientific Questions:** Why can adversarial perturbations in the embedding space help LLMs learn to defend against jailbreak prompts from the original input token space? 

3. **Overall Core Idea and Design Philosophy**

**Overall Core Idea:** To address the research question, the paper establishes a new ICL embedding AT problem under ICL theory to approximate real-world LLM CAT. A novel Linear Self-Attention with Embedding (LSA-E) model is designed to incorporate the embedding process. By deriving a surrogate upper bound for the original AT loss and calculating its closed-form solution, a robust generalization upper bound for linear transformers trained via ICL embedding AT is proved. This bound reveals a negative correlation with the embedding-space perturbation radius and a close relationship with the singular values of the embedding matrix. Based on this finding, an improved method called Embedding-Regularized Continuous AT (ER-CAT) is proposed, which introduces the variance of the embedding matrix singular values as an additional regularization term into the objective function of the original CAT.

**Design Philosophy:** The design philosophy is to bridge the gap between theoretical ICL analysis and real-world LLM CAT by showing that LSA-E models and real-world LLMs share similar embedding processes and training goals. By proving that the robustness depends on "not too large nor too small singular values" of the embedding matrix, the philosophy is to explicitly regularize the variance of singular values during AT to simultaneously reduce large singular values and increase small singular values, thereby improving the robustness-utility tradeoff.

4. **Core Innovation Points**

1. **First Theoretical Analysis of CAT based on ICL Theory:** This paper presents the first theoretical analysis of continuous adversarial training for LLMs based on in-context learning theory, bridging the gap between empirical CAT success and theoretical understanding.
2. **Construction of LSA-E Model and ICL Embedding AT Framework:** A novel Linear Self-Attention with Embedding (LSA-E) model is designed to approximate real-world CAT, and an ICL embedding AT problem is formalized to simulate the embedding space adversarial perturbation process.
3. **Proof of Robust Generalization Upper Bound:** A robust generalization upper bound is proved for linear transformers trained via ICL embedding AT. The bound has a negative correlation with the embedding space adversarial perturbation radius, which clearly explains why CAT can defend against jailbreak prompts from the token space.
4. **Discovery of the Role of Embedding Matrix Singular Values:** The robust bound shows that the robustness of an adversarially trained LLM is closely related to the singular values of its embedding matrix, suggesting that an embedding matrix with "not too large nor too small singular values" enjoys strong adversarial robustness.
5. **Proposal of ER-CAT Method:** Based on the theoretical finding, Embedding Regularized Continuous AT (ER-CAT) is proposed by introducing the variance of the embedding matrix singular values as an additional regularization term into the objective function of CAT, achieving a better jailbreak robustness-utility tradeoff.

5. **Overview of the Overall Technical Solution**

The overall technical solution consists of three phases: Theoretical Framework Construction, Theoretical Derivation and Bound Proof, and Method Improvement.
1. **Theoretical Framework Construction:** Define the ICL linear regression setup, design the LSA-E model with a trainable embedding matrix, formulate the ICL embedding adversarial examples, and define the ICL embedding AT minimax optimization problem. Define the robust generalization risk based on ICL suffix adversarial attacks.
2. **Theoretical Derivation and Bound Proof:** Derive a closed-form surrogate upper bound for the original ICL embedding AT loss. Under specific initialization assumptions, calculate the closed-form optimal solution for the surrogate problem. Prove a robust generalization upper bound for the optimal model, analyzing its correlation with the perturbation radius and embedding matrix singular values.
3. **Method Improvement:** Based on the insight that singular values should not be too large or too small, design the ER-CAT objective function by adding a singular value variance regularization term to the original CAT loss. Implement this on real-world LLMs using LoRA and PGD.

6. **Detailed Module Design**

**Module 1: LSA-E Model and ICL Embedding AT Formulation**
*   **Embedding Function:** An embedding function $E(\cdot)$ maps any ICL input $Z_\tau \in \mathbb{R}^{(d_0+1) \times (N+1)}$ to its ICL embedding matrix.
*   **LSA-E Model:** The model $f_{LSAE,\theta}$ is formalized using the embedding function, fused key-query matrix $W_{KQ}$, and value projection matrix $W_V$. The prediction for the query point is the right-bottom entry of the model output.
*   **ICL Embedding Adversarial Example:** Defined by adding adversarial perturbations $\Delta^E_\tau$ to the embeddings of the $N$ in-context training points.
*   **ICL Embedding AT:** Formalized as a minimax optimization problem minimizing the expected squared error between the adversarial prediction and the true label, where the inner maximization searches for perturbations within an $\epsilon$-ball.
*   **Robust Generalization Risk:** Uses ICL suffix adversarial attacks (perturbing the last $M$ points in the original input space with radius $\rho$) to define the robust risk $R_{\rho,M}^{adv}(\theta)$.

**Module 2: Surrogate ICL Embedding AT and Closed-form Solution**
*   **Surrogate Objective:** Since the original AT loss is difficult to tackle in a closed-form manner, a surrogate objective function $\tilde{L}_{LSAE}^{adv}(\theta)$ is derived as a closed-form upper bound, consisting of four terms $\ell_1(\theta) + \ell_2(\theta) + \ell_3(\theta) + \ell_4(\theta)$.
*   **Initialization Assumption:** Assumes specific block matrix structures for $W_V(0)$ and $W_{KQ}(0)$ with parameter $\zeta > 0$ and matrix $\Theta$, zeroing out non-contributing terms and ensuring symmetric initialization.
*   **Optimal Solution:** Under the initialization assumption and continuous gradient flow, certain parameters stay zero. The global minimizer is derived where the product of optimal $w^*_{V,22}$, $(W_E^*)^\top$, $W^*_{KQ,11}$, and $W_E^*$ equals $(W_E^*)^\top(W_E^* \Gamma_N \Lambda (W_E^*)^\top + \text{Tr}(\Lambda)\epsilon^2 I_d)^{-1} W_E^* \Lambda$.

**Module 3: Robust Generalization Bound and Implications**
*   **Bound Derivation:** By inserting the optimal prediction function into the robust risk and applying inequalities, the robust risk $R_{\rho,M}^{adv}(\theta^*)$ is bounded. The bound shows a negative correlation with $\epsilon$ and depends on the sum of the 4th power of singular values of $W_E^*$ divided by the 4th power of the minimum singular value.
*   **Implication:** An embedding matrix with "not too large nor too small singular values" reduces the bound. Minimizing the variance of singular values helps achieve this.

**Module 4: ER-CAT Implementation**
*   **Regularization Term:** Introduces the variance of all singular values of the LLM embedding matrix, calculated as $\sum_{i=1}^d [\sigma_i(W_E) - \bar{\sigma}(W_E)]^2 / d$.
*   **Optimization:** Solves the ER-CAT minimization problem using AdamW. Applies LoRA to the embedding layer and query/key projection matrices. Uses PGD with a fixed perturbation radius $\epsilon=0.05$, step size $\eta$, and update number $K=10$ to search for embedding space adversarial perturbations. Applies a loss cut-off technique to prevent over-optimizing.

7. **All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
*   $x \in \mathcal{V}^{|x|}$: Input prompt of length $|x|$, where $\mathcal{V} := \{1, \cdots, |\mathcal{V}|\}$ is the token space.
*   $p_\theta$: Parameterized distribution of LLM.
*   $x^{(h)}, y^{(h)}$: Harmful instruction and targeted harmful response.
*   $x^{(s)}$: Synthesized adversarial suffix.
*   $\mathcal{A}$: Attack oracle for prompt-level attacks.
*   $\mathcal{D}^{(h)}$: Safety dataset with samples $(x, y, \tilde{y})$.
*   $\mathcal{D}^{(u)}$: Utility dataset with samples $(x, y)$.
*   $E(\cdot)$: Embedding function of LLM.
*   $W_E \in \mathbb{R}^{d \times |\mathcal{V}|}$: Embedding matrix.
*   $\delta^*$: Most adversarial perturbation in embedding space.
*   $\epsilon > 0$: Embedding space perturbation radius.
*   $\alpha > 0$: Weight parameter in CAT.
*   $\tau$: Task index in ICL.
*   $w_\tau \sim \mathcal{N}(0, I_{d_0})$: Task weight.
*   $\Lambda \in \mathbb{R}^{d_0 \times d_0}$: Covariance matrix for ICL inputs.
*   $Z_\tau$: ICL input sequence.
*   $N$: Number of in-context training samples.
*   $W_{KQ} \in \mathbb{R}^{(d+1) \times (d+1)}$: Fused key and query projection matrix.
*   $W_V \in \mathbb{R}^{(d+1) \times (d+1)}$: Value projection matrix.
*   $\theta := (W_E, W_{KQ}, W_V)$: Trainable parameters of LSA-E.
*   $\Delta^E_\tau$: Adversarial perturbations added to embeddings.
*   $M$: Number of suffix points perturbed in robust risk evaluation.
*   $\rho > 0$: Adversarial perturbation radius for suffix attack.
*   $\beta > 0$: Coefficient for the regularization term in ER-CAT.
*   $\sigma_i(W_E)$: $i$-th singular value of $W_E$.
*   $\bar{\sigma}(W_E) := \frac{1}{d} \sum_{i=1}^d \sigma_i(W_E)$: Mean of singular values.

**Mathematical Formulas:**
1. Token-level attack synthesis:
$$ \min_{x^{(s)} \in \mathcal{V}^{|x^{(s)}|}} - \log p_\theta(y^{(h)}|x^{(h)} \oplus x^{(s)}) $$
2. Prompt-level attack synthesis:
$$ \min_{\hat{x}^{(h)} \sim \mathcal{A}(x^{(h)})} - \log p_\theta(y^{(h)}|\hat{x}^{(h)}) $$
3. Vanilla LLM AT optimization problem:
$$ \min_\theta \left\{ - \mathbb{E}_{(x,y,\tilde{y}) \in \mathcal{D}^{(h)}} \left[ \log p_\theta(\tilde{y}|\hat{x}^*) + \log(1 - p_\theta(y|\hat{x}^*)) \right] - \mathbb{E}_{(x,y) \in \mathcal{D}^{(u)}} \log p_\theta(y|x) \right\} $$
4. Continuous AT (CAT) optimization problem:
$$ \min_\theta \left\{ -\alpha \cdot \mathbb{E}_{(x,y,\tilde{y}) \in \mathcal{D}^{(h)}} \log \frac{p_\theta(\tilde{y}|E(x) + \delta^*)}{p_\theta(y|E(x) + \delta^*)} - \mathbb{E}_{(x,y) \in \mathcal{D}^{(u)}} \log p_\theta(y|x) \right\} $$
5. ICL input $Z_\tau$:
$$ Z_\tau := \begin{pmatrix} x_{\tau,1} & \cdots & x_{\tau,N} & x_{\tau,q} \\ y_{\tau,1} & \cdots & y_{\tau,N} & 0 \end{pmatrix} \in \mathbb{R}^{(d_0+1) \times (N+1)} $$
6. Embedding function $E(Z_\tau)$:
$$ E(Z_\tau) := \begin{pmatrix} W_E x_{\tau,1} & \cdots & W_E x_{\tau,N} & W_E x_{\tau,q} \\ y_{\tau,1} & \cdots & y_{\tau,N} & 0 \end{pmatrix} \in \mathbb{R}^{(d+1) \times (N+1)} $$
7. LSA-E model prediction $ŷ_{q,\theta}(Z_\tau)$:
$$ \hat{y}_{q,\theta}(Z_\tau) := \begin{pmatrix} (w_{21}^V)^\top & w_{22}^V \end{pmatrix} \frac{E(Z_\tau) E(Z_\tau)^\top}{N} \begin{pmatrix} W_{11}^{KQ} \\ (w_{21}^{KQ})^\top \end{pmatrix} W_E x_{\tau,q} $$
8. ICL embedding adversarial example $E_{adv}(Z_\tau, \Delta^E_\tau)$:
$$ E_{adv}(Z_\tau, \Delta^E_\tau) := \begin{pmatrix} W_E x_{\tau,1} + \delta_{\tau,1}^E & \cdots & W_E x_{\tau,N} + \delta_{\tau,N}^E & W_E x_{\tau,q} \\ y_{\tau,1} & \cdots & y_{\tau,N} & 0 \end{pmatrix} \in \mathbb{R}^{(d+1) \times (N+1)} $$
9. Adversarial prediction $\hat{y}_{q,\theta}^{adv}(Z_\tau, \Delta^E_\tau)$:
$$ \hat{y}_{q,\theta}^{adv}(Z_\tau, \Delta^E_\tau) := \begin{pmatrix} (W_{21}^V)^\top & w_{22}^V \end{pmatrix} \frac{E_{adv}(Z_\tau, \Delta^E_\tau) E_{adv}(Z_\tau, \Delta^E_\tau)^\top}{N} \begin{pmatrix} W_{11}^{KQ} \\ (W_{21}^{KQ})^\top \end{pmatrix} W_E x_{\tau,q} $$
10. ICL embedding AT minimax problem:
$$ \min_\theta \mathcal{L}_{LSAE}^{adv}(\theta) := \min_\theta \left\{ \mathbb{E}_\tau \max_{\|(\Delta^E_\tau)^\top\|_{2,\infty} \leq \epsilon} \frac{1}{2} |\hat{y}_{q,\theta}^{adv}(Z_\tau, \Delta^E_\tau) - y_{\tau,q}|^2 \right\} $$
11. ICL suffix adversarial example $Z_{\tau,M}^{adv}$:
$$ Z_{\tau,M}^{adv} := \begin{pmatrix} x_{\tau,1} & \cdots & x_{\tau,N-M} & x_{\tau,N-M+1} + \delta_{\tau,1}^O & \cdots & x_{\tau,N} + \delta_{\tau,M}^O & x_{\tau,q} \\ y_{\tau,1} & \cdots & y_{\tau,N-M} & y_{\tau,N-M+1} & \cdots & y_{\tau,N} & 0 \end{pmatrix} $$
12. Robust generalization risk $R_{\rho,M}^{adv}(\theta)$:
$$ R_{\rho,M}^{adv}(\theta) = \mathbb{E}_\tau \max_{\|(\Delta^O_\tau)^\top\|_{2,\infty} \leq \rho} \frac{1}{2} |\hat{y}_{q,\theta}(Z_{\tau,M}^{adv}) - y_{\tau,q}|^2 $$
13. Surrogate ICL embedding AT objective:
$$ \min_\theta \tilde{\mathcal{L}}_{LSAE}^{adv}(\theta) = \min_\theta \left\{ \ell_1(\theta) + \ell_2(\theta) + \ell_3(\theta) + \ell_4(\theta) \right\} $$
Where:
$$ \ell_1(\theta) := 2 \cdot \mathbb{E}_\tau \left| \begin{pmatrix} (w_{21}^V)^\top & w_{22}^V \end{pmatrix} \frac{E(Z_\tau) E(Z_\tau)^\top}{N} \begin{pmatrix} W_{11}^{KQ} \\ (w_{21}^{KQ})^\top \end{pmatrix} W_E x_{\tau,q} - y_{\tau,q} \right|^2 $$
$$ \ell_2(\theta) := \frac{2\epsilon^2}{N} \cdot \|w_{21}^V\|_2^2 \cdot \mathbb{E}_\tau \left[ \left\| \begin{pmatrix} W_E X_\tau \\ Y_\tau \end{pmatrix}^\top \begin{pmatrix} W_{11}^{KQ} \\ (w_{21}^{KQ})^\top \end{pmatrix} W_E x_{\tau,q} \right\|_2^2 \right] $$
$$ \ell_3(\theta) := \frac{2\epsilon^2}{N} \cdot \mathbb{E}_\tau \left[ \left\| \begin{pmatrix} (w_{21}^V)^\top & w_{22}^V \end{pmatrix} \begin{pmatrix} W_E X_\tau \\ Y_\tau \end{pmatrix} \right\|_2^2 \right] \cdot \mathbb{E}_\tau \left[ \|W_{11}^{KQ} W_E x_{\tau,q}\|_2^2 \right] $$
$$ \ell_4(\theta) := 2\epsilon^4 \cdot \|w_{21}^V\|_2^2 \cdot \mathbb{E}_\tau \left[ \|W_{11}^{KQ} W_E x_{\tau,q}\|_2^2 \right] $$
14. Assumption 1 (Initialization):
$$ W_V(0) = \begin{pmatrix} 0_{d \times d} & 0_{d \times 1} \\ 0_{1 \times d} & \zeta \end{pmatrix}, \quad W_{KQ}(0) = \begin{pmatrix} \zeta \Theta \Theta^\top & 0_{d \times 1} \\ 0_{1 \times d} & 0 \end{pmatrix} $$
15. Theorem 1 (Optimal solution of surrogate ICL embedding AT):
$$ w_{22}^{V*} (W_E^*)^\top W_{11}^{KQ*} W_E^* = (W_E^*)^\top \left( W_E^* \Gamma_N \Lambda (W_E^*)^\top + \text{Tr}(\Lambda)\epsilon^2 I_d \right)^{-1} W_E^* \Lambda $$
16. Optimal prediction function:
$$ \hat{y}_{q,\theta^*}(Z_\tau) := \frac{1}{N} Y_\tau (X_\tau)^\top (W_E^*)^\top \left( W_E^* \Gamma_N \Lambda (W_E^*)^\top + \text{Tr}(\Lambda)\epsilon^2 I_d \right)^{-1} W_E^* \Lambda x_{\tau,q} $$
17. Theorem 2 (Robust generalization upper bound):
$$ R_{\rho,M}^{adv}(\theta^*) \leq \mathcal{O} \left( (1 + M\rho^2/N^2) \cdot \frac{\sum_{i=1}^d \sigma_i(W_E^*)^4}{\sigma_{\min}(W_E^*)^4 + \epsilon^4} \right) + \mathcal{O}(1) $$
18. ER-CAT optimization problem:
$$ \min_\theta \mathcal{L}^{\text{ER-CAT}}(\theta, \alpha, \beta) := \min_\theta \left\{ \mathcal{L}^{\text{CAT}}(\theta, \alpha) + \beta \cdot \frac{\sum_{i=1}^d [\sigma_i(W_E) - \bar{\sigma}(W_E)]^2}{d} \right\} $$

8. **Algorithm Pseudocode**

The paper describes the Projected Gradient Descent (PGD) update for searching embedding space adversarial perturbations in Appendix C.1. The iterative process is formalized as follows:

Initialize $\delta^{(0)} := 0_{d \times |x|}$.
For $k = 1, \dots, K$:
    For $i = 1, \dots, |x|$:
        $$ \delta_i^{(k)} = \prod_{\|\delta_i\|_2 \leq \epsilon} \left[ \delta_i^{(k-1)} + \eta \cdot \frac{\partial_{\delta_i} \log p_\theta(y|E(x) + \delta^{(k-1)})}{\|\partial_{\delta_i} \log p_\theta(y|E(x) + \delta^{(k-1)})\|_2} \right] $$
Return $\delta^* := \delta^{(K)}$