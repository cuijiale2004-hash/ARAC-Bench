## 1. Research Background and Existing Pain Points

**Research Background:**
Transformers have rapidly become a cornerstone in modern machine learning, demonstrating exceptional performance and versatility across diverse applications, including natural language processing, computer vision, and reinforcement learning. As a critical component, self-attention layers assign varying weights to features based on their relevance and embedded positional context, intuitively endowing transformers with the ability to efficiently process structural and positional information.

**Existing Pain Points:**
Despite their profound empirical impact, the theoretical foundations of transformers, especially the mechanisms of how self-attention layers work, remain largely unexplored due to their intricate architecture. Recent theoretical studies have aimed to understand transformers by analyzing their capability in solving specific tasks, such as in-context linear regression (Zhang et al., 2024b), in-context linear classification (Frei & Vardi, 2025), patch association (Jelassi et al., 2022), sparse token selection (Wang et al., 2024), and group sparse linear models (Zhang et al., 2025c). However, these works focus on very specific learning tasks, which limits the generality of their theoretical findings. This specificity prompts the need to seek a unified theoretical framework accounting for a broader range of examples and models.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The authors observe that despite the distinctions among the model simplifications and technical assumptions in previous works, for some learning tasks discussed—including a variant of the sparse token selection, the group sparse linear predictors, and patch association—their true responses are essentially given by bilinear functions. In addition, the linear attention studied in prior works inherently constitutes a bilinear structure with respect to its parameter matrices. Motivated by this observation, there is a need to define a general class of "teacher models" that employ a bilinear structure to provide a unified theoretical guarantee.

**Scientific Questions:**
Can one-layer transformers trained via gradient descent provably learn a general class of teacher models employing a bilinear structure? Can a unified theoretical framework be established to guarantee the convergence of parameters and loss for diverse learning tasks, including previously unexplored models such as convolution layers with average pooling and graph convolution layers?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Define a general class of "teacher models" that employ a bilinear structure $f^*(X) = \sigma(V^*XS^*)$, and investigate the setting where one-layer transformers are trained as "student" models under the supervision from these teacher models. The framework encompasses learning tasks from prior works and covers popular, previously unexplored models such as convolution layers with average pooling and graph convolution layers on regular graphs.

**Design Philosophy:**
The analysis focuses on a one-layer transformer with a moderately simplified "position-only" softmax self-attention. This design is justified by empirical observations that when a typical one-layer transformer is used to learn the teacher model, substantial training predominantly occurs in specific parameter blocks (the left block of $W_V$ and the bottom-right block of $W_{KQ}$), while other blocks exhibit negligible changes. By fixing these rarely updated blocks to zero, the simplified model is essentially equivalent to the typical transformer, which allows for a rigorous optimization analysis focusing on global optimization trajectories under the population loss setting.

## 4. Core Innovation Points

1.  **Unified Theoretical Guarantee for Diverse Tasks:** The paper theoretically demonstrates that one-layer transformers trained via gradient descent can effectively recover a general class of teacher models. The framework covers a wide range of learning tasks, including convolution layers with average pooling, graph convolution layers, sparse token selection models, and group-sparse linear predictors, providing a unified guarantee previously unavailable for such a broad class.
2.  **Tight Convergence Rate with Matching Bounds:** The paper establishes a tight convergence guarantee for the population loss, with matching upper and lower bounds at the rate of $\Theta\left(\frac{1}{T}\right)$. This is sharper than the $O\left(\frac{\log(T)}{T}\right)$ guarantee obtained under different problem formulations in prior work (Wang et al., 2024).
3.  **Provable Parameter Recovery:** The paper proves that the attention scores and the value matrix of the one-layer transformer align with the ground truth softmax scores and value matrix of the teacher model, respectively, demonstrating that the student model can successfully recover all parameter blocks of the teacher.
4.  **Out-of-Distribution Generalization Bounds:** Building upon the efficient mimicry of trained transformers towards teacher models, the paper further demonstrates that they can generalize well to a broad class of out-of-distribution data under mild assumptions, establishing bounds on the OOD loss that show the trained transformer is competitive with the teacher model.
5.  **Identification of Bilinear Structure:** The key innovation in analysis is identifying a fundamental bilinear structure shared by various learning tasks, which enables the establishment of unified learning guarantees for these tasks when treating them as teachers for transformers.

## 5. Overview of the Overall Technical Solution

The technical solution follows a rigorous theoretical analysis of gradient descent dynamics on a simplified transformer architecture learning a bilinear teacher. The overview is as follows:
1.  **Model Definition:** Define the teacher model as $f^*(X) = \sigma(V^*XS^*)$ and the student transformer with position-only attention as $TF(Z;W_V;W_{KQ}) = \sigma(W_VXS(P^\top W_{KQ}P/\sqrt{D}))$.
2.  **Optimization Setup:** Formulate the training objective using the population mean squared error and establish the gradient descent iterative update rules for $W_V$ and $W_{KQ}$.
3.  **Dynamics Analysis (Decomposition):** Prove that throughout training, the parameter matrices preserve specific structural decompositions, reducing the optimization analysis regarding full matrices to studying the updates of three scalars $C_1(t), C_2(t), C_3(t)$.
4.  **Convergence Analysis:** Accurately characterize the convergence of the scalar coefficients showing that $C_1(t) \to 1$ and $C_2(t) + C_3(t) \to \infty$ through a three-phase training process.
5.  **Result Derivation:** Combine the decomposition and scalar convergence rates to derive the final tight convergence results for parameters and excess loss, and extend the analysis to establish OOD generalization bounds.

## 6. Detailed Module Design

### 6.1 Teacher Model Module
The teacher model receives an input matrix $X \in \mathbb{R}^{d \times D}$ and computes the output via a bilinear structure:
$$f^*(X) = \sigma(V^*XS^*)$$
where $V^* \in \mathbb{R}^{M \times d}$ is the ground truth value matrix and $S^* \in \mathbb{R}^{D \times D}$ is the ground truth softmax scores. Each column of $S^*$ has $K$ non-zero entries equivalent to $1/K$. The activation $\sigma(\cdot)$ is either an identity map, ReLU, or Leaky ReLU. This module covers:
*   **Single convolutional layer with average pooling:** $F_{CNN}(X) = \sigma(V^*X[1_{g_1}, \dots, 1_{g_J}]/K)$.
*   **Single graph convolution layer:** $F_{GCN}(X) = \sigma(V^*X\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2})$.
*   **Sparse token selection model:** $F_{STS}(X) = \frac{1}{K}\sum_{i \in g} x_i$.
*   **Group sparse linear predictors:** $F_{GSLP} = \langle v^*, x_{i^*} \rangle$.

### 6.2 Student Model Module
The student model is a one-layer transformer with position-only attention. The input $Z$ is formed by concatenating the feature matrix $X$ and positional encoding matrix $P$. The model computes:
$$TF(Z;W_V;W_{KQ}) = \sigma\left(W_V X S\left(\frac{P^\top W_{KQ} P}{\sqrt{D}}\right)\right) = \sigma(W_V X S)$$
Here, only the positional encoding matrix $P$ is involved in calculating the softmax attention score, and the value matrix $W_V$ only interacts with the feature matrix $X$.

### 6.3 Loss Function and Optimization Module
The observed label is $Y = f^*(X) + E$. The training minimizes the population mean squared error:
$$\mathcal{L}(W_V;W_{KQ}) = \frac{1}{2}\mathbb{E}_{X,Y}\left[\|Y - TF(Z;W_V;W_{KQ})\|_F^2\right]$$
Gradient descent is utilized to derive the optimal solutions:
$$W_V^{(t+1)} = W_V^{(t)} - \eta \nabla_{W_V} \mathcal{L}(W_V^{(t)};W_{KQ}^{(t)})$$
$$W_{KQ}^{(t+1)} = W_{KQ}^{(t)} - \eta \nabla_{W_{KQ}} \mathcal{L}(W_V^{(t)};W_{KQ}^{(t)})$$

### 6.4 Training Dynamics Analysis Module
The proof relies on structural preservation of parameter matrices during training, reducing the analysis to scalar coefficients.
*   **Decomposition:** Throughout training, parameters preserve the forms:
    $$w_{V,m}^{(t)} = C_1(t) \cdot v_m^*$$
    $$W_{KQ}^{(t)} = C_2(t)\sum_{i=1}^D \sum_{i_1 \in G_i} p_{i_1} p_i^\top - C_3(t)\sum_{i=1}^D \sum_{i_1 \notin G_i} p_{i_1} p_i^\top$$
    The attention score $S_{i_1,i}^{(t)}$ is determined by a scalar $p(t) = \frac{1}{K + (D-K)\exp(-(C_2(t)+C_3(t))/\sqrt{D})}$.
*   **Three Phases Training:**
    1.  **Phase 1:** $C_1(t)$ monotonically increases, approaching $C_1^*(t)$ while $p(t)$ remains close to $1/D$.
    2.  **Phase 2:** $C_1(t)$ remains in a neighborhood of $C_1^*(t)$, while $p(t)$ monotonically increases until reaching $1/2K$.
    3.  **Phase 3:** $p(t)$ eventually converges to $1/K$, and $C_1^*(t)$ converges to 1, leading the loss to converge.

## 7. All Mathematical Formulas and Symbol Definitions

*   **$X$**: Input feature matrix $\in \mathbb{R}^{d \times D}$.
*   **$V^*$**: Ground truth value matrix of the teacher model $\in \mathbb{R}^{M \times d}$.
*   **$S^*$**: Ground truth softmax scores of the teacher model $\in \mathbb{R}^{D \times D}$, with each column having $K$ non-zero entries $1/K$.
*   **$\sigma(\cdot)$**: Activation function (Identity, ReLU, or Leaky ReLU).
*   **$f^*(X)$**: Teacher model definition:
    $$f^*(X) = \sigma(V^*XS^*)$$
*   **$Z$**: Input matrix of the transformer, $Z = [X^\top, P^\top]^\top$.
*   **$P$**: Positional encoding matrix $\in \mathbb{R}^{D \times D}$, orthogonal.
*   **$W_V$**: Trainable value matrix of the student transformer.
*   **$W_{KQ}$**: Trainable key-query matrix of the student transformer.
*   **$TF(Z;W_V;W_{KQ})$**: Student model definition:
    $$TF(Z;W_V;W_{KQ}) = \sigma\left(W_V X S\left(\frac{P^\top W_{KQ} P}{\sqrt{D}}\right)\right)$$
*   **$Y$**: Observed label:
    $$Y = f^*(X) + E$$
*   **$E$**: Noise matrix $\in \mathbb{R}^{M \times D}$, zero-mean.
*   **$\mathcal{L}(W_V;W_{KQ})$**: Population mean squared error loss:
    $$\mathcal{L}(W_V;W_{KQ}) = \frac{1}{2}\mathbb{E}_{X,Y}\left[\|Y - TF(Z;W_V;W_{KQ})\|_F^2\right]$$
*   **$\mathcal{L}_{opt}$**: Optimal loss:
    $$\mathcal{L}_{opt} = \frac{1}{2}\mathbb{E}_{X,Y}\left[\|Y - f^*(X)\|_F^2\right] = \frac{1}{2}\mathbb{E}\left[\|E\|_F^2\right]$$
*   **Gradient Descent Updates**:
    $$W_V^{(t+1)} = W_V^{(t)} - \eta \nabla_{W_V} \mathcal{L}(W_V^{(t)};W_{KQ}^{(t)})$$
    $$W_{KQ}^{(t+1)} = W_{KQ}^{(t)} - \eta \nabla_{W_{KQ}} \mathcal{L}(W_V^{(t)};W_{KQ}^{(t)})$$
*   **$G_i$**: Index set of entries of value $1/K$ in $i$-th column of $S^*$.
*   **$C_1(t)$**: Scalar coefficient for $W_V$ decomposition.
*   **$C_2(t)$**: Scalar coefficient for $W_{KQ}$ decomposition (positive part).
*   **$C_3(t)$**: Scalar coefficient for $W_{KQ}$ decomposition (negative part).
*   **$p(t)$**: Time dependent scalar determining attention scores:
    $$p(t) = \frac{1}{K + (D-K)e^{-(C_2(t)+C_3(t))/\sqrt{D}}}$$
*   **Convergence Bounds (Theorem 3.1)**:
    For $D \ge \Omega(\text{poly}(M,K))$, $\eta \le O(M^{-1}D^{-5/2})$, and $T \ge T^* = \Theta\left(\frac{KD^2}{\eta\|V^*\|_F^2}\right)$:
    $$\|S^{(T)} - S^*\|_F = \Theta\left(\frac{D^{5/2}\|V^*\|_F}{\sqrt{\eta T}}\right)$$
    $$\|W_V^{(T)} - V^*\|_F = \Theta\left(\frac{D^2\sqrt{K}}{\sqrt{\eta T}}\right)$$
    $$\frac{cKD^4}{\eta T} \le \mathcal{L}(W_V^{(T)};W_{KQ}^{(T)}) - \mathcal{L}_{opt} \le \frac{\bar{c}KD^4}{\eta T}$$

## 8. Algorithm Pseudocode

```text
Input: 
  Learning rate $\eta$
  Number of iterations $T$
  Initialization $W_V^{(0)} = 0, W_{KQ}^{(0)} = 0$

Process:
1: for t = 0, 1, ..., T-1 do
2:   // Compute Gradient with respect to Value Matrix
3:   $\nabla_{W_V} \mathcal{L} \gets -\sum_{i=1}^D \sum_{i_1=1}^D \mathbb{E}\Big[ \big[ Y_{m,i} - \sigma(\sum_{i_1=1}^D \langle w_{V,m}^{(t)}, x_{i_1}\rangle S_{i_1,i}^{(t)}) \big] \sigma'(\sum_{i_1=1}^D \langle w_{V,m}^{(t)}, x_{i_1}\rangle S_{i_1,i}^{(t)}) x_{i_1} S_{i_1,i}^{(t)} \Big]$
4:   
5:   // Compute Gradient with respect to Key-Query Matrix
6:   $\nabla_{W_{KQ}} \mathcal{L} \gets -\frac{1}{\sqrt{D}} \sum_{m=1}^M \sum_{i=1}^D \mathbb{E}\Big[ \big[ Y_{m,i} - \sigma(\sum_{i_1=1}^D \langle w_{V,m}^{(t)}, x_{i_1}\rangle S_{i_1,i}^{(t)}) \big] \sigma'(\sum_{i_1=1}^D \langle w_{V,m}^{(t)}, x_{i_1}\rangle S_{i_1,i}^{(t)}) \sum_{i_1=1}^D \sum_{i_2=1}^D \langle w_{V,m}^{(t)}, x_{i_1}\rangle S_{i_1,i}^{(t)} S_{i_2,i}^{(t)} (p_{i_1}-p_{i_2})p_i^\top \Big]$
7:   
8:   // Update Parameters
9:   $W_V^{(t+1)} \gets W_V^{(t)} - \eta \nabla_{W_V} \mathcal{L}$
10:  $W_{KQ}^{(t+1)} \gets W_{KQ}^{(t)} - \eta \nabla_{W_{KQ}} \mathcal{L}$
11: end for

Output: 
  Trained parameters $W_V^{(T)}, W_{KQ}^{(T)}$
```