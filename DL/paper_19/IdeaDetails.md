1. Research Background and Existing Pain Points
Transformers are a dominant architecture in modern machine learning, powering applications across vision, language, and beyond. At the core of their success lies the attention layer, where the query, key, and value matrices determine how token dependencies are captured. While considerable work has focused on scaling and optimizing Transformers, comparatively little attention has been paid to how the weights of the queries, keys and values are initialized. 

Currently, Transformers typically adopt simple random initializations (such as normal or truncated normal distributions), without consideration of the unique structure of attention layers. Recent alternatives such as mimetic initialization, which imitates weight patterns from converged models, and weight selection, which transfers weights from a teacher model, highlight a growing recognition that initialization matters. However, the existing pain points are that these methods remain heuristic and lack a principled connection to the conditioning of the attention mechanism itself. Poor initialization can introduce an optimization bias that fundamentally shapes training dynamics negatively, potentially causing unstable optimization and poor generalization due to the ill-conditioning of the attention layer's Jacobian.

2. Core Research Motivation and Scientific Questions
The core research motivation stems from the fact that initialization plays a critical role in shaping the optimization landscapes of deep neural networks. Classical schemes such as Xavier and Kaiming initialization showed that carefully chosen scaling of weights at the start of training can dramatically improve gradient optimization and stability in deep networks by preventing vanishing or exploding gradients. Given that self-attention relies on query, key, and value projections whose interplay directly governs the stability of token interactions, it is natural to ask whether initialization schemes tailored specifically to attention could offer similar benefits.

The scientific questions addressed are: How does the initialization of the query, key, and value matrices affect the conditioning of the self-attention Jacobian? Can we design a principled initialization scheme that explicitly targets the spectral properties of these matrices to improve the conditioning of the attention mechanism? Furthermore, does improving the conditioning of the attention Jacobian at initialization lead to more stable optimization, accelerated convergence, and improved generalization?

3. Overall Core Idea and Design Philosophy
The overall core idea is to revisit Transformer initialization from a theoretical perspective by establishing a connection between the conditioning of self-attention Jacobians and the spectral structure of the query, key, and value matrices. The design philosophy is to intervene at initialization—rather than directly modifying training objectives—to provide an inductive bias that promotes more stable optimization dynamics. 

By analyzing the Jacobian of the self-attention layer with respect to its parameters, it is shown that its condition number is bounded by the condition numbers of the query, key, and value weight matrices. Therefore, the core idea is "conditioned initialization": a principled scheme designed to improve the spectral conditioning of attention blocks at the start of training. This is achieved by initializing the weight matrices such that their condition numbers are minimized (i.e., equal to 1), specifically using rectangular identity matrices and semi-orthogonal matrices, thereby tightening the upper bound on the attention Jacobian's condition number and biasing training toward stable optimization.

4. Core Innovation Points
1. Theoretical Framework: Establishment of a rigorous theoretical connection between the conditioning of self-attention Jacobians and the spectral structure (condition numbers) of the query, key, and value matrices. This motivates initialization schemes that explicitly target this property.
2. Conditioned Initialization: Proposal of a simple yet principled initialization method that reduces the upper bound on the attention Jacobian’s condition number, thereby biasing training toward stable optimization. This addresses the heuristic nature of prior methods like mimetic initialization and weight selection.
3. Differentiated Matrix Initialization Strategy: Recognition that WQ, WK, and WV play distinct algebraic roles in attention, justifying different treatments. Specifically, WV is initialized with a rectangular identity to preserve scale and avoid Jacobian distortion, while WQ and WK are initialized as semi-orthogonal projections to provide near-isometric embeddings, preventing the anisotropic logits and unstable softmax dynamics that would arise from identity initialization.
4. Broad Applicability and Empirical Validation: Demonstration that conditioned initialization is simple to apply, integrates seamlessly into a wide range of Transformer architectures (including those with normalization like QKNorm and RMSNorm), and consistently accelerates convergence and improves generalization across diverse applications spanning vision, language, and long-range sequence learning.

5. Overview of the Overall Technical Solution
The overall technical solution begins by analyzing the self-attention map $A(X)$ and computing its Jacobian $J(A(X))$ with respect to the parameter matrices $W_Q, W_K, W_V$. Through matrix calculus and properties of the softmax function, the partial derivatives of the attention output with respect to each weight matrix are derived. Using these derivatives, an upper bound for the condition number of the attention Jacobian is established.

This upper bound reveals a direct dependency on the condition numbers of the input matrix $X$, the value matrix $W_V$, the query matrix $W_Q$, the key matrix $W_K$, and terms related to the softmax operation. Since $W_Q, W_K, W_V$ are the only components directly controlled at initialization, the solution proposes minimizing their condition numbers to tighten the bound.

Matrices with a condition number of 1 are either scalar multiples of the identity or semi-orthogonal matrices. The solution implements $W_V$ as a rectangular identity matrix and $W_Q, W_K$ as independent semi-orthogonal projections for each attention head. The semi-orthogonal matrices are constructed via the truncated Singular Value Decomposition (SVD) of random matrices. This initialization improves the spectral properties of the attention layer, leading to a better-conditioned Jacobian and thus more stable optimization dynamics as supported by Neural Tangent Kernel (NTK) theory.

6. Detailed Module Design
1. Self-Attention Layer Module: The standard self-attention mechanism takes an input sequence $X \in \mathbb{R}^{N \times D}$ and applies three learnable matrices: query $W_Q \in \mathbb{R}^{D \times d}$, key $W_K \in \mathbb{R}^{D \times d}$, and value $W_V \in \mathbb{R}^{D \times d}$. The output is defined by $A(X) = \text{softmax}(XW_QW_K^TX^T)XW_V$. Multiple heads are concatenated to form the multi-head attention layer.

2. Value Matrix Initialization Module ($W_V$): The value map enters linearly into the output $(\text{softmax}(XW_QW_K^TX^T))(XW_V)$. Initializing it with the rectangular identity $I_{D \times d}$ preserves the scale of the input representations ($XW_V = X$), keeps $\kappa(W_V) = 1$, and avoids unnecessary distortion of the Jacobian.

3. Query and Key Matrix Initialization Module ($W_Q, W_K$): The query and key maps interact bilinearly through $S = XW_QW_K^TX^T$. Initializing them as rectangular identities can bias projections toward coordinate subspaces, yielding anisotropic logits and unstable softmax dynamics. Instead, a semi-orthogonal initialization for $W_Q$ and $W_K$ provides near-isometric embeddings, giving each head balanced representations of $X$ and supporting more diverse and stable attention patterns. For the $i$-th head, independent semi-orthogonal projections are used: $(W_Q^{(i)})^TW_Q^{(i)} = I_d$, $(W_K^{(i)})^TW_K^{(i)} = I_d$. This produces near-isometric embeddings into distinct subspaces, diversifying the logits $S^{(i)} = (XW_Q^{(i)})(XW_K^{(i)})^T$ and attention patterns.

4. Normalization Compatibility Module: The conditioned initialization operates on top of existing normalization layers (like RMSNorm, QKNorm, Layer Normalization) rather than replacing them. Normalization controls the scale of activations and gradients, whereas conditioning concerns the spectral structure of the Jacobian. These mechanisms are distinct and complementary. The fixed scaling factor $1/\sqrt{d}$ in attention does not depend on model parameters and simply scales the Jacobian by a constant, which does not change the condition number.

7. All Mathematical Formulas and Symbol Definitions
Symbol Definitions:
- $X \in \mathbb{R}^{N \times D}$: Input sequence.
- $W_Q \in \mathbb{R}^{D \times d}$: Query weight matrix.
- $W_K \in \mathbb{R}^{D \times d}$: Key weight matrix.
- $W_V \in \mathbb{R}^{D \times d}$: Value weight matrix.
- $A(X)$: Self-attention output.
- $\text{softmax}$: Softmax activation acting row-wise.
- $h$: Number of attention heads.
- $J(A(X))$: Jacobian of $A(X)$ with respect to parameters $W_Q, W_K, W_V$.
- $\text{vec}(Z) \in \mathbb{R}^{mn \times 1}$: Vectorization of matrix $Z \in \mathbb{R}^{m \times n}$.
- $T_{mn} \in \mathbb{R}^{mn \times mn}$: Commutation matrix such that $T_{mn}\text{vec}(Z) = \text{vec}(Z^T)$.
- $\sigma_{max}(Z)$: Maximum singular value of matrix $Z$.
- $\sigma_{min}(Z)$: Minimum singular value of matrix $Z$.
- $\kappa(Z)$: Condition number of matrix $Z$, defined as $\kappa(Z) = \frac{\sigma_{max}(Z)}{\sigma_{min}(Z)}$.
- $I_{m \times n}$: Rectangular identity matrix with 1's on main diagonal.
- $O_{m \times n}$: Real $m \times n$ semi-orthogonal matrices.

Mathematical Formulas and Derivations:
Self-Attention Definition:
$$A(X) = \text{softmax}(XW_QW_K^TX^T)XW_V$$

Jacobian Definition:
$$J(A(X)) = \left[\frac{\partial A(X)}{\partial W_Q}, \frac{\partial A(X)}{\partial W_K}, \frac{\partial A(X)}{\partial W_V}\right]^T$$

Lemma 3.1 (Softmax Derivative): Let $\Lambda : \mathbb{R}^n \to \mathbb{R}^{n \times n}$ denote the function $\Lambda(z) = \text{Diag}(z) - z \cdot z^T$. Then:
$$\frac{\partial \text{softmax}}{\partial x}(z) = \Lambda(\text{softmax}(z))$$

Lemma A.1 (Matrix Derivative): Let $A \in \mathbb{R}^{n \times m}, B \in \mathbb{R}^{k \times l}, C \in \mathbb{R}^{m \times k}$. Then:
$$\frac{\partial ACB}{\partial C} = B^T \otimes A$$

Lemma A.2 (Transpose Derivative): Let $A \in \mathbb{R}^{n \times m}$. Then:
$$\text{vec}(A^T) = T_{mn}\text{vec}(A)$$
$$\frac{\partial \text{vec}(A^T)}{\partial \text{vec}(A)} = T_{mn}$$

Proposition 3.1 (Attention Derivatives):
$$\frac{\partial A(X)}{\partial W_Q} = (W_V^TX^T \otimes I_N)(\Lambda(\text{softmax}(XW_QW_K^TX^T)))(XW_K \otimes X)$$
$$\frac{\partial A(X)}{\partial W_K} = (W_V^TX^T \otimes I_N)(\Lambda(\text{softmax}(XW_QW_K^TX^T)))(X \otimes XW_Q) \cdot T_{Dd}$$
$$\frac{\partial A(X)}{\partial W_V} = I_d \otimes \text{softmax}(XW_QW_K^TX^T)X$$

Theorem 3.1 (Condition Number Bound): Let $A(X)$ be a self-attention matrix with input $X$ and let $J(A(X))$ denote its Jacobian. Assume $J(A(X))$ has full rank so $\kappa(J(A(X)))$ is finite. Then:
$$\kappa(J(A(X))) \le \kappa(X)^3 \kappa(\Lambda(\text{softmax}(XW_QW_K^TX^T))) \kappa(W_V)(\kappa(W_Q) + \kappa(W_K)) + \kappa(X)\kappa(\text{softmax}(XW_QW_K^TX^T))$$

Surrogate Upper Bound $B(J(A))$:
$$B(J(A)) := \kappa(X)^3 \kappa(\Lambda(\text{softmax}(XW_QW_K^TX^T))) \kappa(W_V)(\kappa(W_Q) + \kappa(W_K)) + \kappa(X)\kappa(\text{softmax}(XW_QW_K^TX^T))$$

Proposition 3.2 (Bound Improvement): Let $A(X)$ denote an attention matrix with $W_Q, W_K, W_V$ initialized from a Gaussian or uniform distribution, and let $\bar{A}(X)$ denote one where $W_Q, W_K, W_V$ are initialized from either $\{\lambda I_{D \times d}\}$ or $O_{D \times d}$. Then:
$$B(J(\bar{A})) \le B(J(A))$$

Conditioned Initialization Constraints:
For Value Matrix:
$$W_V = I_{D \times d}$$
For Query and Key Matrices per head $i = 1, \dots, h$:
$$(W_Q^{(i)})^TW_Q^{(i)} = I_d$$
$$(W_K^{(i)})^TW_K^{(i)} = I_d$$

SVD Implementation Details:
Let $r = \min(D, d)$. For each head $i$, form random matrices $R_{Q_i}, R_{K_i} \in \mathbb{R}^{D \times d}$ and take the truncated SVD:
$$U_{Q_i}^{(r)} S_{Q_i}^{(r)} (V_{Q_i}^T)^{(r)} \text{ and } U_{K_i}^{(r)} S_{K_i}^{(r)} (V_{K_i}^T)^{(r)}$$
where $U_{Q_i}^{(r)} \in \mathbb{R}^{D \times r}$ and $(V_{Q_i})^{(r)} \in \mathbb{R}^{r \times d}$. The semi-orthogonal initializations are:
$$O_{Q_i} := U_{Q_i}^{(r)} \cdot (V_{Q_i}^{(r)})^T$$
$$O_{K_i} := U_{K_i}^{(r)} \cdot (V_{K_i}^{(r)})^T$$

8. Algorithm Pseudocode
Algorithm 1: Conditioned Initialization for Attention
Input: Input dimension D, Head dimension d, Number of heads h
Output: Initialized weight matrices $W_Q, W_K, W_V$ for all heads

1: for $i = 1$ to $h$ do
2:   // Initialize Value Matrix
3:   $W_V^{(i)} \leftarrow I_{D \times d}$  // Rectangular identity matrix
4: 
5:   // Initialize Query Matrix via Semi-Orthogonal Projection
6:   Generate random matrix $R_{Q_i} \in \mathbb{R}^{D \times d}$
7:   $r \leftarrow \min(D, d)$
8:   Compute SVD: $R_{Q_i} = U S V^T$
9:   Truncate SVD to rank r: $U_{Q_i}^{(r)} \in \mathbb{R}^{D \times r}, V_{Q_i}^{(r)} \in \mathbb{R}^{r \times d}$
10:  $W_Q^{(i)} \leftarrow U_{Q_i}^{(r)} \cdot (V_{Q_i}^{(r)})^T$
11: 
12:  // Initialize Key Matrix via Semi-Orthogonal Projection
13:  Generate random matrix $R_{K_i} \in \mathbb{R}^{D \times d}$
14:  Compute SVD: $R_{K_i} = U S V^T$
15:  Truncate SVD to rank r: $U_{K_i}^{(r)} \in \mathbb{R}^{D \times r}, V_{K_i}^{(r)} \in \mathbb{R}^{r \times d}$
16:  $W_K^{(i)} \leftarrow U_{K_i}^{(r)} \cdot (V_{K_i}^{(r)})^T$
17: end for
18: Return $\{W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}\}_{i=1}^h$