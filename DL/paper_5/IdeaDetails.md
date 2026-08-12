# 1. Research Background and Existing Pain Points

**Research Background:**
Learning divergence-free vector fields is of particular interest at the interface of physics and machine learning, as such fields naturally emerge in systems governed by fundamental conservation laws (e.g., incompressible flows in fluid dynamics, magnetic fields). In image processing, specifically self-supervised denoising, divergence-free estimators are particularly well-suited based on Stein’s Unbiased Risk Estimation (SURE) theory. SURE establishes that the mean square error between an estimator and the clean signal can be reformulated to depend solely on noisy observations and noise levels, involving the divergence of the estimator. This allows training without ground-truth clean data.

**Existing Pain Points:**
1.  **Scalability Challenges in High Dimensions:** Existing parameterizations for learning divergence-free fields (e.g., using stream functions or differential forms) rely heavily on automatic differentiation to compute Jacobian matrices. As dimensionality increases (e.g., to image sizes), computing the Jacobian becomes intractable, limiting the application of such hard constraints to low-dimensional problems.
2.  **Approximation Errors in SURE-based Methods:** When the noise level is known, methods like MC-SURE approximate the divergence using Monte Carlo methods (Equation 6). This introduces approximation errors, leading to a slight performance gap compared to supervised methods.
3.  **Instability in Min-Max Optimization:** Methods like UNSURE, which enforce zero expected divergence (Constant Expected Divergence) for unknown noise levels, formulate the problem as a min-max optimization (saddle point). This approach does not guarantee strict constraint satisfaction, is highly sensitive to learning-rate pairs, and can lead to instabilities or oscillatory dynamics during training.
4.  **Expressivity Limitations of Blind-Spot Networks:** To avoid the need for noise level estimation, blind-spot networks (Noise2Self) impose the constraint that the output pixel does not depend on the input pixel ($\partial f_i / \partial y_i = 0$). While this guarantees zero divergence, it severely limits the expressivity of the estimator (since $y_i$ is usually highly informative about $x_i$) and tends to introduce checkerboard artifacts.
5.  **Failure under Variable Noise Levels:** The UNSURE approach (Constant Expected Divergence) cannot be used when the noise level $\sigma$ varies across noisy observations. In full generality, $E_{y,\sigma}[\sigma^2 \text{div} f(y)] \neq E_\sigma[\sigma^2] E_y[\text{div} f(y)]$ because $\sigma^2$ and $\text{div} f(y)$ are statistically dependent (since $y$ depends on $\sigma$), violating the required constraint condition for self-supervised learning.

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To design a neural network architecture that is strictly divergence-free by construction, remains resource-efficient and scalable to high-dimensional problems (like images), and enables self-supervised denoising even when the noise level is unknown and varies across the training data.

**Scientific Questions:**
1.  How to represent divergence-free vector fields in a way that avoids the intractable computation of full Jacobian matrices in high dimensions?
2.  How to construct a constraint set for self-supervised denoising that balances the expressivity lost in blind-spot designs while allowing training when the noise level is unknown and non-constant?
3.  How to parameterize a neural network to guarantee constant (zero) divergence everywhere by design, using structured combinations of conservative fields?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Establish a representer theorem for divergence-free vector fields, characterizing them as structured combinations of conservative fields (gradients of scalar fields multiplied by skew-symmetric matrices). Based on this theorem, incorporate sparsity constraints to reduce the quadratic number of terms to a computationally tractable subset. Construct a neural network parameterization that guarantees zero divergence analytically by design, avoiding the need for Monte Carlo divergence approximations or min-max optimization.

**Design Philosophy:**
1.  **Hard Constraint Satisfaction:** Enforce the divergence-free property strictly through the network architecture and parameterization, rather than soft penalties or approximations.
2.  **Sparsity for Efficiency:** Deliberately simplify the universal representation by using a sparse combination of conservative fields ($K' \ll K$) to ensure computational tractability in high dimensions while retaining sufficient expressivity.
3.  **Affine Structure Exploitation:** Leverage the affine space structure of the constraint sets ($S_c = c \text{id} + S_0$) to focus the learning on the divergence-free component $S_0$.
4.  **Adaptability to Variable Noise:** Define an intermediate constraint set (Constant Divergence) that satisfies the self-supervised learning condition for both constant and variable noise levels, bridging the gap between overly restrictive blind-spot constraints and loose expected divergence constraints.

# 4. Core Innovation Points

1.  **Representer Theorem for Divergence-Free Fields:** The establishment of a representer theorem (Theorem 1) proving that any smooth divergence-free vector field can be represented as a sum of skew-symmetric matrices multiplied by the gradients of scalar fields ($f = \sum_{k=1}^K A_k \nabla \psi_k$).
2.  **CENSURE Estimator (Divergence-Constant Constraint):** Introduction of the constraint set $S_c^{DC}$ (Constant Normalized Divergence) where $\text{div} f(y) = nc$ everywhere. This set lies strictly between the blind-spot set $S_c^{BS}$ and the constant expected divergence set $S_c^{CED}$, preserving expressivity while allowing training with unknown, non-constant noise levels.
3.  **Resource-Efficient Sparse Architecture:** A scalable neural network architecture that enforces zero divergence by design. It implements a sparse version of the representer theorem ($K' \ll K$) using shared backbone networks and parameterized skew-symmetric matrices, avoiding intractable Jacobian computations.
4.  **Theoretical Framework for Self-Supervised Denoising:** A theoretical analysis (Propositions 1-3) of optimal constants and affine space structures, demonstrating that divergence-free estimators can be converted into noise-level-aware variants without additional training, and providing lower bounds on reconstruction error.

# 5. Overview of the Overall Technical Solution

The method, termed CENSURE (Concealed and Erratic Noise level with Stein’s Unbiased Risk Estimate), constructs a divergence-free vector field $f$ as a sparse sum of terms $A_k \nabla \psi_k$.
1.  **Constraint Formulation:** The estimator is constrained to $S_0^{DC}$, requiring the divergence to be zero everywhere ($c=0$). This allows the self-supervised objective $\arg\min_f E_y\|f(y)-y\|_2^2$ to be valid even for variable noise levels.
2.  **Representation:** The vector field $f$ is represented as $f = \sum_{k=1}^{K'} A_k \nabla \psi_k$, where $K'$ is a small number (e.g., 8).
3.  **Skew-Symmetric Matrices ($A_k$):** Parameterized as sparse, shared-block diagonal matrices modified by fixed permutation matrices to ensure skew-symmetry ($A^\top = -A$).
4.  **Scalar Fields ($\psi_k$):** Parameterized using a shared U-Net backbone $D_\theta$ and parameterized matrices $B_k$. The scalar potential is defined such that its gradient can be computed via automatic differentiation of a specific loss-like formulation.
5.  **Training:** The network is trained to minimize measurement consistency $E_y\|f(y)-y\|_2^2$. The divergence-free property is guaranteed by construction, eliminating the need for divergence terms in the loss function or Monte Carlo approximations.

# 6. Detailed Module Design

**Constraint Set Module ($S_c^{DC}$):**
The set of weakly differentiable vector fields on $\mathbb{R}^n$ with constant (normalized) divergence $c \in \mathbb{R}$.
$$S_c^{DC} = \{f \in L^1(\mathbb{R}^n,\mathbb{R}^n) : \forall y \in \mathbb{R}^n, \text{div} f(y) = nc\}$$
By setting $c=0$, the estimator is divergence-free everywhere, satisfying the condition $E_{y,\sigma}[\sigma^2 \text{div} f(y)] = 0$ for variable $\sigma$.

**Divergence-Free Representation Module:**
Based on Lemma 2, a simple divergence-free vector field is $A\nabla\psi$ where $A$ is skew-symmetric.
Theorem 1 extends this: any divergence-free field can be represented as a sum of such terms using a basis of skew-symmetric matrices.
$$f = \sum_{k=1}^K A_k \nabla\psi_k$$
To ensure scalability, the architecture uses a sparse subset $K' \ll K$:
$$f = \sum_{k=1}^{K'} A_k \nabla\psi_k$$

**Skew-Symmetric Matrix Parameterization Module ($A_k$):**
To guarantee skew-symmetry by design and share parameters, $A_k$ is constructed using a shared parameterized repeated-block diagonal matrix $\Theta$ and fixed permutation matrices $P_k$:
$$A_k = P_k^\top \frac{\Theta - \Theta^\top}{2} P_k$$

**Scalar Field Parameterization Module ($\psi_k$):**
The scalar fields are parameterized to incorporate a neural network backbone $D_\theta$ (a U-Net) specific to image processing. The parameters $\theta$ are shared across all scalar fields.
$$\psi_{\theta,B_k}: y \in \mathbb{R}^n \mapsto \frac{1}{2}(\|B_k y\|_2^2 - \|B_k y - D_\theta(y)\|_2^2)$$
Here, $B_k \in \mathbb{R}^{n \times n}$ are parameterized matrices. The gradient of this scalar field is computed via automatic differentiation:
$$\nabla\psi_{\theta,B_k}(y) = B_k^\top D_\theta(y) + J_{D_\theta}(y)^\top(B_k y - D_\theta(y))$$

**Matrix B Parameterization Module:**
The matrices $B_k$ are parameterized analogously to $A_k$ via a shared repeated-block diagonal matrix $\Theta' \in \mathbb{R}^{n \times n}$:
$$B_k = P_k^\top \Theta' P_k$$
The permutation matrices $P_k$ are the same as those used for $A_k$.

**Optimization Module:**
The objective is to minimize the measurement consistency term:
$$\arg\min_f E_y\|f(y)- y\|_2^2 \text{ s.t. } f \in S_0^{DC}$$
Since the constraint is enforced by design, the optimization reduces to minimizing the loss function $E_y\|f(y)-y\|_2^2$ using standard gradient descent.

# 7. All Mathematical Formulas and Symbol Definitions

**Formulas:**
(1) $\text{div} f(y) \triangleq \text{tr}(Jf(y)) = \sum_{i=1}^n \frac{\partial f_i}{\partial y_i}(y)$
(2) $y = x + \sigma\epsilon$
(3) $\arg\min_f E_{x,y}\|f(y)- x\|_2^2$
(4) $E_{x,y}\|f(y)- x\|_2^2 = E_{y,\sigma}[-n\sigma^2 + \|f(y)- y\|_2^2 + 2\sigma^2 \text{div} f(y)]$
(5) $\arg\min_f E_{y,\sigma}[\|f(y)- y\|_2^2 + 2\sigma^2 \text{div} f(y)]$
(6) $\text{div} f(y) = \lim_{\tau\to 0} E_{h\sim N (0,I)}[\frac{h^\top f(y + \tau h)- f(y)}{\tau}]$
(7) $\exists \lambda \in \mathbb{R}, \forall f \in S, E_{y,\sigma}[\sigma^2 \text{div} f(y)] = \lambda$
(8) $\arg\min_f E_y\|f(y)- y\|_2^2 \text{ s.t. } f \in S$
(9) $S_c^{BS} = \{f \in L^1(\mathbb{R}^n,\mathbb{R}^n) : \forall y \in \mathbb{R}^n, \frac{\partial f_i}{\partial y_i}(y) = c\}$
(10) $S_c^{CED} = \{f \in L^1(\mathbb{R}^n,\mathbb{R}^n) : E_y \text{div} f(y) = nc\}$
(11) $S_c^{DC} = \{f \in L^1(\mathbb{R}^n,\mathbb{R}^n) : \forall y \in \mathbb{R}^n, \text{div} f(y) = nc\}$
(12) $\arg\min_{f\in S_c} E_y\|f(y)- y\|_2^2 = c \text{id} + (1-c) \arg\min_{f\in S_0} E_y\|f(y)- y\|_2^2$
(13) $c^* = \arg\min_{c\in\mathbb{R}} \min_{f\in S_c} E_{x,y}\|f(y)- x\|_2^2 = 1 - \frac{nE_\sigma[\sigma^2]}{\min_{f\in S_0} E_y\|f(y)- y\|_2^2} \in [0, 1]$
(14) $\min_{f\in S_c^{DC}} E_{x,y} \frac{1}{n}\|f(y)- x\|_2^2 \ge \min_{f\in S_c^{CED}} E_{x,y} \frac{1}{n}\|f(y)- x\|_2^2 = \text{MMSE} + \frac{\sigma^2}{1 - \text{MMSE}/\sigma^2} (c - \text{MMSE}/\sigma^2)^2 \ge \text{MMSE}$
(15) $f = \sum_{k=1}^K A_k \nabla\psi_k$
(16) $(A_1,A_2,A_3) = \left(\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 0 & -1 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}\right)$
(17) $f = \sum_{k=1}^3 A_k \nabla\psi_k = \begin{pmatrix} \frac{\partial \psi_3}{\partial y_2} - \frac{\partial \psi_2}{\partial y_3} \\ \frac{\partial \psi_1}{\partial y_3} - \frac{\partial \psi_3}{\partial y_1} \\ \frac{\partial \psi_2}{\partial y_1} - \frac{\partial \psi_1}{\partial y_2} \end{pmatrix} \triangleq \nabla \times \begin{pmatrix} \psi_1 \\ \psi_2 \\ \psi_3 \end{pmatrix}$
(18) $f = \sum_{k=1}^{K'} A_k \nabla\psi_k$
(19) $A_k = P_k^\top \frac{\Theta - \Theta^\top}{2} P_k$
(20) $\psi_{\theta,B_k}: y \in \mathbb{R}^n \mapsto \frac{1}{2}(\|B_k y\|_2^2 - \|B_k y - D_\theta(y)\|_2^2)$
(21) $\nabla\psi_{\theta,B_k}(y) = B_k^\top D_\theta(y) + J_{D_\theta}(y)^\top(B_k y - D_\theta(y))$
(22) $B_k = P_k^\top \Theta' P_k$

**Symbol Definitions:**
*   $y \in \mathbb{R}^n$: Noisy observation.
*   $x \in \mathbb{R}^n$: Noise-free signal.
*   $\sigma > 0$: Noise level.
*   $\epsilon \sim N(0, I_n)$: Standard Gaussian noise vector.
*   $f$: Estimator function.
*   $\text{div} f(y)$: Divergence of $f$ at $y$.
*   $Jf(y)$: Jacobian matrix of $f$ at $y$.
*   $S_c^{BS}$: Blind-spot constraint set.
*   $S_c^{CED}$: Constant expected divergence constraint set.
*   $S_c^{DC}$: Constant (normalized) divergence constraint set (CENSURE).
*   $c \in \mathbb{R}$: Constant divergence value.
*   $A_k$: Skew-symmetric matrices ($A_k^\top = -A_k$).
*   $\psi_k$: Smooth scalar fields.
*   $K$: Dimension of the space of real skew-symmetric $n \times n$ matrices ($\binom{n}{2}$).
*   $K'$: Number of terms used in the sparse representation ($K' \ll K$).
*   $P_k \in \mathbb{R}^{n \times n}$: Fixed permutation matrices.
*   $\Theta, \Theta'$: Shared parameterized repeated-block diagonal matrices.
*   $D_\theta$: Neural network backbone (e.g., U-Net) with parameters $\theta$.
*   $B_k$: Parameterized matrices for scalar fields.

# 8. Algorithm Pseudocode

The original paper does not provide a standard algorithm pseudocode block. The implementation plan is described by the mathematical construction of the CENSURE estimator. The logic is as follows:
1.  Define the backbone network $D_\theta$.
2.  Define the shared block diagonal matrices $\Theta$ and $\Theta'$.
3.  Select fixed permutation matrices $P_k$ for $k=1, \dots, K'$.
4.  Construct $A_k = P_k^\top \frac{\Theta - \Theta^\top}{2} P_k$.
5.  Construct $B_k = P_k^\top \Theta' P_k$.
6.  For an input $y$:
    *   Compute $D_\theta(y)$.
    *   For each $k=1, \dots, K'$:
        *   Compute scalar potential $\psi_k(y) = \frac{1}{2}(\|B_k y\|_2^2 - \|B_k y - D_\theta(y)\|_2^2)$.
        *   Compute gradient $\nabla\psi_k(y)$ via automatic differentiation of $\psi_k$ with respect to $y$.
        *   Compute term $A_k \nabla\psi_k(y)$.
    *   Sum terms to get estimator output: $f(y) = \sum_{k=1}^{K'} A_k \nabla\psi_k(y)$.
7.  Update parameters $\theta, \Theta, \Theta'$ by minimizing the loss $E_y\|f(y) - y\|_2^2$.