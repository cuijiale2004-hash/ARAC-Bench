## 1. Research Background and Existing Pain Points

**Research Background:**
With the increasing demand for interpretability in machine learning, functional ANOVA decomposition has gained renewed attention as a principled tool for breaking down high-dimensional functions into low-dimensional components that reveal the contributions of different variable groups. The functional ANOVA model decomposes a high-dimensional function into a sum of low-dimensional functions called components or interactions. Notable examples include the generalized additive Model, SS-ANOVA, and MARS. Because complex structures of a given high-dimensional model can be understood by interpreting low-dimensional components, functional ANOVA models have been extensively used in interpretable AI applications. Recently, Tensor Product Neural Network (TPNN) has been developed and applied as basis functions in the functional ANOVA model, referred to as ANOVA-TPNN. ANOVA-TPNN estimates the components under the uniqueness constraint (sum-to-zero condition) and thus provides a stable estimate of each component.

**Existing Pain Points:**
A disadvantage of ANOVA-TPNN is that the components to be estimated must be specified in advance, which makes it difficult to incorporate higher-order TPNNs into the functional ANOVA model due to computational and memory constraints. Existing neural-network approaches to the functional ANOVA model require prohibitive computation when the input dimension $p$ is large, because the number of components—and thus the required networks—grows exponentially. As a result, only 1–2 dimensional components are typically used, yielding suboptimal prediction when higher-order interactions matter. In addition, choosing the number of basis functions $K_S$ for each component is not easy.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To address the limitations of ANOVA-TPNN, this work proposes a Bayesian neural network (BNN) for the functional ANOVA model which can estimate higher-order interactions (i.e., components whose input dimension is greater than 2) without requiring huge amounts of computing resources. The main idea is to infer the architecture (the architectures of neural networks for each component) as well as the parameters (the weights and biases in each neural network). By treating the architecture as a learnable parameter and developing a specially designed MCMC algorithm that searches architectures in a stepwise manner (i.e., growing or pruning the current architecture), huge computing resources for memorizing and processing all of the predefined neural networks for the components can be avoided. Furthermore, BNN offers stronger generalization and better-calibrated uncertainty estimates compared to frequentist approaches, enhancing decision making.

**Scientific Questions:**
How to design a Bayesian Tensor Product Neural Network (Bayesian-TPNN) that treats the component order and variable set as learnable parameters? How to develop an efficient MCMC algorithm that explores high-posterior regions of the architecture, specifically encouraging the detection of higher-order components without dramatic loss of accuracy? How to theoretically prove the posterior consistency of the prediction model as well as each component?

## 3. Overall Core Idea and Design Philosophy

The core idea is to formulate the functional ANOVA model using a flexible sum of Tensor Product Neural Networks (TPNNs) where the subset of variables $S_k$ for each basis function is not fixed in advance but treated as a parameter to be inferred. Instead of fixing $S \subseteq [p]$, the model considers $f(x) = \sum_{k=1}^K \beta_k \phi(x|\Theta_k)$, where $\Theta_k = (S_k, b_{S_k,k}, \Gamma_{S_k,k})$, and aims to learn $K$ and $(S_k, k \in [K])$ as well as the other parameters. Since $K$ and $S_k$ are not numeric parameters and cannot be updated by gradient descent, a Bayesian approach is adopted where $K$ and $S_k$ are explored via a custom MCMC algorithm. The design philosophy of the MCMC algorithm is to use a specially designed proposal distribution that encourages the algorithm to visit important higher-order interactions more frequently, utilizing a stepwise search (copying an existing node and adding an edge) and a pretrained probability mass function $p_{input}(\cdot)$ on input variables to guide the search.

## 4. Core Innovation Points

1.  **Bayesian-TPNN for Functional ANOVA:** Proposes a BNN for the functional ANOVA model called Bayesian-TPNN which treats the architecture (number of nodes and edge connections/variable subsets) as learnable parameters, overcoming the exponential growth of components in prior works.
2.  **Efficient MCMC Algorithm:** Develops an MCMC algorithm which efficiently explores high-posterior regions of the architecture. The algorithm introduces a "Stepwise search" move (proposing higher-order interactions by extending existing nodes) and uses a pretrained input variable importance probability $p_{input}(\cdot)$ to guide the addition of variables, ensuring higher-order components are found with reduced computational cost compared to ANOVA-TPNN.
3.  **Theoretical Justification (Posterior Consistency):** Proves the posterior consistency of the prediction model as well as each individual component under specific regularity conditions, providing a rigorous theoretical foundation for the proposed Bayesian-TPNN.
4.  **Superior Performance and Uncertainty Quantification:** Empirically demonstrates that the proposed BNN provides more accurate and stable estimation and uncertainty quantification than other neural networks for the functional ANOVA model, and effectively estimates important higher-order components which baseline models fail to capture.

## 5. Overview of the Overall Technical Solution

The overall technical solution consists of three main parts: the Bayesian-TPNN model formulation with hierarchical priors, a specialized MCMC algorithm for posterior sampling, and theoretical consistency analysis.
The model replaces the fixed component structure of ANOVA-TPNN with a flexible sum of TPNN basis functions where the number of functions $K$ and their input variable subsets $S_k$ are random variables. Hierarchical priors are defined for node-sparsity ($K$), edge-sparsity ($S_k$), and numeric parameters ($\beta_k, b_{j,k}, \gamma_{j,k}$). 
For inference, an MCMC algorithm is developed that iteratively updates $K$, $S_k$, numeric parameters, and nuisance parameters using Metropolis-Hastings algorithms. The key mechanism for updating $K$ and $S_k$ involves proposal distributions that combine "Random" moves with "Stepwise" moves (incrementally increasing interaction order) weighted by $p_{input}$. Numeric parameters are updated using Langevin dynamics. Finally, the posterior consistency of the model is proven by verifying sieve conditions and applying existing Bayesian nonparametric theory.

## 6. Detailed Module Design

**Model Definition Module:**
Bayesian-TPNN defines the regression function as $f(x) = \sum_{k=1}^K \beta_k \phi(x|\Theta_k)$, where $\Theta_k = (S_k, b_{S_k,k}, \Gamma_{S_k,k})$, $S_k \subseteq [p]$. Here, $K$ and $S_k$ define the architecture. The basis function $\phi(x|\Theta_k)$ is a TPNN that satisfies the sum-to-zero condition. Bayesian-TPNN can be understood as an edge-sparse shallow neural network where $K$ is the number of hidden nodes and $S_k$ is the set of input variables linked to the $k$-th hidden node through active edges.

**Prior Specification Module:**
The parameters are categorized into three groups: (1) $K$ for node-sparsity, (2) $S_k$ for edge sparsity, and (3) numeric parameters.
*   **Prior for K:** $\pi(K = k) \propto \exp(-C_0 k \log n)$, for $k = 0, \dots, K_{max}$.
*   **Prior for $S_K | K$:** Conditional on $K$, $S_k$ are independent and follow a mixture distribution: $\sum_{d=1}^p w_d \text{Uniform}(\text{power}([p], d))$, where $w_d \propto (1 - p_{adding}(d)) \prod_{\ell < d} p_{adding}(\ell)$ with $p_{adding}(\ell) := \alpha_{adding}(1 + \ell)^{-\gamma_{adding}}$.
*   **Prior for numeric parameters given $K, S_K$:**
    *   $\beta_k \sim N(0, \sigma^2_\beta)$
    *   $b_{j,k} \sim \text{Uniform}(0, 1)$
    *   $\gamma_{j,k} \sim \text{Gamma}(a_\gamma, b_\gamma)$
    *   For Gaussian regression nuisance parameter $\eta = \sigma^2$: $\sigma^2 \sim IG(v/2, v\lambda/2)$.

**MCMC Update Module for $K$:**
Updates $K$ to $K_{new} = K - 1$ or $K + 1$ with probabilities $K/K_{max}$ and $1 - K/K_{max}$.
*   **Case $K_{new} = K - 1$:** Removes one of the existing nodes uniformly.
*   **Case $K_{new} = K + 1$:** Generates a new node $(S_{K+1}^{new}, b_{S_{K+1}^{new},K+1}^{new}, \Gamma_{S_{K+1}^{new},K+1}^{new}, \beta_{K+1}^{new})$. $S_{K+1}^{new}$ is proposed via:
    *   **Random:** Sample $S_{K+1}^{new}$ from the prior distribution.
    *   **Stepwise:** Propose $S_{K+1}^{new} = S_{k^*} \cup \{j_{k^*}\}$, where $k^* \sim \text{Uniform}[K]$ and $j_{k^*} \sim p_{input}(\cdot | S_{k^*}^c)$.
    The algorithm selects Random or Stepwise with probabilities $M/(M+K)$ and $K/(M+K)$. The acceptance probability uses the prior ratio and proposal ratio (Section A.1).

**MCMC Update Module for $S_k, b_{S_k,k}, \Gamma_{S_k,k}$:**
For a given $k$, proposes new states using:
*   **Adding:** Add $j_{new} \sim p_{input}(\cdot | S_k^c)$ to $S_k$. Generate $b_{j_{new},k}, \gamma_{j_{new},k}$ from priors.
*   **Deleting:** Uniformly select and delete $j \in S_k$.
*   **Changing:** Select $j \in S_k$ uniformly, replace with $j_{new} \sim p_{input}(\cdot | S_k^c)$.
Moves are chosen with probabilities $(q_{add}, q_{delete}, q_{change})$. The acceptance probability involves the likelihood ratio, prior ratio, and proposal ratio (Section A.2).

**MCMC Update Module for Numeric Parameters:**
Updates $(b_{S_k,k}, \Gamma_{S_k,k}, \beta_k)$ using Langevin dynamics:
$(b_{S_k,k}^{new}, \Gamma_{S_k,k}^{new}, \beta_k^{new}) = (b_{S_k,k}, \Gamma_{S_k,k}, \beta_k) + \frac{\epsilon^2}{2} U(b_{S_k,k}, \Gamma_{S_k,k}, \beta_k) + \epsilon M$, where $M \sim N(0, I)$ and $U = \nabla \log \pi$.

**MCMC Update Module for Nuisance Parameter $\eta$:**
For Gaussian regression, $\sigma^2_g | \text{others} \sim IG(v/2, \frac{1}{n}\sum_{i=1}^n (y_i - f(x_i))^2 + v\lambda/2)$.

## 7. All Mathematical Formulas and Symbol Definitions

**Notation:**
$x = (x_1, \dots, x_p)^\top \in \mathcal{X}$, $\mathcal{X} = \mathcal{X}_1 \times \cdots \times \mathcal{X}_p \subseteq [0,1]^p$.
$[p] = \{1, \dots, p\}$.
$x_S = (x_j, j \in S)^\top$.
$\|f\|_{2,n} := (\sum_{i=1}^n f(x_i)^2 / n)^{1/2}$.
$\sigma(x) := 1 / (1 + \exp(-x))$.
$\mu_n$: empirical distribution of $\{x_1, \dots, x_n\}$. $\mu_{n,j}$: marginal distribution.

**Probability Model:**
$Y_i | x_i \sim Q_{f(x_i), \eta}$ (1)
$q_{f(x), \eta}(y) = \exp\left(\frac{f(x)y - A(f(x))}{\eta} + S(y, \eta)\right)$ (2)

**Functional ANOVA:**
$\forall j \in S, \int_{\mathcal{X}_j} f_S(x_S) \mu_j(dx_j) = 0$ (3)
$f(x) = \sum_{S \subseteq [p]} f_S(x_S)$ (4)

**TPNN:**
$\phi(x_S | S, B_{S,k}, R_{S,k}) := \prod_{j \in S} \left( 1 - \sigma\left(\frac{x_j - b_{S,j,k}}{\gamma_{S,j,k}}\right) + c_j(b_{S,j,k}, \gamma_{S,j,k}) \sigma\left(\frac{x_j - b_{S,j,k}}{\gamma_{S,j,k}}\right) \right)$ (5)
$c_j(b, \gamma) := - \left( 1 - \int_{\mathcal{X}_j} \sigma\left(\frac{x_j - b}{\gamma}\right) \mu_{n,j}(dx_j) \right) / \int_{\mathcal{X}_j} \sigma\left(\frac{x_j - b}{\gamma}\right) \mu_{n,j}(dx_j)$ (6)
$f(x) = \sum_{S \subseteq [p], |S| \le d} \sum_{k=1}^{K_S} \beta_{S,k} \phi(x_S | S, B_{S,k}, R_{S,k})$ (7)

**Bayesian-TPNN:**
$f(x) = \sum_{k=1}^K \beta_k \phi(x|\Theta_k)$ (8)

**Priors:**
$\pi(K = k) \propto \exp(-C_0 k \log n)$ (9)
$\sum_{d=1}^p w_d \text{Uniform}(\text{power}([p], d))$ (10), where $w_d \propto (1 - p_{adding}(d)) \prod_{\ell < d} p_{adding}(\ell)$, $p_{adding}(\ell) := \alpha_{adding}(1 + \ell)^{-\gamma_{adding}}$.
$\beta_k \sim N(0, \sigma^2_\beta)$, $b_{j,k} \sim \text{Uniform}(0, 1)$, $\gamma_{j,k} \sim \text{Gamma}(a_\gamma, b_\gamma)$.
$\sigma^2 \sim IG(v/2, v\lambda/2)$, $\sigma^2_g | \text{others} \sim IG(v/2, \frac{1}{n}\sum_{i=1}^n (y_i - f(x_i))^2 + v\lambda/2)$ (14)

**MCMC Acceptance Probability Components:**
For Adding move:
$\frac{\pi(\Theta_k^{new})}{\pi(\Theta_k)} \frac{q_\omega(\Theta_k | \Theta_k^{new})}{q_\omega(\Theta_k^{new} | \Theta_k)} = \frac{\alpha^{|S_k^{new}| - \gamma} (1 - \alpha (1+|S_k^{new}|)^{-\gamma})}{1 - \alpha^{|S_k^{new}| - \gamma}} \frac{1}{p - |S_k^{new}| + 1} \frac{Pr(\text{Deleting})}{Pr(\text{Adding})} \frac{\sum_{l \in S_k^c} \omega_l}{\omega_{j_{adding}}}$
For Deleting move:
$\frac{\pi(\Theta_k^{new})}{\pi(\Theta_k)} \frac{q_\omega(\Theta_k | \Theta_k^{new})}{q_\omega(\Theta_k^{new} | \Theta_k)} = \frac{1}{\alpha(1+|S_k^{new}|)^{-\gamma}} \frac{1 - \alpha(1+|S_k^{new}|)^{-\gamma}}{1 - \alpha(2+|S_k^{new}|)^{-\gamma}} (p - |S_k^{new}|) \frac{Pr(\text{Adding})}{Pr(\text{Deleting})} \frac{\omega_{j_{deleting}}}{\sum_{l \in S_k^c} \omega_l}$
For Changing move:
$\frac{\pi(\Theta_k^{new})}{\pi(\Theta_k)} \frac{q_\omega(\Theta_k | \Theta_k^{new})}{q_\omega(\Theta_k^{new} | \Theta_k)} = \frac{\omega_{j_{change}}}{\sum_{l \in S_k^c} \omega_l} \frac{\omega_{j_{new}}}{\sum_{l \in (S_k^{new})^c} \omega_l}$
Langevin proposal: $(b_{S_k,k}^{new}, \Gamma_{S_k,k}^{new}, \beta_k^{new}) = (b_{S_k,k}, \Gamma_{S_k,k}, \beta_k) + \frac{\epsilon^2}{2} U(b_{S_k,k}, \Gamma_{S_k,k}, \beta_k) + \epsilon M$

**Posterior Consistency:**
$\pi_\xi(\{f : \|f_{0,S} - f_S\|_{2,n} > \epsilon \mid X^{(n)}, Y^{(n)}\}) \overset{Q_0^n}{\to} 0$ (11)
$\pi_\xi(\{f : \|f_0 - f\|_{2,n} > \epsilon \mid X^{(n)}, Y^{(n)}\}) \overset{Q_0^n}{\to} 0$ (15)

## 8. Algorithm Pseudocode

**Algorithm 1: MCMC algorithm of Bayesian TPNN**
Input $\{(x_i, y_i)\}_{i=1}^n$: data, $K$: initial number of hidden nodes, $M_{mcmc}$: the number of MCMC iterations,
1: for $i : 1$ to $M_{mcmc}$ do
2:     Update $K$
3:     for $k : 1$ to $K$ do
4:         Update $S_k, b_{S_k,k}, \Gamma_{S_k,k}$
5:         Update $b_{S_k,k}, \Gamma_{S_k,k}, \beta_k$
6:     end for
7:     Update $\eta$
8: end for