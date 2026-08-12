## 1. Research Background and Existing Pain Points

Graph Neural Networks (GNNs) have achieved strong empirical performance across various domains such as social networks, molecular property prediction, and recommendation systems. Their success relies on message passing, where node representations are iteratively updated using features from local neighborhoods. However, the statistical foundations and fundamental generalization behavior of GNNs remain poorly understood.

Existing pain points include:
1.  **Broken Independence Assumptions:** Classical feed-forward networks assume i.i.d. samples, where minimax analyses show generalization error scales as $1/\sqrt{n}$ with $n$ samples. GNNs break these independence assumptions because node samples are correlated through edges, message passing couples distant regions, and the effective number of statistically independent observations differs significantly from the number of labeled nodes.
2.  **Lack of Structure-Aware Theory:** Prior theoretical work mostly addresses the inductive (graph-level) regime where each training example is an independent graph. However, widely used benchmarks (like ogbn_arxiv, ogbn_products, Reddit) operate in the transductive (node-level) setting on a single fixed graph. These regimes differ sharply in statistical difficulty, as node labels on a single slowly mixing graph may exhibit substantial redundancy. A principled minimax analysis reflecting the interplay between architecture and graph topology is missing.
3.  **Absence of Lower Bounds:** While upper bounds via VC-dimension or PAC-Bayesian approaches exist, sharp characterizations of sample complexity and lower bounds on generalization—which are critical for understanding statistical limitations—remain scarce.

## 2. Core Research Motivation and Scientific Questions

The core research motivation is to determine the fundamental statistical limits of GNNs. Specifically, the paper seeks to answer: How many training samples are needed for a GNN to generalize on a given graph, and how does graph structure shape this requirement?

The central scientific questions are:
1.  What are the minimax limits for GNNs, and under what structural conditions do they arise?
2.  When do GNNs follow classical $1/\sqrt{n}$ behavior, and when do structural properties of the graph impose stricter limits?
3.  How do graph topology (homophily, spectral expansion, mixing time) and mixing geometry fundamentally constrain the statistical efficiency of GNNs?

## 3. Overall Core Idea and Design Philosophy

The overall idea is to develop a minimax analysis framework for ReLU message-passing GNNs with explicit architectural assumptions, covering both inductive (graph-level) and transductive (node-level) settings. The design philosophy relies on information-theoretic methods (Fano's inequality, packing sets, Varshamov–Gilbert bound) to derive lower bounds that reflect the hardness of estimation.

The framework differentiates between:
1.  **Worst-Case Regime:** For arbitrary graphs without structural constraints (e.g., path graphs), the worst-case generalization error scales as $\sqrt{\log d/n}$.
2.  **Structural Regime:** Under a "Spectral–homophily" condition (combining strong label homophily and bounded spectral expansion), the effective sample size collapses due to overlapping message-passing neighborhoods, leading to a tighter bound of $d/\log n$.

The design also incorporates a systematic empirical validation strategy using "ratio-based scaling diagnostics" to determine which theoretical regime real-world graphs fall into.

## 4. Core Innovation Points

1.  **Dual-Regime Analysis:** The paper analyzes both inductive (graph-level) and transductive (node-level) prediction settings, providing minimax lower bounds tailored to each regime, addressing the gap in prior work that focused mostly on inductive settings.
2.  **Worst-Case Lower Bound:** For arbitrary graphs without structural assumptions, the paper proves a lower bound of $R = \Omega(\sqrt{\log d/n})$, matching the classical rate for ReLU networks and establishing that GNNs cannot universally generalize faster than $1/\sqrt{n}$.
3.  **Structure-Aware Lower Bound:** Under the spectral–homophily condition $\lambda_2 \le \kappa/\log n$, the paper proves the minimax risk tightens to $R = \Omega(d/\log n)$. This reflects the collapse of effective sample size on slowly mixing graphs where message-passing neighborhoods overlap heavily.
4.  **Empirical Validation via Ratio Diagnostics:** The paper introduces and utilizes ratio-based scaling diagnostics (Err(n)/$\sqrt{\log d/n}$ vs Err(n)/(d/log n)) combined with structural diagnostics and synthetic stress tests to demonstrate that real graphs lie in the structural regime and empirically follow the $d/\log n$ scaling.
5.  **Topology-Driven Sample Complexity:** The results demonstrate that the effective sample complexity of GNNs is governed not only by architecture but by graph topology—specifically homophily, spectral expansion, and mixing time—highlighting the need for structure-aware generalization theory.

## 5. Overview of the Overall Technical Solution

The technical solution follows a theoretical-empirical pipeline:
1.  **Definition of Hypothesis Class:** Define the class of ReLU GNNs $\mathcal{F}_{GNN}(v_s, L)$ with specific norm constraints ($\ell_1$-norm budget).
2.  **Risk Formulation:** Define Graph-level (inductive) risk and Node-level (transductive) risk using minimax formulation.
3.  **Theorem 1 (Inductive Bound):** Construct a packing set on path (chain) graphs using the Varshamov-Gilbert bound. Apply Fano's inequality with controlled KL divergence to derive the $\sqrt{\log d/n}$ rate.
4.  **Theorem 2 (Transductive Bound):** Define the spectral–homophily condition. Construct a packing set on $K = \Theta(\log n)$ anchor nodes within the graph. Use a shared-parameter subclass and Fano's inequality to show the effective sample size collapse, yielding the $d/\log n$ rate.
5.  **Empirical Verification:** Construct synthetic datasets (worst-case and bottlenecked) and evaluate real benchmarks. Use ratio diagnostics to test stability against theoretical rates and perform curve fitting.

## 6. Detailed Module Design

### 6.1 GNN Architecture and Hypothesis Class
A ReLU-based GNN with $L$ message-passing layers realizes a function $f: \mathcal{G} \mapsto \hat{Y}$. Each layer updates hidden node representations as defined in Equation (1).
**Assumptions:**
*   **(A1) Input-independent aggregation:** Theorem 1 applies to message-passing GNNs satisfying input-independent, 1-hop permutation-invariant aggregation (e.g., SUM, MEAN).
*   **(A2) Uniform layerwise Lipschitz/variation control:** Instantiated as the $\ell_1$-norm budget $\sum_{\ell=0}^{L-1} (\|W^{(\ell)}\|_1 + \|B^{(\ell)}\|_1) \le v_s$.
*   **Class Definition:** $\mathcal{F}_{GNN}(v_s, L)$ is the class of $L$-layer ReLU GNNs satisfying this constraint.

### 6.2 Spectral–Homophily Condition
Let $L = I - D^{-1/2}AD^{-1/2}$ be the normalized Laplacian and $\lambda_2(L)$ be its second-smallest eigenvalue. The condition is defined as $\lambda_2(L) \le \kappa/\log n$ for a constant $\kappa > 0$. A small $\lambda_2(L)$ indicates weak expansion and slow mixing (bottleneckedness), preventing information injected in one region from propagating globally.

### 6.3 Packing Construction for Theorem 1 (Inductive)
The proof constructs a family $\{f_S\}$ indexed by $r$-subsets $S \subset [d]$.
*   **Realizable subclass:** $f_S(x) := a \sum_{j \in S} \tilde{\phi}(x_j)$ with $a = \frac{c_0 v_s}{Lr}$, where $\tilde{\phi}(z) := \phi(z) - \mathbb{E}[\phi(Z)] = \max\{0, z\} - \frac{1}{\sqrt{2\pi}}$.
*   **L2 separation:** $\|f_S - f_T\|_{L^2(P_X)}^2 = a^2 v_\phi m$, where $v_\phi = \frac{1}{2} - \frac{1}{2\pi}$ and $m = |S \triangle T|$.
*   **Constant-weight code:** Using Varshamov-Gilbert, exists $C \subset \{S \subset [d] : |S|=r\}$ with $|S \triangle T| \ge r/2$ and $|C| \ge (c d/r)^r$.

### 6.4 Packing Construction for Theorem 2 (Transductive)
The proof selects $K = \lceil c_{blk} \log n \rceil$ anchor nodes $A = \{i_1, \dots, i_K\}$.
*   **Feature model:** $X_v = (Z_v \tilde{X}_v, Z_v) \in \mathbb{R}^{d+1}$, where $Z_v = 1\{v \in A\}$ and $\tilde{X}_v \sim \mathcal{N}(0, I_d)$.
*   **Shared-parameter subclass:** Define odd activation $\psi(z) := \phi(z) - \phi(-z)$ (where $\psi(z)=z$). For $\theta \in \{0,1\}^d$: $f_\theta(x) := \frac{\sigma v_s}{LK} \sum_{j=1}^d \theta_j \psi(x_j)$.
*   **Packing set:** $\Theta \subset \{0,1\}^d$ is a constant-weight code with $\|\theta\|_0 = d/4$ and Hamming distance $d_H \ge d/8$.

### 6.5 Synthetic Dataset Construction (WorstCase_Bottleneck_20k)
To instantiate the spectral–homophily condition:
*   **Graph:** $N=20,000$ nodes, $K=40$ communities. Edges added with probability $p_{in} = 0.03$ (intra-community) and $p_{out} = 0.0003$ (inter-community).
*   **Features:** $X_{u,:} \sim \mathcal{N}(0, I_d)$.
*   **Teacher Logits:** $z_u = \langle X_{u,:}, w_{global} \rangle + 0.5 \cdot \text{community}(u) + \varepsilon_u$, $\varepsilon_u \sim \mathcal{N}(0, \sigma^2_{noise})$.
*   **Labels:** Quantile-based binning into $C=4$ classes.

## 7. All Mathematical Formulas and Symbol Definitions

**GNN Layer Update:**
$$h_i^{(\ell+1)} = \phi(W^{(\ell)} \text{Agg}_{j \in N(i)} h_j^{(\ell)} + B^{(\ell)} h_i^{(\ell)}), \quad \phi(z) = \max\{0, z\}, \quad \ell = 0, \dots, L-1$$

**Graph-level (Inductive) Risk:**
$$R_n^{\text{graph}}(\mathcal{F}_{GNN}) := \inf_{\hat{f}} \sup_{f^* \in \mathcal{F}_{GNN}} \mathbb{E}_{\text{train}} \mathbb{E}_{G \sim P_G} [(\hat{f}(G) - f^*(G))^2]$$

**Node-level (Transductive) Risk:**
$$R_{(n,G)}^{\text{node}}(\mathcal{F}_{GNN}) := \inf_{\hat{f}} \sup_{f^* \in \mathcal{F}_{GNN}} \mathbb{E}_S [\frac{1}{|V|} \sum_{v \in V} (\hat{f}(v) - f^*(v))^2]$$

**Theorem 1 (Graph-level Minimax Lower Bound):**
$$R_n^{\text{graph}}(\mathcal{F}_{GNN}) \ge K_{\text{new}} \frac{\sigma v_s}{L} \sqrt{\frac{\log d}{n}}$$

**Sample-size Implication (Theorem 1):**
$$\epsilon^2 \ge K_{\text{new}} \frac{\sigma v_s}{L} \sqrt{\frac{\log d}{n}} \implies n \ge \frac{K_{\text{new}}^2 \sigma^2 v_s^2 L^2 \log d}{\epsilon^4}$$

**Packing Number Bound (Lemma 2):**
$$\log M(2\epsilon, \mathcal{F}_{GNN}, \|\cdot\|_{L^2}) \ge \frac{C_A v_s^2 \log d}{L^2 \epsilon^2}$$

**Fano Application (Theorem 1 Proof):**
$$R(n,|V|) \ge \frac{\epsilon^2}{2} \left(1 - \frac{n C_{KL} \epsilon^2}{\sigma^2} + \frac{\epsilon^2 \log 2}{A_0}\right)$$

**Exact Quadratic Solution for $\epsilon^2$:**
$$\epsilon^2 = x = \frac{\sigma^2}{2n C_{KL}} \left(-\log 2 + \sqrt{(\log 2)^2 + \frac{2n A_0 C_{KL}}{\sigma^2}}\right)$$

**Theorem 2 (Structured-Graph Minimax Lower Bound):**
$$R_{(n,G)}^{\text{node}}(\mathcal{F}_{GNN}) \ge \frac{\sigma^2 v_s^2}{\Gamma L^2} \cdot \frac{d}{\log n}$$

**Sample-size Implication (Theorem 2):**
$$\frac{\sigma^2 v_s^2}{\Gamma L^2} \frac{d}{\log n} \le \epsilon^2 \implies n \ge \exp\left(\frac{\sigma^2 v_s^2 d}{\Gamma L^2 \epsilon^2}\right)$$

**General Minimax Risk:**
$$M_n(\Theta) = \inf_{\hat{\theta}} \sup_{\theta \in \Theta} \mathbb{E}_{P_\theta} [L(\hat{\theta}, \theta)]$$

**Regression Model:**
$$Y = f^*(G,X) + U, \quad U \sim \mathcal{N}(0, \sigma^2)$$

**Regression Minimax Risk:**
$$M_n(\mathcal{F}_{GNN}) = \inf_{\hat{f}} \sup_{f^* \in \mathcal{F}_{GNN}} \mathbb{E} [(\hat{f}(G,X) - f^*(G,X))^2]$$

**KL Divergence for Gaussian Regression:**
$$KL(P_f^{(n)} \| P_g^{(n)}) = \frac{1}{2\sigma^2} \sum_{i=1}^n (f(X_i) - g(X_i))^2$$

**L2 Separation (Theorem 2 Proof):**
$$\|f_\theta - f_{\theta'}\|_{L^2}^2 \ge \frac{\sigma^2 v_s^2}{L^2 K} \frac{d}{8} =: d_0^2$$

**KL Divergence (Theorem 2 Proof):**
$$KL(P_\theta \| P_{\theta'}) \le \frac{v_s^2}{2L^2 K} d =: KL_{\max}$$

**Normalized Laplacian:**
$$L_n = I - D_n^{-1/2} A_n D_n^{-1/2}$$

**Empirical $\kappa$ Certificate:**
$$\kappa_0 := \max_{n \in \mathcal{N}} \lambda_2(L_n) \log n$$

## 8. Algorithm Pseudocode and Implementation Procedures

The paper does not contain explicit algorithm pseudocode blocks but details a rigorous subsampling procedure and experimental methodology implemented in Python (PyTorch Geometric).

### 8.1 Subsampling Procedure (for ogbn_products_50k and Reddit_50k)
**Step 1:** Load full dataset (ogbn-products or Reddit2). Exclude nodes without incident edges.
**Step 2:** Random candidate subsample. Draw a random subset of $C = 200,000$ non-isolated nodes from the full dataset using a fixed random seed.
**Step 3:** Induced subgraph on 200 K candidates. Construct the induced subgraph on the candidate set and compute connected components via `scipy.sparse.csgraph.connected_components`.
**Step 4:** Extract the largest connected component (LCC). Identify the LCC of the candidate subgraph (size > 50K).
**Step 5:** Randomly select exactly 50 K nodes from the LCC. Sample $N = 50,000$ nodes uniformly at random (with a new fixed seed).
**Step 6:** Build the final induced 50 K-node graph. Construct the induced subgraph on the selected 50 K nodes. Retain edges $(u, v)$ iff both endpoints lie in the selected set.
**Step 7:** Preserve OGB-style dataset splits. Map original splits to the 50 K subgraph by checking original indices.

### 8.2 Experimental Methodology
1.  **Subset Sampling:** For node tasks, subgraphs were constructed using `torch_geometric.utils.subgraph`.
2.  **Data Splits:** Fixed 80%/20% train/test split.
3.  **Training Config:** Optimizer: Adam (lr=0.01, weight decay=$10^{-4}$). Epochs: 200. Batch Size: 32.
4.  **Scaling Diagnostics:** Compute $\text{Ratio1}(n) = \text{Err}(n) / \sqrt{\log d/n}$ and $\text{Ratio2}(n) = \text{Err}(n) / (d/\log n)$. A stable ratio indicates consistency with the theoretical rate.
5.  **Curve Fitting:** Fit models $c_1 + \frac{\alpha}{\sqrt{n}}$, $c_2 + \frac{\beta}{n}$, $c_3 + \frac{\delta}{\log n}$, $c_4 + \frac{1}{n^\gamma}$ using weighted least squares ($w_i = 1/\sigma_i^2$).
6.  **Statistical Significance:** 20 independent seeds per configuration. Error reported as mean $\pm$ standard deviation.