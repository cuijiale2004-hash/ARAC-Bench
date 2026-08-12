1. Research Background and Existing Pain Points

High-dimensional data visualization (HDV) plays an important role in data science and engineering applications, allowing humans to intuitively interpret large, complex datasets by projecting them into two or three dimensions. Over the years, numerous algorithms for dimensionality reduction and HDV have been developed, broadly classified into linear methods (PCA, MDS, LDA) and non-linear methods (SOM, Isomap, Kernel PCA, autoencoders, t-SNE, UMAP, TriMap, PaCMAP). However, existing traditional and recent HDV methods face critical limitations:

*   Sensitive to hyper-parameter tuning: Traditional data visualization methods are sensitive to hyper-parameter selection, such as perplexity in t-SNE and the number of neighbors in UMAP and PaCMAP. Different hyper-parameter selections will significantly change the visualization results. A fixed or default hyper-parameter may not always lead to the best performance. The absence of labeled data in unsupervised tasks poses a unique challenge to tuning the HDV hyper-parameter.
*   Re-training overhead: They need to re-run the training algorithm iteratively, i.e., re-training, on every new dataset. It brings significant computational overhead, especially when handling a large number of datasets.
*   Lack of cross-domain and cross-dimension generalization: Some studies tried to solve re-training overhead using a parametric model by optimizing $\min_\theta \mathbb{E}_i[\|f_\theta(x_i) - z_i\|_2^2]$, where $f: \mathbb{R}^d \to \mathbb{R}^{d'}$ is the HDV model. They still failed to adapt the model to new datasets with different domains and dimensions and suffered from overfitting to the training datasets. They failed to effectively utilize the historical low-dimensional representations.

2. Core Research Motivation and Scientific Questions

The core research motivation is to overcome the limitations of traditional HDV methods by leveraging historical visualization results of labeled datasets in a meta-learning manner, effectively capturing the underlying data structure via graph-based representations. The goal is to build an end-to-end deep learning model (AutoDV) that generalizes robustly to unseen datasets without requiring re-training or additional hyper-parameter tuning during inference. 

There are two primary scientific questions/challenges (C1 and C2) when solving this problem:
*   C1: Designing the end-to-end model $f_\phi$ is non-trivial, especially when the model is expected to handle input datasets with different sizes and dimensions, i.e., $N_i \neq N_j$ and $d_i \neq d_j$ for some $i$ and $j$. Most deep neural networks are designed for a fixed input dimensionality and cannot be directly applied to datasets with different numbers of features.
*   C2: There exist multiple optimal low-dimension embeddings $Z^*_i$ for the same input $X_i$ if we apply translation, rotation, and scaling to $Z^*_i$, e.g., $Z^*_i$ and $Z^*_iQ$ with any orthonormal matrix $Q$ are equivalent in terms of visualization. This will result in a one-to-many problem and will be difficult to converge.

3. Overall Core Idea and Design Philosophy

The overall core idea is to propose AutoDV, an end-to-end data visualization approach leveraging graph neural networks (GNNs) and graph transformers. AutoDV builds a deep learning model using historical visualization results of labeled datasets in a meta-learning manner, effectively capturing the underlying data structure via graph-based representations. The design philosophy involves converting datasets into multi-weight graphs to mitigate the constraints imposed by varying dataset dimensions on end-to-end visualization methods. To tackle the one-to-many problem brought by translation and rotation, AutoDV introduces an affine invariant loss function that aligns the geometric structure between the low-dimensional embeddings by constructing pairwise squared similarity matrices. To further realize the scaling invariant, it pre-processes $Z^*_i$ using z-score normalization.

4. Core Innovation Points

*   Introduce AutoDV, an end-to-end data visualization method that eliminates the need for hyper-parameter tuning and re-training when visualizing new datasets.
*   Compared with existing parametric visualization methods such as parametric UMAP and inductive t-SNE, AutoDV can adapt to datasets of varying dimensionality, exhibiting superior generalization performance on unseen datasets from any domain. It utilizes multi-scale graph representation to remove dimension constraints.
*   Introduce an affine invariant loss function using Bregman divergence on pairwise similarity matrices to solve the one-to-many problem caused by translation, rotation, and scaling.
*   Propose a sign-count based method to eliminate sign ambiguity in positional encoding derived from spectral techniques, which is lightweight to implement and shows good performance.

5. Overview of the Overall Technical Solution

Given a historical dataset $X_i$, AutoDV first constructs a weighted graph whose adjacency matrix is a pairwise similarity matrix between samples using a Gaussian kernel function. To preserve as much structural information as possible, it further generates $k$-scale graphs for each dataset by using different bandwidth parameters in the Gaussian kernel. The resulting graphs, $S^{(1)}_i, S^{(2)}_i, \ldots, S^{(k)}_i$, capture the data structure at different scales. A dataset in $\mathbb{R}^{N_i \times d_i}$ is transformed into multiple graphs represented by weighted adjacency matrices in $\mathbb{R}^{N_i \times N_i}$. AutoDV derives graph positional encodings (PE) from the weighted adjacency matrices as node features, utilizing SVD. It proposes a multi-graph GNN where for each graph, there is an individual GIN to process it. The $k$ sets of node embeddings extracted by GINs are concatenated together to get the final hidden node embedding. A graph transformer is introduced to explore the inter-node relationship, and the final output is produced by a multi-layer perceptron (MLP). The model is trained using an affine invariant loss function to mimic the optimal low-dimensional embedding from t-SNE or UMAP.

6. Detailed Module Design

*   Task Definition Module: Suppose we have $L$ labeled historical datasets $(X_1, y_1), (X_2, y_2), \ldots, (X_L, y_L)$. Let $A_\theta$ be an effective data visualization algorithm. For each $X_i$, an optimal low-dimensional embedding $Z^*_i$ is obtained by $Z^*_i = A_{\theta^*_i}(X_i)$, where $\theta^*_i$ denotes the optimal hyper-parameters selected using $y_i$ with respect to some metric $M$. The goal is to train an end-to-end model $f_\phi: \mathbb{R}^{N_i \times d_i} \to \mathbb{R}^{N_i \times d'}$ such that $\text{dist}(f_\phi(X_{new}), Z^*_{new}) \le \epsilon$.
*   Graph Construction Module: Given a historical dataset $X_i$, construct a weighted graph using a Gaussian kernel function. Generate $k$-scale graphs by using different bandwidth parameters $\gamma^{(j)}$. The similarity matrix is computed as: $S^{(j)}_i[u, v] = \exp\left(-\frac{\|X_i[u] - X_i[v]\|_2^2}{\gamma^{(j)}}\right), \forall j=1,2,\ldots,k$.
*   Positional Encoding Module: Derive graph positional encodings from the weighted adjacency matrices as node features, utilizing SVD. $P^{(i)}_j = h(S^{(j)}_i), \forall j=1,\ldots,k$. To address sign ambiguity, a sign count-based method is proposed: if the majority value in one column of $P$ is negative, flip the sign of this column. Otherwise, keep the sign.
*   Multi-Graph GNN Module: For each graph, there is an individual GIN to process it. Then, there are $k$ sets of node embeddings extracted by GINs. By concatenating them together, we get the final hidden node embedding.
*   Graph Transformer and Output Module: A graph transformer is introduced to explore the inter-node relationship, and the final output of the model is produced by an MLP. The model $f_\phi$ is expressed as: $f_\phi\left(\{(P^{(j)}_i, S^{(j)}_i)\}_{j=1}^k\right) = \text{MLP}\left(\text{GT}\left(\text{GIN}_1(P^{(1)}_i, S^{(1)}_i) \oplus \cdots \oplus \text{GIN}_k(P^{(k)}_i, S^{(k)}_i)\right)\right)$.
*   Affine Invariant Loss Module: Instead of direct alignment between the output low-dimensional embeddings, align the geometric structure by constructing pairwise squared similarity matrices $\hat{\Delta}_i$ and $\Delta^*_i$ for $Z_i$ and $Z^*_i$. A general training loss function is defined using Bregman divergence $K_\psi(\cdot, \cdot)$. To further realize scaling invariant, pre-process $Z^*_i$ using z-score normalization. For t-SNE, use Kullback-Leibler divergence loss. For UMAP, consider both local structure and global structure consistency with a regularization coefficient $\lambda(t)$ decreasing as the training step $t$ grows.
*   Theoretical Robustness Module: Proved that if the eigen gap $\delta^{(j)}$ is large, $d_e/\lambda^{(j)}_{min}$ is small, and $\gamma^{(j)}$ is large, AutoDV is stable, provided the neural network is not too complex. This guarantees generalization ability to some extent.
*   Large-Scale Extension Module: Split $X_i$ into subsets. Construct an anchor subset $A$ with a size smaller than $M$, where intra-distance is small. Simultaneously input $A \cup X^{(j)}_i$. The final low-dimensional embedding is the corresponding union of outputs, utilizing a fixed anchor set to calibrate outputs from different batches.

7. All Mathematical Formulas and Symbol Definitions

*   Task Definition:
    $\text{dist}(f_\phi(X_{new}), Z^*_{new}) \le \epsilon$
    Where $\text{dist}(\cdot, \cdot)$ denotes some distance metric and $\epsilon > 0$ is a small constant.

*   Similarity Matrix (k-scale graphs):
    $S^{(j)}_i[u, v] = \exp\left(-\frac{\|X_i[u] - X_i[v]\|_2^2}{\gamma^{(j)}}\right), \forall j=1,2,\ldots,k$
    Where $S^{(j)}_i[u, v]$ denotes the entry in the $u$-th row and $v$-th column of $S^{(j)}_i$, $\|X_i[u] - X_i[v]\|_2^2$ represents the squared Euclidean distance between $u$-th data point and the $v$-th data point in dataset $X_i$, and $\gamma^{(j)}$ is the bandwidth parameter of the $j$-th scale graph.

*   Positional Encoding:
    $P^{(i)}_j = h(S^{(j)}_i), \forall j=1,\ldots,k$
    Where $h(\cdot)$ is the function for extracting the positional encoding, $P_i \in \mathbb{R}^{N_i \times d_e}$, and $d_e$ is the dimensionality of the positional encoding.

*   Model Expression:
    $f_\phi\left(\{(P^{(j)}_i, S^{(j)}_i)\}_{j=1}^k\right) = \text{MLP}\left(\text{GT}\left(\text{GIN}_1(P^{(1)}_i, S^{(1)}_i) \oplus \cdots \oplus \text{GIN}_k(P^{(k)}_i, S^{(k)}_i)\right)\right)$
    Where $\oplus$ represents the concatenation along the feature dimension, $\text{GIN}_j(\cdot)$ is the $j$-th GIN network, and $\text{GT}(\cdot)$ is the graph transformer. $f_\phi$ is a GNN model, $\mathbb{R}^{k \times N_i \times d_e} \to \mathbb{R}^{N_i \times d'}$.

*   Pairwise Squared Similarity Matrices:
    $\hat{\Delta}_i[u, v] = \sigma\left(\|\hat{Z}_i[u] - \hat{Z}_i[v]\|_2^2\right), \quad \Delta^*_i[u, v] = \sigma\left(\|Z^*_i[u] - Z^*_i[v]\|_2^2\right)$
    Where $\sigma(\cdot)$ is a transformation function turning a distance to a similarity.

*   General Training Loss Function:
    $L = \frac{1}{L} \sum_{i=1}^L \sum_{u=1}^{N_i} \sum_{v=1}^{N_i} K_\psi(\hat{\Delta}_i[u, v], \Delta^*_i[u, v])$
    Where $K_\psi(\cdot, \cdot)$ denotes the Bregman divergence with a strictly convex function $\psi$ and $K_\psi(x, y) = \psi(x) - \psi(y) - \langle \nabla\psi(y), x-y \rangle$.

*   t-SNE Loss Function (KLD):
    $L_{tsne} = \frac{1}{L} \sum_{i=1}^L \sum_{u=1}^{N_i} \sum_{v=1}^{N_i} p^{(i)}_{u,v} \log\left(\frac{p^{(i)}_{u,v}}{q^{(i)}_{u,v}}\right)$
    $p^{(i)}_{u,v} = \frac{(1 + \|Z^*_i[u] - Z^*_i[v]\|_2^2)^{-1}}{\sum_{u',v'}(1 + \|Z^*_i[u'] - Z^*_i[v']\|_2^2)^{-1}}, \quad q^{(i)}_{u,v} = \frac{(1 + \|\hat{Z}_i[u] - \hat{Z}_i[v]\|_2^2)^{-1}}{\sum_{u',v'}(1 + \|\hat{Z}_i[u'] - \hat{Z}_i[v']\|_2^2)^{-1}}$

*   UMAP Loss Function:
    $L_{umap} = \frac{1}{L} \sum_{i=1}^L \left( \sum_{(u,v) \in \mathcal{N}_i} \ell(u,v) + \lambda(t) \sum_{(u,v) \notin \mathcal{N}_i} \ell(u,v) \right)$
    Where $\ell(u,v) = \left( \frac{1}{1 + a\|\hat{Z}_i[u'] - \hat{Z}_i[v']\|_2^{2b}} - \frac{1}{1 + a\|Z^*_i[u'] - Z^*_i[v']\|_2^{2b}} \right)^2$. $a$ and $b$ are two hyper-parameters mapping the distance value into a family of student-t distribution. $\lambda(t)$ is a regularization coefficient decreasing as the training step $t$ growing, $\mathcal{N}_i$ is the index set of the k-nearest pairs in dataset $X_i$.

*   Theoretical Robustness Bound:
    $\|Z - \tilde{Z}\|_F \le 2kL_\phi \sqrt{2n} e \max_j \left( c^{(j)} + \frac{1}{\sqrt{\gamma^{(j)}}} \right) \|X - \tilde{X}\|_F$
    Where $c^{(j)} = \frac{2\sqrt{2}\lambda^{(j)}_{max}}{\delta^{(j)}} + \frac{1}{2}\sqrt{\frac{d_e}{\lambda^{(j)}_{min}}}$. $\delta^{(j)}$ is the eigen gap between $S^{(j)}$ and $\tilde{S}^{(j)}$, $\lambda^{(j)}_{max}$ is the maximum eigenvalue of $S^{(j)}$, and $\lambda^{(j)}_{min}$ is the minimum the $d_e$-th eigenvalue between $S^{(j)}$ and $\tilde{S}^{(j)}$. $L_\phi$ is the Lipschitz constant.

8. Algorithm Pseudocode

Algorithm 1 Sign-Count based Sign Flipping for Positional Encoding
Input: positional encoding $P \in \mathbb{R}^{N \times d_e}$ from Eq. (3);
1: for $i \in \{1, \ldots, d_e\}$ do
2:     $v \leftarrow P[:, i]$;
3:     $n_{pos} \leftarrow \text{sum}(\text{bool}(v > 0))$; //count the number of positive elements in v
4:     $n_{neg} \leftarrow \text{sum}(\text{bool}(v < 0))$; //count the number of negative elements in v
5:     if $n_{pos} < n_{neg}$ then
6:         $v \leftarrow -1 \cdot v$;
7:     end if
8:     $P[:, i] \leftarrow v$;
9: end for
Output: Sign Flipped Positional Encoding $P$.

Algorithm 2 Batch-based AutoDV for Extending to Large Datasets
Input: Trained AutoDV model $f_\phi$, input dataset $X$, anchor size $M_A$;
1: $\{X^{(1)}, X^{(2)}, \ldots, X^{(q)}\} \leftarrow \text{split}(X, q)$; // split dataset into q chunks.
2: $c \leftarrow X.\text{size} // M_A$;
3: $A \leftarrow \text{KMeans}(X, c)[0]$;
4: $\hat{Z} \leftarrow \emptyset$;
5: for $i \in \{1, \ldots, q\}$ do
6:     $X_{in} = A \cup X^{(i)}$;
7:     $\{P\}, \{S\} \leftarrow \text{Extract PEs and similarity matrix from } X_{in}$;
8:     $\hat{Z}_{out} \leftarrow f_\phi(\{P\}, \{S\})$
9:     $\hat{Z} \leftarrow \hat{Z} \cup \hat{Z}_{out}[M_A :]$
10: end for
Output: Low-dimensional embedding for large dataset $\hat{Z}$.

Algorithm 3 Dataset Downsampling in Training Data Preparation
Input: dataset $X$, the number of class $C$, maximum subset size $N_{max}$;
1: $D_{train} \leftarrow \emptyset$;
2: for $i \in \{1, \ldots, C\}$ do
3:     $C \leftarrow \text{combination}(C, i)$;
4:     $X_c \leftarrow \text{select all points with label in } C$;
5:     $s \leftarrow \text{linspace}(100, N_{max}, 10)$; // generate 10 subsets with different sizes.
6:     for $j \in \{1, \ldots, 10\}$ do
7:         $N' \leftarrow \text{random}(s[j], s[j + 1])$;
8:         $X' \leftarrow \text{randomly downsample } N' \text{ samples from } X_c$;
9:         $D_{train} \leftarrow D_{train} \cup \{X'\}$;
10:    end for
11: end for
Output: All sampled subsets $D_{train}$.