## ABSTRACT

Graph generation plays an important role in various domains such as molecular design, protein prediction, and drug discovery. However, generating graphstructured data poses challenges due to the complex dependencies inherent in graphs, spanning from intricate local substructures to broad global topologies. Although recent advances in graph-generative models have made notable progress, traditional node-level generative paradigms may have difficulty simultaneously capturing the multiscale dependencies in graphs. To address these challenges, we propose a unified latent diffusion model that jointly learns local and global topological information, enabling effective and efficient graph generation. Besides, our approach introduces a dual conditioning mechanism designed to promote dynamic interaction between local and global information, equipping the generative model with global and local awareness to better capture the coupled dependencies within graphs. Our method can largely promote the joint modeling of global and local information and substantially improve the quality of the generated graphs. Extensive experiments consistently demonstrate the effectiveness of our method.

## 1 INTRODUCTION

Graph-structural data provide powerful representations for describing objects and their relationships in diverse real-world domains, such as social networks (Singh et al., 2024), biological systems (Liu et al., 2024; Zang & Wang, 2020), and traffic networks (Cui et al., 2019). Among the core tasks in graph machine learning, graph generation has emerged as a powerful tool with significant potential in diverse applications such as molecular structure modeling (Xu et al., 2023; Vignac et al., 2022; Luo et al., 2023), circuit design (Shahane et al., 2023), and source code generation (Dai et al., 2018).

Currently, advances in deep models, including generative adversarial networks (Goodfellow et al., 2020), diffusion-based models (Ho et al., 2020; Jo et al., 2022; Luo et al., 2023), and flow-based methods (Chen et al., 2018; Bengio et al., 2023; Zhang et al., 2024b), have paved the way for effective graph generation. Unlike densely distributed image data, graph data can be extremely sparse and often exhibits complex topological information, especially the coupled local and global dependencies within distinct structures (Guo & Zhao, 2022). For instance, social networks exhibit intricate dependencies both within individual communities and between different communities (Singh et al., 2024); likewise, molecular graphs demonstrate complex relationships both within and among substructures (Zhao et al., 2012). Specifically, the local structure of a molecule, such as the specific functional groups, plays a crucial role in determining its chemical reactivity. However, the global information, including the overall structural topology and the spatial distribution of functional groups, also significantly impacts its overall properties and functional performance (Livingstone, 2000). Therefore, the intrinsic and distinct dependencies present in graphs highlight the importance of integrating both global and local information to enable a more precise graph generation process.

Recently, some effective models have incorporated global information to enhance the graph generation process. For instance, SubgDiff (Zhang et al., 2024a) introduces a subgraph prediction module that integrates substructure information into diffusion models, thereby improving molecular representation learning. Similarly, Graphusion (Yang et al., 2024) leverages graph-level pseudo-labels derived from clustering algorithms to provide informative guidance during the generation process.

However, these approaches may overlook the dependencies among substructures and have limitations in modeling the joint distribution of global and local information. This raises a natural and important question: How can we design a unified generative model that can dynamically capture both global and local topological information to enable effective graph generation?

To answer this question, we propose a unified dual conditioning latent diffusion model, termed DualDiff, which is capable of effectively modeling the joint distribution of global and local information, facilitating a robust and stable generation process. Concretely, DualDiff first maps the original graph space and diverse graph features into a unified latent space via a pretrained graph autoencoder, then it extracts the global information according to the local representations. Subsequently, DualDiff employs a two-branch diffusion process to learn topological dependencies at both the node and subgraph levels within a unified framework. To further advance the joint modeling of global and local information, a dual conditioning mechanism is introduced to promote interaction between these two branches, wherein global and local information are alternately utilized as conditions to guide the learning process of the complementary branch.

By leveraging the two-branch diffusion process and dual conditioning, DualDiff can be equipped with both global and local topological awareness, enabling it to dynamically capture the dependencies within the coarse-grained global structures and the fine-grained local details. Experimentally, we conduct comprehensive evaluations on graph generation benchmarks, further demonstrating the effectiveness of our proposed method. Our main contributions are summarized as follows:

• We highlight the limitations of conventional graph generative models in capturing the joint distribution of global and local topological information within graphs. To this end, we propose DualDiff, an effective latent diffusion-based generative model capable of simultaneously learning the dependencies inherent in both global and local structures.

• By incorporating the proposed two-branch diffusion process and dual conditioning mechanism, DualDiff can effectively capture the joint global and local topological dependencies.

• Extensive experiments across diverse benchmarks, including generic graphs and real-world molecular datasets, demonstrate that DualDiff consistently outperforms existing methods, largely illustrating the effectiveness of our proposed method.

## 2 RELATED WORK

Generative Models on Graph. Graph generative models have been widely adopted across various domains. Early approaches include graph-based variants of variational autoencoders (VAEs) (Kingma et al., 2013; Kipf & Welling, 2016; Jin et al., 2018) and generative adversarial networks (GANs) (Goodfellow et al., 2020; De Cao & Kipf, 2018; Krawczuk et al., 2020), flow-based models (Lippe & Gavves, 2021; Luo et al., 2021), and autoregressive models (You et al., 2018; Liao et al., 2019). More recently, diffusion-based models have demonstrated state-of-the-art performance on graph generation tasks (Jo et al., 2022; Qiang et al., 2023; Yan et al., 2023; Luo et al., 2023). Despite these advancements, effectively modeling the complex and coupled topological dependencies inherent in graphs remains a significant challenge. To address this limitation, we propose a unified framework, DualDiff, designed to simultaneously capture both global and local topological information, thereby facilitating a more efficient and expressive graph generation process.

Latent Graph Diffusion Model. Latent diffusion model (LDM) (Rombach et al., 2022; Podell et al., 2023) leverages the idea of performing the diffusion process in a lower-dimensional and compact latent space instead of the original data space, which greatly improves efficiency and scalability. Building on the success of LDM in images, GEOLDM (Xu et al., 2023), EDM-SYCO (Ketata et al., 2025), and LGD (Zhou et al., 2024) extend LDM to molecular and graph generation tasks. They embed the graph structures and features into a latent space by a pretrained autoencoder and conduct a diffusion process in the latent space, exhibiting competitive performance and high computational efficiency. However, these methods still rely on node-level generative methods, which treat each node as an independent entity and may overlook dependency within substructures.

Conditioning on Diffusion Model. Conditioning (Ho et al., 2022; Nichol & Dhariwal, 2021) is a widely used technique for guiding diffusion models to generate samples in a targeted and meaningful manner. Traditional conditioning methods typically incorporate external information, such as class labels or auxiliary inputs like low-resolution images (Zhang et al., 2023). More recently, selfconditioning (Chen et al., 2023) allows models to leverage their own previous predictions, resulting in notable improvements in sample quality. In addition, GPrinFlowNet (Mo et al., 2024) introduces a low-to-high frequency generation curriculum, effectively preserving semantic information and achieving high-quality conditional graph generation. Distinct from these prior approaches, we propose a dual conditioning mechanism that facilitates interaction between global and local information. This design endows our model with both global and local topological awareness, enabling more effective learning of the diverse dependencies present in graph-structured data.

## 3 PRELIMINARY

Diffusion Models. Diffusion models have demonstrated strong capabilities in modeling complex data distributions (Sohl-Dickstein et al., 2015; Ho et al., 2020). In these models, the forward process gradually corrupts a data sample x<sub>0</sub> by adding Gaussian noise over $T$ steps, producing a sequence $\mathbf { x } _ { t }$ according to $q ( \mathbf { \bar { x } } _ { t } | \mathbf { x } _ { t - 1 } ) = \bar { \mathcal { N } } ( \mathbf { x } _ { t } ; \sqrt { 1 - \beta _ { t } } \mathbf { x } _ { t - 1 } , \beta _ { t } \pmb { I } )$ , where $\beta _ { t }$ is the noise schedule. The objective of diffusion is to learn the reverse process $p _ { \theta } ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } )$ , which denoises $\mathbf { x } _ { t }$ to recover the original distribution. Elucidated Diffusion Models (EDMs) (Karras et al., 2022) extend this framework by generalizing the noise schedule and reverse process parameterization. In EDM, the model is trained to predict clean data from noisy inputs across different noise levels. Specifically, given a noise level $\sigma ,$ , the denoising loss is defined as $\mathbb { E } _ { p _ { d a t a } ( \mathbf { x } ) p _ { \sigma } ( \tilde { \mathbf { x } } | \mathbf { x } ) } | | D _ { \theta } ( \tilde { \mathbf { x } } , \sigma ) - \mathbf { x } | |$ , where $D _ { \theta } ( \tilde { \mathbf { x } } , \sigma )$ is the denoising network, and x˜ is the noisy data sampled as $p _ { \sigma } ( \tilde { \mathbf { x } } | \mathbf { x } ) = \mathcal { N } ( \tilde { \mathbf { x } } ; \mathbf { x } , \sigma ^ { 2 } I )$ . During reverse process, the learned denoising network approximates the score function ∇x log $\upsilon ( { \bf x } ; \sigma ) = \bar { ( } D _ { \theta } ( \tilde { \bf x } , \bar { \sigma } ) - { \bf x } ) / \sigma ^ { 2 }$ enabling sample generation via numerical solvers as described in (Song et al., 2021).

Self-Conditioning. Self-conditioning is a tool to improve diffusion models by allowing the denoising network to utilize its own previous predictions as conditions (Chen et al., 2023). Specifically, self-conditioning feeds the previously predicted clean sample $\hat { \mathbf { x } } _ { 0 }$ back into the model as an additional input, which can largely enhance sample quality and consistency with minimal computational overhead. During training, self-conditioning is applied with a probability $p$ to prevent the model from over-relying on self-conditioned inputs, thereby improving the robustness of the learning process.

## 4 METHODOLOGY

Problem Definition. In this paper, we consider graph generation from scratch. Let N denote the number of nodes in a given graph, then a graph can be defined as $\mathcal { G } = ( H , A )$ , where $\pmb { H } \in \mathbb { R } ^ { N \times d _ { h } }$ denotes the node feature matrix, and $\pmb { A } \in \overline { { \mathbb { R } } } ^ { \hat { N } \times N }$ represents the adjacency matrix. For tasks related to biology and chemistry, the 3D coordinate matrix $\pmb { X } \in \mathbb { R } ^ { N \times 3 }$ can also be available. The graph autoencoder utilized in our latent diffusion framework consists of an encoder $\mathcal { E } _ { \phi }$ and a decoder $\mathcal { D } _ { \psi }$ The encoder $\mathcal { E } _ { \phi }$ transforms G into a latent representation $\boldsymbol { Z } \in \mathbb { R } ^ { N \times d }$ . We use $\dot { Z _ { l } } \in \mathbb { R } ^ { N \times d }$ to denote the local information and $\pmb { Z } _ { q } \in \mathbb { R } ^ { K \times d }$ to represent the global information, where $Z _ { g }$ is obtained by applying clustering to $\boldsymbol { Z } _ { l }$ and $K$ is the number of clusters. By employing advanced graph clustering methods, $Z _ { g }$ can effectively encode important global information within graphs. Our objective is to employ a unified latent diffusion model to model the joint distribution $p ( Z _ { l } , Z _ { g } )$

Overview. The primary motivation behind DualDiff is to endow the model with both global and local structural awareness. For example, in molecular generation tasks, DualDiff can accurately capture local details, such as specific functional groups or molecular fragments, while simultaneously considering the molecule’s overall topology and the distribution of substructures. The architecture of DualDiff is depicted in Figure 1. We first describe the construction of the autoencoder and global information extraction in Section 4.1. Then, the details of DualDiff will be illustrated in Section 4.2. Finally, we summarize the training and sampling scheme of DualDiff in Section 4.3.

## 4.1 AUTOENCODER CONSTRUCTION AND GLOBAL INFORMATION EXTRACTION

AutoEncoder. In graph-structured data, there often exist many meaningful features, such as molecular conformation positions and charges. To fully utilize these informative features, we leverage the Latent Graph Generation (LGD) paradigms to map graphs and the features into a unified latent space, thereby integrating rich feature information to facilitate generation. Specifically, we follow the settings in previous work (Rombach et al., 2022; Xu et al., 2023; Ketata et al., 2025) and pretrain an autoencoder to enable the transformation between graph space and latent space. The encoding and decoding processes can be formulated by $q _ { \phi } ( Z | \mathcal { G } ) \stackrel { - } { = } \dot { \mathcal { N } } ( \dot { \mathcal { E } _ { \phi } } ( \mathcal { G } ) , \sigma _ { 0 } I )$ and $\bar { p _ { \psi } } ( \mathcal { G } | Z _ { l } , Z _ { g } )$

![](images/6864122981b4ae3070b667dadce9d6716f6cf8c9f544739c8a02773040ee2d11.jpg)  
Figure 1: The workflow of DualDiff (Left) and details of the dual conditioning mechanism (Right).

Regarding the encoder architecture, in chemical and biological domains, nodes are generally endowed with 3D coordinate information, which inherently encodes spatial adjacency relationships. To exploit this property, we incorporate equivariance into $\mathcal { E } _ { \phi }$ by parameterizing the encoder with Equivariant Graph Neural Networks (EGNNs) (Satorras et al., 2021), thereby ensuring that the induced latent representations are SE(3)-equivariant. For general graph-structured data lacking explicit coordinate information, $\mathcal { E } _ { \phi }$ can be instantiated with various established graph neural network architectures, such as GIN (Xu et al., 2019), GCN (Kipf & Welling, 2017), and GAT (Velickoviˇ c´ et al., 2018), to effectively facilitate message passing among nodes. To reconstruct the original graph, our decoder exploits both global and local structural information, which can be implemented as lightweight MLPs. The overall framework is summarized as follows:

$$
\mathbf {Z} = \mathcal {E} _ {\phi} (\mathcal {G}) \implies \left\{ \begin{array}{l} \mathbf {Z} _ {l} = \mathbf {Z} + \sigma_ {0} \mathbf {I} \\ \mathbf {Z} _ {g} = \text { GlobalExtraction } (\mathbf {Z}, \mathcal {G}) \end{array} \right. \implies \hat {\mathcal {G}} = \mathcal {D} _ {\psi} (\mathbf {Z} _ {l}, \mathbf {Z} _ {g}),\tag{1}
$$

where GlobalExtraction represents the operation of global information extraction, which will be explained later. Then, the whole framework can be effectively optimized by:

$$
\mathcal {L} _ {r e c} = - \mathbb {E} _ {q (\mathcal {G}) q _ {\phi} (\mathbf {Z} | \mathcal {G})} \left[ p _ {\psi} (\mathcal {G} | \mathbf {Z} _ {l}, \mathbf {Z} _ {g}) \right].\tag{2}
$$

In practice, the reconstruction loss can be calculated as the $L _ { 2 }$ norm or the cross-entropy function. More details of the autoencoder and latent space can be found in Appendix A.1.

Global Information Extraction. Recent studies have demonstrated the critical role of global information in graph representation learning, particularly in applications like molecular property prediction and social network analysis (Yang et al., 2019; Fang et al., 2022). By incorporating global information, models can capture long-range dependencies and holistic structural patterns that cannot be fully represented by local features, thus largely improving the representation capability.

To extract global topological information from a graph, we leverage graph clustering methods (Schaeffer, 2007), which are powerful tools for uncovering intrinsic structural patterns within graphs. To obtain topologically-enhanced global information, we first employ different clustering strategies based on the nature of the graph. Specifically, for molecular graphs, we apply the K-means algorithm in the atom coordinate space to generate geometry-enhanced class labels. For generic graphs, we utilize spectral clustering (Von Luxburg, 2007) that partitions nodes according to the eigenvectors of the graph Laplacian matrix. Following the clustering process, we aggregate node features within each cluster using a pooling operation to derive cluster-level embeddings. The overall process of global information extraction can be formulated as follows:

$$
\boldsymbol {S} _ {g} = \text { Clustering } (\mathcal {G}) \implies \boldsymbol {Z} _ {g} = \text { Pooling } (\boldsymbol {S} _ {g}, \boldsymbol {Z}) \in \mathbb {R} ^ {K \times d},\tag{3}
$$

where $S _ { g } \in \{ 0 , 1 \} ^ { N \times K }$ is the affiliation matrix indicating the cluster assignment of each node. By introducing topologically-enhanced global information, our model can effectively capture dependencies among global structures and guide the generation of local details more efficiently. We provide more detailed explanations regarding the effectiveness of the clustering methods in Appendix C.8.

## 4.2 TWO-BRANCH DIFFUSION WITH DUAL CONDITIONING

To integrate global and local information, we propose a unified diffusion framework that explicitly models both node-level and cluster-level dependencies. Furthermore, a dual conditioning mechanism is proposed to facilitate effective information exchange between global and local information, thereby promoting the model to capture the joint distribution of global and local representations.

Two-Branch Diffusion on Global and Local Information. Given that global structures and local details within a graph generally exhibit distinct topological characteristics, we introduce a unified two-branch diffusion process operating at both the node and cluster levels, which enables the model to effectively capture local and global information, respectively. Following (Song et al., 2021), the forward process of this diffusion can be formulated as a system of stochastic differential equations (SDEs) defined in the latent space, as follows:

$$
\left\{ \begin{array}{l} d \mathbf {Z} _ {l, t} = f _ {l, t} (\mathbf {Z} _ {l, t}) d t + s _ {l, t} d \mathbf {W} _ {l, t}, \\ d \mathbf {Z} _ {g, t} = f _ {g, t} (\mathbf {Z} _ {g, t}) d t + s _ {g, t} d \mathbf {W} _ {g, t}, \end{array} \right.\tag{4}
$$

where $f _ { l , t } ( \cdot )$ and $f _ { g , t } ( \cdot )$ denote the drift coefficients, $s _ { l , t }$ and $s _ { g , t }$ are the diffusion coefficients, which are typically formulated as deterministic functions of time t, $W _ { l , t }$ and $W _ { g , t }$ represent the standard Wiener processes. Correspondingly, the reverse-time SDE system of (4) can be given by:

$$
\left\{ \begin{array}{l} d \bar {\mathbf {Z}} _ {l, t} = \left(f _ {l, t} (\bar {\mathbf {Z}} _ {l, t}) - s _ {l, t} ^ {2} \nabla_ {\mathbf {Z} _ {l}} \log p _ {t} (\bar {\mathbf {Z}} _ {l, t})\right) d \bar {t} + s _ {l, t} d \bar {\mathbf {W}} _ {l, t}, \\ d \bar {\mathbf {Z}} _ {g, t} = \left(f _ {g, t} (\bar {\mathbf {Z}} _ {g, t}) - s _ {g, t} ^ {2} \nabla_ {\mathbf {Z} _ {g}} \log p _ {t} (\bar {\mathbf {Z}} _ {g, t})\right) d \bar {t} + s _ {g, t} d \bar {\mathbf {W}} _ {g, t}, \end{array} \right.\tag{5}
$$

where $d \bar { t } = - d t$ is the negative infinitesimal time step, $\hat { W } _ { l , t }$ and $\bar { W } _ { g , t }$ represent the reverse-time Wiener processes, and $\nabla \log p _ { t } ( \cdot )$ is the score function. Under the EDM framework, we can set $f _ { l , t } ( Z _ { l , t } ) = \mathbf { 0 } , f _ { g , t } ( Z _ { g , t } ) = \mathbf { 0 }$ , and $s _ { l , t } = s _ { g , t } = \sqrt { 2 t }$ . To predict the clean global and local latent embeddings, we employ two separate denoising networks, $D _ { \theta _ { l } }$ and $D _ { \theta _ { q } }$ , which can be instantiated as GNNs in practice. Then, the overall training objective can be formulated as:

$$
\mathbb {E} _ {(\pmb {Z} _ {l, 0}, \pmb {Z} _ {g, 0}) \sim q _ {\phi} (\cdot | \mathcal {G}) p _ {\sigma} (\tilde {\pmb {Z}} _ {l}, \tilde {\pmb {Z}} _ {g} | \pmb {Z} _ {l, 0}, \pmb {Z} _ {g, 0})} [ \| D _ {\theta_ {l}} (\tilde {\pmb {Z}} _ {l}, \sigma) - \pmb {Z} _ {l, 0} \| ^ {2} + \| D _ {\theta_ {g}} (\tilde {\pmb {Z}} _ {g}, \sigma) - \pmb {Z} _ {g, 0} \| ^ {2} ].\tag{6}
$$

In the proposed two-branch diffusion process, the denoising networks are designed to capture nodelevel and cluster-level dependencies independently. However, this architecture has limitations in modeling the joint distribution between global and local information. To further promote dynamic information exchange and enable joint modeling of global and local representations, we introduce a dual conditioning mechanism, as described in the following part.

Dual Conditioning Mechanism. In this part, we propose a dual conditioning mechanism to achieve dynamic interactions between different topological knowledge, thereby equipping the model with comprehensive global and local topological awareness. According to the conditional probabil ity formulation, the joint distribution of global and local information can be expressed as:

$$
p (\mathbf {Z} _ {l}, \mathbf {Z} _ {g}) = p (\mathbf {Z} _ {l} | \mathbf {Z} _ {g}) p (\mathbf {Z} _ {g}) = p (\mathbf {Z} _ {g} | \mathbf {Z} _ {l}) p (\mathbf {Z} _ {l}).\tag{7}
$$

This equality implies that modeling the joint distribution of global and local information can be decomposed as two complementary processes: (i) introducing global information into the modeling of local details, associated with $p ( \dot { Z } _ { l } \dot { } | Z _ { q } )$ ; and (ii) incorporating local information into the modeling of global structures, associated with $\dot { p } ( \check { Z } _ { g } | Z _ { l } )$ .

Specifically, we first follow the self-conditioning and obtain the previously estimated $\hat { Z } _ { l , 0 }$ and $\hat { Z } _ { g , 0 }$ produced by $D _ { \theta _ { l } }$ and $D _ { \theta _ { q } }$ , respectively. Then, we combine local and global information as conditions to alternately guide the local and global diffusion process. Let $C _ { l }$ and $C _ { g }$ respectively denote the conditions introduced to the local and global diffusion processes, then they can be given by:

$$
(\boldsymbol {C} _ {l}, \boldsymbol {C} _ {g}) = \left\{ \begin{array}{l l} ((\hat {\boldsymbol {Z}} _ {l, 0}, \hat {\boldsymbol {Z}} _ {g, 0}), \mathbf {0}), & \text {with prob. p ,} \\ (\mathbf {0}, (\hat {\boldsymbol {Z}} _ {l, 0}, \hat {\boldsymbol {Z}} _ {g, 0})), & \text {with prob. 1 - p ,} \end{array} \right. \quad \text {related to p(Z_{l} |Z_{g})p(Z_{g})}\tag{8}
$$

where p denotes the frequency of process (i). In detail, during process (i), to match the first decomposition $p ( Z _ { l } | Z _ { g } ) p ( Z _ { g } )$ in (7), we use $C _ { l } = ( \hat { Z } _ { l , 0 } , \hat { Z } _ { g , 0 } )$ to incorporate both self-condition and global information to guide the learning of Z<sub>l</sub>, thereby modeling $p ( \boldsymbol { \hat { Z } } _ { l } | Z _ { g } )$ . Meanwhile, $C _ { g }$ is set to 0 to align with $p ( Z _ { g } )$ . Similarly, we can obtain $( \boldsymbol { C } _ { l } , \boldsymbol { C } _ { g } ) = ( \mathbf { 0 } , ( \hat { Z } _ { l , 0 } , \hat { Z } _ { g , 0 } ) )$ during process (ii). More interestingly, this strategy naturally matches the self-conditioning, where the condition is set to zero with a certain probability to enhance the robustness of the generation process, further guaranteeing the effectiveness of the proposed dual conditioning. A comparison of different conditioning methods is provided in Figure 2. Next, we will present a detailed description of the two processes used in the dual conditioning mechanism, as illustrated in the right part of Figure 1.

(i) Incorporating global information into the local structure. According to the conditional probability $p ( Z _ { l } \vert \bar { Z } _ { g } )$ we can introduce the global information to promote the generation process of local topologies. Concretely, inspired by FiLM (Perez et al., 2018), we first pass the predicted global representation $\hat { Z } _ { g , 0 }$ through linear transformation to yield scale and shift parameters, $\gamma ^ { i } \in \mathbb { R } ^ { \check { d } }$ and $\beta ^ { i } \in \mathbb { R } ^ { d } ( i \mathrm { ~ = ~ } 1 , \dots , K )$ Next, each node is assigned to a cluster $y _ { i }$ based on the similarity between $\hat { Z } _ { l , 0 }$ and $\hat { Z } _ { g , 0 }$

![](images/a73093b63e89f9e3494da9702d02a52ba1bfcdbdef0a6b6c72e1015c5de24874.jpg)  
(a)

![](images/59c63d09cf30db0ed02267985b544801a94553d6acbc0c3b75118708d02d86da.jpg)  
(b)

![](images/5bd2ea39ce37b8f586c7e24760b97ae670d71ac96057f0803a9891e3d307e144.jpg)  
(c)  
Figure 2: Comparison between different conditioning methods during the reverse process. (a) diffusion without conditioning; (b) self-conditioning; (c) dual conditioning.

Finally, the scale and shift parameters associated with each cluster are utilized to steer the distribution of the final output $\hat { Z } _ { l , 0 } ^ { \prime } ,$ thereby integrating global awareness into the modeling of local structures. Formally, we have:

$$
\hat {\mathbf {Z}} _ {l, 0} ^ {\prime , i} = \gamma^ {y _ {i}} \odot \hat {\mathbf {Z}} _ {l, 0} ^ {i} + \beta^ {y _ {i}}, \text {where} y _ {i} = \underset {j = 1, \ldots , K} {\arg \max} \text {sim} (\hat {\mathbf {Z}} _ {l, 0} ^ {i}, \hat {\mathbf {Z}} _ {g, 0} ^ {j}), \hat {\mathbf {Z}} _ {l, 0} = \text {GNN} (\mathbf {Z} _ {l, t}, \hat {\mathbf {Z}} _ {l, 0}, \sigma_ {t}).\tag{9}
$$

(ii) Incorporating local information into the global topology. In addition to process (i), local information can also be leveraged to help predict global structures, thus modeling $p ( Z _ { g } | Z _ { l } )$ . Specifically, the estimated local details $\hat { \boldsymbol Z } _ { l , 0 }$ can be processed through message passing (MP) and pooling (Pool) operations to obtain a low-dimensional global condition C. This condition is then concatenated with the self-condition $\hat { Z } _ { g , 0 }$ as the final condition of the denoising network. Formally, we have:

$$
\boldsymbol {C} = \operatorname{Linear} (\operatorname{Pool} (\mathrm{MP} (\hat {\boldsymbol {Z}} _ {l, 0}))) \Rightarrow \boldsymbol {C} = \operatorname{Concat} (\hat {\boldsymbol {Z}} _ {g, 0}, \boldsymbol {C}) \Rightarrow \hat {\boldsymbol {Z}} _ {g, 0} ^ {\prime} = \operatorname{GNN} (\boldsymbol {Z} _ {g, t}, \boldsymbol {C}, \sigma_ {t}).\tag{10}
$$

In conclusion, the dual conditioning mechanism can naturally integrate processes (i) and (ii), enabling dynamic interaction between global and local information and equipping DualDiff with global and local topological awareness, which largely promotes the effectiveness of the generation process. We also include more details of the dual conditioning mechanism in Appendix A.4.

## 4.3 TRAINING AND SAMPLING

In this section, we present the detailed training and sampling procedures for DualDiff. Consistent with the latent diffusion model (LDM) framework, we employ a two-stage training strategy: an autoencoder is first pretrained, and remains fixed throughout both the diffusion training and sampling phases. To facilitate a more stable sampling process, we draw inspiration from the server-client communication paradigm, where the server updates global parameters after accumulating several updates from clients to promote training stability (McMahan et al., 2017). We analogize the global clusters to the role of the global server and their corresponding nodes to the local clients. Correspondingly, process (i) and process (ii) can be viewed as local update and global aggregation, respectively. Therefore, we alternate m steps of process (i) with a single step of process (ii) to enhance the stability of the sampling process. Besides, this alternating strategy can significantly improve sampling stability and model performance, compared to executing two processes simultaneously. Further details regarding the effectiveness of dual conditioning and the corresponding theoretical insights can be found in Appendix D. Finally, with the integration of dual conditioning, the overall loss objectives are updated as follows:

$$
\mathbb {E} _ {(\pmb {Z} _ {l, 0}, \pmb {Z} _ {g, 0}) \sim q _ {\phi} (\cdot | \mathcal {G}) p _ {\sigma} (\tilde {\pmb {Z}} _ {l}, \tilde {\pmb {Z}} _ {g} | \pmb {Z} _ {l, 0}, \pmb {Z} _ {g, 0})} [ \| D _ {\theta_ {l}} (\tilde {\pmb {Z}} _ {l}, \pmb {C} _ {l}, \sigma) - \pmb {Z} _ {l, 0} \| ^ {2} + \| D _ {\theta_ {g}} (\tilde {\pmb {Z}} _ {g}, \pmb {C} _ {g}, \sigma) - \pmb {Z} _ {g, 0} \| ^ {2} ].\tag{11}
$$

Overall, the training and sampling process can be shown in Algorithm 1 and Algorithm 2.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Training Algorithm of DualDiff
Input: input graph G, denoising networks  $D_{\theta_l}$  and  $D_{\theta_g}$, autoencoder  $E_\phi$  and  $D_\psi$, noise scheduler  $\{\sigma_t\}$, diffusion steps T, training epochs E.
Output: learned  $D_{\theta_l}$  and  $D_{\theta_g}$.
1: Pretrain  $E_\phi$  and  $D_\psi$  by optimizing (2), and then fix the parameters.
2:  $Z_{l,0}, Z_{g,0} \sim q_\phi(\cdot|G)$
3: for i = 0 to E - 1 do
4:    $t \sim U(0,T)$
5:    $Z_{l,t}, Z_{g,t} \sim p_{\sigma_t}(\tilde{Z}_l, \tilde{Z}_g | Z_{l,0}, Z_{g,0})$
6:    Obtain  $C_l$  and  $C_g$  by (8).
7:    Update  $D_{\theta_l}$  and  $D_{\theta_g}$  by (11).
8: end for
9: Return:  $D_{\theta_l}, D_{\theta_g}$

Algorithm 2 Sampling Algorithm of DualDiff
Input: trained  $D_{\theta_l}$  and  $D_{\theta_g}$, trained decoder  $D_\psi$, noise scheduler  $\{\sigma_t\}$, sampling steps T.
Output: generated graph  $\hat{G}$.
1:  $\hat{Z}_{l,T}, \hat{Z}_{g,T} \sim prior$
2: for t = T - 1 to 0 do
3:    if t % (m + 1) == 0 then
4:    $(C_l, C_g) = (\mathbf{0}, (\hat{Z}_{l,0}, \hat{Z}_{g,0}))$
5:    else  $(C_l, C_g) = ((\hat{Z}_{l,0}, \hat{Z}_{g,0}), \mathbf{0})$
6:    $\hat{Z}_{l,t} = solver(D_{\theta_l}(\hat{Z}_{l,t+1}, C_l, \sigma_{t+1}))$
7:    $\hat{Z}_{g,t} = solver(D_{\theta_g}(\hat{Z}_{g,t+1}, C_g, \sigma_{t+1}))$
8: end for
9:  $\hat{G} = p_\psi(\hat{Z}_{l,0}, \hat{Z}_{g,0})$
10: Return:  $\hat{G}$
</div>

## 5 EXPERIMENT

Experimentally, we conduct comprehensive evaluations of our proposed DualDiff framework on two widely adopted graph generation tasks: generic graph generation and molecular generation.

## 5.1 EXPERIMENTAL SETUP

Datasets. Following previous methods, we leverage eight graph datasets to evaluate our proposed DualDiff extensively. For generic graph generation tasks, we include Ego-small, Community-small, Grid, Planar, and SBM datasets. For molecular graphs, we involve mainstream ZINC250k, QM9, and MOSES datasets. For a fair comparison, we follow the experimental and evaluation setting of (Liu et al., 2025; Jo et al., 2022; Vignac et al., 2022) with the same train/test split. More details and descriptions of datasets can be found in Appendix B.2.

Baselines. We compare DualDiff with competitive generative models. Specifically, for generic graph generation, we choose competitive GraphRNN (You et al., 2018), GDSS (Jo et al., 2022), DiGress (Vignac et al., 2022), GSDM (Luo et al., 2023), Graphusion (Yang et al., 2024), GruM (Jo et al., 2023), GBD (Liu et al., 2025), and GraphBFN (Song et al., 2025a). For molecular generation, we additionally select advanced MoLeR (Maziarz et al., 2021), GraphArm (Kong et al., 2023), MicCaM (Kong et al., 2023), MAGNet (Geng et al., 2023), , SwinGNN (Yan et al., 2023), LGD (Zhou et al., 2024), and EDM-SyCo (Ketata et al., 2025).

Evaluation Metrics. Following (You et al., 2018; Du et al., 2021; Jo et al., 2022), for generic graph generation, we use the maximum mean discrepancy (MMD) to compare the distributions of statistics between the same number of generated and test graphs, such as distributions of degree (Deg.), clustering coefficient (Clus.), the number of occurrences of orbits with 4 nodes (Orbit), and the eigenvalues of the graph Laplacian (Spec.). In addition, we report V.U.N. scores, representing the percentages of valid, unique, and novel graphs, to evaluate how well the model captures both intrinsic characteristics and global graph properties. For molecular generation, we follow the metrics used in Guacamol (Brown et al., 2019), and report scores of KL divergence (KL), the Frechet´ ChemNet Distance (FCD) (Preuer et al., 2018), Novelty, Uniqueness, and Validity.

Table 1: Comparison of advanced models on Planar and SBM datasets. More experiments on the Ego-Small, Community-small, and Grid datasets are included in Appendix C.1.

<table><tr><td rowspan="2">DatasetsMetrics</td><td colspan="5">Planar</td><td colspan="5">SBM</td></tr><tr><td>Deg.↓</td><td>Clus.↓</td><td>Orbit↓</td><td>Spec.↓</td><td>V.U.N.↑</td><td>Deg.↓</td><td>Clus.↓</td><td>Orbit↓</td><td>Spec.↓</td><td>V.U.N.↑</td></tr><tr><td>Training set</td><td>0.0002</td><td>0.0310</td><td>0.0005</td><td>0.0052</td><td>100.0</td><td>0.0008</td><td>0.0332</td><td>0.0255</td><td>0.0063</td><td>100.0</td></tr><tr><td>GraphRNN</td><td>0.0049</td><td>0.2779</td><td>1.2543</td><td>0.0459</td><td>0.0</td><td>0.0055</td><td>0.0584</td><td>0.0785</td><td>0.0065</td><td>5.0</td></tr><tr><td>GDSS</td><td>0.0041</td><td>0.2676</td><td>0.1720</td><td>0.0370</td><td>0.0</td><td>0.0212</td><td>0.0646</td><td>0.0894</td><td>0.0128</td><td>5.0</td></tr><tr><td>DiGress</td><td>0.0003</td><td>0.0372</td><td>0.0009</td><td>0.0106</td><td>75.0</td><td>0.0013</td><td>0.0498</td><td>0.0434</td><td>0.0400</td><td>74.0</td></tr><tr><td>GBD</td><td>0.0003</td><td>0.0353</td><td>0.0135</td><td>0.0059</td><td>92.5</td><td>0.0013</td><td>0.0493</td><td>0.0043</td><td>0.0047</td><td>75.0</td></tr><tr><td>GruM</td><td>0.0005</td><td>0.0353</td><td>0.0009</td><td>0.0062</td><td>90.0</td><td>0.0007</td><td>0.0492</td><td>0.0448</td><td>0.0050</td><td>85.0</td></tr><tr><td>GraphBFN</td><td>0.0005</td><td>0.0294</td><td>0.0002</td><td>0.0046</td><td>96.7</td><td>0.0005</td><td>0.0560</td><td>0.0370</td><td>0.0053</td><td>87.5</td></tr><tr><td>DualDiff</td><td>0.0003</td><td>0.0275</td><td>0.0002</td><td>0.0038</td><td>97.5</td><td>0.0004</td><td>0.0473</td><td>0.0365</td><td>0.0042</td><td>85.0</td></tr></table>

Table 2: Experiments on ZINC250K dataset. Following previous studies, we report the FCD and KL scores, where higher values indicate better performance. The results are reported from EDM-SyCo and the original papers. Methods that do not report the KL metric are denoted as N.A.

<table><tr><td></td><td>Method</td><td>FCD score (↑)</td><td>KL score (↑)</td><td>Novelty (↑)</td><td>Uniqueness (↑)</td><td>Validity (↑)</td></tr><tr><td rowspan="5">Autoreg.</td><td>GraphAF</td><td>0.05 ± 0.00</td><td>0.67 ± 0.01</td><td>0.91 ± 0.01</td><td>0.91 ± 0.01</td><td>1.00 ± 0.00</td></tr><tr><td>MoLeR</td><td>0.83 ± 0.00</td><td>0.97 ± 0.00</td><td>0.99 ± 0.00</td><td>0.99 ± 0.00</td><td>1.00 ± 0.00</td></tr><tr><td>GraphArm</td><td>0.04 ± 0.00</td><td>N.A.</td><td>1.00 ± 0.00</td><td>0.99 ± 0.00</td><td>0.88 ± 0.00</td></tr><tr><td>MiCaM</td><td>0.63 ± 0.02</td><td>0.94 ± 0.00</td><td>0.98 ± 0.00</td><td>0.98 ± 0.00</td><td>1.00 ± 0.00</td></tr><tr><td>MAGNet</td><td>0.76 ± 0.00</td><td>0.95 ± 0.00</td><td>0.99 ± 0.00</td><td>0.99 ± 0.00</td><td>1.00 ± 0.00</td></tr><tr><td rowspan="6">One-shot</td><td>GDSS</td><td>0.10 ± 0.01</td><td>N.A.</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>0.97 ± 0.01</td></tr><tr><td>DiGress</td><td>0.65 ± 0.00</td><td>0.91 ± 0.00</td><td>0.99 ± 0.00</td><td>0.99 ± 0.00</td><td>0.85 ± 0.01</td></tr><tr><td>GruM</td><td>0.64 ± 0.01</td><td>N.A.</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>0.99 ± 0.00</td></tr><tr><td>SwinGNN</td><td>0.67 ± 0.00</td><td>N.A.</td><td>0.96 ± 0.00</td><td>1.00 ± 0.00</td><td>0.91 ± 0.00</td></tr><tr><td>EDM-SyCo</td><td>0.85 ± 0.01</td><td>0.96 ± 0.00</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>0.88 ± 0.01</td></tr><tr><td>DualDiff</td><td>0.91 ± 0.02</td><td>0.98 ± 0.01</td><td>1.00 ± 0.00</td><td>1.00 ± 0.00</td><td>0.92 ± 0.02</td></tr></table>

## 5.2 GENERIC GRAPH GENERATION

From Table 1, we can observe that our proposed DualDiff achieves competitive performance across diverse metrics. Essentially, generic graphs demonstrate pronounced geometric properties. The topologically-enhanced global information can effectively capture the core features of the dataset, thereby providing valuable guidance for the generation of local information. We further validate our model’s performance on the mainstream Community-small, Ego-small, and Grid datasets, as shown in Table 8, further illustrating the effectiveness and applicability of our model.

## 5.3 MOLECULAR GENERATION

We also evaluate DualDiff on molecular generation tasks. As shown in Table 2, DualDiff surpasses other advanced methods on the FCD and KL metrics, and exhibits competitive performance on the novelty and validity metrics. Fundamentally, molecular structures demonstrate dependencies across multiple scales, encompassing both atomic dependencies within motifs and interactions between distinct motifs. By equipping the generative model with both global and local awareness, DualDiff can substantially improve the quality of the generated molecules. Besides, most methods achieve novelty and uniqueness scores approaching 100%, which can be primarily attributed to the large molecular sizes. The resulting exponential growth in the number of possible molecular structures makes it difficult to sample molecules identical to those in the training and test sets.

Due to the page limits, we present the performance of our model on the QM9 dataset in Appendix C.2 and on the large-scale molecular dataset MOSES in Appendix C.4. In both cases, DualDiff achieves competitive results compared to other advanced methods, further demonstrating the effectiveness of our approach. In addition to unconditional graph generation, DualDiff can also be applied to conditional generation tasks. Detailed experimental results are provided in Appendix C.5, where DualDiff likewise demonstrates strong conditional generation performance.

## 5.4 COMPREHENSIVE COMPARISON WITH HIERARCHICAL MODELS

Hierarchical graph generation methods Karami (2024); Bergmeister et al. (2024), which model complex graphs by decomposing them into multiple levels and progressively generating structures from coarse to fine, have achieved notable success in graph generation. In contrast to these autoregressive and coarse-to-fine modeling approaches, our dual conditioning mechanism enables dynamic interaction between global and local information. This allows for a more accurate capture of the joint distribution of global and local features compared to hierarchical methods. We provide a comparison of model performance between our method and hierarchical approaches, as shown in Table 3, further demonstrating the effectiveness of our approach.

Table 3: Comparison between advanced hierarchical models.

<table><tr><td rowspan="2">MethodsMetrics</td><td colspan="4">Planar</td><td colspan="4">SBM</td></tr><tr><td>Deg.</td><td>Clus.</td><td>Orbit</td><td>Spec.</td><td>Deg.</td><td>Clus.</td><td>Orbit</td><td>Avg.</td></tr><tr><td>PPGN (Bergmeister et al., 2024)</td><td>0.0003</td><td>0.0245</td><td>0.0006</td><td>0.0104</td><td>0.0119</td><td>0.0517</td><td>0.0669</td><td>0.0067</td></tr><tr><td>HiGen (Karami, 2024)</td><td>0.0012</td><td>0.0435</td><td>0.0234</td><td>0.0025</td><td>0.0017</td><td>0.0503</td><td>0.0604</td><td>0.0068</td></tr><tr><td>DualDiff (Ours)</td><td>0.0003</td><td>0.0275</td><td>0.0002</td><td>0.0038</td><td>0.0004</td><td>0.0473</td><td>0.0365</td><td>0.0042</td></tr></table>

## 5.5 EXPERIMENTS ON GENERATING 3D MOLECULES

In this subsection, we will evaluate the effectiveness of our approach in 3D molecular tasks. Following GEOLDM (Xu et al., 2023), we replace the lightweight MLP decoder with an EGNN to ensure that both latent embedding and coordinates are simultaneously generated. The experimental results on the QM9 dataset are shown in Table 4. Our method achieves competitive performance compared to the advanced GEOLDM and EQUIFM (Song et al., 2023), which further illustrates the effectiveness of our methods on generating 3D molecules.

Table 4: Experiments of 3D molecular generation on the QM9 Dataset.

<table><tr><td>Methods</td><td>Atom Sta (%)</td><td>Mol Sta (%)</td><td>Valid &amp; Unique (%)</td></tr><tr><td>Data</td><td>99.0</td><td>95.2</td><td>97.7</td></tr><tr><td>GEOLDM</td><td>98.9</td><td>89.4</td><td>92.7</td></tr><tr><td>EQUIFM</td><td>98.9</td><td>88.3</td><td>93.5</td></tr><tr><td>DualDiff</td><td>98.9</td><td>88.7</td><td>99.3</td></tr></table>

## 5.6 MODEL ANALYSIS

Ablation study. We conduct an ablation study on the proposed dual conditioning mechanism, as shown in Table 5. Specifically, we compare the model performance of DualDiff without any conditioning, with self-conditioning, with only global-to-local conditioning, and with full dual conditioning on the ZINC250k dataset.

Table 5: Ablation study of dual conditioning.

The results demonstrate that dual conditioning can substantially improve the quality of the generated graphs and significantly outperforms the conventional self-conditioning approach, underscoring the importance of the interaction between global and local topological information.

<table><tr><td>Method</td><td>FCD (↑)</td><td>KL (↑)</td></tr><tr><td>DualDiff (w/o any cond.)</td><td>0.65 ± 0.01</td><td>0.82 ± 0.02</td></tr><tr><td>DualDiff (w self cond.)</td><td>0.72 ± 0.01</td><td>0.89 ± 0.02</td></tr><tr><td>DualDiff ( $Z_l \to Z_g$ )</td><td>0.75 ± 0.03</td><td>0.95 ± 0.02</td></tr><tr><td>DualDiff ( $Z_g \to Z_l$ )</td><td>0.83 ± 0.01</td><td>0.95 ± 0.01</td></tr><tr><td>DualDiff ( $Z_g \leftrightarrow Z_l$ )</td><td>0.91 ± 0.02</td><td>0.98 ± 0.01</td></tr></table>

Parameter Analysis. We conduct an

extensive evaluation of model performance under varying key parameters, including the frequency p of process (i) during training, m steps used in Section 4.3, and the number of clusters K. Detailed results on the ZINC250k dataset are presented in Figure 3. The left panel indicates that selecting a moderate value for $p$ facilitates effective interaction between global and local information. The middle panel shows that increasing m during sampling enables the model to focus more on complex local details, leading to enhanced performance. Finally, the right panel highlights the significance of an appropriate cluster number $\begin{array} { r } { \dot { K } \colon \qquad } \end{array}$ a smaller K not only reduces computational overhead but also preserves essential global information. Besides, we also include the detailed ablation studies of different clustering methods, and cluster numbers in Appendix C.6.

![](images/57ab4f23a64aee9d8073a38e4fdc080e5f59d32f17232231b4e769f7c8e86a86.jpg)  
Figure 3: Experiments of model parameters on ZINC250k dataset.

Model Efficiency. Apart from the advanced generation performance, our model also demonstrates high efficiency. As shown in Table 6, DualDiff achieves competitive results using only approximately 200 steps. We further evaluate the actual time to generate 10,000 molecular graphs on the QM9 and ZINC250k datasets, as reported in Table 7, where ”w/o dual.” refers to the time without dual conditioning. The results show that the dual conditioning mechanism introduces acceptable computational overhead, and our model consistently maintains high efficiency. Additionally, we provide further explanations and experiments regarding the efficiency of DualDiff in Appendix C.7.

Table 6: Model performance at different diffusion steps on Community-small and Ego-small datasets.  
Table 7: Experimental evaluation of the model’s sampling efficiency.

<table><tr><td>Dataset</td><td>Models</td><td>50</td><td>100</td><td>200</td><td>500</td><td>1000</td></tr><tr><td rowspan="3">Comm.</td><td>GDSS</td><td>0.110</td><td>0.104</td><td>0.072</td><td>0.061</td><td>0.046</td></tr><tr><td>GSDM</td><td>0.109</td><td>0.079</td><td>0.021</td><td>0.011</td><td>0.009</td></tr><tr><td>DualDiff</td><td>0.098</td><td>0.043</td><td>0.009</td><td>0.007</td><td>0.006</td></tr><tr><td rowspan="3">Ego.</td><td>GDSS</td><td>0.040</td><td>0.031</td><td>0.023</td><td>0.021</td><td>0.017</td></tr><tr><td>GSDM</td><td>0.044</td><td>0.026</td><td>0.024</td><td>0.019</td><td>0.016</td></tr><tr><td>DualDiff</td><td>0.032</td><td>0.012</td><td>0.005</td><td>0.005</td><td>0.004</td></tr></table>

<table><tr><td>Dataset</td><td>Model</td><td>Sampling (s)</td></tr><tr><td rowspan="3">QM9</td><td>GDSS</td><td>109</td></tr><tr><td>DualDiff</td><td>41</td></tr><tr><td>- w/o dual.</td><td>32</td></tr><tr><td rowspan="3">ZINC250k</td><td>GDSS</td><td>1870</td></tr><tr><td>DualDiff</td><td>223</td></tr><tr><td>- w/o dual.</td><td>169</td></tr></table>

## 6 LIMITATIONS AND FUTURE DIRECTIONS

The proposed DualDiff is currently primarily designed for generating generic and simple molecular graphs. However, as a general paradigm, DualDiff can effectively model local and global information simultaneously, and thus can be broadly transferred to other data with hierarchical characteristics, such as proteins, peptides, and even circuit design. We consider these extensions as promising directions for future work and will explore the applicability of DualDiff to diverse domains.

## 7 CONCLUSION

In this paper, we design a unified global and local topology-aware generative model, i.e., DualDiff. DualDiff employs a two-branch diffusion framework and dual conditioning mechanism, which can effectively model the joint distribution between global and local information, thereby naturally capturing the complex and entangled multi-scale relationships inherent in graphs. Empirical results demonstrate that our proposed model significantly enhances the quality of generated graphs while further ensuring training stability and sampling efficiency.

## ACKNOWLEDGMENTS

The research work described in this paper was conducted in the JC STEM Lab of Machine Learning and Symbolic Reasoning, funded by The Hong Kong Jockey Club Charities Trust.