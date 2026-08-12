1. Research Background and Existing Pain Points
Graph-structural data provide powerful representations for describing objects and their relationships in diverse real-world domains, such as social networks, biological systems, and traffic networks. Among the core tasks in graph machine learning, graph generation has emerged as a powerful tool with significant potential in diverse applications such as molecular structure modeling, circuit design, and source code generation. Currently, advances in deep models, including generative adversarial networks, diffusion-based models, and flow-based methods, have paved the way for effective graph generation. However, generating graph-structured data poses challenges due to the complex dependencies inherent in graphs, spanning from intricate local substructures to broad global topologies. Unlike densely distributed image data, graph data can be extremely sparse and often exhibits complex topological information, especially the coupled local and global dependencies within distinct structures. For instance, social networks exhibit intricate dependencies both within individual communities and between different communities; likewise, molecular graphs demonstrate complex relationships both within and among substructures. Specifically, the local structure of a molecule, such as the specific functional groups, plays a crucial role in determining its chemical reactivity. However, the global information, including the overall structural topology and the spatial distribution of functional groups, also significantly impacts its overall properties and functional performance. Traditional node-level generative paradigms may have difficulty simultaneously capturing the multiscale dependencies in graphs. Although recent advances like SubgDiff and Graphusion incorporate global information, these approaches may overlook the dependencies among substructures and have limitations in modeling the joint distribution of global and local information.

2. Core Research Motivation and Scientific Questions
The intrinsic and distinct dependencies present in graphs highlight the importance of integrating both global and local information to enable a more precise graph generation process. Therefore, a natural and important scientific question arises: How can we design a unified generative model that can dynamically capture both global and local topological information to enable effective graph generation? To answer this question, the research aims to equip the generative model with both global and local topological awareness, enabling it to dynamically capture the dependencies within the coarse-grained global structures and the fine-grained local details.

3. Overall Core Idea and Design Philosophy
The overall core idea is to propose a unified dual conditioning latent diffusion model, termed DualDiff, which is capable of effectively modeling the joint distribution of global and local information, facilitating a robust and stable generation process. The design philosophy centers on jointly learning local and global topological information. Concretely, DualDiff first maps the original graph space and diverse graph features into a unified latent space via a pretrained graph autoencoder, then it extracts the global information according to the local representations. Subsequently, DualDiff employs a two-branch diffusion process to learn topological dependencies at both the node and subgraph levels within a unified framework. To further advance the joint modeling of global and local information, a dual conditioning mechanism is introduced to promote interaction between these two branches, wherein global and local information are alternately utilized as conditions to guide the learning process of the complementary branch. By leveraging the two-branch diffusion process and dual conditioning, DualDiff can be equipped with both global and local topological awareness.

4. Core Innovation Points
1. Highlight the limitations of conventional graph generative models in capturing the joint distribution of global and local topological information within graphs. Propose DualDiff, an effective latent diffusion-based generative model capable of simultaneously learning the dependencies inherent in both global and local structures.
2. Incorporate a two-branch diffusion process operating at both the node and cluster levels, which enables the model to effectively capture local and global information, respectively, within a unified framework.
3. Introduce a novel dual conditioning mechanism that facilitates effective information exchange between global and local information. This design endows the model with both global and local topological awareness, enabling more effective learning of the diverse dependencies present in graph-structured data, and promoting the model to capture the joint distribution of global and local representations.
4. Design a stable sampling strategy inspired by the server-client communication paradigm, where the global clusters are analogized to the global server and their corresponding nodes to the local clients. Alternating m steps of local update (incorporating global info to local) with a single step of global aggregation (incorporating local info to global) enhances the stability of the sampling process and model performance compared to executing two processes simultaneously.

5. Overview of the Overall Technical Solution
The overall technical solution follows a unified latent diffusion framework with a two-stage training strategy. First, an autoencoder is pretrained to map graphs and diverse features into a unified latent space, enabling the transformation between graph space and latent space. Global topological information is extracted via graph clustering methods (spectral clustering for generic graphs, K-means for molecular graphs), aggregating node features within clusters. Second, a two-branch diffusion process is employed on global and local information in the latent space, formulated as a system of SDEs. Two separate denoising networks are used to predict clean global and local latent embeddings. To promote dynamic information exchange and enable joint modeling, a dual conditioning mechanism is introduced. During training, global and local information are alternately utilized as conditions: process (i) introduces global information into the modeling of local details, and process (ii) incorporates local information into the modeling of global structures. The overall loss objective is updated accordingly. During sampling, to facilitate a more stable process, an alternating strategy is adopted: m steps of process (i) are alternated with a single step of process (ii), mimicking a server-client communication paradigm for stability. The trained decoder finally reconstructs the generated graph from the sampled latent embeddings.

6. Detailed Module Design
- AutoEncoder Construction and Global Information Extraction:
  - AutoEncoder: Maps the original graph space and diverse graph features into a unified latent space. The encoder Eϕ transforms G into a latent representation Z ∈ RN×d. For molecular graphs with 3D coordinate information, Eϕ is parameterized with Equivariant Graph Neural Networks (EGNNs) to ensure SE(3)-equivariance. For general graphs, Eϕ can be instantiated with GIN, GCN, or GAT. The latent representation Zv for each node v is computed via layer-wise message passing. The decoder Dψ exploits both global and local structural information to reconstruct the original graph, implemented as lightweight MLPs. It first utilizes a FiLM layer to integrate global and local information, obtaining global context-enhanced node embeddings. Then, it employs two separate lightweight MLPs to recover graph-structured data, i.e., the node features and adjacency matrix. For molecular graphs, coordinate information is further leveraged to facilitate more accurate adjacency matrix reconstruction. The latent space allows the model to fully utilize informative node features and implicitly model relationships among nodes with O(N) complexity instead of O(N2).
  - Global Information Extraction: Leverages graph clustering methods to uncover intrinsic structural patterns. For molecular graphs, the K-means algorithm is applied in the atom coordinate space to generate geometry-enhanced class labels. For generic graphs, spectral clustering partitions nodes according to the eigenvectors of the graph Laplacian matrix. Following clustering, node features within each cluster are aggregated using a pooling operation to derive cluster-level embeddings.

- Two-Branch Diffusion with Dual Conditioning:
  - Two-Branch Diffusion on Global and Local Information: Introduces a unified two-branch diffusion process operating at both the node and cluster levels. The forward process is formulated as a system of stochastic differential equations (SDEs) defined in the latent space. The reverse-time SDE system is correspondingly defined. Under the EDM framework, drift coefficients are set to 0, and diffusion coefficients to √(2t). Two separate denoising networks, Dθl and Dθg (GNNs), are employed to predict the clean global and local latent embeddings.
  - Dual Conditioning Mechanism: Achieves dynamic interactions between different topological knowledge. According to the conditional probability formulation, modeling the joint distribution can be decomposed into two complementary processes: (i) introducing global information into the modeling of local details (p(Zl|Zg)p(Zg)) and (ii) incorporating local information into the modeling of global structures (p(Zl)p(Zg|Zl)).
  - Process (i): Inspired by FiLM, the predicted global representation Ẑg,0 is passed through linear transformation to yield scale and shift parameters (γi and βi). Each node is assigned to a cluster yi based on similarity between Ẑl,0 and Ẑg,0. The affiliation matrix Sg ∈ RN×K is calculated based on Ẑl,0 and Ẑg,0, where Si,jg = 1, if j = argmaxj sim(Ẑil,0, Ẑjg,0) and Si,jg = 0, otherwise. The scale and shift parameters associated with each cluster are utilized to steer the distribution of the final output Ẑ'l,0.
  - Process (ii): The estimated local details Ẑl,0 are processed through message passing (MP) and pooling (Pool) operations to obtain a low-dimensional global condition C. Specifically, Cj = ∑Ni=1 Sijg ·Ẑil,0 / ∑Ni=1 Sijg. This condition is concatenated with the self-condition Ẑg,0 as the final condition of the denoising network to predict Ẑ'g,0.
  - Theoretical Insights: The joint conditional probability path pt(Zl, Zg) can be factorized into three equivalent forms: 1) simultaneously incorporating both global and local conditions; 2) incorporating local information into the generation of global information; 3) incorporating global information into the generation of local information. pt(Zl|Zl,0) can be modeled as N (Zl;µt(Zl,0), σt(Zl,0)), corresponding to a standard diffusion process without additional conditions, i.e., Cl = 0 in process (ii). Besides, pt(Zg|Zl, Zg,0) = N (Zg;µt(Zl, Zg,0), σt(Zl, Zg,0)), which can be seen as a conditional diffusion process with Zl as the condition. By further integrating self-conditioning, we obtain Cg = (Ẑl,0, Ẑg,0). Similarly, in process (i), (Cl, Cg) = (((Ẑl,0, Ẑg,0), 0)) corresponds to modeling pt(Zl|Zg, Zl,0)pt(Zg|Zg,0).

- Training and Sampling:
  - Training: A two-stage training strategy. Autoencoder is pretrained first and fixed. The dual conditioning mechanism is applied with a probability p during training to prevent over-reliance.
  - Sampling: Inspired by the server-client communication paradigm, alternating m steps of process (i) with a single step of process (ii) enhances stability. Global clusters are the global server, nodes are local clients.

7. All Mathematical Formulas and Symbol Definitions
- N: number of nodes
- G = (H,A): graph definition, H ∈ RN×dh denotes the node feature matrix, A ∈ RN×N represents the adjacency matrix
- X ∈ RN×3: 3D coordinate matrix
- Eϕ, Dψ: encoder and decoder of the graph autoencoder
- Z ∈ RN×d: latent representation
- Zl ∈ RN×d: local information
- Zg ∈ RK×d: global information, where K is the number of clusters
- q(xt|xt−1) = N (xt;√1− βtxt−1, βtI): forward process of diffusion
- βt: noise schedule
- pθ(xt−1|xt): reverse process of diffusion
- Dθ(x̃, σ): denoising network in EDM
- x̃: noisy data sampled as pσ(x̃|x) = N (x̃;x, σ2I)
- Epdata(x)pσ(x̃|x)||Dθ(x̃, σ)−x|| : denoising loss in EDM
- ∇x log p(x;σ) = (Dθ(x̃, σ) − x)/σ2: score function
- Eq 1: Z = Eϕ(G) =⇒{Zl = Z + σ0I; Zg = GlobalExtraction(Z,G) =⇒ Ĝ = Dψ(Zl,Zg)
- Eq 2: Lrec = −Eq(G)qϕ(Z|G) [pψ(G|Zl,Zg)]
- Eq 3: Sg = Clustering(G) =⇒ Zg = Pooling(Sg,Z) ∈ RK×d
- Sg ∈ {0, 1}N×K: affiliation matrix indicating the cluster assignment of each node
- Eq 4: {dZl,t = fl,t(Zl,t)dt+ sl,tdWl,t; dZg,t = fg,t(Zg,t)dt+ sg,tdWg,t}
- fl,t(·) and fg,t(·): drift coefficients
- sl,t and sg,t: diffusion coefficients
- Wl,t and Wg,t: standard Wiener processes
- Eq 5: {dZ̄l,t = (fl,t(Z̄l,t)− s2l,t∇Zl log pt(Z̄l,t))dt̄+ sl,tdW̄l,t; dZ̄g,t = (fg,t(Z̄g,t)− s2g,t∇Zg log pt(Z̄g,t))dt̄+ sg,tdW̄g,t}
- dt̄ = −dt: negative infinitesimal time step
- W̄l,t and W̄g,t: reverse-time Wiener processes
- ∇ log pt(·): score function
- Eq 6: E(Zl,0,Zg,0)∼qϕ(·|G)pσ(Z̃l,Z̃g|Zl,0,Zg,0)[∥Dθl(Z̃l, σ)−Zl,0∥2 + ∥Dθg (Z̃g, σ)−Zg,0∥2]
- Eq 7: p(Zl,Zg) = p(Zl|Zg)p(Zg) = p(Zg|Zl)p(Zl)
- Eq 8: (Cl,Cg) ={((Ẑl,0, Ẑg,0),0), with prob. p, related to p(Zl|Zg)p(Zg); (0, (Ẑl,0, Ẑg,0)), with prob. 1− p, related to p(Zl)p(Zg|Zl)}
- p: frequency of process (i)
- Eq 9: Ẑ ′,i l,0 = γyi ⊙ Ẑi l,0 + βyi , where yi = argmaxj=1,...,K sim(Ẑi l,0, Ẑj g,0), Ẑl,0 = GNN(Zl,t, Ẑl,0, σt)
- γi ∈ Rd and βi ∈ Rd: scale and shift parameters
- Eq 10: C = Linear(Pool(MP(Ẑl,0))) ⇒ C = Concat(Ẑg,0,C) ⇒ Ẑ ′ g,0 = GNN(Zg,t,C, σt)
- C: low-dimensional global condition
- Eq 11: E(Zl,0,Zg,0)∼qϕ(·|G)pσ(Z̃l,Z̃g|Zl,0,Zg,0)[∥Dθl(Z̃l,Cl, σ)−Zl,0∥2+∥Dθg (Z̃g,Cg, σ)−Zg,0∥2]
- Eq 12: Z(k)v = UPDATE(k)(Z(k−1)v ,AGG(k)(Z(k−1)u : u ∈ N (v)))
- Eq 13: γ, β = Linear(SgZg), Z̃l = γ ⊙Zl + β
- Eq 14: X̂ = MLPnode(Z̃l)
- Eq 15: Â = σ(MLPadj(Z̃lW Z̃Tl))
- Eq 16: Âij = MLPadj(Concat[Z(h)i ,Z(h)j , ||Z(x)i −Z(x)j ||22])

8. Algorithm Pseudocode
Algorithm 1 Training Algorithm of DualDiff
Input: input graph G, denoising networks Dθl and Dθg , autoencoder Eϕ and Dψ , noise scheduler {σt}, diffusion steps T , training epochs E.
Output: learned Dθl and Dθg .
1: Pretrain Eϕ and Dψ by optimizing (2), and then fix the parameters.
2: Zl,0,Zg,0 ∼ qϕ(·|G)
3: for i = 0 to E − 1 do
4: t ∼ U(0, T )
5: Zl,t,Zg,t ∼ pσt(Z̃l, Z̃g|Zl,0,Zg,0)
6: Obtain Cl and Cg by (8).
7: Update Dθl and Dθg by (11).
8: end for
9: Return: Dθl , Dθg

Algorithm 2 Sampling Algorithm of DualDiff
Input: trained Dθl and Dθg , trained decoder Dψ , noise scheduler {σt}, sampling steps T .
Output: generated graph Ĝ.
1: Ẑl,T , Ẑg,T ∼ prior
2: for t = T − 1 to 0 do
3: if t % (m+ 1) == 0 then
4: (Cl,Cg) = (0, (Ẑl,0, Ẑg,0))
5: else (Cl,Cg) = ((Ẑl,0, Ẑg,0),0)
6: Ẑl,t = solver(Dθl(Ẑl,t+1,Cl, σt+1))
7: Ẑg,t = solver(Dθg (Ẑg,t+1,Cg, σt+1))
8: end for
9: Ĝ = pψ(Ẑl,0, Ẑg,0)
10: Return: Ĝ