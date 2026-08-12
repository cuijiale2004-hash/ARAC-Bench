### 1. Research Background and Existing Pain Points

**Research Background:**
Multimodal Large Language Models (MLLMs) face significant computational overhead when processing long videos due to the massive number of visual tokens required. The rapid advancement of Large Language Models (LLMs) has significantly propelled progress in multimodal understanding, leading to the emergence of numerous prominent Large Vision Language Models (LVLMs). These LVLMs typically process images or video frames by converting them into visual tokens via pre-trained visual encoders, which are then fed into LLMs. Particularly in video understanding, they show promise in processing longer, higher-resolution complex videos. However, this powerful capability also introduces significant computational challenges, especially in video understanding scenarios. Due to the need to process continuous sequences of frames, the number of visual tokens grows exponentially, far exceeding that of static images, posing a severe challenge to the training and inference efficiency of LVLMs. A standard Large Vision-Language Model (LVLM) first encodes the video into a full sequence of N visual tokens, Tfull = {t1, t2, . . . , tN}. This process is computationally demanding, as the self-attention mechanism’s complexity scales quadratically with the token count N , i.e., O(N2). Therefore, how to efficiently process massive video visual tokens and reduce computational overhead remains a critical issue in current LVLM research.

**Existing Pain Points:**
To address the substantial overhead of video processing, visual token compression has become a critical method for improving the efficiency of LVLMs. Existing methods primarily approach compression from two aspects: token importance and similarity. One category is importance-based token pruning, aiming to identify and remove visual tokens that contribute less to model performance. These methods typically utilize attention scores or other feature metrics to quantify token importance. The other category is similarity-based merging/selection, recognizing that even important tokens can exhibit high similarity, leading to informational redundancy. Despite the success of existing methods in visual token reduction, they face two key limitations:
1. **Lack of collaborative modeling for spatio-temporal relationships:** Existing token pruning methods often focus on spatial correlations within the same frame or temporal correlations at the same positions across different frames. This lack of joint modeling and analysis of spatio-temporal correlations prevents them from effectively capturing complex dynamic events. Existing similarity-driven methods mainly only focus on the temporal similarity between adjacent frames, failing to utilize complex spatio-temporal relevant information. This limits their ability to fully exploit inherent video redundancy, restricting performance in extreme token reduction scenarios.
2. **Neglect of crucial changes and differences:** These methods share a potential blind spot: they mainly focus on informational commonalities, like similarity and importance, while neglecting crucial changes and differences. Since the narrative of the video is often driven by turning events, a model that only seeks similarity might smooth over a sudden action, leading to a misinterpretation of the content. Existing hybrid baseline methods' common optimization goal is still to identify and preserve representative “commonality” information, with a theoretical blind spot of potentially overlooking key narrative information driven by “turning points” and “abrupt events”.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Based on the limitations of existing methods, we propose a new perspective: **similarity is for identifying redundancy, while difference is for capturing key moments.** We believe that an ideal video token compression algorithm should achieve two goals simultaneously: representing the stable content of a video with the fewest tokens, and precisely preserving the key changes. If similarity defines the “norm” of a video, then difference defines its “events”. A model focusing only on similarity may excel at understanding “what is” but struggle with “what happened”. A video’s plot is driven by key events, and the essence of an event is change—the appearance of a new object, the start of an action, or a scene transition. In our model, these changes manifest as a sharp difference in the features of temporally adjacent tokens. Therefore, accurately capturing these “turning points” of significant difference is crucial for understanding the video’s dynamic content and correctly answering questions like “when” and “why”.

**Scientific Questions:**
- How to uniformly model the complex spatio-temporal relationships between video tokens to capture both spatiotemporal similarity and temporal difference?
- How to design a dual token selection function that can simultaneously screen for both representative and pivotal event tokens, balancing spatiotemporal similarity and difference?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Balancing spatiotemporal similarity and difference for efficient video understanding with MLLMs. Spatiotemporal Similarity can be utilized to compress redundant information both spatially within a frame and temporally across adjacent frames. Temporal Difference should be detected to capture the key actions or events that define the plot.

**Design Philosophy:**
To this end, we designed a training-free framework named ST-SimDiff. We first construct a spatio-temporal graph from the visual tokens to uniformly model their complex associations. Subsequently, we employ a parallel dual-selection strategy: 1) similarity-based selection uses community detection to retain representative tokens, compressing static information; 2) temporal difference-based selection precisely locates content-changing points to preserve tokens that capture key dynamic shifts. This allows it to preserve both static and dynamic content with a minimal number of tokens.

### 4. Core Innovation Points

- We are the first to propose that the similarity and difference of tokens should be given equal importance in VLM efficiency research.
- We design a spatio-temporal graph framework to uniformly model the complex spatio-temporal relationships between video tokens.
- We propose a novel dual token selection strategy that can simultaneously screen for both representative and pivotal event tokens.
- We conduct extensive experiments on video understanding benchmarks, and the results show that ST-SimDiff significantly reduces the visual token budget while maintaining or even improving performance.

### 5. Overview of the Overall Technical Solution

Given a video V and a text query Q, a standard Large Vision-Language Model (LVLM) first encodes the video into a full sequence of N visual tokens, Tfull = {t1, t2, . . . , tN}. Our task is therefore to design an efficient token selection function, f(·), that takes the full sequence Tfull and a compression ratio r to produce a much smaller subset Tsub = f(Tfull, r). The objective is to maximize the LVLM’s downstream task performance using this subset, subject to the constraint that its size |Tsub| is approximately r · N . 

The proposed ST-SimDiff method comprises two key components that operate in parallel on a spatio-temporal graph constructed from the video’s visual tokens: Similarity-based Representative Token Selection (SRTS) and Difference-based Event Token Selection (DETS). First, in the SRTS module, we identify and condense the video’s stable and redundant content. By applying community detection algorithms to the graph, we group highly similar tokens into clusters, which correspond to persistent visual elements like static backgrounds. From each cluster, we then select a few highly central tokens to form a representative set, Trep. Second, the DETS module is designed to capture the video’s crucial dynamic shifts. It analyzes the temporal edges of the graph, identifying moments where the similarity between corresponding tokens in adjacent frames drops sharply. These points of high difference signify turning points, and the tokens framing these transitions are preserved as critical event tokens, forming the set Ttrans. By synergistically combining these two components, ST-SimDiff generates a final token subset Tsub = Trep ⋃ Ttrans, which is both highly compact and information-rich, retaining stable content while precisely capturing key events.

### 6. Detailed Module Design

**6.1 Spatio-Temporal Graph Construction**
Given a video, the vision encoder transforms it into a sequence of tokens T = {t1, t2, . . . , tN}. Each token tk ∈ T corresponds to a specific spatio-temporal location and is associated with a feature vector xk. We define the location of each token tk by its frame index T (tk), its spatial height index H(tk), and its spatial width index W (tk).
We model the relationships between these tokens using a spatio-temporal graph G = (V,E), where the vertex set V represents all tokens in T . The edge set E is the union of two distinct subsets, E = ES ∪ET , which capture spatial and temporal relationships respectively. The spatial edges ES connects tokens that are spatially adjacent within the same frame. The temporal edges ET connects tokens that are at the same spatial position but in adjacent frames, capturing temporal continuity. The weight of any edge w(vi, vj) ∈ E is defined by the cosine similarity of the feature vectors of the corresponding tokens. This sparse graph structure efficiently encodes local spatial relationships and temporal continuity, providing a unified foundation for our dual-path token selection strategy.

**6.2 Similarity-based Representative Token Selection**
Video content is inherently characterized by substantial spatio-temporal redundancy. Our spatio-temporal graph is designed to capture these joint similarities. Within this graph, semantically related tokens form dense “communities” or “clusters” based on their feature similarity, irrespective of their specific frame or location. Therefore, selecting representatives from each community emerges as a highly efficient strategy for compressing redundant information while preserving core semantics.
To accurately identify strongly correlated token clusters, we first threshold the graph G. We set a similarity threshold τsim and retain only the edges with weights above this threshold, forming a new graph G′ = (V,E′). Next, we apply a graph community detection algorithm (e.g., the Louvain method or connected components) on G′ to identify token clusters C = {c1, c2, . . . , cm}. For each community ck ∈ C, we rank and filter its internal tokens based on their centrality. The centrality score Sc(ta) of a token ta ∈ ck is defined as its average similarity with all other tokens within the community. Subsequently, we set the intra-community retention rate to the globally preset compression ratio r, and preserve the top ⌈|ck| · r⌉ tokens with the highest centrality scores from each community ck. This uniform filtering process naturally handles all communities, including single-node communities composed of unique tokens (in which case the sole node is retained). In summary, this strategy efficiently compresses the static and persistent content of the video by identifying and retaining the most central tokens within each semantic cluster.

**6.3 Difference-based Event Token Selection**
Difference, particularly along the temporal dimension, often signals the occurrence of a key event. For example, the entry of a new object, a scene change, or the beginning/end of an action will cause drastic changes in the visual features at corresponding positions in adjacent frames. We specifically analyze the temporal edges (ET ) in our spatio–temporal graph. For any temporal edge (vk, vl) ∈ ET in the graph, which connects two temporally adjacent tokens tk and tl, we set a dynamic threshold τ (e.g., the 95th percentile of all temporal edge difference scores). When the difference score of a temporal edge (vk, vl) exceeds this threshold, i.e., w(tk, tl) < τdiff, we consider the subsequent token tl as a critical event token and retain it. With this strategy, tokens that signify moments of significant temporal change can be preserved. By capturing these key transitions, we ensure that the crucial dynamic aspects of the video’s narrative are not overlooked during the compression process.

**6.4 Overall Reduction Process**
In the proposed framework, we first compute the representative token set Trep and the event token set Tevent in parallel, and take their union, Tcandidate = Trep ∪ Tevent, as the initial candidate set. To precisely meet the target token count Ntarget = ⌈r · N⌉, we introduce a final pruning step. If the size of the candidate set, |Tcandidate|, exceeds Ntarget, we remove the |Tcandidate| − Ntarget tokens with the lowest importance. Following existing work, the importance of a token is determined by its attention score in the shallow layers of the LLM. This step ensures that our method can flexibly meet any computational budget while preserving critical information. In summary, our overall method adeptly combines two strategies: graph-based token selection and attention-based dynamic pruning. The former leverages the intrinsic structure of the video content (similarity and difference) to ensure the retention of core information, while the latter provides a flexible mechanism to precisely meet any given computational budget. This design makes our token compression process both principled and adaptive, thereby achieving a robust balance between efficiency and performance.

**6.5 Computational Complexity Analysis**
The initial stage involves building the graph. For each of the N visual tokens, we compute the cosine similarity with its spatially and temporally adjacent neighbors. As each token has a small, constant number of neighbors, this step requires a single pass over all tokens, resulting in a complexity of O(Nd), where d is the feature dimension of each token. Following graph construction, the SRTS module identifies token clusters using a connected components algorithm, which operates with a near-linear time complexity of O(N + |E|). Since the number of edges |E| is at most 3N , this simplifies to O(N). To prevent the subsequent filtering step from being computationally intensive, we impose a constraint on the maximum size of any community. In the rare event that a community exceeds a predefined threshold (e.g., |ck| > √N ), we partition it. The DETS module involves a single pass through all temporal edges in the graph to identify significant changes by comparing token similarities against a threshold. This process has a linear complexity of O(Nd). In summary, the total complexity of our ST-SimDiff framework is governed by these linear-time operations, culminating in an overall complexity of O(Nd). This is substantially more efficient than the quadratic complexity of the self-attention mechanism, O(N2d), used in the Large Language Model.

### 7. All Mathematical Formulas and Symbol Definitions

**Mathematical Formulas:**
- Spatial edges ES:
  ES ={(vi, vj) ∈ V × V | T (ti) = T (tj) and |H(ti)−H(tj)|+ |W (ti)−W (tj)| = 1}
- Temporal edges ET:
  ET ={(vi, vj) ∈ V × V | H(ti) = H(tj), W (ti) = W (tj), and |T (ti)− T (tj)| = 1}
- Edge weight w(vi, vj):
  w(vi, vj) = (xi · xj) / (∥xi∥∥xj∥)
- Thresholded edge set E′:
  E′ = {(vi, vj) ∈ E | w(vi, vj) > τsim}
- Centrality score Sc(ta):
  Sc(ta) = (1 / (|ck| − 1)) ∑_{tb∈ck,b̸=a} w(ta, tb)
- Representative token set Trep:
  Trep = ⋃_{k=1}^{m} TopK_{t∈ck,score=Sc(t)} (⌈|ck| · r⌉)
- Event token set Tevent:
  Tevent ={tl | ∃tk s.t. (vk, vl) ∈ ET, T (tl) > T (tk), and w(vk, vl) < τdiff}
- Target token count Ntarget:
  Ntarget = ⌈r · N⌉
- Token subset Tsub:
  Tsub = f(Tfull, r)
- Constraint on subset size:
  |Tsub| ≈ r · N
- Initial candidate set Tcandidate:
  Tcandidate = Trep ∪ Tevent

**Symbol Definitions:**
- V: Input video
- Q: Text query
- Tfull: Full sequence of N visual tokens, Tfull = {t1, t2, . . . , tN}
- N: Total number of visual tokens
- f(·): Efficient token selection function
- r: Compression ratio
- Tsub: Much smaller subset of tokens
- G = (V,E): Spatio-temporal graph
- V: Vertex set representing all tokens in T
- E: Edge set, E = ES ∪ET
- ES: Spatial edges
- ET: Temporal edges
- tk: Token corresponding to a specific spatio-temporal location
- xk: Feature vector of token tk
- T(tk): Frame index of token tk
- H(tk): Spatial height index of token tk
- W(tk): Spatial width index of token tk
- w(vi, vj): Weight of any edge defined by the cosine similarity of the feature vectors
- xi, xj: Feature vectors of tokens ti, tj
- τsim: Similarity threshold
- G′ = (V,E′): Thresholded graph
- C = {c1, c2, . . . , cm}: Token clusters/communities identified by community detection algorithm
- ck: A specific community within C
- Sc(ta): Centrality score of a token ta within community ck
- Trep: Set of representative tokens
- τ: Dynamic threshold for temporal edge difference (e.g., the 95th percentile of all temporal edge difference scores)
- τdiff: Difference threshold (implemented as 0.2)
- Tevent: Set of event tokens
- Tcandidate: Initial candidate set
- Ntarget: Target token count

### 8. Algorithm Pseudocode

The original paper does not provide explicit algorithm pseudocode blocks, but the flowchart logic and iterative process are extracted exactly as follows:

1. **Input:** Full sequence of N visual tokens Tfull = {t1, t2, . . . , tN}, compression ratio r, similarity threshold τsim, difference threshold τdiff.
2. **Spatio-Temporal Graph Construction:**
   - Initialize vertex set V representing all tokens in T.
   - Construct spatial edge set ES: For any (vi, vj) ∈ V × V, if T (ti) = T (tj) and |H(ti)−H(tj)|+ |W (ti)−W (tj)| = 1, add (vi, vj) to ES.
   - Construct temporal edge set ET: For any (vi, vj) ∈ V × V, if H(ti) = H(tj), W (ti) = W (tj), and |T (ti)− T (tj)| = 1, add (vi, vj) to ET.
   - Compute edge weights: For any (vi, vj) ∈ ES ∪ ET, compute w(vi, vj) = (xi · xj) / (∥xi∥∥xj∥).
3. **Similarity-based Representative Token Selection (SRTS):**
   - Form thresholded graph G′ = (V,E′) where E′ = {(vi, vj) ∈ E | w(vi, vj) > τsim}.
   - Apply graph community detection algorithm on G′ to identify token clusters C = {c1, c2, . . . , cm}.
   - Initialize Trep = ∅.
   - For each community ck ∈ C:
     - Compute centrality score Sc(ta) = (1 / (|ck| − 1)) ∑_{tb∈ck,b̸=a} w(ta, tb) for each token ta ∈ ck.
     - Select TopK_{t∈ck,score=Sc(t)} (⌈|ck| · r⌉) and add to Trep.
4. **Difference-based Event Token Selection (DETS):**
   - Initialize Tevent = ∅.
   - For each temporal edge (vk, vl) ∈ ET:
     - If T (tl) > T (tk) and w(vk, vl) < τdiff:
       - Add token tl to Tevent.
5. **Overall Reduction Process:**
   - Compute initial candidate set: Tcandidate = Trep ∪ Tevent.
   - Compute target token count: Ntarget = ⌈r · N⌉.
   - If |Tcandidate| > Ntarget:
     - Compute importance of each token in Tcandidate using its attention score in the shallow layers of the LLM.
     - Remove the |Tcandidate| − Ntarget tokens with the lowest importance.
   - **Output:** Final token subset Tsub.