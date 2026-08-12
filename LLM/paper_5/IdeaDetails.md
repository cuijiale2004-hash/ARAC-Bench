# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
Large language models (LLMs) perform strongly on many language tasks but still struggle with complex multi-step reasoning across disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, as well as guiding principles for question synthesis. Existing question synthesis methods fall into two categories, both possessing significant pain points:
1. **Query-centric approaches**: These methods expand seed questions through rewriting, added constraints (e.g., Evol-Instruct), or incorporating chain-of-thought reasoning. However, they are limited by the coverage of the seed pool and inherent model biases.
2. **Document-centric approaches**: These methods generate questions from unstructured or structured source documents, ensuring broad disciplinary coverage grounded in authentic knowledge. However, they struggle to control difficulty and diversity, often degenerating into factual recall. 

Meanwhile, post-training (e.g., SFT) and even mid-training all rely heavily on exam-style data. This raises a key challenge: how can we rapidly synthesize large volumes of high-quality, human-like multidisciplinary exam questions while controlling their difficulty, diversity, and question types?

## 2. Core Research Motivation and Scientific Questions
**Core Research Motivation**: The scarcity of large-scale, high-quality, and diverse training data limits the development of LLMs’ multidisciplinary reasoning capabilities. Existing datasets focus mainly on math and programming, while most other disciplines lack comparable resources. 
**Scientific Question**: How can we leverage naturally available, extensive raw documents to generate multidisciplinary questions that possess complex reasoning patterns, while maintaining explicit control over difficulty, diversity, and question types?

## 3. Overall Core Idea and Design Philosophy
The central insight is the notion of **Design Logic**, a form of reusable meta-knowledge that encapsulates the structured process human experts use to transform knowledge into complex exam questions. When human education experts design challenging and insightful questions, they do not merely state facts. Instead, they follow a structured design process: identifying objectives, constructing contexts, designing reasoning paths, formulating answers, adding distractors, and validating the questions. Solvers must engage in multi-step reasoning beyond memorization. 

The Design Logic abstracts the underlying reasoning structure, enabling LLMs to generate new questions with the same complex reasoning patterns from entirely different source texts. By reverse-engineering the meta-knowledge of human educators, the approach provides structured, reusable, and abstract control over difficulty and diversity that prior document-centric methods lacked.

## 4. Core Innovation Points
1. **Proposal of "Design Logic"**: A fundamentally new question synthesis method guided by "Design Logic", which abstracts the strategic process that human experts use to create challenging questions. This approach enables the generation of truly complex, multi-step reasoning questions from raw text by providing the structured, reusable, and abstract control over difficulty and diversity that prior document-centric methods lacked.
2. **Reverse-engineering Design Logics**: Using LLMs to reverse-engineer and abstract over 120,000 Design Logics from existing questions across various disciplines to construct a reusable Design Logic library. This addresses the absence of guiding principles in prior data synthesis methods.
3. **Two-stage Retrieve-and-Generate Mechanism**: Designing a coarse-to-fine matching mechanism that utilizes vector similarity for coarse retrieval of candidate logics and an LLM for fine-grained evaluation to select the optimal logic and generate reasoning questions, ensuring precise alignment between text and logic.
4. **Construction of Large-Scale Multidisciplinary Datasets**: Constructing two large-scale reasoning datasets spanning 75 disciplines: DLR-Book (3.04 million questions) and DLR-Web (1.66 million questions). Data analysis indicates these synthesized questions exhibit greater difficulty and diversity compared to baseline datasets. SFT on Qwen3 and Llama3 with this data substantially improves multidisciplinary reasoning, even surpassing their official final models that have undergone full post-training.

## 5. Overview of the Overall Technical Solution
The overall technical solution, DESIGNER, is a design-logic-guided reasoning data synthesis pipeline that consists of three phases:
* **Phase 1: Data Extraction and Preprocessing**: Processing large-scale book and web corpora with multi-dimensional labeling and filtering to construct a high-quality source material library. Processing a question bank via clustering-based question selection for Design Logic extraction.
* **Phase 2: Core Synthesis**: Extracting and deduplicating Design Logics from the question bank. Adopting a two-stage retrieve-and-generate mechanism to match Design Logics with raw corpus and synthesize diverse, multidisciplinary questions with reference answers.
* **Phase 3: Filtering and Output**: Performing question deduplication, decontamination, and discipline labeling to output the final datasets (DLR-Book and DLR-Web). Synthesizing long CoT responses for the questions.

## 6. Detailed Module Design

### 6.1 Data Curation Module
This module processes three data sources aligned to a unified 75-discipline taxonomy:
* **Question Bank Processing**: Annotate over 150 million questions with discipline, difficulty, and question types using Qwen3-30B-A3B (non-thinking mode). Compute embeddings with Qwen3-Embedding-4B and apply K-means clustering within each discipline (cluster numbers determined by silhouette search). From each cluster, draw an equal number of questions following a fixed difficulty ratio of 3:2:1 (Very Hard:Hard:Medium), aligning per-discipline sizes with the overall bank distribution. If higher-difficulty questions are insufficient, backfill with lower-difficulty ones. This yields 132,409 questions.
* **Book Corpus Processing**: Process at the chapter level. Split chapters over 5,000 words into smaller blocks and deduplicate via MinHash. Assign discipline labels by a ModernBERT-large classifier. Predict readability with a BERT-based model to filter incoherent text; score helpfulness (0–5) by the fineweb-edu-classifier. Remove segments with negative readability, rank remaining by helpfulness, and select to meet discipline quotas proportional to frequencies in the book corpus and question bank. Yields three million high-quality segments (most with helpfulness $\ge$ 2).
* **Web Corpus Processing**: Apply reasoning-oriented filtering and discipline relabeling to the FineFineWeb corpus. Score 6.5 billion texts with Qwen3-30B-A3B using a five-level rubric and retain those with scores $\ge$ 3. Relabel selected texts to align with the 75-discipline taxonomy.

### 6.2 Design Logic Extraction Module
Using a specific prompt, instruct an LLM (DeepSeek-R1-0528) to analyze authentic questions to: (i) infer the designer’s thought process, (ii) trace construction from knowledge points, and (iii) abstract underlying design principles. The output is expressed in Mermaid format, producing a reusable pool of Design Logics.

### 6.3 Design Logic Deduplication Module
Deduplicate extracted logics using semantic similarity. Embed each logic with Qwen3-Embedding-4B, and pairwise similarities yield a matrix $S \in \mathbb{R}^{n \times n}$. Within each discipline, construct a graph $G = (V,E)$ where nodes represent logics and edges connect pairs with $S_{ij} \ge \tau$. Connected components correspond to redundant Design Logic groups. From each group, retain the item with the highest similarity sum. With $\tau = 0.85$, this yields 125,328 unique Design Logics.

### 6.4 Question Synthesis Module (Two-stage Retrieve-and-Generate)
To avoid combinatorial explosion, adopt a retrieval-augmented approach:
1. **Coarse Ranking**: For each discipline-specific corpus, compute the cosine similarity between embeddings of a text segment $t$ and a Design Logic $d$ using Qwen3-Embedding-4B with task-specific instructions. Retain the top-5 logics with the highest similarity as candidates.
2. **Fine Ranking & Generation**: Prompt DeepSeek-R1-0528 to: (i) select the most suitable logic from the top-5 candidates, and (ii) synthesize a graduate-level exam question strictly following its steps. For each question, the LLM also generates a concise reference answer.

### 6.5 Filtering and Response Synthesis Module
* **Question Deduplication and Decontamination**: A two-stage filtering pipeline: (i) MinHash-based deduplication to remove near-duplicates, and (ii) 13-gram decontamination against all evaluation benchmarks to prevent leakage.
* **Response Synthesis**: Employ Qwen3-235B-A22B-Thinking-2507-FP8 to generate a corresponding long CoT response for each synthesized question.

## 7. All Mathematical Formulas and Symbol Definitions

### 7.1 Design Logic Retrieval Similarity
* $s(t, d)$: Cosine similarity between a text segment $t$ and a Design Logic $d$.
* $e(t)$: Embedding of text segment $t$.
* $e(d)$: Embedding of Design Logic $d$.
$$s(t, d) = \cos(e(t), e(d))$$

### 7.2 Design Logic Deduplication Variables
* $S \in \mathbb{R}^{n \times n}$: Pairwise semantic similarity matrix of Design Logics.
* $G = (V,E)$: Undirected graph where nodes $V$ represent logics and edges $E$ connect pairs with similarity $S_{ij} \ge \tau$.
* $\tau$: Similarity threshold ($\tau = 0.85$).
* $C$: A connected component (cluster of duplicates) in graph $G$.
* $i^*$: Centroid index, the most representative item in a cluster.
$$i^* = \text{argmax}_{i \in C} \sum_{j \in C, j \neq i} S_{ij}$$

### 7.3 Semantic Diversity Metrics
Let $X = \{x_1, x_2, \ldots, x_N\}$ be a given question set, and $E = \{e_1, e_2, \ldots, e_N\}$ be the corresponding set of embedding vectors, where $e_i \in \mathbb{R}^d$ is the d-dimensional embedding for question $x_i$. $N = 300,000$.

* **Mean Cosine Distance**: The average cosine distance between all unique pairs of embeddings. Higher value indicates greater semantic dissimilarity.
$$M_{cosine} = \frac{2}{N(N-1)} \sum_{i<j} (1 - \cos(e_i, e_j))$$

* **Mean L2 Distance**: The average Euclidean distance between all unique pairs of embeddings. Measures the average separation of questions in the embedding space.
$$M_{L2} = \frac{2}{N(N-1)} \sum_{i<j} \|e_i - e_j\|_2$$

* **1-Nearest Neighbor Distance (1-NN Distance)**: The average cosine distance from each embedding to its single nearest neighbor, where $d_1(e_i)$ is the cosine distance from $e_i$ to its closest neighbor. Highlights tightly clustered, near-identical questions.
$$M_{1\text{-}NN} = \frac{1}{N} \sum_{i} d_1(e_i)$$

* **Cluster Inertia**: The total squared distance of samples to their closest cluster center after applying K-means, where $c_j$ are the cluster centroids. Measures the overall spread and density of the data clusters.
$$M_{inertia} = \sum_{i=1}^N \min_j \|e_i - c_j\|_2^2$$

* **Radius**: The geometric mean of the standard deviations of the embedding dimensions, modeling the data as a multi-dimensional Gaussian distribution, where $\sigma_j$ is the standard deviation along the j-th dimension. Directly quantifies the spread of the data in the semantic space.
$$M_{Radius} = \left(\prod_{j=1}^d \sigma_j\right)^{1/d}$$

## 8. Algorithm Pseudocode

**Algorithm 1: Graph-based Deduplication via Centroid Selection**

1: **Input:** A set of items $D = \{d_1, \ldots, d_n\}$, a similarity matrix $S \in \mathbb{R}^{n \times n}$, a similarity threshold $\tau$.
2: **Output:** A deduplicated set of representative items $R$.
3: 
4: Initialize an undirected graph $G = (V,E)$ where $V = \{1, \ldots, n\}$ and $E = \emptyset$.
5: Initialize the set of representatives $R = \emptyset$.
6: 
7: // Build a similarity graph where nodes are items and edges connect similar items.
8: **for** $i = 1$ **to** $n$ **do**
9:   **for** $j = i+1$ **to** $n$ **do**
10:     **if** $S_{ij} > \tau$ **then**
11:       Add edge $(i, j)$ to $E$.
12:     **end if**
13:   **end for**
14: **end for**
15: 
16: // Identify clusters of duplicates by finding connected components.
17: Let $C \leftarrow \text{FindConnectedComponents}(G)$.
18: 
19: // Select the most representative item (centroid) from each cluster.
20: **for** each connected component $C \in \mathcal{C}$ **do**
21:   Find centroid index $i^* = \text{argmax}_{i \in C} \sum_{j \in C, j \neq i} S_{ij}$.
22:   Add item $d_{i^*}$ to $R$.
23: **end for**
24: 
25: **return** $R$.