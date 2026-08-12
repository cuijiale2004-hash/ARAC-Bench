## 1. Research Background and Existing Pain Points

**Research Background:** 
Controllable generation is a fundamental task in the era of Large Language Models (LLMs). It provides the foundation for stable tool use, agentic communication, and interaction with existing application programming interfaces (APIs). Specifically, structured output generation requires models to generate in predefined formats (e.g., code, JSON, XML, or tables), supporting tasks such as entity extraction, classification, and correlation prediction. 

**Existing Pain Points:**
1. **Limitations of Autoregressive (AR) LLMs:** Even state-of-the-art autoregressive LLMs exhibit unreliability when required to generate structured output. This stems from architectural constraints: (1) Left-to-right generation prevents global structural coherence, as early tokens are generated without full sequence knowledge; (2) Commitment to previously generated tokens limits backtracking when structural violations occur; (3) Sequential dependencies inhibit simultaneous satisfaction of multiple constraints. 
2. **Limitations of Existing Constrained Decoding Methods:** Prior autoregressive-based methods pair a grammar-driven finite-state automata (FSA) with constrained decoding to enforce structural constraints. When no token satisfies the grammar, all beams are pruned and generation halts. Prompt-engineering heuristics are labor-intensive and yield inconsistent results across domains and complexity levels. Existing approaches rely solely on language models’ intrinsic capabilities without additional mechanisms to guide generation trajectories.
3. **Limitations of Existing Diffusion-based LLMs (dLLMs):** Current open-sourced instruction-tuned dLLMs fail to produce well-structured outputs, often generating hallucinated content or breaking structural constraints. Their inference speed remains limited. Existing implementations (e.g., semi-autoregressive approaches) systematically undermine dLLMs’ advantages in global awareness and parallel generation.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** 
Inspired by the current new diffusion-based large language models (dLLM), the architectural difference—especially the global information-sharing mechanism for language modeling—may be the key to unlock next-level controllable generation. Unlike AR models, dLLMs iteratively denoise corrupted inputs, enabling global context modeling potential and parallel token generation. The multi-step generation process can potentially be extended for iterative token edition and refinement. The global attention mechanism of dLLM can largely improve its global context awareness and even include future planning capabilities. Therefore, dLLMs can serve as a natural alternative to traditional autoregressive generation for controllable tasks.

**Scientific Questions:** 
How can we fully unveil dLLM's potential in controllable generation? Specifically, how can we enable dLLM to stably generate reliable structured outputs (e.g., JSON) by utilizing its innate reverse reasoning capability and global context awareness, while overcoming the challenges of hallucination, structural constraint breaking, and slow inference speed associated with current dLLMs?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:** 
The core idea is to transform unconstrained generation into a structured fill-in-the-blank task. Instead of relying on intricate prompt optimization or constrained decoding, the proposed framework, Self-adaptive Schema Scaffolding (S3), initiates a schematic template directly in the output context as a starting state for dLLM. This schematic template acts as a structural prior that constrains the model’s generation space while preserving semantic flexibility. 

**Design Philosophy:** 
1. **Leveraging Global Awareness:** Utilize dLLM's global attention mechanism to allow the model to "see" the entire structure (scaffold) before filling in the missing tokens, addressing the AR limitation of lacking future token information.
2. **Warm-Start Initialization:** Initialize the reverse denoising process from a partially denoised state (the scaffold) rather than a fully random/masked one. This provides a principled way to guide the denoising process, reducing the expected denoising error.
3. **Adaptive Length Management:** To address the issue of fixed-length scaffolds causing hallucinations or over-generation (filling surplus slots with fabricated content), introduce a semantic token `null` as a placeholder. This transforms the fixed-length scaffolding problem into an adaptive generation task where dLLMs can naturally handle variable-length fields and missing values without fine-tuning.
4. **Efficient Iterative Refinement:** Use a selective remasking strategy (Top-K) to allow the model to iteratively refine its predictions within the scaffold, achieving structural adherence with significantly fewer denoising steps, thus reducing computational complexity.

## 4. Core Innovation Points

1. **Architectural Advantage Analysis:** The paper analyzes the architectural advantages of diffusion-based large language models (dLLMs) compared with autoregressive models for controllable generation, focusing on the prior’s global attention mechanism and iterative refinement capabilities.
2. **Self-adaptive Schema Scaffolding (S3) Method:** A novel, training-free method that enables dLLMs to achieve higher structure output performance with fewer denoising steps. It injects a schematic template into the output context instead of instruction, providing a more compelling language prior.
3. **Adaptive Null Token Mechanism:** The introduction of the semantic token `null` as a flexible placeholder to address variable-length fields and missing values. This transforms unfamiliar test-time scenarios into familiar training-time cases, significantly reducing hallucination rates caused by distributional shifts from rigid scaffolding.
4. **Comprehensive Evaluation Framework:** The establishment of a comprehensive structure output evaluation framework focusing on structural adherence, content fidelity, and faithfulness metrics, which enables a more nuanced and reliable assessment of a model’s ability to generate coherent, informative, and trustworthy structured texts.
5. **Theoretical Guarantee:** The provision of Theorem 4.1 (Scaffold-Guided Denoising Convergence), which rigorously proves that initializing the denoising process with structural scaffold S reduces the expected denoising error proportionally to the scaffold coverage ratio.

## 5. Overview of the Overall Technical Solution

The overall technical solution follows a pipeline that decomposes the task and leverages the dLLM's denoising process:
1. **Decomposition:** Begin by decomposing the original task instruction into two components: a problem description and a set of structural constraints.
2. **Schema Compilation:** Compile the structural constraints into a schema. Parse the structural specification to identify invariant structural elements (e.g., brackets, delimiters, field names) and replace variable content positions with mask tokens $M$. This creates a structural scaffold $A_s$.
3. **Noisy Scaffold Initialization:** Initialize a noisy scaffold where mask tokens serve as placeholders for missing content. The dLLM's generation context starts with this scaffold rather than a fully masked sequence.
4. **Context-Conditioned Denoising:** The dLLM completes this scaffold by predicting the masked tokens, using the problem description as context to generate structured outputs.
5. **Selective Remasking:** Apply a selective remasking strategy (Top-K) that allows the model to iteratively refine its predictions and further improve generation quality.
6. **Self-Adaptive Generation:** Guide the model to adopt `null` tokens to indicate absent values or variable lengths, preventing over-generation and hallucination in surplus slots.

## 6. Detailed Module Design

### 6.1 Task Formalization Module
Given a user query $Q$, and a structural specification $S$, the objective is to generate a response $A$ that satisfies both the semantic requirements of $Q$ and the structural constraints defined by $S$. The structural specification $S$ can take various forms (schema-based, format constraints, etc.). The task is formulated as a constrained optimization problem over the space of all valid outputs conforming to structure $S$, denoted as $\mathcal{A}(S)$.

### 6.2 Schema Scaffolding (S2) Module
This module explicitly incorporates structural constraints by pre-populating the generation context with structural templates. It operates by parsing the structural specification $S$ to identify invariant structural elements and replacing variable content positions with mask tokens $M$. This creates a structural scaffold $A_s$ that constrains the model’s generation space while preserving semantic flexibility.
*   **Schema Compiling:** Given a schema $S$ and a target format $F$ (e.g., JSON, XML), the compiler function $C(S,F)$ generates a scaffold sequence $x_{scaffold}$. For example, in JSON, field names and braces are kept, while values become mask tokens like `{"name": [MASK], "born": [MASK]}`.

### 6.3 Self-adaptive Schema Scaffolding (S3) Module
Vanilla S2 introduces the challenge of determining the appropriate number of mask tokens for variable content. Allocating ample mask tokens distorts generation quality (dLLMs are sensitive to sequence length), leading to under-utilization or hallucinated content. S3 extends the scaffolding framework to incorporate adaptive length management:
*   **Null Token Mechanism:** Leverages the semantic token `null` as a placeholder. This preserves the training-free property while guiding dLLMs to naturally represent absent or variable-length content.
*   **Augmented Prompting:** Uses an augmented prompt $Q^+$ which guides the model to adopt `null` tokens to indicate absent values. This transforms the fixed-length scaffolding problem into an adaptive generation task.

### 6.4 Remasking Strategy Module
By default, dLLMs generate all tokens in parallel. S3 adopts a simple yet effective **Top-K remask strategy**:
*   **Definition:** $K = O/n$, where $O$ denotes the total number of tokens to be generated and $n$ is a tunable number of denoising steps.
*   **Mechanism:** At each step, only the top-K lowest confidence tokens are remasked for future regeneration, while the rest are kept. This provides a more adaptive number of iterations and precise iteration control. Unlike Low-Confidence remasking (which may re-mask too few/many tokens in early steps leading to unstable convergence), Top-K ensures a predictable computational budget and steady refinement of the scaffolded structure.

### 6.5 Evaluation Framework Module
A comprehensive evaluation framework assessing outputs across three key dimensions:
*   **Structural Adherence:** Measures how well-generated outputs conform to the target schema.
    *   Structure Validity (SV): Proportion of outputs syntactically valid and parse without errors.
    *   Field Completeness (FC): Percentage of required fields correctly populated.
    *   Schema Compliance (SC): Proportion of outputs fully adhering to the predefined schema (data types, value constraints, nested structures).
*   **Content Fidelity:** Evaluates semantic accuracy of information within structurally valid outputs.
    *   Precision/Recall (PR/RE): Computed over individual field types.
    *   F1 Score: Harmonic mean of field-level precision and recall (using exact and fuzzy match).
*   **Faithfulness:** Assesses the degree to which generated content remains grounded in the source input.
    *   Hallucination Rate (HR): Proportion of output fields including information not present in or not reasonably inferable from the source text.

## 7. All Mathematical Formulas and Symbol Definitions

### 7.1 Autoregressive Language Modeling
*   **Formula:** 
    $$\log P_\theta(A|Q) = \sum_{t=1}^{|A|} \log P_\theta(a_t|a_{<t}, Q)$$
*   **Symbols:**
    *   $A = (a_1, \ldots, a_{|A|})$: The response sequence.
    *   $a_t$: The token at position $t$.
    *   $a_{<t}$: The preceding tokens.
    *   $|A|$: The sequence length.
    *   $\theta$: Model parameters.
    *   $Q$: The query/instruction.

### 7.2 Diffusion Language Modeling
*   **Forward/Reverse Process Definition:** Given tokenized sequence $x_0$, forward process corrupts by masking producing $x_t$ for $t \in [0, 1]$. As $t \to 1$, $x_1$ becomes fully masked. Reverse process trains network to predict original tokens at masked positions.
*   **Pre-training Objective Formula:**
    $$\min_\phi -\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t} \sum_{i=1}^L \mathbb{1}[x_t^i = M] \log P_\phi(x_0^i|x_t)\right]$$
*   **Symbols:**
    *   $M$: Denotes the mask token.
    *   $L$: The sequence length.
    *   $\phi$: Represents the model parameters.
    *   $x_t^i$: The token at position $i$ at timestep $t$.
    *   $x_0^i$: The original token at position $i$.
*   **Inference Step Factorization Formula:** (Concatenating user instruction $Q$ as fixed prefix to masked target $A_t$)
    $$\log P_\phi(Q \oplus A) = \sum_{i=1}^{|A|} \mathbb{1}[a_i = M] \log P_\phi(a_i|Q \oplus A_t)$$
*   **Symbols:**
    *   $\oplus$: Concatenation operation.
    *   $A_t$: Masked target sequence at step $t$ (initially $t=1$, all masked).

### 7.3 Task Formalization
*   **Formula:**
    $$A^* = \arg\max_{A \in \mathcal{A}(S)} P_{LM}(A|Q,S)$$
*   **Symbols:**
    *   $A^*$: The optimal response.
    *   $\mathcal{A}(S)$: The space of all valid outputs conforming to structure $S$.
    *   $P_{LM}$: A conditional language model.
    *   $Q$: User query.
    *   $S$: Structural specification.

### 7.4 Theorem 4.1: Scaffold-Guided Denoising Convergence
*   **Theorem Statement:** Let $x_0$ be a target structured sequence and $x_t$ be a partially masked sequence at timestep $t$ with scaffold $S$ defining fixed structural positions. For a diffusion language model trained with objective (Eq. 2), initializing the denoising process with structural scaffold $S$ reduces the expected denoising error by:
    $$\mathbb{E}[\|\hat{x}_0 - x_0\|_M] \leq \mathbb{E}[\|\tilde{x}_0 - x_0\|_M] \cdot \left(1 - \frac{|S|}{L}\right)$$
*   **Symbols:**
    *   $\hat{x}_0$: Generated output with scaffolding.
    *   $\tilde{x}_0$: Generated output without scaffolding.
    *   $\|\cdot\|_M$: Error over masked positions only.
    *   $|S|/L$: Scaffold coverage ratio.
*   **Proof Logic:**
    *   Partition positions into: Fixed scaffold positions $S = \{i : x_t^i \neq M, \text{fixed by structure}\}$ and Variable positions $V = \{i : x_t^i = M\}$.
    *   Model prediction: $p_\phi(x_0^i|x_t) = p_\phi(x_0^i|\{x_t^j\}_{j \neq i})$.
    *   With scaffolding, context includes correct structural tokens: $p_\phi(x_0^i|x_t \text{ with } S) = p_\phi(x_0^i|S \cup \{x_t^j\}_{j \in V \setminus \{i\}})$.
    *   Error reduction: $\mathbb{E}[\epsilon_i|S] \leq \mathbb{E}[\epsilon_i] \cdot (1 - I(S;x_0^i))$, where $I(S;x_0^i)$ is mutual information.
    *   Aggregating: $\mathbb{E}[\|\hat{x}_0 - x_0\|_M] \leq \mathbb{E}[\|\tilde{x}_0 - x_0\|_M] \cdot \left(1 - \frac{|S|}{L}\right)$. Equality holds when scaffold and content are independent.

### 7.5 Schema Scaffolding Objective
*   **Formula:**
    $$A^* \approx \arg\max_{A_s \in \mathcal{S}_C} P_\phi(A_s|Q) = \arg\max_{A_s \in \mathcal{S}_C} \sum_{a_i \in A_s} \mathbb{1}[a_i = M] \log P_\phi(a_i|Q,A_s)$$
*   **Symbols:**
    *   $\mathcal{S}_C \subset \mathcal{A}(S)$: The constrained subspace of outputs sharing the structural template derived from $S$.
    *   $A_s$: The scaffold with all non-masked tokens fixed except position $i$.

### 7.6 Self-adaptive Schema Scaffolding Objective
*   **Formula:**
    $$A^* \approx \arg\max_{A_s \in \mathcal{S}_C} \sum_{a_i \in A_s} \mathbb{1}[a_i = M] \log P_\phi(a_i|Q^+, A_s)$$
*   **Symbols:**
    *   $Q^+$: The augmented prompt which guides the model to adopt null tokens to indicate absent values.

### 7.7 Complexity Analysis
*   **Standard dLLM Complexity:** The overall computational complexity of the reverse process scales as $\mathcal{O}(L^3)$ (inference latency grows with diffusion steps, global attention incurs quadratic cost w.r.t context length $L$).
*   **S3 Complexity:** Asymptotically, S3 reduces the decoding complexity to $\mathcal{O}(nL^2)$, where $n$ is a tunable hyperparameter (number of steps) significantly smaller than $L$.

## 8. Algorithm Pseudocode

The paper does not provide an explicit algorithm pseudocode block, but describes the complete algorithmic pipeline and iterative update rules in Section 4 and Figure 2. The logic is extracted as follows:

**Algorithm: Self-adaptive Schema Scaffolding (S3)**

**Input:** User query $Q$, Structural specification $S$, Total denoising steps $n$, dLLM parameters $\phi$.
**Output:** Structured output $A^*$.

1.  **Decompose Instruction:** Split input into Problem Description $Q$ and Structural Constraints.
2.  **Compile Schema:** $x_{scaffold} \leftarrow C(S, F)$ where $F$ is target format. Identify invariant structural elements and replace variable positions with mask token $M$.
3.  **Initialize Scaffold:** $A_s \leftarrow x_{scaffold}$.
4.  **Augment Prompt:** $Q^+ \leftarrow \text{AppendNullInstruction}(Q)$ (Guide model to use `null` for missing/variable length content).
5.  **Calculate Remasking Budget:** $K \leftarrow O/n$, where $O$ is the total number of masked tokens in $A_s$.
6.  **Initialize State:** $A_t \leftarrow A_s$ (Start with partially masked scaffold instead of fully masked sequence).
7.  **For** $step = 1$ **to** $n$ **do:**
8.  $\quad$ **Parallel Prediction:** For all masked positions $a_i \in A_t$ where $a_i = M$, compute $P_\phi(a_i|Q^+, A_t)$.
9.  $\quad$ **Token Unmasking:** Fill masked positions with tokens having highest probability (or use `null` token if probability indicates absence).
10. $\quad$ **Confidence Evaluation:** Calculate confidence score (e.g., log-probability) for all currently unmasked tokens in the variable positions.
11. $\quad$ **Selective Remasking:** Select the top-$K$ tokens with the lowest confidence scores and remask them (replace with $M$).
12. $\quad$ **Update State:** $A_t \leftarrow$ updated sequence with new tokens and remasked low-confidence tokens.
13. **End For**
14. **Return** $A_t$ as $A^*$