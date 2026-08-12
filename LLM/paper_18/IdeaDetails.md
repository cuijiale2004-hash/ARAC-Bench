**1. Research Background and Existing Pain Points**

**Research Background:** Chain-of-Thought (CoT) has become a core technique for enhancing the reasoning ability of large language models (LLMs) on complex tasks. Through prompting step-by-step reasoning, CoT enables LLMs to decompose complex problems into simpler sub-tasks, thus improving their problem-solving capabilities. Recent studies, including OpenAI o1, QwQ, and DeepSeek-R1, demonstrate that scaling up CoT length can further enhance the reasoning abilities of LLMs. 

**Existing Pain Points:** Since most current LLMs are built on the Transformer architecture, the computational complexity of their attention grows quadratically with context length, and the memory overhead of their KV-cache increases linearly with context length. Hence, generating long CoT substantially increases the computational and memory cost of LLMs, limiting their practical efficiency on complex reasoning tasks. To improve the reasoning efficiency of LLMs, previous studies employ prompting, supervised fine-tuning (SFT), or reinforcement learning (RL) to encourage LLMs toward generating shorter CoT sequences. However, these methods often impair the reasoning ability of LLMs, since CoT shortening conflicts with test-time scaling, limiting the reasoning capacity of LLMs. To preserve the reasoning ability of LLMs, some studies express the CoT in more concise text (e.g., by removing less important tokens or rewriting with GPT-4) to reduce its length. However, they risk losing critical reasoning information or reducing interpretability when simplifying long CoT. Additionally, LLMs often produce noisy reasoning steps, which may mislead subsequent reasoning steps and lead to the overthinking issue.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:** The core research motivation is to address the significant computational and memory costs associated with generating long Chain-of-Thought (CoT) sequences in Large Language Models (LLMs), which currently limit their practical efficiency on complex reasoning tasks. Crucially, this efficiency improvement must be achieved without sacrificing the reasoning ability or interpretability of the models, avoiding the pitfalls of existing CoT shortening and simplification methods. 

**Scientific Questions:** How can the reasoning process of LLMs be modeled to efficiently retain and retrieve historical reasoning information without the quadratic computational cost and linear memory overhead of standard softmax attention? How can the model effectively transition between reasoning steps using a compressed state representation that separates linguistic fluency from core reasoning information? How can noisy reasoning steps that lead to overthinking issues be identified and mitigated during this state-transition process to ensure the model stays on the correct global reasoning trajectory?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:** The paper proposes an efficient reasoning framework that models the reasoning process of LLMs as a state-transition process. A long CoT is regarded as a sequence of reasoning steps, where LLMs perform a specific thinking pattern in each step. Each reasoning step contains two types of information: substantial linguistic information to ensure its fluency, and limited reasoning information to support subsequent reasoning or answer generation. The framework first compresses the reasoning information from previously generated reasoning steps into a matrix, termed the reasoning state matrix. Then, based on the query prompt and the state matrix, the LLM can efficiently generate the current reasoning step and update the state matrix accordingly. With linear attention, each token in the current reasoning step can directly retrieve relevant historical reasoning information from the reasoning state, without explicitly attending to tokens in previous reasoning steps.

**Design Philosophy:** The design philosophy centers on reducing the computational complexity of attention from quadratic to linear and the memory overhead of the KV-cache from linear to constant, while strictly preserving reasoning ability and interpretability by not shortening or simplifying the CoT sequences. By using a Mixed Attention Module (MAM) combining softmax attention for local linguistic context and linear attention for global reasoning state retrieval, the model maintains fluency while drastically cutting costs. Furthermore, recognizing the state-transition process as a gradient-descent learning procedure (test-time training), a state-based reasoning strategy is designed using momentum accumulation to guide the global reasoning direction and correct noisy steps, thereby mitigating overthinking.

**4. Core Innovation Points (List all innovations in the original text completely)**

1. **State-Transition Reasoning Framework:** Proposes an efficient reasoning framework that models the reasoning process of LLMs as a state-transition process. This reduces the computational complexity of attention from quadratic O(C^2) to linear O(C) and the memory usage of the KV-cache from linear O(C) to constant O(1), significantly improving reasoning efficiency without sacrificing reasoning ability or interpretability by preserving the full CoT sequence length.
2. **Mixed Attention Module (MAM):** Designs a Mixed Attention Module (MAM) consisting of a Softmax-Attention (SA) submodule and a Linear-Attention (LA) submodule to replace the original softmax attention module in LLMs. The SA submodule restricts attention to the query prompt and current reasoning step for linguistic fluency, while the LA submodule captures the LLM’s reasoning state matrix, allowing tokens to directly retrieve historical reasoning information without explicitly attending to previous tokens.
3. **State-Based Reasoning Strategy:** Proposes a state-based reasoning strategy to mitigate the overthinking issue caused by noisy reasoning steps. Based on the theoretical perspective that linear attention state updates are a gradient-descent process, it aggregates reasoning directions of all previous steps into a global reasoning direction using momentum accumulation and corrects the reasoning direction of the current step using this global direction.
4. **Long CoT Segmentation and Thinking Pattern Annotation:** Introduces a long CoT segmentation method that uses high-entropy transition tokens occurring at the start of sentences to segment thinking sequences into reasoning steps. It further clusters reasoning steps using K-means to identify thinking patterns and introduces special tokens to enclose them, enabling structured thinking sequences and precise tracking and control of the reasoning processes.
5. **Training Strategy with Knowledge Distillation:** Employs a training strategy that fine-tunes only the newly added LA submodule parameters and special tokens corresponding to thinking patterns using LoRA. It optimizes a joint loss function combining autoregressive loss and knowledge distillation loss (KL divergence) between the base model and the proposed model to ensure the LA submodule effectively captures global reasoning information while preserving the base model's capabilities.

**5. Overview of the Overall Technical Solution**

The overall technical solution begins by reconstructing the training data: a long CoT sample is segmented into reasoning steps based on high-entropy transition tokens and clustered into thinking patterns, which are enclosed by special tokens. The core architectural modification is replacing the original softmax attention module in LLMs with the Mixed Attention Module (MAM). The MAM contains a Softmax-Attention (SA) submodule and a Linear-Attention (LA) submodule. In the SA submodule, each token attends only to tokens in the query prompt and previously generated tokens in its current reasoning step; the KV-cache of completed steps is cleared. In the LA submodule, a linear attention mechanism captures the LLM's reasoning state matrix, which records the reasoning information from previously completed reasoning steps. The state is updated token-by-token, and a query vector extracts relevant historical information, which is controlled by a gating mechanism. To mitigate noisy reasoning steps, a state-based reasoning strategy is applied. It computes the reasoning direction (gradient) of each step, accumulates a global reasoning direction using momentum, and corrects the current step's direction by linearly interpolating it with the global direction. Finally, the model is trained by fine-tuning only the newly added LA submodule and special tokens, optimizing an objective that combines autoregressive loss and knowledge distillation loss.

**6. Detailed Module Design (Complete mechanisms of each layer/sub-module)**

**Long CoT Segmentation Module:**
A long CoT sample (Q, T, A) comprises a query prompt Q, a thinking sequence T, and a final answer A. The framework first extracts high-entropy transition tokens from the long CoT training set, which occur at the start of a sentence. Next, these tokens are used to segment the thinking sequence T in each training sample into a sequence of reasoning steps. Meanwhile, all reasoning steps in the entire training set are clustered using MiniBatch K-Means clustering (e.g., 128 clusters), with each cluster corresponding to a thinking pattern (i.e., type). Cosine similarity is used as the distance metric. Finally, for each thinking pattern, two special tokens are introduced to enclose its corresponding reasoning steps (e.g., `<type i>` and `</type i>`). 

**Mixed Attention Module (MAM):**
The MAM replaces the original softmax attention module and consists of an SA submodule and an LA submodule. Taking the generation of the i-th token xt,i in the t-th reasoning step as an example at the l-th layer:
- **SA Submodule:** The KV-cache retains only the KV vectors of the query prompt tokens and ones of the previously generated tokens in the current reasoning step, while those of tokens in completed reasoning steps have been removed. Through the softmax attention mechanism, the input token xt,i-1 attends to tokens retained in the KV-cache, yielding an updated feature \hat{h}_{t,i-1}^{(l)}.
- **LA Submodule:** After completing the first t−1 reasoning steps, the model state is updated to S_{t-1}^{(l)}. The initial state for the current reasoning step is S_{t,0}^{(l)} = S_{t-1}^{(l)}. It is updated token-by-token, yielding the current model state S_{t,i-1}^{(l)} = S_{t,0}^{(l)} + \sum_{j=1}^{i-1} k_j^{(l)\top} v_j^{(l)}. The query vector q_{i-1}^{(l)} of the input token xi-1 is used to extract the relevant historical reasoning information o_{i-1}^{(l)} from S_{t,i-1}^{(l)}: o_{i-1}^{(l)} = q_{i-1}^{(l)} S_{t,i-1}^{(l)}. A gating mechanism is applied to control reliance on historical info: \check{h}_{t,i-1}^{(l)} = \sigma(W_g h_{t,i-1}^{(l-1)}) * o_{i-1}^{(l)}, where \sigma(\cdot) denotes the sigmoid function and W_g is a learnable weight. The LA submodule is implemented via the LoRA strategy.
- **Combination and Output:** The outputs of the two submodules are combined and passed through a linear output layer: \tilde{h}_{t,i-1}^{(l)} = W_o(\check{h}_{t,i-1}^{(l)} + \hat{h}_{t,i-1}^{(l)}). This is subsequently processed using the FFN module to produce the output h_{t,i-1}^{(l)} for the l-th layer. After generating the end token corresponding to the thinking pattern, the state matrix is updated to S_{t,n_t}^{(l)}, denoted as S_t^{(l)} and used as the initial state for the next reasoning step. KV vectors of the current step are cleared.

**State-Based Reasoning Strategy:**
- The state transition process at the l-th layer is formalized as: S_0^l \rightarrow S_1^l \rightarrow \cdots \rightarrow S_{|T|}^l.
- The total gradient contributed by each reasoning step is represented as [\nabla_1^l, \nabla_2^l, \cdots, \nabla_{|T|}^l], where \nabla_t^l = S_t^l - S_{t-1}^l indicates the reasoning direction of the t-th step.
- Momentum accumulation aggregates reasoning directions into a global reasoning direction \bar{\nabla}_{t-1}^l: \bar{\nabla}_{t-1}^l = (1 - \frac{1}{t-1})\bar{\nabla}_{t-2}^l + \frac{1}{t-1}\nabla_{t-1}^l = \frac{1}{t-1}\sum_{i=1}^{t-1}\nabla_i^l, where \bar{\nabla}_0^l is initialized as a zero matrix.
- The global direction corrects the current reasoning direction: \hat{\nabla}_t^l = (1 - \alpha)\nabla_t^l + \alpha\bar{\nabla}_{t-1}^l, where \alpha = \max\{\alpha_{\max}, \frac{t}{|T|}\} and |T| denotes the maximum number of reasoning steps.
- The corrected reasoning direction updates the model state: S_t^l = S_{t-1}^l + \hat{\nabla}_t^l.
- Thinking patterns diversity: Special markers indicating thinking patterns are applied to guide the model toward adopting different thinking patterns across consecutive steps.

**Model Training:**
Fine-tune only the parameters of the newly added LA submodule and ones of the special tokens corresponding to thinking patterns. The training objective is defined as L = L_{AR} + \beta L_{KD}. A customized mask matrix is applied in the SA submodule to ensure each token attends only to the tokens in the query prompt and those from its corresponding reasoning step.

**7. All Mathematical Formulas and Symbol Definitions**

**Softmax Attention (SA):**
Given the input sequence X=[x1, · · · , x|X|], the softmax attention is formulated as:
o_t = \frac{\sum_{i=1}^t \exp(q_t k_i^\top / \sqrt{d}) v_i}{\sum_{i'=1}^t \exp(q_t k_{i'}^\top / \sqrt{d})} ; q_t, k_t, v_t = x_t W_Q, x_t W_K, x_t W_V 
where WQ, WK, WV are learnable weight matrices.

**Linear Attention (LA):**
Replaces the exponential function exp(·) with a kernel function φ(·):
o_t = \frac{\sum_{i=1}^t \phi(q_t)\phi(k_i)^\top v_i}{\sum_{i'=1}^t \phi(q_t)\phi(k_{i'})^\top} = \frac{\phi(q_t) \sum_{i=1}^t \phi(k_i)^\top v_i}{\phi(q_t) \sum_{i'=1}^t \phi(k_{i'})^\top} 
When φ(·) is the identity function and the normalizer is removed:
o_t = q_t \sum_{i=1}^t k_i^\top v_i = q_t S_t ; S_t = \sum_{i=1}^t k_i^\top v_i 
where St is the state matrix storing historical information.

**Test-Time Training (TTT) Perspective of LA:**
Objective: L(S) = -\langle S k_t, v_t \rangle,
SGD update: S_t = S_{t-1} - \beta \nabla L_t(S_{t-1}) = S_{t-1} + \beta v_t k_i^\top 
where ⟨·, ·⟩ denotes the inner product, and β is the learning rate.

**LA Submodule State Update:**
S_{t,i-1}^{(l)} = S_{t,0}^{(l)} + \sum_{j=1}^{i-1} k_j^{(l)\top} v_j^{(l)} 

**LA Submodule Information Extraction:**
o_{i-1}^{(l)} = q_{i-1}^{(l)} S_{t,i-1}^{(l)} 

**Gating Mechanism:**
\check{h}_{t,i-1}^{(l)} = \sigma(W_g h_{t,i-1}^{(l-1)}) * o_{i-1}^{(l)} 
where σ(·) denotes the sigmoid function and Wg is a learnable weight.

**Combination of Submodules:**
\tilde{h}_{t,i-1}^{(l)} = W_o(\check{h}_{t,i-1}^{(l)} + \hat{h}_{t,i-1}^{(l)}) 

**Reasoning Step Gradient/Direction:**
\nabla_t^l = S_t^l - S_{t-1}^l 

**Momentum Accumulation for Global Reasoning Direction:**
\bar{\nabla}_{t-1}^l = (1 - \frac{1}{t-1})\bar{\nabla}_{t-2}^l + \frac{1}{t-1}\nabla_{t-1}^l = \frac{1}{t-1}\sum_{i=1}^{t-1}\nabla_i^l 

**Correction of Reasoning Direction:**
\hat{\nabla}_t^l = (1 - \alpha)\nabla_t^l + \alpha\bar{\nabla}_{t-1}^l, \alpha = \max\{\alpha_{\max}, \frac{t}{|T|}\} 

**State Update with Corrected Direction:**
S_t^l = S_{t-1}^l + \hat{\nabla}_t^l 

**Training Objective:**
L = L_{AR} + \beta L_{KD} 

**Autoregressive Loss:**
L_{AR} = - \log P(A,T|Q) 

**Knowledge Distillation Loss:**
L_{KD} = KL(\hat{P}(A,T|Q) || P(A,T|Q)) 

**8. Algorithm Pseudocode**

The paper does not contain explicit algorithm pseudocode blocks.