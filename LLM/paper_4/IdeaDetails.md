# Research Idea and Full Implementation Plan Extraction

## 1. Research Background and Existing Pain Points

Large Language Models (LLMs) are transforming AI for Science (AI4S) research paradigm by automating complex scientific code generation for simulations, data analysis, and related science tasks. While models such as Codex, AlphaCode, and CodeLlama effectively lower technical barriers for researchers, several challenges hinder their reliable application in AI4S research. These include: (1) potentially unclear prompt descriptions from domain scientists without computer science backgrounds, (2) complex execution pipelines for scientific tasks, and (3) the need to maintain adherence to physical laws and domain-specific constraints. Standard prompting and self-refinement techniques are often inadequate for handling the subtle error patterns in complex scientific workflows.

Crucially, there is a lack of strong empirical evidence to fully trust LLMs’ capabilities in deep understanding and complex reasoning, particularly for professional scientific research tasks. Their decision-making processes remain opaque, and while their outputs often appear plausible, they may contain subtle inaccuracies or conceptual misunderstandings. These limitations fundamentally constrain the performance ceiling of LLM-based coding platforms, as their capabilities are inherently bounded by the underlying LLM’s intelligence level. Such inherent uncertainty demands the development of frameworks that operate without requiring absolute confidence in the LLM’s intelligence level.

Recent advances in LLM-based multi-agent systems attempt to address these limitations through distributed reasoning and specialized agent roles, where different LLM agents focus on specific sub-tasks while coordinating through structured communication protocols and/or a master agent. However, while such multi-agent architectures can effectively distribute computational complexity and domain expertise across components of the underlying domain task, they introduce new challenges in error propagation and validation. The system’s overall reliability becomes constrained by its weakest agent, as flawed intermediate outputs from one of the agents can be uncritically accepted by downstream agents, potentially amplifying rather than mitigating the above limitations of individual LLMs. Furthermore, evaluating the code of domain-specific tasks is often difficult. Standard unit tests may miss critical scientific constraints, theoretical foundations, or domain-specific limitations. This evaluation gap stems from three key issues: (1) scientific correctness often requires deeper domain knowledge than standard unit tests can verify; (2) comprehensive evaluation metrics may be prohibitively expensive or fundamentally intractable to define; and (3) LLM-generated tests may inherit the same reliability issue as the code they aim to validate.

## 2. Core Research Motivation and Scientific Questions

The core research motivation is to overcome the fundamental limitations where the probabilistic nature of LLMs produces hallucinations in both generated code and its corresponding test cases. In a multi-agent architecture, these errors can propagate and compound, leading to flawed final outputs. There is a critical need to develop frameworks that operate without requiring absolute confidence in the LLM’s intelligence level and systematically reduce evaluation uncertainty. 

Additionally, outside the Machine Learning community, an average scientist cannot be expected to be aware of or skilled in extensive prompt engineering techniques. A typical domain researcher’s prompt might be vague, assume implicit domain knowledge, or use specialized terminology and abbreviations that an LLM may not fully grasp, leading to misinterpretations, suboptimal outputs, or complete system failures. This demands bridging the gap between AI-generated code and domain-specific needs for non-technical experts.

The scientific questions addressed are: How to adversarially co-evolve test case generation and code improvement to mutually refine each other through competitive optimization, replacing traditional static verification approaches? How to structure agent interactions and evolve prompt distributions using Bayes’ Theorem to systematically reduce evaluation uncertainty and mitigate dependence on any single LLM’s reliability? How to empower domain experts by translating high-level natural language prompts into executable, domain-specific requirements without intricate prompt engineering?

## 3. Overall Core Idea and Design Philosophy

The core design philosophy is an adversarial co-evolution framework where test case generation and code improvement mutually refine each other through competitive optimization, replacing traditional static verification approaches. The proposed framework structures agent interactions and evolves prompt distributions using Bayes’ Theorem, reducing dependence on the base LLM’s inherent capabilities.

The framework comprises three specialized agents: a Task Manager (TM) serving as Challenger, a Solution Generator (SG) as Solver, and an Evaluator for comprehensive assessment. Unlike conventional multi-agent code generation systems that depend entirely on LLM-based evaluation and decision-making, this approach introduces an adversarial dynamic between TM and SG. The TM actively constructs and refines test cases to probe the SG’s current limitations, while the SG iteratively improves its code generation based on Evaluator feedback to meet these evolving challenges. By continuously probing and validating solutions against dynamically refined test cases, the framework overcomes evaluation barriers and progressively converges on solutions that satisfy both explicit requirements from domain experts and implicit domain constraints from the specific domain or application scenarios.

Furthermore, the framework incorporates a specialized scheme within the TM agent that actively structures raw user requests, resolves ambiguities through interactive clarification, and transforms potentially vague prompts into precise task plans and scientifically valid initial test cases. It maintains accessibility for non-technical domain experts while fully leveraging their domain expertise without requiring any computer science or professional prompt engineering skills.

## 4. Core Innovation Points

1. **A Novel AI4S Low-Code Platform with Bayesian Adversarial Framework**: Introduction of a multi-agent framework that employs a Bayesian recursive co-updating strategy to iteratively refine generated code and test cases using a non-LLM-based adversarial score. This method significantly enhances scientific coding performance across a spectrum of base models (from open-source to commercial) and allows smaller LLMs to achieve results competitive with larger counterparts.

2. **Bayesian Optimization for code performance estimation**: Proposal of a Bayesian Optimization method to estimate the performance of a given code based on its structure similarity with the tested codes, which enables the framework to handle and evaluate complicated code without incurring the computational expense of executing all generated codes for testing.

3. **Domain Knowledge Refinement for scientific tasks**: The LCP facilitates scientific exploration for non-coding professionals by enabling the generated code to better reflect domain knowledge and constraints through iteratively refining, adding and updating domain knowledge in the specially structured prompt. This ensures scientific consistency, such as minimal deviation from established ocean dynamics in Earth Science applications.

4. **Adversarial Co-evolution of Test Cases and Code**: An adversarial dynamic between the Task Manager (Challenger) and Solution Generator (Solver) where test case generation and code improvement mutually refine each other. The TM adapts weights and selects test cases based on their "True Hardness" to probe SG's limitations, replacing traditional static verification approaches and mitigating cumulative error by treating tests and code with equivalent confidence.

5. **Interactive Planning and Prompt Refinement for Non-Experts**: A specialized scheme that actively structures raw user requests, resolves ambiguities through interactive clarification loops, and transforms potentially vague prompts into precise task plans and scientifically valid initial test cases, entirely eliminating the need for intricate prompt engineering by domain scientists.

## 5. Overview of the Overall Technical Solution

The proposed Bayesian adversarial multi-agent framework is designed for AI4S tasks, incorporating subjective prior knowledge and addressing complex task abilities. The framework comprises three core component agents: a Task Manager (TM), a Solution Generator (SG), and an Evaluator(Eval). Within this structure, code generation becomes a dynamic interaction, primarily between the Task Manager (acting as a Challenger) and the SG agent (acting as a Solver), with the Evaluator providing the performance metrics that guide learning and adaptation. The game concludes when the SG agent produces code that successfully passes all defined validation tests.

The process initiates prior knowledge P with a task description (provided by a scientist user) and relevant subject materials (e.g., prior domain knowledge, including reference code samples). P is the initial input to the TM agent, which develops a structured plan of the scientific task, decomposing the main task into an ordered set of Sub-tasks. This plan is iteratively refined based on users’ feedback F and refinements until user’s approval and denoted as Plans. Subsequently, the TM agent generates an initial set of test cases (Test Case0) corresponding to these sub-tasks and other criteria derived from prior knowledge. These initial test cases, along with user-provided reference code base are serving as initial sample codes (Sample Code0). Sample Code0 is followed by the user approved plan (Plans) to form the initial prompt. Both the test cases and the sample codes can be independently updated in subsequent iterations. This update mechanism, guided by Bayesian principles, leverages the performance of candidate codes generated previously and the effectiveness of past test cases. The objective is to iteratively refine the prompt to guide the SG agent towards producing a solution that meets all test criteria and user requirements. 

The Evaluator agent is responsible for assessing the candidate codes, the effectiveness of the test cases, and the overall quality of the prompts used in each iteration. The core of the framework’s learning capability lies in its iterative refinement loop, characterized by an adversarial dynamic between the TM agent (Challenger) and the SG agent (Solver), and guided by Bayesian updates for prompt composition. The selection of which specific test cases and sample codes to include in the prompt for the next iteration is governed by a non-LLM-based Bayesian update rule, which systematically reduces evaluation uncertainty and mitigates the system’s dependence on any single LLM’s reliability.

## 6. Detailed Module Design

**Planning and Initial Code Generation**
The TM agent engages in a comprehensive planning phase. In particular, this involves: 1. Decomposing the primary task into sub-tasks. 2. Posing Sanity Checks for data structure and range. 3. Reasoning about the logical workflow and dependencies between sub-tasks. 4. Formulating strategic advice for the generation of effective test cases. This detailed plan is presented to the user in natural language for review and potential refinement. This interactive feedback loop continues until the user approves the plan. Once the plan is finalized, the TM agent generates an initial set of test cases. These test cases are designed to cover the specified sub-tasks and incorporate domain-specific prior knowledge, such as sanity checks (e.g., for expected data ranges) and out-of-range detection. Following the planning phase, the system constructs the initial prompt. The code generation agent then uses this prompt to produce N candidate solutions (codes). For each candidate, the system automatically generates comprehensive documentation containing: algorithm explanations, execution instructions, and detailed specifications for all functions, variables, and parameters.

**A Priori Estimation with Bayesian Optimization**
Executing all the generated codes for testing can be computationally expensive. To address this issue, as well as to leverage both the accuracy of evaluation and the range of exploration in the solution space, a Bayesian Optimization method is employed to estimate the performance score relates to the structural difference from all the tested codes. As an initiation, all the generated code in the first iteration get tested against the initial test cases, where all test cases start from the same initial difficulty score. We store the test results as a score vector S2. We then embed each Codei to a vector xi through a structural embedding that captures features from its Abstract Syntax Tree (AST) and code embedding vectors. We then use a Bayesian optimization process to predict the code’s performance based on its structural similarity with the tested code. For code embeddings, we use OpenAI’s text-embedding-3-large model in the surrogate modeling pipeline. This embedding model is used to map generated code candidates into a semantic-structural vector space, after which the GP surrogate estimates candidate quality before expensive full execution. In this embedding space, we can further obtain the pair-wise similarity between all the code embeddings. With the scores S2 and the kernel matrix K, we follow a standard Bayesian Optimization practice and fit a Gaussian Processes (GP) model. This model allows us to estimate the score of any new, untested code x∗ without running the full evaluation. This also gives the ‘likelihood’ to guide the Bayesian update. To decide which untested code to evaluate next, we employ a standard acquisition function that balances exploiting codes with high expected scores (exploitation) and exploring codes where the model is uncertain (exploration). We use the Upper Confidence Bound (UCB) acquisition function. The next code selected for full evaluation is the one that maximizes this UCB score.

**Evaluation and Feedback**
The Evaluator agent is responsible for assessing the candidate codes, the effectiveness of the test cases, and the overall quality of the prompts used in each iteration.
- **Test Case Score (S1)**: This score quantifies the "True Hardness" of the jth test case Test Casej , representing its capacity to be challenging yet ultimately solvable. An effective test case should successfully discriminate between code solutions of varying quality. S1(i)t is set as 1 by default for t = 0, and α is a hyperparameter to control the momentum of updating, which in experiments we set α = 0.8.
- **Code Score (S2)**: Each generated code Cj : 1 ≤ j ≤ N receives a composite score based on several factors.
- **Prompt Score (S3)**: The overall score for a prompt used in an iteration is a function of the performance of the codes and test cases generated by this prompt. If multiple prompt configurations are tested within a single logical iteration, the iteration’s representative prompt score might be the highest achieved. For the Bayesian update, we are interested in the score of a specific prompt configuration is then denoted.

**Iterative Refinement: Adversarial Dynamics and Bayesian Prompt Updates**
The core of the framework’s learning capability lies in its iterative refinement loop, characterized by an adversarial dynamic between the TM agent (Challenger) and the SG agent (Solver), and guided by Bayesian updates for prompt composition.
- **Adversarial interaction**: The TM agent’s role evolves to that of a Challenger. Based on the ’True hardness’, which is measured by S1, the TM adapts its weights for future evaluation and selects test cases for subsequent prompts. It aims to create test suites that are optimally challenging for the SG’s current learned capabilities—difficult enough to drive further learning and expose weaknesses, yet generally solvable to provide a positive learning signal. The SG agent, as the Solver, implicitly adapts by producing code in response to these evolving challenges. Its success or failure provides the feedback signal that shapes the TM’s subsequent challenging strategy.
- **Bayesian prompt updates**: The selection of which specific test cases (indexed by i) and sample codes (indexed by j) to include in the prompt for the next iteration is governed by a Bayesian update rule. p(Prompttij) is the prior probability of selecting the pair (Test Casei, Sample Codej) for the prompt. This prior can be uniform initially and can adapt over time based on the historical effectiveness of these components. p(St3|Prompttij) is the likelihood of observing the score St3 given that the prompt was formed using Test Casei and Sample Codej. This term captures how well this specific combination performed. The underlying intuition is to identify and prioritize "teacher-subject" pairs-specific combinations of sample code and test cases that consistently yield high-scoring prompts. 
- **Sample code pool management**: The pool of available sample codes (Sample Code) is not static, but recursively updated. Initially, it contains user-provided reference codes. As the SG agent generates new codes, those that achieve high S2 can be added to the Sample Code pool. The selection of a Sample Codej for a prompt can then be influenced not only by its initial status (as a reference) but also by an evolving measure of its "guidance quality," learned from its impact on past prompt scores when it was included.
- **Final results**: Within each round, if there exists a code that can pass all the test cases, the System will output it as the final result to the user. Otherwise, the System will keep using the Bayesian Adversarial method recursively till generation of a satisfying code or reaching the maximum round of iterations.

## 7. All Mathematical Formulas and Symbol Definitions

**Initial Prompt Definition:**
Prompt0 := Plans⊕ Test Case0 ⊕ Sample Code0
Where ⊕ is the direct concatenation operator.

**Bayesian Update Rule:**
p(Promptt+1ij |St3) ∝ p(St3|Prompttij)p(Prompttij)

**Test Case Score (S1):**
S1(i)t+1 = (1− α) · S1(i)t + α · (∑j′ s.t. pass S2(j′)/|{Codej′}| − ∑j† s.t. fail S2(j†)/|{Codej†}|)
Where S1(i)t is set as 1 by default for t = 0, and α is a hyperparameter to control the momentum of updating (α = 0.8).

**Code Score (S2):**
S2(j)t = ∑i I(Cj passes Ti) · S1(i)t−1 / ∑i S1(i)t−1

**Prompt Score (S3):**
S3t = 1/M ∑Mj=1 S1(i)t + 1/N ∑Ni=1 S2(j)t
Where M is the number of test cases and N is the number of generated codes. The score of a specific prompt configuration is denoted {(S3)tij}ij.

**Likelihood Formulation:**
p(St3|Prompttij) ∝ exp(E[S3t−11(i, j)])
Where E[S31(i, j)] is the expected score for the generated code with Testi, Codej in the prompt based on past performance or a baseline.

**Squared Exponential Kernel:**
k(xi,xj) = exp(−d(xi,xj)2/2l2)
Where d(xi,xj) is the distance between the two code embeddings and l is a length-scale parameter. These pairwise similarities form the kernel matrix K.

**Gaussian Processes Mean Function:**
µ(x∗) = kT∗ (K+ σ2nI)−1S2
Where k∗ is the vector of kernel similarities between the new code x∗ and all previously tested codes, and σ2n is the noise term.

**Gaussian Processes Variance Function:**
σ2(x∗) = k(x∗,x∗)− kT∗ (K+ σ2nI)−1k∗

**Upper Confidence Bound (UCB) Acquisition Function:**
UCB(x∗) = µ(x∗) + κσ(x∗)
Where parameter κ controls the trade-off between exploitation and exploration. The next code selected for full evaluation is xnext = argmaxx∗ UCB(x∗).

## 8. Algorithm Pseudocode

**Algorithm 1 Bayesian Adversarial Multi-Agent Framework**
1: Input: Task description and domain knowledge P , reference code base, max iterations Tmax
2: // Planning till User’s approval
3: Generate Plans = TM(P)
4: while NOT User Approval do
5:     Plans← TM(P,F) iteratively updates the plan given user feedback.
6: // Make initial prompts
7: Generate Test Case0 := ({Sanity Checks}, {Sub-tasks})← TM(Plans)
8: Form Sample Code0 ← {Test Case0, Reference Code}
9: Generate Prompt0 according to equation 1
10: // Code generation and evaluation
11: t← 0
12: Initialize test case scores ∀i ∈ {1, . . . ,M}, S1(i)0 ← 1
13: while Not ∃ Code : pass all checks ∧ t < Tmax do:
14:     t← t+ 1
15:     Generate code batch {Cj}Nj=1 ← SG(Promptt−1)
      // Score Calculation & Updates
16:     Compute Code Scores S2(j)t using St−11 according to Eq. equation 3
17:     Update Test Case Scores S1(i)t using S2(j)t according to Eq. equation 2
18:     Compute Prompt Score St3 using St1, St2 according to Eq. equation 4
19:     Cbest = argmaxj S2(j)t
      // Bayesian updates for next iteration
20:     Test Caset ← TM(Test Caset−1, St1) ▷ Refine test cases based on St1
21:     Promptt ∼ p(Promptt|St3) ▷ Update prompt based on St3
22: return Cbest