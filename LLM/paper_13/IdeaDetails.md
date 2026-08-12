**1. Research Background and Existing Pain Points**
Reinforcement Learning with Verifiable Reward (RLVR) has recently become an essential tool for post-training of large language models (LLMs), equipping them with the capability of reasoning over complicated logical problems through policy optimization. Policy optimization algorithms (such as Group Relative Policy Optimization, GRPO) have significantly enhanced the logical reasoning capabilities of LLMs. For logical problems, directly verifiable answers provide straightforward rewards for reinforcement learning, enabling effective outcome supervision of LLMs. However, conventional approaches demand complete annotation over the entire dataset for verification, and often allocate computation resources uniformly across the full dataset. This requires tedious annotation efforts and computational redundancy, as many recent studies and data subset selection works suggest that the instances in the training set are not equally important, and training on a small subset may also lead to satisfactory results.

**2. Core Research Motivation and Scientific Questions**
Based on the finding that not all instances are equally important for policy optimization, this paper articulates the **lottery sample hypothesis** in RLVR of LLMs: A large training set for RLVR on LLMs contains a small subset that, when trained alone, can achieve performance comparable to that of the full dataset. With this hypothesis, it is possible to break conventional approaches from two aspects: (i) full annotation of the dataset is no longer required, and ground truth answers of several lottery-winning samples are sufficient; (ii) computation can be concentrated on several critical instances. 
Therefore, the core scientific question explored in this paper is: **How can we find the critical instances (the lottery-winning samples) for RLVR on LLMs from the original training set without annotation?**

**3. Overall Core Idea and Design Philosophy**
The overall core idea is the unsupervised discovery of critical instances using a novel framework named Complementary Conformal Selection (CONST). The design philosophy evaluates the importance of samples from two complementary perspectives without access to ground truth answers: **Procedural Volatility** and **Outcome Volatility**. Procedural volatility measures how the final answer changes when reasoning paths are truncated at different stages, capturing the complexity and fragility of the reasoning process. Outcome volatility captures inconsistencies in the final answers produced by different possible reasoning paths sampled from the policy. These volatilities yield multisets of results, which are merged and fed into a conformal prediction module. The conformal prediction produces a prediction set, whose cardinality serves as the criterion for selecting lottery-winning samples. A larger prediction set size indicates richer and more effective optimization signals. Additionally, clustering is used to encourage sample diversity before selection.

**4. Core Innovation Points**
*   **New Perspective:** Presenting a probabilistic perspective for the unsupervised identification of critical instances in the full dataset for further annotation and RLVR optimization on LLMs, enabling an annotation-minimal, data-efficient, and performance-competitive optimization procedure compared with training on the entire fully annotated dataset.
*   **Novel Methodology:** Proposing CONST, a probabilistic approach based on conformal prediction that considers both procedural volatility and outcome volatility in LLM reasoning, to select lottery-winning samples for annotation and optimization.
*   **Theoretical Analysis:** Providing a rigorous theoretical analysis of CONST demonstrating that it can effectively approximate the optimal policy parameter setup under the lottery sample hypothesis, establishing a generalization bound for the proposed method in ergodic Markov decision processes.
*   **Empirical Validation:** Conducting extensive experiments showing that CONST is (1) annotation-efficient (achieving similar performance to the full dataset with < 0.5% of the annotation), (2) high-performing (outperforming competitive baselines by 10.97% on average), and (3) model-agnostic (showing consistent improvement across three different architectures).
*   **Scoring Function Design:** Designing a specific scoring function for conformal prediction in LLMs that combines negative frequency and entropy to quantify the disagreement and certainty between input questions and predicted answers.

**5. Overview of the Overall Technical Solution**
Given an unlabeled training set of questions $Q$, the original policy $\pi_0$, and a calibration set $D_{\text{cal}}$:
1.  **Procedural Volatility Evaluation:** For each question $X \in Q$, sample a deterministic output, truncate the reasoning path at $n_P$ stages to form $T(X)$, and re-query the LLM for final answers to obtain a multiset $B_P(X)$.
2.  **Outcome Volatility Evaluation:** Sample $n_O$ outputs from the policy $\pi_0$ for question $X$ to obtain a multiset $B_O(X)$.
3.  **Conformal Prediction:** Combine the multisets $B(X) = B_P(X) \uplus B_O(X)$. Compute a scoring function (combining negative frequency and entropy) for each answer in $B(X)$. Use the calibration set to determine a threshold $\hat{\rho}$ and form a prediction set $\hat{C}_{1-\alpha}(X)$ for each instance.
4.  **Selection:** Cluster the question set $Q$ into $b$ groups. From each group, select the question with the largest prediction set size to form the critical subset $Q'$.
5.  **Optimization:** Annotate $Q'$ to get $A'$, and optimize the original LLM using the RLVR algorithm (GRPO) with $Q'$ and $A'$ to obtain the optimized model $\pi_P$.

**6. Detailed Module Design**
*   **Procedural Volatility Module:** This module assesses potential variations in reasoning chains. Given a question $X$, a deterministic output $O = [t_1, t_2, ..., t_L; \hat{Y}]$ is sampled. The reasoning path is truncated at different stages to obtain a set of truncated reasoning trajectories $T(X)$. The LLM is then queried with these truncated trajectories to output a single final answer for each, forming a multiset $B_P(X)$. Simple reasoning paths yield consistent answers, while complex ones exhibit volatility.
*   **Outcome Volatility Module:** This module measures inconsistencies in the final answers (outcomes) produced by different possible reasoning trajectories. It directly samples $n_O$ outputs from the original policy $\pi_0$ for a question $X$ to obtain a multiset $B_O(X)$. Diverse answers induce gradients from multiple directions, helping the model avoid pitfalls.
*   **Conformal Prediction Module:** This module quantifies how many of the answers are likely to be the correct answer. It merges the multisets $B_P(X)$ and $B_O(X)$. A scoring function $f_{\pi_0}(X, \hat{Y})$ is designed to quantify the disagreement between the input and the predicted answer. It uses negative frequency and entropy. It is calibrated on $D_{\text{cal}}$ to find a threshold $\hat{\rho}$, producing a prediction set $\hat{C}_{1-\alpha}(X)$ containing answers with scores below the threshold. If the correct answer does not appear in the multiset, the score is set to $+\infty$.
*   **Selection Module:** The size of the conformal prediction set naturally measures how many answers the model considers likely correct. A larger size indicates richer optimization signals. To encourage diversity, the questions are clustered into $b$ groups using Sentence-BERT and K-means. The instance with the largest prediction set size is selected from each cluster.
*   **Optimization Module:** The selected set of questions $Q'$ is annotated with ground truth answers $A'$. The standard RLVR algorithm (GRPO) is then used to optimize the model $\pi_0$ with $Q'$ and $A'$ to obtain the final model $\pi_P$.

**7. All Mathematical Formulas and Symbol Definitions**
*   **Problem Definition:** Training set of questions $Q = \{X_1, X_2, . . . , X_N\}$, ground truth answers $A = \{Y_1, Y_2, . . . , Y_N\}$, original policy $\pi_0$. Optimized policy with full data: $\pi_F = \Phi(\pi_0,Q,A)$. Goal: Find subset $Q' \subset Q$ with budget $b$ ($|Q'| = b$), annotate $A'$, optimize policy $\pi_P = \Phi(\pi_0,Q',A')$ such that $\pi_P$ achieves comparable performance to $\pi_F$.
*   **RLVR Advantage Function:**
    $$a_i = \frac{r_i - mean(\{r_j\}_{j=1}^n)}{std(\{r_j\}_{j=1}^n)}$$
*   **GRPO Optimization Objective:**
    $$\mathcal{L}_{GRPO} = \mathbb{E}_{O_i \sim \pi_{\theta'}} [- \frac{1}{n} \sum_{i=1}^n ( \min (\frac{\pi_{\theta}(O_i|X)}{\pi_{\theta'}(O_i|X)} a_i, clip(\frac{\pi_{\theta}(O_i|X)}{\pi_{\theta'}(O_i|X)}, 1-\epsilon, 1+\epsilon) a_i ))] + \beta D_{KL}(\pi_{\theta}||\pi_0)$$
*   **Conformal Prediction Set:**
    $$C_{1-\alpha}(X) = \{Y | f_{\pi}(X,Y) \le \rho\}$$
    where $\rho$ is the $\frac{\lceil(m+1)(1-\alpha)\rceil}{m}$ quantile of calibration scores.
*   **Truncated Reasoning Trajectories:**
    $$T(X) = \{[t_1, t_2, . . . , t_{\lceil \frac{iL}{n_P} \rceil}] | i = 1, 2, . . . , n_P \}$$
*   **Procedural Volatility Multiset:**
    $$B_P(X) = \{\{\hat{Y} = \pi_0(\hat{Y}|X, \tau) | \tau \in T(X)\}\}$$
*   **Outcome Volatility Multiset:**
    $$B_O(X) = \{\{\hat{Y}_i | i = 1, 2, . . . , n_O, \hat{Y}_i \sim \pi_0(\hat{Y}|X)\}\}$$
*   **Scoring Function - Negative Frequency:**
    $$f_{NF}(X, \hat{Y}) = - freq(\hat{Y}; B(X)) = -\frac{count_{B(X)}(\hat{Y})}{|B(X)|}$$
*   **Scoring Function - Entropy:**
    $$f_{ent}(X, \hat{Y}) = \frac{H(B(X))}{\log |B(X)|} = \frac{-\sum_{Y' \in set(B(X))} freq(Y';B(X)) \log freq(Y';B(X))}{\log |B(X)|}$$
*   **Combined Scoring Function:**
    $$f_{\pi_0}(X, \hat{Y}) = f_{NF}(X, \hat{Y}) + \lambda \cdot f_{ent}(X, \hat{Y})$$
*   **Conformal Prediction Set for Selection:**
    $$\hat{C}_{1-\alpha}(X) = \{\hat{Y} \in set(B(X)) | f_{\pi_0}(X, \hat{Y}) \le \hat{\rho}\}$$
*   **Selection Process:**
    $$Q' = \{ \arg\max_{X \in Q_i} |\hat{C}_{1-\alpha}(X)| | i = 1, 2, . . . , b \}$$
*   **Ergodic MDP Mixing Time:**
    $$t_{mix}(\epsilon) \triangleq min\{t \ge 0 : max_{s \in S} (max_{A \subseteq S} (|Pr(Y_t \in A|Y_0 = s) - \rho^{\infty}(A)|)) \le \epsilon\}$$
*   **Assumption 3.1 (Lottery Sample Hypothesis):** Subset $Q'$ is an $\epsilon$-approximation of the full training set $Q$ if $\|\nabla \hat{\mathcal{L}}^{Q'}_{GRPO}(\theta) - \nabla \hat{\mathcal{L}}^{Q}_{GRPO}(\theta)\|_2 \le \epsilon$ holds for any parameter vector $\theta$.
*   **Assumption 3.2 (Smoothness):** The objective $\hat{\mathcal{L}}^{Q}_{GRPO}(\theta)$ is $L$-smooth, i.e., $\|\nabla \hat{\mathcal{L}}^{Q}_{GRPO}(x) - \nabla \hat{\mathcal{L}}^{Q}_{GRPO}(y)\|_2 \le L\|x-y\|_2$.
*   **Assumption 3.3 (Polyak-Lojasiewicz Condition):** $2\mu (\hat{\mathcal{L}}^{Q}_{GRPO}(\theta) - \hat{\mathcal{L}}^{Q}_{GRPO}(\theta^*_{GRPO})) \le \|\nabla \hat{\mathcal{L}}^{Q}_{GRPO}(\theta)\|_2^2$ where $\theta^*_{GRPO} \triangleq \arg\min_{\theta} \hat{\mathcal{L}}^{Q}_{GRPO}(\theta)$.
*   **Theorem 3.1 Generalization Bound:** Under Assumption 3.1-3.3, if the underlying MDP is ergodic with mixed time $t_{mix}$ and the gradient is bounded ($\|\nabla \hat{\mathcal{L}}^{Q}_{GRPO}(\theta)\|_2 \le G$), then with probability $> 1-\delta$:
    $$\mathcal{L}_{GRPO}(\theta_k) - \mathcal{L}_{GRPO}(\theta^*) \le 4\mathcal{R}(\mathcal{F}_{GR}) + O(\sqrt{\frac{t_{mix}\sigma^2_R(1-\frac{1}{n}) \ln(\frac{1}{\delta})}{Nn}} + \frac{\ln(\frac{1}{\delta})}{Nn(1-\gamma)}) + (1-\frac{\mu}{L})^{k+1}\hat{\mathcal{L}}^{Q}_{GRPO}(\theta_0) + \frac{2G}{\mu}\epsilon + \frac{\epsilon^2}{2\mu}$$
    where $\theta^* \triangleq \arg\min_{\theta} \mathcal{L}_{GRPO}(\theta)$, $\mathcal{R}(\mathcal{F}_{GR})$ is Rademacher complexity, $N$ is size of $Q$, $n$ is number of outputs, and $\sigma^2_R$ is upper bound of variance of return.
*   **Lemma A.1 Convergence:** Gradient Descent with step-size $1/L$, $\theta_{k+1} \triangleq \theta_k - \frac{1}{L}\nabla \hat{\mathcal{L}}^{Q'}_{GRPO}(\theta)$, has linear convergence:
    $$\hat{\mathcal{L}}^{Q}_{GRPO}(\theta_k) - \hat{\mathcal{L}}^{Q}_{GRPO}(\theta^*_{GRPO}) \le (1-\frac{\mu}{L})^k \hat{\mathcal{L}}^{Q}_{GRPO}(\theta_0) + \frac{2G}{\mu}\epsilon + \frac{\epsilon^2}{2\mu}$$
*   **Lemma A.2 Generalization of GRPO:** With probability $1-\delta$:
    $$\sup_{\theta \in \Theta} |\hat{\mathcal{L}}^{Q}_{GRPO}(\theta) - \mathcal{L}_{GRPO}(\theta)| \le 2\mathcal{R}(\mathcal{F}_{GR}) + O(\sqrt{\frac{t_{mix}\sigma^2_R(1-\frac{1}{n}) \ln(\frac{1}{\delta})}{Nn}} + \frac{\ln(\frac{1}{\delta})}{Nn(1-\gamma)})$$

**8. Algorithm Pseudocode**
```text
Algorithm 1: The execution pipeline of CONST
Input: The set of questionsQ, the original policy π0, the calibration set Dcal = {(Xcal_i , Y cal_i )}^m_i=1
1 Initialize CalScoreList and SizeList as empty lists
2 for iteration i← 1 to m do // Calibrate the scoring function
3     Calculate the score fπ0(Xcal_i , Y cal_i ) with Eq. 9 and append it to CalScoreList
4 end
5 Find the ⌈(m+1)(1−α)⌉_m quantile of CalScoreList as ρ̂
6 for iteration i← 1 to N do // Obtain the conformal prediction sets
7     Calculate the multiset of possible answers B(Xi) according to Eq. 5 and Eq. 6
8     Calculate the score fπ0(Xi, Ŷi) for each Ŷi ∈ B(Xi) with Eq. 9
9     Obtain the prediction set Ĉ1−α(X) with ρ̂ using Eq. 10 and append the size of it to SizeList
10 end
11 ClusterQ into b groupsQ1,Q2, . . . ,Qb
12 Select the question with the largest size from each group to formQ′ // Select critical samples
13 AnnotateQ′ with ground truth answers A′ // Annotate several samples
14 Optimize π0 withQ′ and A′ to obtain the optimized model πP // Optimize the policy using RL
15 return πP as the optimized model
```