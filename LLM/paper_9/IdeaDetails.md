1. Research Background and Existing Pain Points
The performance of a Large Language Model (LLM) depends heavily on how well the training data domain matches the downstream evaluation task. For instance, if users are interested in layman science questions, training the LLM with more Wikipedia data allows it to converse better with these users. Hence, knowing the evaluation task is important for curating a more relevant training data mixture. 
However, in many practical settings, there are significant pain points:
1. Unseen Evaluation Tasks: The data (e.g., its domain, distribution, or labels) involved in an unseen evaluation task are often unknown. For example, due to privacy concerns, conversations between a deployed LLM and users might be end-to-end encrypted. The LLM owner does not know the actual evaluation data seen during test-time.
2. Limited to Coarse and Noisy Feedback: LLM owners can only receive coarse, noisy feedback on how well the LLM has performed (e.g., user ratings or duration spent on the application), rather than fine-grained data information.
3. Computational Expense of Naive Search: A naive idea to simply iterate through all possible data mixtures and observe the resulting LLM performance is too computationally expensive.
4. Limitations of Conventional Domain Adaptation (DA): Prior DA works assume fine-grained knowledge of data (e.g., labeled/unlabeled data or data distribution) from the evaluation task for selecting relevant training data, which is unavailable in this setting.
5. Limitations of Domain Generalization (DG): DG considers a rigid setting with no knowledge (not even feedback) of the evaluation task.
6. Limitations of Existing Data Mixing/Selection Methods: Works such as DoReMi, BiMix, LESS, and Aioli assume some availability of fine-grained evaluation data information, such as evaluation gradients, labels, or distribution, or naively assume the training data shares the same distribution as the task. In practice (like the unseen task setting), these are not always available. When applied directly to the unseen task setting, they perform worse.

2. Core Research Motivation and Scientific Questions
Core Research Motivation: To bridge the gap between the need for task-specific training data and the unavailability of fine-grained information about downstream evaluation tasks, by exploiting the coarse and noisy feedback loop that is naturally available during LLM deployment. Subjecting models to multiple rounds of training or fine-tuning in a feedback loop is a natural part of the deployment life-cycle to improve LLMs.
Scientific Question: How can we exploit such noisy feedback efficiently to optimize the LLM training data-mixture when the data in the evaluation task is unknown?

3. Overall Core Idea and Design Philosophy
The core idea is DUET (Data mixture for Unseen Evaluation Task), an efficient algorithm that exploits a noisy feedback loop to optimize the training data mixture. The design philosophy is a global-to-local approach that interleaves data selection with Bayesian optimization (BO):
Globally (Outer Problem): BO in DUET uses coarse, noisy feedback from the unseen evaluation task to automatically refine the mixing ratio of data domains in the training data mixture iteratively. BO is chosen because it is sample-efficient and is the only way to exploit such coarse and noisy feedback iteratively.
Locally (Inner Problem): DUET uses data selection (specifically, an Influence Function-driven estimator) to retrieve high-quality data points from each data domain until the proposed mixing ratio is reached. This results in an algorithm that can optimize training data iteratively even without having access to fine-grained data information from the evaluation task.

4. Core Innovation Points
1. Novel and Realistic Problem Setting: Introduction of a setting where the data involved in an unseen evaluation task is unknown but we can deploy our LLM to gather multiple rounds of coarse and noisy feedback.
2. Novel Algorithm (DUET): The first work that interleaves data selection with BO to iteratively optimize training data mixture based on feedback from an unseen evaluation task.
3. IF-Driven Estimator: Introduction of an Influence Function (IF)-driven estimator for the inner optimization problem that improves the quality of estimating the optimal data mixture for a given ratio, showing lower variance and bias compared to uniform random estimators.
4. Theoretical Guarantee: Theoretical analysis showing DUET’s convergence to the optimal training data mixture by analyzing DUET’s attained cumulative regret under the BO framework, even without any fine-grained data information.
5. Flexibility: The algorithm is flexible enough to incorporate different data selection methods in its inner loop (e.g., LESS, coresets, diversity-driven measures), each with different performance-compute tradeoffs.

5. Overview of the Overall Technical Solution
The overall technical solution reformulates the high-dimensional discrete optimization problem of selecting the optimal data mixture into a bilevel optimization problem:
1. Reparameterization: The original problem of finding the optimal set of data points is reparameterized into an outer problem (finding the optimal mixing ratio $r$) and an inner problem (finding the best data subset satisfying ratio $r$).
2. Outer Optimization (Global): Use Bayesian Optimization to solve the outer problem. BO constrains the mixing ratio $r$ to a probability simplex and uses an acquisition function (LCB) to propose a candidate mixing ratio $r_t$ at each iteration based on historical feedback.
3. Inner Optimization (Local): Use an IF-driven estimator to solve the inner problem. For a proposed mixing ratio $r_t$, perform IF-weighted sampling from each domain dataset based on pre-computed IF scores to retrieve data points until the mixing ratio is satisfied. This forms a data mixture sample.
4. Feedback Loop: Fine-tune the LLM on the selected data mixture and observe the noisy feedback (evaluation task loss) from the unseen task. Update the Gaussian Process posterior with the new observation $(r_t, \tilde{y}^*_{r_t})$.
5. Iteration: Repeat the interleaving process of BO and data selection until the budget $T$ of BO iterations is exhausted, recovering the best-performing data mixture.

6. Detailed Module Design
Reparameterization Module: Transforms the original problem $\min_{X \in \mathcal{D}} L_{eval}(\theta_X)$ into $\min_{r \in \mathbb{R}^n} \min_{X \in S_r} L_{eval}(\theta_X)$. This decomposes the problem into a continuous ratio optimization and a discrete data selection problem.
Data Selection Module (Inner Loop): 
- Uniform Random Estimator: A baseline method that randomly samples $k$ different data mixtures from $S_r$ and takes the minimum evaluation loss as the estimate. It is consistent but has high variance.
- IF-Driven Estimator: For each dataset $D_i$, fine-tune a separate LLM. Derive the IF score of each training data point w.r.t. the trained LLM for its respective domain. Given a mixing ratio $r$, perform weighted sampling from each domain based on each data point’s IF score until the mixing ratio is satisfied. By performing IF-weighted sampling $k$ times, obtain $k$ samples, and take the minimum evaluation loss as the estimate. This estimator emphasizes selecting data with high IF scores (high quality) and has lower variance and bias.
Bayesian Optimization Module (Outer Loop):
- Models the unknown objective function $f(r)$ as a realization of a Gaussian Process (GP) specified by prior mean $\mu(r)$ and covariance $\kappa(r, r')$.
- Uses the Lower Confidence Bound (LCB) acquisition function to select the next mixing ratio query: $r_{t+1} = \arg\min_r \mu_t(r) - \beta_{t+1}\sigma_t(r)$.
- Handles the constraint that $r$ is a probability simplex ($\|r\|_1 = 1$).
Feedback Integration: The estimation error of the IF-driven estimator and the noisy feedback from the evaluation task are treated as observation noise $\epsilon$ in the BO framework, which BO handles gracefully.
IF Score Calculation: The influence of a data point $z$ on the loss of a test data point $z_{test}$ is given by $\text{IF}_{z,z_{test}} = -\nabla_\theta L(z_{test}, \theta)^\top H_\theta^{-1} \nabla_\theta L(z, \theta)$. Scores are pre-computed, normalized, and used as weights for sampling.

7. All Mathematical Formulas and Symbol Definitions
Notation and Symbols:
- $n$: Number of training datasets (domains).
- $\mathcal{D} \triangleq \{D_1, D_2, \dots, D_n\}$: Training datasets from $n$ domains.
- $L_{eval}(\theta)$: Unseen evaluation task loss w.r.t. LLM parameters $\theta$.
- $X \in \mathcal{D}$: Set of training data points (data mixture).
- $M$: Practical constraint on the size of the selected data mixture.
- $\theta_X \triangleq \arg\min_\theta L_{train}(X, \theta)$: Model parameters learned from data mixture $X$.
- $r \in \mathbb{R}^n$: Mixing ratio of training data domains.
- $S_r \triangleq \{X : X \in \mathcal{D}, \text{ratio}(X) = r, |X| = M\}$: Set of data mixtures satisfying ratio $r$ and size $M$.
- $X^*_r \triangleq \arg\min_{X \in S_r} L_{eval}(\theta_X)$: Optimal data mixture for a given ratio $r$.
- $y^*_r$: Optimal unseen evaluation task loss for ratio $r$.
- $\kappa(r, r')$: Kernel function (e.g., SE kernel $\kappa(r, r') \triangleq \exp(-\|r - r'\|_2^2 / (2m^2))$).
- $\mu_t(r')$, $\sigma_t(r')$: GP posterior mean and variance.
- $\beta_t$: Exploration parameter in LCB.
- $\zeta > 0$: Free hyperparameter in GP posterior.
- $k$: Sampling size for estimators.
- $\epsilon_t$: Sub-Gaussian noise / estimation error.

Mathematical Formulas:
1. Original Problem:
$$\min_{X \in \mathcal{D}} L_{eval}(\theta_X) \quad \text{s.t.} \quad |X| = M$$

2. Reparameterized Problem (Theorem 3.1):
$$\min_{r \in \mathbb{R}^n} \min_{X \in S_r} L_{eval}(\theta_X)$$

3. GP Posterior Mean and Variance:
$$\mu_t(r') \triangleq \kappa_t^\top(r')(K_t + \zeta I)^{-1}y_t$$
$$\sigma_t(r') \triangleq \kappa(r', r') - \kappa_t^\top(r')(K_t + \zeta I)^{-1}\kappa_t(r')$$

4. LCB Acquisition Function:
$$r_{t+1} = \arg\min_r \mu_t(r) - \beta_{t+1}\sigma_t(r)$$

5. Cumulative Regret (BO):
$$R_T \triangleq \sum_{t=1}^T [f(r_t) - f(r^*)]$$

6. Uniform Random Estimator:
$$\tilde{y}^*_r = \min_{X_i} \{L_{eval}(\theta_{X_1}), \dots, L_{eval}(\theta_{X_k})\}$$

7. IF-driven Estimator:
$$\tilde{y}^*_r = \min_{X_i} \{L_{eval}(\theta_{X^{IF}_1}), \dots, L_{eval}(\theta_{X^{IF}_k})\}$$

8. IF Score Formula:
$$\text{IF}_{z,z_{test}} = -\nabla_\theta L(z_{test}, \theta)^\top H_\theta^{-1} \nabla_\theta L(z, \theta)$$

9. Theorem 3.2 (IF-Driven Estimator Error PDF):
Let $\{X^{IF}_1, \dots, X^{IF}_k\}$ be $k$ data mixture samples drawn from $S_r$ using IF-weighted sampling. Assume each independent sample $L_{eval}(\theta_{X^{IF}_i})$ follows the shifted truncated exponential distribution $y^*_r + \text{expt}(\lambda, c)$. The IF-driven estimator $\tilde{y}^*_r$ is a random variable: $y^*_r + \epsilon$, where $\epsilon$ has PDF:
$$\text{PDF}_\epsilon(u) = \frac{\lambda k e^{-\lambda u}}{1 - e^{-\lambda c}} \left( \frac{e^{-\lambda u} - e^{-\lambda c}}{1 - e^{-\lambda c}} \right)^{k-1} \quad \text{on} \quad u \in [0, c]$$

10. Theorem 4.1 (Attained Average Regret Bound):
Let $A_{c,k} = \frac{c^2(1 - e^{-c} - c/2)^{k-1}}{(1 - e^{-c})^k}$. With probability at least $1 - \delta$:
$$\lim_{T \to \infty} \frac{\tilde{R}_T}{T} \le \frac{6(\sqrt[4]{\delta} + \sqrt{k})}{\sqrt[4]{\delta}k} + 2A_{c,k} + \frac{\sqrt{2A_{c,k}}}{\sqrt[4]{\delta}}$$

11. Supporting Lemma B.2 (Bound on GP error):
$$|\mu_t(x) - f(x)| \le \left( B + R\sqrt{2(\gamma_t + 1 + \ln(1/\delta))} \right) \sigma_t(x)$$

12. Supporting Lemma B.3 (Bound on Estimator Error):
$$E(\epsilon_t) \le \frac{6}{k} + \frac{2c^2((1-e^{-c}) - c/2)^{k-1}}{(1-e^{-c})^k}$$
$$\text{Var}(\epsilon_t) \le E(\epsilon_t)$$

8. Algorithm Pseudocode
Algorithm 1 DUET: Optimizing training Data Mixtures for an Unseen Evaluation Task
1: Input: n training datasets from n domains {D1, . . . , Dn}. Computed IF scores of each data point (App. A.4) w.r.t. its domain dataset and locally trained model. Initial observation of data mixing ratio and evaluation task performance: D0 ≜ {(r0, ỹ0)}, SE kernel κ, sampling size k, parameter βt for acquisition step and total number of BO iterations T .
2: for t = 1, . . . , T do
3: rt = argminr µt(r)− βtσt(r) (BO acquisition step)
4: IF-weighted sampling to obtain k samples of data mixtures {X IF 1 , . . . ,X IF k } (Sec. 3.2).
5: IF-driven estimator at iteration t:
ỹ∗rt = minXi {Leval(θX IF 1 ), . . . ,Leval(θX IF k )}.
6: Keep track of best performing data mixture X ∗ t = argminXi {Leval(θX IF 1 ), . . . ,Leval(θX IF k )}.
7: Dt = Dt−1 ∪ {(rt, ỹ∗rt )}
8: Update the GP posterior and κ with updated observations Dt+1 (Sec. 2.1).
9: end for
10: X ∗ = argminX∗ i ∈{X∗ 1 ,...,X∗ T } Leval(θX∗ i )

Algorithm 2 IF-weighted sampling for one data domain containing dataset D
1: Input: number of data points n required for the given data domain (taken from the mixing ratio proposed at current iteration). Dataset D = {x1, x2, ..., x|D|}, Influence value of each data point in data domain dataset D: I ≜ [I1, I2, . . . , I|D|], small constant ϵ to avoid degenerate-case normalization.
2: Normalize the IF scores into probabilities: Inormalized ≜ [ (I1+min(I)+ϵ)/∑I , (I2+min(I)+ϵ)/∑I , . . . , (I|D|+min(I)+ϵ)/∑I ]
3: Perform weighed sampling from dataset D according to weights given by Inormalized n times.