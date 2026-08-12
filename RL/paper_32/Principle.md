**Principle 1: Controlled Bias Analysis and Mechanistic Validation of Online-Target Network Hybridization in Temporal-Difference Learning**

**Definition:**  
In deep temporal-difference reinforcement learning, hybridizing online and target network estimates introduces fundamental tensions between learning speed and stability. The subfield demands rigorous mechanistic validation that goes beyond aggregate performance metrics to dissect how the proposed operator shapes estimation bias, variance, and training dynamics over time. Because bootstrapping from online networks risks severe overestimation and divergence, reviewers must assess whether the work provides controlled empirical evidence—ideally in environments with computable ground-truth values—demonstrating that the hybrid rule avoids catastrophic instability or excessive conservatism. This principle further requires an explicit causal account linking the operator’s mathematical structure to observed learning behaviors, rather than inferring mechanism from end-task returns alone. A thorough analysis should inspect value-estimation error trajectories, bias overshoots during early training, and loss-curve morphologies to ensure that reported gains are rooted in stabilized temporal-difference updates. Ultimately, such mechanistic scrutiny prevents the acceptance of black-box heuristics that appear effective on standard benchmarks yet fail on contrived but diagnostically revealing tasks.

**Core Evaluation Criteria:**
- **Ground-Truth Bias Quantification**: Does the work evaluate empirical bias and value-estimation error against exactly known optimal action-values in controlled tasks (e.g., Car-on-the-Hill, Baird’s counterexample), rather than relying solely on asymptotic returns?
- **Training Dynamic Inspection**: Are learning curves, loss trajectories, and gradient behaviors analyzed to verify that introducing online estimates does not induce instability, cyclic behavior, or pathological loss increase-then-decrease patterns?
- **Distinction of Mechanism from Performance**: Does the paper clearly separate "stability enhancement" from "performance improvement," establishing a causal chain between the operator and reduced estimation bias rather than correlating them?
- **Failure Mode Disclosure**: Does the study explicitly identify scenarios where hybrid online-target bootstrapping fails, such as suboptimal fixation in low-noise or strongly optimistic exploration regimes?

---

**Principle 2: Cross-Algorithmic and Cross-Domain Empirical Generality of Bootstrapping Modifications**

**Definition:**  
Because modifications to bootstrapping targets are typically positioned as drop-in improvements applicable across the full spectrum of deep RL, claims of generality must be matched by commensurate empirical breadth. The subfield expects validation spanning multiple algorithmic families—including value-based, actor-critic, and distributional methods—as well as both discrete and continuous control domains, online and offline learning paradigms, and diverse network architectures. Reviewers should evaluate whether the experimental matrix is sufficiently dense to detect context-dependent failures, rather than cherry-picking favorable settings. Reporting standards matter: aggregate metrics such as interquartile mean (IQM) with confidence intervals, per-task performance profiles, and statistically principled seed counts are necessary to distinguish robust improvements from noise. The principle also demands that authors identify boundary conditions where the method offers limited benefit, thereby establishing honest scope rather than overstated universality.

**Core Evaluation Criteria:**
- **Algorithmic Coverage**: Is the method evaluated across distinct algorithm classes (e.g., DQN, IQN, CQL, SAC, Simba, CrossQ+WN) and both discrete and continuous action spaces, rather than being confined to a single algorithmic niche?
- **Benchmark Diversity**: Does the empirical suite span multiple environment suites (e.g., Atari, MuJoCo, DMC-Hard, HumanoidBench) and learning modes (online, offline), with per-task breakdowns in addition to aggregate scores?
- **Statistical Rigor**: Are results reported with standardized robust metrics (e.g., IQM, confidence intervals) across sufficient random seeds following community conventions (e.g., Dopamine protocols)?
- **Scope Boundary Identification**: Does the work candidly report settings where gains diminish or disappear, such as specific task suites or estimator configurations where the modification is neutral?

---

**Principle 3: Fairness and Comprehensiveness of Baseline Comparisons for Bias-Reduction Methods**

**Definition:**  
Novel target-network operators do not exist in a vacuum; they compete against an expanding toolbox of bias-reduction techniques, including double estimators, clipped updates, ensemble minima, functional regularization, and self-correcting variants. A rigorous submission must situate itself within this landscape by comparing against the strongest relevant baselines, correctly adapting them to each experimental setting rather than restricting comparisons to a single algorithmic context. Reviewers should verify that baseline identities are accurately labeled and that methodological cousins—such as clipped double Q-learning or online-network trust regions—are acknowledged and empirically evaluated where feasible. Fairness further requires that all methods share common hyperparameter protocols drawn from official codebases or standardized frameworks, preventing confounding due to tuning disparities. Without such comprehensive benchmarking, it is impossible to ascertain whether the proposed operator provides incremental value or merely replicates existing mechanisms under a different name.

**Core Evaluation Criteria:**
- **Baseline Relevance and Strength**: Does the work include state-of-the-art bias-reduction baselines (e.g., Double DQN, Maxmin DQN, Clipped Double Q, FR-DQN, ScDQN) and adapt them properly to all tested settings (e.g., FR in continuous control, Maxmin in offline RL)?
- **Correct Baseline Nomenclature**: Are baseline labels precise and uncontroversial (e.g., distinguishing "Clipped Double Q" from ambiguous "Double Q"), with proper lineage citations to prior work such as TD3?
- **Related Work Integration**: Are conceptually similar mechanisms (e.g., online-network trust regions, ensemble minimum operators) discussed and, where possible, included as empirical baselines to clarify differentiation?
- **Hyperparameter Parity**: Is there evidence that baselines and the proposed method operate under identical tuning protocols sourced from canonical frameworks, avoiding hidden tuning advantages?

---

**Principle 4: Transparency in Hyperparameter Protocols and Computational Reproducibility**

**Definition:**  
Deep reinforcement learning remains notoriously sensitive to implementation details and hyperparameter choices, making transparency a cornerstone of trustworthy research in target-network design. This principle obliges authors to explicitly document the provenance of every hyperparameter—whether inherited from canonical frameworks like Dopamine or SimbaV2, newly introduced, or tuned—and to release complete code with run configurations. Reviewers must assess whether computational constraints that necessitate protocol deviations, such as differing frame counts across architectures, are justified and applied uniformly to all compared methods. Adherence to community evaluation standards, including seed counts, environment preprocessing, and aggregation statistics, is essential to ensure that results are reproducible rather than idiosyncratic artifacts of a bespoke pipeline. In this subfield, reproducibility is not merely a courtesy but a strict epistemological requirement for validating algorithmic innovations.

**Core Evaluation Criteria:**
- **Hyperparameter Provenance**: Does the paper explicitly state whether hyperparameters are taken from official codebases (e.g., Dopamine, SimbaV2), prior publications, or newly tuned, and are these sources cited?
- **Code and Configuration Release**: Is the full codebase—including run scripts, random seeds, and environment wrappers—released alongside the paper to enable exact replication?
- **Protocol Uniformity**: If computational limits force asymmetric training regimes (e.g., 50M vs. 100M frames), are these deviations justified and applied consistently across all methods sharing the same architecture?
- **Standard Compliance**: Does the evaluation follow established community protocols for random seeds, evaluation intervals, and statistical aggregation, rather than ad hoc custom procedures?

---

**Principle 5: Theoretical Convergence Characterization and Claim Calibration for Heuristic Bootstrapping Operators**

**Definition:**  
Heuristic modifications to Bellman bootstrapping, however intuitive, require theoretical scaffolding to justify their insertion into the learning pipeline. At minimum, submissions should establish convergence properties under controlled conditions—such as tabular or linear function approximation—by proving non-expansion or contraction of the proposed operator within established generalized Q-learning frameworks. Reviewers must also scrutinize the calibration of verbal claims, penalizing overstated language like "eliminates the need" when the evidence supports only risk mitigation, and demanding honest disclosure of limitations such as missing bias bounds or unstudied exploration interactions. Theoretical honesty extends to clearly demarcating the boundary between proven guarantees and empirical conjecture, particularly when neural network function approximation invalidates tabular assumptions. A paper that balances elegant theoretical intuition with candid acknowledgment of its formal limits is more valuable than one that obscures theoretical gaps behind strong empirical results.

**Core Evaluation Criteria:**
- **Convergence Guarantees**: Does the work provide theoretical analysis (e.g., contraction, non-expansion) under interpretable assumptions (tabular, linear), mapping the operator onto recognized frameworks such as Generalized Q-learning?
- **Claim Precision**: Are verbal claims carefully calibrated (e.g., "mitigates" rather than "eliminates"), and are prior works described with accurate attributions of capability?
- **Limitation Candor**: Does the paper openly discuss unresolved theoretical questions (e.g., absence of explicit bias bounds, unknown exploration interactions) rather than omitting them?
- **Theory-Empiricism Boundary**: Is the distinction between formally proven behavior and conjectured behavior under deep function approximation clearly delineated for the reader?