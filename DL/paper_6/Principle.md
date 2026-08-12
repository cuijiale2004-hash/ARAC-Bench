**Principle 1: Conceptual Distinctness Between Combinatorial Hardness and Graph-Theoretic Expressivity in GNN Analysis**

**Definition:**  
In the analysis of GNN expressivity for combinatorial problems such as SAT, it is crucial to maintain a rigorous conceptual separation between computational complexity (e.g., NP-hardness) and structural distinguishability (e.g., WL-indistinguishability). These two notions are fundamentally orthogonal: a problem may be NP-complete yet fully identifiable by low-order WL, while another may admit structurally indistinguishable instances despite being computationally easy. Reviewers must assess whether the paper explicitly articulates this distinction and avoids the common fallacy that hardness implies expressivity limitations. The work should demonstrate that its negative results arise from structural symmetries or non-local properties that elude the WL hierarchy, rather than merely restating that the problem is difficult. Providing concrete counterexamples—such as NP-hard families that are fully WL-distinguishable—is essential to validate the conceptual framework. This principle ensures that expressivity analysis contributes novel structural insights rather than rediscovering complexity-theoretic facts.

**Core Evaluation Criteria:**
- **Explicit separation of concepts:** Does the paper clearly distinguish between computational complexity classes and structural distinguishability under WL, rather than conflating NP-hardness with expressivity limits?
- **Novelty beyond hardness:** Are the negative results derived from structural properties (e.g., symmetry, locality) rather than presented as corollaries of SAT being NP-complete?
- **Counterexample provision:** Does the work provide explicit problem families (e.g., PlanarSAT) that are NP-hard yet fully WL-distinguishable, proving the orthogonality of the two concepts?
- **Resistance to trivialization:** Do the theoretical claims survive the critique that they are "unsurprising because the problem is hard," and does the rebuttal or preemptive clarification successfully address this?

---

**Principle 2: Constructive Rigor and Structural Mechanism in Expressivity Lower Bounds**

**Definition:**  
Papers establishing expressivity lower bounds must provide constructive, mechanistic explanations for why certain architectures fail, rather than relying on existential or black-box arguments. The evaluation centers on whether the authors explicitly build families of indistinguishable instances and identify the precise structural properties—such as global parity constraints, local regularity, or symmetry—that thwart the WL hierarchy. Furthermore, the work should connect its constructions to canonical theoretical frameworks (e.g., Cai-Fürer-Immerman graphs, Tseitin formulas, or pebbling games) to demonstrate depth and continuity with existing literature. The rigor of the proof and the tightness of the bound (e.g., achieving indistinguishability under the full n-WL hierarchy) are critical measures of quality. Finally, the paper should explain why the identified symmetries are inherent to the problem domain and not artifacts of pathological encoding choices.

**Core Evaluation Criteria:**
- **Explicitness of construction:** Are the indistinguishable instances explicitly constructed (e.g., via CFI-like gadgets or parity encodings) rather than proven to exist by non-constructive means?
- **Mechanistic attribution:** Does the paper identify the precise structural feature (e.g., global parity, regularity, local symmetry) that causes WL failure, explaining why the hierarchy cannot detect it?
- **Canonical framework linkage:** Are the constructions tied to established theoretical frameworks (e.g., CFI graphs, Tseitin formulas, resolution complexity, pebbling games)?
- **Tightness and generality:** Does the lower bound hold for the full WL hierarchy or a specific k-WL, and is the result tight with respect to the problem parameters (e.g., clause size, variable count)?

---

**Principle 3: Theoretical-Experimental Alignment and Necessary-Condition Interpretability**

**Definition:**  
When empirical evaluation accompanies theoretical expressivity analysis, reviewers must scrutinize whether the experiments are designed to validate the specific theoretical claims without exceeding their interpretive scope. Because WL-distinguishability provides only necessary conditions for GNN success, experiments that measure WL-convergence or color refinement directly test architectural capacity bounds, not learned performance. The paper must clearly frame its empirical contribution as diagnostic or illustrative, explicitly stating that high WL-distinguishability does not guarantee that a trained GNN will learn the target function. If trained models are omitted, the justification—such as unavailability of suitably scaled industrial benchmarks—should be principled rather than perfunctory. Overall, the alignment between what the theory proves and what the experiments validate must be explicit, transparent, and free of overstated causal inferences from WL color counts to practical solver efficacy.

**Core Evaluation Criteria:**
- **Scope alignment:** Do the experiments directly measure quantities relevant to the theory (e.g., WL iteration counts, color refinement stability) rather than conflating diagnostics with end-to-end learning benchmarks?
- **Necessary-condition framing:** Is the one-directional nature of WL bounds clearly stated—i.e., that WL failure implies GNN failure, but WL success does not guarantee GNN learnability?
- **Omission justification:** If trained GNN experiments are absent, is the omission justified by principled limitations (e.g., scale, data scarcity) rather than convenience, and is the empirical contribution still framed as valid?
- **Transparency of limitations:** Does the paper explicitly acknowledge that WL diagnostics are illustrative and do not demonstrate actual learning performance or generalization gaps?

---

**Principle 4: Justification of Graph Representation and Invariance Properties for Logical Structures**

**Definition:**  
The choice of graph representation for logical or combinatorial structures profoundly impacts what GNNs can theoretically express, making rigorous justification of the encoding scheme a core evaluation criterion. Authors must demonstrate that their chosen representation is information-theoretically lossless and stable under semantic-preserving transformations, such as variable flips that produce logically equivalent formulas. The paper should systematically compare alternative encodings—such as Literal-Clause Graphs versus Variable-Clause Graphs—explaining whether alternatives are lossy, produce non-unique representations for equivalent instances, or otherwise preclude clean expressivity analysis. This justification should extend beyond convenience or common practice to principled arguments about invariance, isomorphism correspondence, and the preservation of problem structure. Failure to address why a specific encoding is necessary and sufficient for the theoretical framework substantially weakens the validity of the derived expressivity bounds.

**Core Evaluation Criteria:**
- **Losslessness and invariance:** Is the chosen graph representation proven to preserve all logical information and remain stable under semantic-preserving transformations (e.g., variable flips)?
- **Comparative encoding analysis:** Does the paper discuss alternative representations (e.g., VCG, LIG, CIG) and explain whether they introduce non-uniqueness, information loss, or instability?
- **Expressivity impact assessment:** Does the work analyze how the encoding choice directly affects WL/GNN expressivity limits, including potential pathologies in alternative schemes?
- **Justification for scope:** If the analysis is restricted to one encoding, is the restriction motivated by fundamental theoretical barriers (e.g., non-isomorphic representations of equivalent formulas) rather than mere convention?

---

**Principle 5: Transferability of Static Expressivity Bounds to Dynamic and Sequential Algorithmic Settings**

**Definition:**  
For GNN-based solvers that operate in sequential or interactive modes—such as iterative variable assignment, restart policies, or query-based architectures—static expressivity bounds must be rigorously extended to dynamic execution traces to remain relevant. Reviewers evaluate whether the paper proves that indistinguishability persists even after partial assignments or state transitions, thereby transferring theoretical limits from the initial graph to the solver's runtime behavior. This requires quantifying how many sequential steps can be executed before symmetry breaking or partial labeling resolves the indistinguishability, if ever. The analysis should explicitly connect to practical solver paradigms, demonstrating that the expressivity bottleneck is not circumvented by temporal unfolding or node labeling tricks. Ultimately, the work must bridge the gap between abstract graph isomorphism and the algorithmic reality of combinatorial solving, ensuring that static negative results carry meaningful implications for deployed systems.

**Core Evaluation Criteria:**
- **Dynamic extension proof:** Are static WL indistinguishability results formally extended to sequential settings, such as partial variable assignments or restarts, with explicit lemmas or theorems?
- **Quantitative persistence:** Does the paper quantify the number of sequential steps (e.g., Θ(n) assignments) for which indistinguishability persists, establishing a concrete dynamic lower bound?
- **Practical solver relevance:** Are the dynamic results tied to realistic solver architectures (e.g., query-based GNNs, CDCL restarts, local search) rather than abstract sequential processes?
- **Architectural implication analysis:** Does the work discuss whether simple modifications (e.g., randomized symmetry breaking, auxiliary features) can overcome dynamic expressivity limits, and prove whether they remain insufficient?