### 1. Research Background and Existing Pain Points

**Research Background:**
Boolean Satisfiability (SAT) is a central reasoning problem in computer science and one of the canonical NP-complete problems. Classic SAT solvers are highly optimized and can handle instances with millions of variables. Part of their success comes from carefully engineered heuristics, such as branching and restart strategies, conflict-driven clause learning (CDCL), and learned clause management. These heuristics are based on recurring patterns that differ across distributions: CDCL solvers are effective on industrial instances with strong community structure, whereas look-ahead solvers are better suited for random instances.

Machine learning offers a promising alternative, where heuristics can be learned from data. Graph Neural Networks (GNNs) have become the main architecture for learning-based SAT solving, since formulas can be naturally expressed as graphs. A common choice is the Literal Clause Graph (LCG), where literals are connected to clauses in a bipartite graph. Existing GNN methods range from end-to-end SAT solvers, such as NeuroSAT and QuerySAT, to hybrid approaches that augment components of classic solvers.

**Existing Pain Points:**
1.  **Design Complexity:** Designing distribution-specific heuristics is time-consuming and requires expertise in the field.
2.  **Expressive Power Limitations:** GNNs are inherently limited in their expressive power—that is, their ability to distinguish different graph structures. The expressive power of GNNs is characterized by the Weisfeiler-Leman (WL) test and the extended k-WL hierarchy, which provide universal limits on which graphs GNNs can distinguish. In particular, any GNN bounded by the WL hierarchy produces identical outputs on graphs that are WL-equivalent. This poses a concrete limitation in the context of SAT, where solving relies on uncovering structural patterns in graphs representing formulas.
3.  **Theoretical Gap:** A fundamental question remains unanswered: Are GNNs expressive enough to reason about satisfiability? A common misconception is that such indistinguishable instances must exist simply because SAT is NP-hard; however, expressivity and computational hardness are unrelated in this way. There exist NP-hard problem families (like PlanarSAT) that the WL hierarchy fully identifies, meaning computational hardness does not necessitate structural indistinguishability.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To analyze the expressive power of GNNs for SAT solving through the lens of the Weisfeiler-Leman (WL) test. Understanding the theoretical limits of GNNs is crucial for determining whether they can fundamentally replace or augment handcrafted heuristics, and for identifying which families of SAT instances are structurally solvable by such architectures.

**Scientific Questions:**
1.  Can the full Weisfeiler-Leman hierarchy distinguish between satisfiable and unsatisfiable formulas in general?
2.  How do expressivity requirements vary across important families of SAT instances (regular, random, planar)?
3.  What are the practical implications of WL indistinguishability for solvers that assign variables sequentially?
4.  Can WL-powerful GNNs theoretically distinguish literals sufficiently to predict a satisfying assignment in practical benchmarks (random vs. industrial)?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper establishes theoretical limitations and positive guarantees for GNNs in SAT solving by mapping the problem to the Weisfeiler-Leman (WL) hierarchy of graph isomorphism testing. It constructs explicit pairs of SAT formulas that are indistinguishable by the WL hierarchy but differ in satisfiability, proving fundamental limits. Conversely, it identifies specific NP-complete SAT families where the WL test is fully expressive. Finally, it designs an experimental setup to quantify the expressivity required to predict satisfying assignments on standard benchmarks by simulating the constraints imposed by the WL test.

**Design Philosophy:**
1.  **Structural Isomorphism as Expressivity Bound:** Use the WL test and the k-WL hierarchy as rigorous theoretical bounds for the expressive power of Message Passing Neural Networks (MPNNs).
2.  **Construction via CFI Gadgets:** Leverage the seminal construction of Cai, Furer, and Immerman (CFI) to build 3-SAT formulas encoding "even orientations" of graphs, transferring graph isomorphism lower bounds to the SAT domain.
3.  **Lossless Graph Representation:** Define the Literal-Clause Graph with Negation connections (LCN) as the standard graph representation to ensure that no logical information is lost when omitting node labels, a necessary step for analyzing WL distinguishability.
4.  **Simulation of Sequential Assignment:** Theoretically model the process of sequential variable assignment (as done in solvers like QuerySAT) to analyze if partial assignments can break symmetries and increase effective expressivity.
5.  **Experimental Necessity Condition:** Test the necessary condition for WL-powerful GNNs to predict satisfying assignments by augmenting formulas with equality constraints between WL-equivalent literals.

### 4. Core Innovation Points

1.  **Proof of WL Hierarchy Limitation for SAT:** The paper proves that the full WL hierarchy cannot, in general, distinguish between satisfiable and unsatisfiable instances. Specifically, it constructs pairs of 3-SAT instances with O(n) variables that are indistinguishable by the n-WL test, despite one being satisfiable and the other unsatisfiable (Theorem 5.3).
2.  **Transfer of Indistinguishability to Sequential Solvers:** The paper shows that indistinguishability under higher-order WL carries over to practical limitations for WL-bounded solvers that set variables sequentially. Even with Ω(n) variable assignments, satisfiability of the residual formula may remain undecidable with a WL-powerful architecture (Lemma 5.4, Corollary 5.5).
3.  **Introduction and Analysis of RegularSAT:** The paper introduces RegularSAT—a family where all instances of the same size are indistinguishable—and proves it is NP-complete (Theorem 5.1), demonstrating that WL-powerful GNNs are essentially useless for this family.
4.  **Positive Identification Guarantees for Planar and Random SAT:** The paper identifies natural families where GNNs can succeed. It proves that PlanarSAT is fully identified by the 4-WL test (Theorem 6.1) and that random SAT instances (extracted from random literal-incidence graphs) are identified by the WL test with high probability (Theorem 6.3).
5.  **Experimental Methodology for Expressivity Quantification:** The paper proposes a novel experimental setup to quantify expressivity needs by solving augmented formulas with equality constraints between WL-equivalent literals, revealing that industrial instances often require more expressivity than random ones.

### 5. Overview of the Overall Technical Solution

The overall technical solution proceeds in four main stages:
1.  **Graph Representation Definition:** Establish the Literal-Clause Graph with Negation connections (LCN) as the representation that uniquely determines the SAT formula up to isomorphism when node labels are omitted.
2.  **Theoretical Construction of Indistinguishable Instances:**
    *   Analyze 3-regular SAT to show WL indistinguishability in uniform degree structures.
    *   Construct 3-SAT formulas $f_G$ and $\tilde{f}_G$ based on the CFI construction. $f_G$ encodes the existence of an even orientation for a graph $G$ (satisfiable iff $|E(G)|$ is even), while $\tilde{f}_G$ encodes the existence of an even orientation when one edge is twisted/bidirectional.
    *   Prove that these formulas are indistinguishable by n-WL but differ in satisfiability.
3.  **Implications for Sequential Solvers:** Prove that partial assignments of variables do not break the WL-indistinguishability between such pairs, implying WL-powerful GNNs cannot distinguish satisfiable residual formulas from unsatisfiable ones even after sequential assignments.
4.  **Positive Analysis and Experimental Validation:**
    *   Analyze PlanarSAT and Random SAT to provide positive identification bounds (4-WL and 1-WL respectively).
    *   Conduct experiments by running the WL algorithm on LCNs, grouping WL-equivalent literals, adding equality constraints to the formula, and checking satisfiability of the augmented formula to determine the critical iteration round ($r_{crit}$) where the formula becomes solvable.

### 6. Detailed Module Design

**6.1 Graph Representation Module (LCN)**
*   **Mechanism:** A CNF formula is mapped to a bipartite graph where literals form one partition and clauses form the other.
*   **Literal-Clause Edges:** Edges connect literals to the clauses in which they appear.
*   **Negation Connections:** To avoid information loss when node labels (variable names) are omitted, additional edges are added connecting each literal to its negation.
*   **Edge Coloring:** Literal-literal edges (negation connections) are assigned a distinct color from literal-clause edges.
*   **Losslessness:** The LCN without node labels uniquely determines the corresponding SAT formula up to isomorphism. The formula can be reconstructed by grouping nodes into variable pairs via literal-literal edges and reading clauses from literal-clause edges.

**6.2 Indistinguishable Instances Construction Module (CFI-based)**
*   **Subformula Construction ($X_k$):** For a vertex $v \in V(G)$ with degree $k=d(v)$:
    *   **Literals:** $A_k \cup B_k$, where $A_k = \{a_i | 1 \le i \le k\}$ and $B_k = \{b_i | 1 \le i \le k\}$.
    *   **Clauses:** $C_k \cup D_k$, where $C_k = \{c_S=(\lor_{i \in S} a_i) \lor (\lor_{i \notin S} b_i) \mid S \subseteq [k], |S| \text{ is even}\}$ and $D_k = \{a_i \lor b_i \mid i \in \{1, \dots, k\}\}$.
*   **Formula Assembly ($f_G$):** For each vertex $v$, add subformula $X_{d(v)}$. Each edge $\{v, w\}$ of $v$ is associated with a literal pair $(a_{v,w}, b_{v,w})$. The node $w$ on the other side uses the negations: $a_{w,v} = \neg a_{v,w}$ and $b_{w,v} = \neg b_{v,w}$.
*   **Twisted Formula Assembly ($\tilde{f}_G$):** Choose an edge $\{v, w\} \in E(G)$ arbitrarily. Twist the literal-literal connections so that $a_{v,w}$ becomes the negation of $b_{w,v}$ and $b_{v,w}$ becomes the negation of $a_{w,v}$.
*   **Satisfiability Mechanism:** The formula $f_G$ encodes the existence of an "even orientation" (each node has even outdegree) of graph $G$. $f_G$ is satisfiable if and only if $|E(G)|$ is even. $\tilde{f}_G$ is satisfiable if and only if $|E(G)|$ is odd. Since one has even edges and the other odd (effectively), exactly one is satisfiable, yet their LCNs are n-WL indistinguishable.

**6.3 Positive Results Module**
*   **PlanarSAT:** Relies on the fact that 4-WL distinguishes all planar graphs. By reducing SAT to PlanarSAT (adding gadgets for edge crossings), the resulting formulas are identified by 4-WL.
*   **Random SAT:** Uses the canonical labeling algorithm for random graphs by Babai et al. (1980). If the top degrees are unique (high probability), WL identifies the graph in 2 iterations. Since the LCN preserves the literal-incidence structure, WL identifies random SAT instances extracted from random LIGs with probability at least $1 - n^{-1/7}$.

**6.4 Experimental Evaluation Module**
*   **WL Iteration:** Run the WL algorithm on $LCN(f)$ for $r \ge 1$ rounds to obtain a partition of literals $L_1, \dots, L_s$.
*   **Augmentation:** Construct an augmented formula $f_r$ that restricts literals in each partition to the same value. For each equivalence class $L_j = \{\ell_{j1}, \dots, \ell_{jn_j}\}$, add equality constraint clauses $g_j$.
*   **Validation:** A WL-powerful architecture can predict a satisfying assignment within $r$ rounds only if $f_r = f \land \bigwedge_{i=1}^s g_i$ is satisfiable. Find the smallest $r$ (denoted $r_{crit}$) where this holds.

### 7. All Mathematical Formulas and Symbol Definitions

**Boolean Satisfiability Definitions:**
*   Variables: $x_1, \dots, x_n$
*   Literal: $\ell$ is a variable $x$ or its negation $\neg x$.
*   Clause: $c = \{\ell_1, \dots, \ell_s\}$ (disjunction $\ell_1 \lor \dots \lor \ell_s$).
*   Formula: $f = \{c_1, \dots, c_m\}$ (conjunction $c_1 \land \dots \land c_m$). Written as $\bigwedge_{c \in f} \bigvee_{\ell \in c} \ell$.
*   Literals set: $L(f) = \cup_{c \in f} \cup_{\ell \in c} \ell$.

**Graph Notation:**
*   Graph: $G = (V, E)$.
*   Neighbors: $N(v)$.
*   Degree: $d(v) = |N(v)|$, Outdegree: $d_{out}$.
*   Node coloring: $\lambda_V : V \to \mathcal{C}$.
*   Edge coloring: $\lambda_E : E \to \mathcal{C}$.
*   Neighbors through color $c$: $N_c(v) = \{w \in V(G) : \exists \{v, w\} \in E(G) \text{ s.t. } \lambda_E(\{v, w\}) = c\}$.

**Isomorphism Definitions:**
*   Graphs $G$ and $H$ are isomorphic if there exists a bijection $\sigma: V(G) \to V(H)$ such that $\{v, w\} \in E(G) \iff \{\sigma(v), \sigma(w)\} \in E(H)$.
*   CNF formulas $f, g$ are isomorphic if there are bijections $\sigma_L: L(f) \to L(g)$ and $\sigma_C: f \to g$ such that:
    1.  $\sigma_L(\neg \ell) = \neg \sigma_L(\ell)$ for all $\ell \in L(f)$.
    2.  $\sigma_C(c) = \{\sigma_L(\ell) : \ell \in c\}$ for all $c \in f$.

**GNN (MPNN) Update Rules:**
*   Aggregation: $a^\ell_v = \text{agg}(\{\{s^{\ell-1}_w : w \in N(v)\}\})$
*   Update: $s^\ell_v = \text{upd}(s^{\ell-1}_v, a^\ell_v)$

**Weisfeiler-Leman (WL) Algorithm Definitions:**
*   **1-WL Update:**
    $\chi^\ell(v) := (\chi^{\ell-1}(v), \{\{\chi^{\ell-1}(w) : w \in N(v)\}\})$
*   **1-WL Update with Edge Colors:**
    $\chi^\ell(v) := (\chi^{\ell-1}(v), \{\{\chi_{N_c} : c \in C_E\}\})$
    where $\chi_{N_c} = \{\{\chi^{\ell-1}(w) : w \in N_c(v)\}\}$.
*   **k-WL Initialization:**
    $\chi^0(v)$ is the atomic type of $v$ for each $v \in V(G)^k$.
*   **k-WL Update:**
    $\chi^\ell(v) := (\chi^{\ell-1}(v), \chi^{\ell-1}_1(v), \chi^{\ell-1}_2(v), \dots, \chi^{\ell-1}_k(v))$
    where $\chi^{\ell-1}_i(v) = \{\{\chi^{\ell-1}(v_1, \dots, v_{i-1}, u, v_{i+1}, \dots, v_k) : u \in V(G)\}\}$.

**Indistinguishable Instances Construction Formulas:**
*   **Subformula $X_k$ Sets:**
    $A_k = \{a_i \mid 1 \le i \le k\}$
    $B_k = \{b_i \mid 1 \le i \le k\}$
    $C_k = \{c_S = (\lor_{i \in S} a_i) \lor (\lor_{i \notin S} b_i) \mid S \subseteq [k], |S| \text{ is even}\}$
    $D_k = \{a_i \lor b_i \mid i \in \{1, \dots, k\}\}$
*   **Twisted Edge Relations:**
    Normal edge $\{v, w\}$: $a_{v,w} = \neg a_{w,v}$ and $b_{v,w} = \neg b_{w,v}$.
    Twisted edge $\{v, w\}$: $a_{v,w} = \neg b_{w,v}$ and $b_{v,w} = \neg a_{w,v}$.
*   **Equivalence of literals in assignment:**
    $(a_{v,w} \lor b_{v,w}) \land (a_{w,v} \lor b_{w,v}) \equiv (a_{v,w} \lor b_{v,w}) \land (\neg a_{v,w} \lor \neg b_{v,w})$
*   **Even Orientation Condition:**
    $\sum_{v \in V} d_{out}(v) = m$

**Experimental Augmentation Formula:**
*   Equality constraint for partition $L_j = \{\ell_{j1}, \dots, \ell_{jn_j}\}$:
    $g_j := (\neg \ell_{jn_j} \lor \ell_{j1}) \land \bigwedge_{i=1}^{n_j-1} (\neg \ell_{ji} \lor \ell_{j,i+1})$
*   Augmented Formula:
    $f_r = f \land \bigwedge_{i=1}^s g_i$

**Variable Reduction Formulas (Appendix D):**
*   $(x \lor y) \iff (x \lor y \lor z) \land (x \lor y \lor \neg z)$
*   Constraint for splitting variable $x$ into $x_1, \dots, x_s$:
    $(x_1 \lor \neg x_2) \land (x_2 \lor \neg x_3) \land \dots \land (x_s \lor \neg x_1)$

### 8. Algorithm Pseudocode

**Algorithm 1: Weisfeiler-Leman Algorithm (1-WL)**
Input: Graph $G=(V,E)$, Initial vertex coloring $\lambda_V$
Output: Stable vertex coloring $\chi$
1: $\chi^0 \leftarrow \lambda_V$
2: $\ell \leftarrow 0$
3: repeat
4:   $\ell \leftarrow \ell + 1$
5:   for $v \in V$ do
6:     $\chi^\ell(v) \leftarrow (\chi^{\ell-1}(v), \{\{\chi^{\ell-1}(w) : w \in N(v)\}\})$
7:   end for
8: until Partition of nodes by $\chi^\ell$ equals partition by $\chi^{\ell-1}$
9: return $\chi^\ell$

**Algorithm 2: k-Dimensional Weisfeiler-Leman Algorithm (k-WL)**
Input: Graph $G=(V,E)$, Integer $k \ge 2$
Output: Stable tuple coloring $\chi$
1: for $v \in V(G)^k$ do
2:   $\chi^0(v) \leftarrow \text{AtomicType}(v)$
3: end for
4: $\ell \leftarrow 0$
5: repeat
6:   $\ell \leftarrow \ell + 1$
7:   for $v \in V(G)^k$ do
8:     for $i \in \{1, \dots, k\}$ do
9:       $\chi^{\ell-1}_i(v) \leftarrow \{\{\chi^{\ell-1}(v_1, \dots, v_{i-1}, u, v_{i+1}, \dots, v_k) : u \in V(G)\}\}$
10:    end for
11:    $\chi^\ell(v) \leftarrow (\chi^{\ell-1}(v), \chi^{\ell-1}_1(v), \chi^{\ell-1}_2(v), \dots, \chi^{\ell-1}_k(v))$
12:  end for
13: until Partition of tuples by $\chi^\ell$ equals partition by $\chi^{\ell-1}$
14: return $\chi^\ell$

**Algorithm 3: Experimental Evaluation of WL Expressivity for SAT**
Input: Satisfiable SAT formula $f$, Max iterations $R$
Output: Critical iteration $r_{crit}$
1: Construct $LCN(f)$
2: Run 1-WL on $LCN(f)$ until convergence at round $r_{converged}$
3: $r_{crit} \leftarrow \text{None}$
4: for $r = 1$ to $r_{converged}$ do
5:   Run 1-WL for $r$ rounds to get literal partition $L_1, \dots, L_s$
6:   Construct augmented formula $f_r = f$
7:   for $j = 1$ to $s$ do
8:     $L_j \leftarrow \{\ell_{j1}, \dots, \ell_{jn_j}\}$
9:     $g_j \leftarrow (\neg \ell_{jn_j} \lor \ell_{j1}) \land \bigwedge_{i=1}^{n_j-1} (\neg \ell_{ji} \lor \ell_{j,i+1})$
10:    $f_r \leftarrow f_r \land g_j$
11:  end for
12:  if $f_r$ is satisfiable then
13:    $r_{crit} \leftarrow r$
14:    break
15:  end if
16: end for
17: return $r_{crit}$