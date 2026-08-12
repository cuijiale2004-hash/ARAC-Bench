**1. Research Background and Existing Pain Points**

**Research Background:**
LLM-based agents have seen promising advances and excel at leveraging vast pre-trained knowledge in tasks such as robotic planning, software engineering, and web automation. However, they are reportedly limited in hard-exploration problems. Hard exploration problems are typically characterized by large state–action spaces, deceptive local optima, and sparse rewards. These factors often trap naive exploration in local optima, such that exploration fails to reach deeper states with rewards. The Jericho benchmark suite of text-based games remains an unsolved hard-exploration problem, where the text-based game environments provide two fundamental challenges: (1) partial observability, requiring agents to construct models of the world from local textual descriptions, and (2) combinatorial state-action spaces. For example in Zork1, the game vocabulary has 697 words and up to five-word commands, resulting in O(697^5) = 1.64 × 10^14 possible actions per step, though only a tiny fraction are grammatically coherent and contextually relevant.

**Existing Pain Points:**
For LLM agents, hard-exploration problems pose two central challenges:
(1) Global learning: for maintaining long-term knowledge of valuable discoveries during exploration. Current LLM agent approaches such as ReAct or Reflexion support local trial-and-error but lack mechanisms for long-term knowledge accumulation.
(2) Local trial-and-error: for quickly refining exploration policies from sparse environmental feedback.
Consequently, LLM agents fall short on hard-exploration tasks that humans can often solve effectively. Existing RL approaches with simple exploration strategies incur hundreds of thousands of interactions to offset sample inefficiencies in exploration. MCTS-based methods suffer from poor sample efficiency, relying on extensive trial-and-error which requires hundreds of thousands of environment interactions. Existing LLM agents have been insufficient to address the challenge of learning from exploration in Jericho games, showing limited performance compared to humans, and existing Go-Explore variants like IGE use ill-defined selection criteria and limited exploration mechanisms.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
The core insight is that both selection and exploration in hard-exploration require structured learning from past exploration experiences, but at different scales. At the global scale, enriching beyond an archive of isolated states by additionally maintaining a trajectory frontier, which keeps the full temporal context of how high value states were reached and why progress stalled, allows an LLM-based analysis across the frontier to infer high-value regions as well as bottleneck states with high future potential, enabling principled state selection beyond heuristic or LLM-internalized notions of interestingness. At the local scale, drawing insights that advantage-based rewards better capture progress signals than Q-values, a Multi-path Advantage Reflection mechanism explores multiple trajectories from the same starting state and leverages LLM reasoning to infer intermediate advantages at key state-action pairs.

**Scientific Questions:**
How can LLM agents effectively maintain long-term knowledge of valuable discoveries (Global learning) while quickly refining exploration policies from sparse environmental feedback (Local trial-and-error) in hard-exploration problems characterized by large state-action spaces, deceptive local optima, and sparse rewards?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
Introduce Global-Local World Memory (GLoW), a framework enabling effective exploration in hard-exploration problems through dual-scale world memory for global and local learning. Inspired by recent views of LLM world models as building implicit representations of task-relevant world knowledge, the world memory modules capture experiential knowledge from exploration trajectories as structured textual representations, rather than modeling transition dynamics. 

**Design Philosophy:**
Build on the Go-Explore algorithm, which achieves breakthroughs on hard-exploration problems by decomposing hard-exploration into alternating between: (1) a selection phase, choosing a promising state from the archive to return to, and (2) an exploration phase, to continue discovering new states from the selected state. The dual-scale world memory instantiates structured learning for both phases:
- Global World Memory: Maintain a trajectory frontier of high-value discoveries and learn value decomposition across this frontier to balance exploitation and exploration for state selection. Implement a semantic form of optimism under uncertainty where UCB uses statistical bonuses while GLoW derives optimistic values from LLM analysis of bottlenecks.
- Local World Memory: Learn from local trial-and-error in exploration through a Multi-path Advantage Reflection mechanism which infers advantage-based progress signals to guide exploration. Compare multiple trajectories from the same starting state to produce pseudo-dense advantage signals from sparse environmental feedback.

**4. Core Innovation Points**

1. **Dual-Scale World Memory Framework (GLoW):** A novel LLM agent framework for hard-exploration problems through global-local world memory, capturing experiential knowledge as structured textual representations at different scales.
2. **Global World Memory with Trajectory Frontier and Value Decomposition:** Enriching beyond an archive of isolated states by maintaining a trajectory frontier, which keeps the full temporal context of how high value states were reached. Using LLM-based value decomposition (Wglobal) to infer high-value regions as well as bottleneck states with high future potential, enabling principled state selection beyond heuristics.
3. **Local World Memory with Multi-path Advantage Reflection (MAR):** Drawing insights that advantage-based rewards better capture progress signals than Q-values, the MAR mechanism explores multiple trajectories from the same starting state and leverages LLM reasoning to infer intermediate advantages at key state-action pairs, producing pseudo-dense advantage signals from sparse environmental feedback.
4. **Semantic Optimism under Uncertainty:** Implementing a semantic form of optimism under uncertainty for state selection, where potential value is derived from LLM analysis of bottlenecks (converging failed trajectories, partial solution patterns, environmental hints) rather than statistical bonuses used in traditional UCB.

**5. Overview of the Overall Technical Solution**

The GLoW framework instantiates the dual-scale world memory by alternating between a selection phase and an exploration phase:
1. **Initialization:** Initialize an empty trajectory frontier F, and a state archive A starting with the initial state s0 and score 0.
2. **Selection Phase (Global World Memory):** Given the trajectory frontier F, apply an LLM operation gLLM to extract the global world memory Wglobal containing critical global states with achieved values and potential values. Then, for each state s in the state archive A, apply an LLM operation alignLLM to score how well the state aligns with the high-value patterns in Wglobal. Select the state snext with the maximum score.
3. **Exploration Phase (Local World Memory):** From the selected state snext, perform iterative exploration by sampling n trajectories sequentially. After each trajectory, apply the Multi-path Advantage Reflection (MAR) mechanism to extract learnings (Wlocal) that inform the next trajectory. MAR takes the local exploration trajectories and the frontier trajectories as inputs to generate structured textual output containing critical states and semantic advantages.
4. **Update Phase:** Collect the newly generated trajectories. Update the trajectory frontier F using a top-k mechanism based on the value function. Update the state archive A by adding newly discovered states that are not redundant based on domain-specific novelty.

**6. Detailed Module Design**

**6.1 Global World Memory Module**
The global world memory module extracts value signals from accumulated exploration trajectories. Unlike traditional state-based archives, it maintains trajectories in a value-ranked frontier and additionally maintains LLM-generated trajectory analysis.

*Value-Ranked Trajectory Frontier:*
As the source of value information, the global world memory module maintains a trajectory frontier F = {τ1, τ2, ..., τk}, containing the k highest-value trajectories discovered during exploration, ranked by a value function v : T → R. Each trajectory τi = (s0^i, a1^i, r1^i, s1^i, ..., aT^i, rT^i, sT^i) represents a complete episode generated by the exploration policy πexplore. For the trajectory value function v, we use the maximum cumulative reward achieved during the episode: v(τi) = max_{t∈[1,T]} ∑_{j=1}^t rj^i. 
In contrast to state-only representations, preserving complete trajectories enables accurate credit assignment and value estimation in sparse-reward environments where success depends on precise action sequences.
The frontier evolves progressively through iterative exploration. When exploration from selected states produces trajectory τnew with value v(τnew), the frontier is updated:
Ft+1 = top-k(Ft ∪ {τnew}, v)
This sliding window mechanism ensures the frontier maintains diverse high-value strategies. For any state si, we can derive the achieved value v(si) = max_{τ∈F, si∈τ} v(τ).

*Motivation: Decomposing value for select and explore:*
Inspired by UCB’s value decomposition which balances exploitation with exploration bonus: V̄(s) + c√(log(N)/ns), where V̄(s) is the empirical mean value and the second term is the exploration bonus based on visit count ns, we annotate two types of values v and v′, corresponding to each term, by analyzing patterns across all frontier trajectories F, to extract a set of critical global states with value annotations, forming the global world memory:
Wglobal = gLLM(F) = {(s1, v1, v′1), (s2, v2, v′2), . . . , (sk, vk, v′k)}
Here, each (si, vi, v′i) represents a critical global state, key semantic landmarks such as exploration frontiers, bottlenecks, and milestones, identified from frontier analysis by a prompted LLM gLLM, where vi denotes the achieved value from si, while v′i reflects LLM’s estimate of future value potential. This potential value v′i cannot be derived from trajectory scores alone, requiring LLM’s reasoning about why trajectories fail and what progress could be achieved by resolving current bottlenecks (e.g., multiple high-value trajectories converge but fail to progress further, partial solution patterns indicate missing components, or environmental hints suggest valuable areas remain undiscovered).

*Balancing Exploitation and Exploration in State Selection:*
We maintain a state archive A = {(si, score(si))} containing discovered states with their achieved scores. Given Wglobal, we select the next exploration state snext by balancing achieved and potential values via LLM. We leverage alignLLM, an LLM-based state selection operation which evaluates how well each archived state s aligns with the high-value patterns identified in Wglobal using a prompted LLM. Since Wglobal contains both achieved and potential values for key frontier states, this alignment naturally balances exploitation (favoring states similar to proven high-reward regions), with exploration (prioritizing states near identified bottlenecks with high potential). Once a state is chosen, we replay the stored sequence of actions to return to the state.

**6.2 Local World Memory Module**
The local world memory module enhances exploration by learning which actions are likely to lead to further progress.

*Motivation: From Q-values to Advantages for Exploration:*
Existing LLM learning methods like self-reflection can be viewed as estimating state-action values (Q-values) from single trajectories. However, Q-value estimation from sparse rewards is notoriously high-variance, and inferences from entire trajectories with sparse feedback are prone to incorrect causal attribution. Drawing from RL theory, advantage functions A(s, a) = Q(s, a) − V(s) reduce variance by comparing actions to a baseline rather than estimating absolute values. Recent work on process reward models (PRMs) further demonstrates that advantage-based rewards are more suited for exploration, by better capturing progress signals than Q-values.

*Multi-path Advantage Reflection (MAR):*
Inspired by TRPO, which computes robust advantage in sparse-reward setting over multiple rollouts from the same state, MAR compares multiple trajectories from the same starting state, to produce pseudo-dense advantage signals from sparse environmental feedback. This effectively densifies the reward signal by inferring intermediate advantages at key state-action pairs.
Given a state s selected based on the global world memory, we perform iterative exploration by sampling n trajectories sequentially: after each trajectory τi, we perform MAR to extract learnings that inform the next trajectory τi+1, in the form of local world memory Wlocal. This creates a sequence Ts = {τ1, τ2, ..., τn} where each trajectory benefits from insights gained from previous attempts.

*Semantic Advantage Representation:*
Concretely, MAR is an LLM operation taking the local exploration trajectories Ts and frontier trajectories F as inputs, and generating the structured textual output Wlocal = {(s*1, As*1), ..., (s*k, As*k)}, where s*1, ..., s*k are critical states (typically 2-4) and each As*i encodes semantic advantages. MAR features two design principles for enhancing the accuracy of semantic advantage inference: First, multi-trajectory comparison enables LLM reasoning to aggregate over divergent outcomes revealing good/bad actions, or consistent patterns confirming reliable strategies, while focusing analysis on critical states. Second, the frontier trajectories provide a stable reference point, grounding the LLM’s evaluation of whether new trajectories constitute meaningful progress. This implements a functional role analogous to a value baseline through context-based reasoning rather than numerical subtraction. Unlike scalar advantages A(s, a), these semantic advantages capture progress signals not expressed by sparse rewards, while serving an analogous functional role by guiding exploration policy through Wlocal.

*Exploration Policy:*
The local world memory module enhances the exploration phase by guiding a policy defined by an LLM agent, as:
πexplore(a|st, ht) = AgentLLM(st, ht, Wlocal, Ts, F)
where ht is the current trajectory history, Ts contains previous trajectories in the same exploration phase, and the policy leverages both learned advantages from Wlocal and successful strategies from frontier F. To address Jericho’s exponential action space, a hybrid approach combines free generation with soft constraints given by valid actions from the Jericho environment.

**7. All Mathematical Formulas and Symbol Definitions**

- F = {τ1, τ2, ..., τk}: Trajectory frontier containing the k highest-value trajectories.
- τi = (s0^i, a1^i, r1^i, s1^i, ..., aT^i, rT^i, sT^i): Complete trajectory representing a full episode, where st ∈ S are states, at ∈ A are actions, and rt ∈ R are rewards.
- v(τi) = max_{t∈[1,T]} ∑_{j=1}^t rj^i: Trajectory value function, representing the maximum cumulative reward achieved during the episode.
- Ft+1 = top-k(Ft ∪ {τnew}, v): Trajectory frontier update rule using a sliding window mechanism.
- v(si) = max_{τ∈F,si∈τ} v(τ): Achieved value from state si across all frontier trajectories.
- V̄(s) + c√(log(N)/ns): UCB value decomposition balancing exploitation (V̄(s)) with exploration bonus.
- Wglobal = gLLM(F) = {(s1, v1, v′1), (s2, v2, v′2), . . . , (sk, vk, v′k)}: Global world memory, where si is a critical global state, vi is the achieved value from si, and v′i is the LLM's estimate of future value potential.
- A = {(si, score(si))}: State archive containing discovered states with their achieved scores.
- snext = argmax_{s∈A} score[s]: Selected next exploration state based on alignment with Wglobal.
- Ts = {τ1, τ2, ..., τn}: Sequence of trajectories from the selected state in the exploration phase.
- Wlocal = {(s*1, As*1), ..., (s*k, As*k)}: Local world memory, where s*k are critical states and As*k encodes semantic advantages.
- πexplore(a|st, ht) = AgentLLM(st, ht, Wlocal, Ts, F): Exploration policy defined by an LLM agent, where ht is the current trajectory history.
- A(s, a) = Q(s, a) − V(s): Advantage function reducing variance by comparing actions to a baseline.
- Var[Âmulti(s∗)] ≤ Var[Âsingle(s∗)] / m: Theoretical variance reduction proposition for multi-trajectory advantage estimation.

**8. Algorithm Pseudocode**

**Algorithm 1 Go-Explore-based Algorithms**
1: procedure GO-EXPLORE-FAMILY(s0, niter)
2: A ← {(s0, 0)} ▷ Archive of (state, score)
3: T ← ∅ ▷ Collected trajectories
4: F ← ∅ ▷ Trajectory Frontier
5: for i = 1 to niter do
— Go Phase (State Selection) —
6: Go-Explore: snext ∼ h(A) ▷ Hand-crafted heuristic (e.g., visit count, domain score)
7: XTX: snext ← ImitationLearning(T ) ▷ Imitation learning
8: IGE: snext ← LLM.SelectPromising(A) ▷ Ill-defined promising-ness
9: GLoW: Wglobal ← gLLM(F) ▷ Principled value decomposition (Sec. 3.1)
10: snext ← alignLLM(A,Wglobal)
11:
— Explore Phase —
12: Go-Explore: τ ← RandomActions(snext) ▷ No learning
13: XTX: τ ← DQN(snext) ▷ DQN with curiosity reward
14: IGE: τ ← ReAct(snext) ▷ Standard LLM agent
15: GLoW:
16: for j = 1 to n do ▷ LLM agent with advantage-driven exploration (Sec. 3.2)
17: τj ← πexplore(snext,Wlocal, {τ1, . . . , τj−1},F)
18: Wlocal ← MAR({τ1, . . . , τj},F)
19: end for
20: T ← T ∪ {τ1, . . . , τn} ▷ Collect trajectories
21: F ← top-k(F ∪ {τ1, . . . , τn}, v) ▷ Update trajectory frontier
22:
— Archive Update —
23: for each state s′ in τ do
24: if IsNotRedundant(s′,A) then ▷ Domain-specific novelty
25: A ← A∪ {s′}
26: end if
27: end for
28: end for
29: end procedure

**Algorithm 2 Value-aligned State Selection**
Require: Frontier F , State archive A
Ensure: Selected state snext
1: Wglobal ← gLLM (F) where
Wglobal = {(s1, v1, v′1), . . . , (sk, vk, v′k)}
vi: achieved, v′i: potential
2: for each state s ∈ A do
3: score[s]← alignLLM (s,Wglobal)
4: end for
5: snext ← argmaxs∈A score[s]
6: return snext

**Algorithm 3 Exploration with MAR**
Require: Selected state snext, Frontier F , Exploration count n
Ensure: Trajectory set Ts
1: Ts ← ∅
2: Wlocal ← ∅
3: for i = 1 to n do
4: τi ← πexplore(snext,Wlocal, Ts,F)
5: Ts ← Ts ∪ {τi}
6: Wlocal ←MAR(Ts,F)
where Wlocal = {(s∗1, As∗1), ..., (s∗k, As∗k)}
7: end for
8: return Ts

**Algorithm 4 GLoW: Global-Local World Memory**
1: procedure GLOW(s0, niter , n, k)
2: F ← ∅ ▷ Initialize frontier
3: A ← {(s0, 0)} ▷ Initialize state archive
4: for i = 1 to niter do
5: snext ← SELECTSTATE(F , A)
6: T ← EXPLORE(snext, F , n)
7: UPDATEARCHIVE(T , F , A, k)
8: end for
9: return argmaxτ∈F v(τ)
10: end procedure
11:
12: procedure SELECTSTATE(F , A)
13: Wglobal ← gLLM(F)
14: snext ← argmaxs∈A alignLLM(s,Wglobal) ▷ Select state based on decomposed value
15: return snext
16: end procedure
17:
18: procedure EXPLORE(s, F , n)
19: T ← ∅ ▷ Initialize trajectory set for current exploration phase
20: Wlocal ← ∅
21: for j = 1 to n do
22: τj ← πexplore(s,Wlocal, T ,F) ▷ Rollout full trajectory from s
23: T ← T ∪ {τj}
24: Wlocal ← MAR(T , F )
25: end for
26: return T
27: end procedure
28:
29: procedure MAR(T , F )
30: Wlocal ← fLLM(T ,F) ▷ Extract semantic advantages at key states
31: return Wlocal
32: end procedure
33:
34: procedure UPDATEARCHIVES(T , F , A, k)
35: for τ ∈ T do
36: F ← top-k(F ∪ {τ}, v) ▷ Update the trajectory frontier
37: for s′ ∈ τ do
38: A ← A∪ {(s′, score(s′))} ▷ Add states to state archive
39: end for
40: end for
41: end procedure