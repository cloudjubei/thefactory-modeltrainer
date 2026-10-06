# Discovering compact "sure win / sure hold" setups

Research note, 2026-10-05. Question: how small can provably optimal play get, and can compact winning or holding "setups" be **discovered and certified by a generic process**, with no hand-written game knowledge, so that they can serve as the *rules* component of the hybrid (table + net + rules, scored in description-length bits)?

Numbers marked *measured* come from throwaway scripts in the session scratchpad (see [Reproduction](#reproduction)). Everything else is cited with a URL. Anything I could not confirm at the primary source is marked **(unverified)**. The machine was heavily loaded during the measurements (load average 45–180 on 10 cores), so all wall-clock times are pessimistic.

---

## 1. Verdict

1. **Yes, a generic, discoverable rule language is plausible as the rules component, and Connect-4 already proves it.** WeakC4 is a complete, machine-checked first-player win from the empty board. At repo HEAD (`a770c05`, 2026-10-05) it holds **132 trunk decisions plus 522 leaves, which use 481 distinct "steady-state" diagrams**. That is about **11 KB** as shipped, and about **35–45 K bits** with entropy coding (*measured/estimated*, §2.4). The whole game is **15–20x smaller than our 659,712-bit net**. Every diagram was found by search (a genetic algorithm, hill-climbing, SAT/CEGIS, and an exact SAT engine), not written by hand. Each one is verified exhaustively against every reply.
2. **The language is generic in form.** It is "win now; else block now; else play the unique legal move whose target cell has the lowest priority level held by exactly one legal move". The only game-specific content is a per-cell priority digit, and the search discovers that digit. It is a strict extension of our `playbook.py` decision lists. It adds cell-identity predicates, which are action coordinates and not game knowledge, and an **∃! (exactly one)** firing rule in place of ∃.
3. **What makes it work is not the language but closed-loop certification plus co-design of the trunk.** A diagram is correct only on the **closed set of positions it steers into**, so the search must be counter-example guided against an exhaustive verifier. Our current `rule_search.py` fits a fixed set of labelled cases, which is not closed-loop. WeakC4 also picks each trunk move *so that its children are steady*. In its curated tree, 76% of stored first-player nodes at ply 8 are leaves, and 86% at ply 10 (*measured*). On positions whose first-player moves were picked at random among winning moves, only 33–40% at plies 10–14 were won by an *existing* diagram (*measured*, §6).
4. **The size argument is decisive wherever a diagram exists.** The median WeakC4 diagram costs about **70 bits** and replaces a strategy subtree of a median **4,038** first-player decision positions (maximum 89,796; *measured*). That subtree costs ≥4,400 bits even at our best table coding of 1.09 bits an entry. The gain is roughly **60x per leaf**.
5. **Recommendation for the first experiment** (§8): use the WeakC4/dave-zyx priority-map language. Run (a) library reuse, then (b) an exact reachability-guarded SAT existence check (Waffle3z's `dsat` design) on the Red-to-move, Red-win frontier of our exact opening table at plies 10, 12 and 14. Report FOUND / UNSAT / TIMEOUT per ply, bits per diagram and subtree size. Then solve the **MDL-optimal weak solution by dynamic programming**: each node is either a diagram leaf or a table move plus its children. Compare this against table + net. A small pilot already gives **29 of 30 positions at ply 16 covered** (25 by library reuse, 4 FOUND by SAT) and **1 proven impossible in the language** (*measured*).
6. **Outside placement and gravity games the priority-map form does not transfer.** This applies to Kalah, the opening of Dots and Boxes, and chess. The same discovery and certification loop does transfer, but over a richer generic predicate language (one-ply lookahead features, as in our playbook and in Bonet–Geffner style feature grammars). The known compact certified setups for those games are mostly **value bounds** (captured stones, stable discs, unconditional life, chain parity), not complete strategies (§5).

---

## 2. WeakC4 in depth

### 2.1 Language and semantics

Source: [explanation page](https://2swap.github.io/WeakC4/explanation/), [CONTRIBUTING.md](https://github.com/2swap/WeakC4/blob/main/CONTRIBUTING.md), [validate_solution.py](https://github.com/2swap/WeakC4/blob/main/solution/validate_solution.py) (the "machine-readable definition"), and [weakc4lab README](https://github.com/Waffle3z/weakc4lab).

- A diagram is a 6x7 grid. Every empty cell carries a priority level, where `0` is the strongest. Cells under stones are ignored, and the validator requires them to hold `0`.
- Red's move at every position reached:
  1. If Red can win now, play the win (lowest column first).
  2. Else, if Yellow threatens to win, block (lowest column first).
  3. Else, for levels 0, 1, …: collect the **playable** cells carrying that level. If there is exactly one, play it. If there are none, or two or more ("they cancel"), drop to the next level.
  4. If no level resolves, the diagram is invalid at that position, and that counts as a loss.
- The diagram never changes during play. It is a static policy, evaluated in O(w·h) time.
- **The language changed recently.** The current numeric "all-skip" language is dave-zyx's (adopted upstream 2026-09-26 per [weakc4lab](https://github.com/Waffle3z/weakc4lab)). The older 2swap language had seven markers: `!` urgent; `@` miai (play only if exactly one is playable); claimeven/claimodd cells (` ` and `|`); and `+`, `=`, `-` as ordered fallbacks. Ties at a non-miai level were fatal. The lab translates the old markers as `!`→0, `@`→1, live claim cell→2, `+`→3, `=`→4, `-`→5, inert claim cell→9. It reports that "all 1,295 translations match … and every re-verified one still wins". dave-zyx describes the new language as containing the old one, with the claimeven/claimodd patterns unrolled into alternating numbered and blank cells ([dave-zyx.github.io](https://dave-zyx.github.io/)).
- **Discrepancy: the level count.** The explanation page says levels run `0`–`d` (14 levels, "exactly four bits per square"). The validator at HEAD accepts only `0`–`9` (commit `6a3a298e`, 2026-09-29: "Enforce only digits 0-9 in steady states, no hex digits allowed"). The protobuf stores each diagram as one base-10 integer of 42 digits (≤18 bytes).

### 2.2 What the tree stores, and how it is verified

- `branches.json` maps each Red-to-move position, stored once per mirror pair, to either a column (a trunk move) or an index into `steady_states.json`.
- The validator runs nine checks. Two carry the proof:
  - **(5)** At every Yellow-to-move node, every legal reply leads to a stored branch, a stored steady state, or an immediate Red win.
  - **(8)** Every steady state wins from its board against every Yellow continuation. A draw is not enough: a full board is a failure.
- Verification is a memoised exhaustive AND-OR walk. Red follows the diagram and Yellow tries every legal move. The CONTRIBUTING file puts it this way: "No rule asks whether a move is objectively best, because the subtree below a move is its own certificate." **No solver is consulted during verification.**
- *Measured* on HEAD, single process, CPython: the full validation passes all 9 checks in **173 s wall / 121 s CPU, 118 MB RSS**.

### 2.3 Size of the solution (HEAD, *measured* with the repo's own scripts)

| quantity | value |
|---|---|
| Red-to-move trunk entries (mirror-folded) | 132 |
| steady-state entries | 522 |
| distinct diagrams | 481 (sharing saved 41; commit `13af3ca8` "Share steady-state diagrams between nodes: 610 → 500") |
| rendered graph (mirrors expanded) | **1,480 nodes: 1,009 leaves (68%), 471 internal** |
| deepest leaf | ply 18 |
| leaves by ply (canonical) | 4:9, 6:33, 8:106, 10:155, 12:112, 14:83, 16:17, 18:7 |
| fraction of stored Red nodes that are leaves, by ply | 4: 38%, 6: 53%, **8: 76%, 10: 86%, 12: 85%**, 14: 95%, 16: 94%, 18: 100% |

**Discrepancy: node counts.** The explanation page still says "under 2,600 nodes … about two-thirds … leaves". The repo's renderer reports 1,480 nodes at HEAD, and the commit history shows a steady fall: about 2,148 on 09-28, then 1,807 on 10-01, then 1,480 on 10-02 (`a3904e51`). The dave-zyx page reports "425 nodes" for his own graph and "~8500 … since reduced to 3754" for 2swap's at the time of writing. These counts are not comparable, because of mirror folding and DAG sharing.

### 2.4 Bits

| encoding | bits | note |
|---|---|---|
| shipped protobuf (`format.proto`) | ≈ 90 K (≈ 11.2 KB) | *estimate from the format*: each trunk node is 3 bits + 7 x 11-bit child refs = 80 bits; each diagram is ≤ 144 bits; plus field overhead. Agrees with the page's "about 12 kilobytes". |
| diagrams, entropy-coded over empty cells only | **33,771** | *measured*: 15,114 empty cells across the 481 diagrams at 2.23 bits/cell (static distribution: 48% `9`, 24% `0`, 9% `1`, …) |
| per diagram | median 32 empty cells, 6 distinct levels, 17 non-`9` cells; ≈ 70 bits | *measured* |
| trunk | ≈ 0.4–10 K | 132 decisions (3 bits each when the shape is implicit, 80 bits each in the shipped format) |
| **whole weak solution** | **≈ 35–45 K bits** (entropy-coded) / ≈ 90 K (shipped) | vs net 659,712 bits; vs our table through ply 9 alone, 4,089 bits ([game-state-compression.md](game-state-compression.md)) |

### 2.5 How steady states are found

- **Genetic algorithm (2swap).** It generates candidates, which are then "verified by brute force", and it is seeded from **cousin steady states**. Mutation is focused "in spots associated with observed blunders" ([explanation](https://2swap.github.io/WeakC4/explanation/)).
- **Hill-climbing / random walk (dave-zyx).** It starts with random diagrams, which "for late-game (and occasionally mid-game) positions … is a surprisingly effective means of obtaining a starting point". It then extends a diagram to cover sibling positions. The walk keeps an archive of 2ⁿ diagrams indexed by which of n sibling positions each one solves, and stops when one solves all n ([dave-zyx.github.io](https://dave-zyx.github.io/)). He describes it as relying "on little understanding or experience of the Connect 4 game".
- **SAT / CEGIS (Waffle3z).** Two engines are documented in [weakc4lab](https://github.com/Waffle3z/weakc4lab):
  - *In-page auto-complete.* A CDCL solver proposes a diagram, the exhaustive verifier returns a losing line, and the line becomes a clause. It fills partially specified diagrams: on a ply-16 root, 4 undecided cells converge in about 20 candidates, 12 in about 560 and 14 in about 3,900. "Past about fourteen it usually will not converge." With all cells undecided at ply 8, 8,919 candidates in 20 s did not converge. **90% of counterexamples are the diagram naming no unique move**, not the diagram playing a losing one.
  - **`dsat`** ([native/dsat.cpp](https://github.com/Waffle3z/weakc4lab/blob/main/native/dsat.cpp), CaDiCaL 3.0.1). This is an *exact decision procedure* for "does a diagram in this language win from this root?". The encoding is a one-hot level per empty cell, a Tseitin "priority circuit" per quiet height-state, a **reachability variable R[b] per visited board**, and per-path chain clauses `R[s] ∧ C[k(s)][m] → R[t]`. Culprit clauses `R[culprit] → ⋁ C[k][w]` range over the **exact winning-move set from an in-process solver**. Soundness rests on guarding every state-specific obligation by that state's reachability variable: "an unguarded version produces confident false UNSATs". FOUND results are re-verified exhaustively; UNSAT is "a proof only for the level count it was built with". The engine runs in hybrid mode: the full envelope is pre-encoded down to a cut ply (root + 16 is "the measured sweet spot"), and CEGIS handles the rest. Reported native times on ply-4 "parity roots" are 7.6–75 s. The memory cost is about 620 B per envelope state, so a full direct encoding of a ply-14 root is "near 30 GB".
- The lab README says: "Everything past ply 12 is already solved … Ply 12 and below is exactly what remains open and exactly what automated search cannot brute-force." I read this as: deep roots are routinely settled by these engines, and the open problem is the shallow trunk. **(Interpretation; not stated more precisely at the source.)**

### 2.6 How often a late position is "steady"

The only public numbers are the curated-tree fractions in §2.3. These are selection-biased, because trunk moves are chosen to make children steady. My own pilot on *uncurated* positions is in §6.

---

## 3. Families of compact winning and holding setups

For each family the table gives: what a setup *is*, how it is *verified*, whether it *composes*, and how a generic search could *discover* it from solved positions.

| family | setup | verification | composes? | discovery from solved positions |
|---|---|---|---|---|
| **Pairing** (Hales–Jewett; Allis claimeven / baseinverse / vertical; Gardner's Hex pairing) | disjoint pairs of empty cells: "if they take one, take the other" | static: every opponent winning line contains a pair (or an owned cell) → opponent cannot complete a line ([Hales & Jewett 1963](https://www.ams.org/journals/tran/1963-106-02/S0002-9947-1963-0143712-1/S0002-9947-1963-0143712-1.pdf)); gravity adds playability constraints (claimeven needs an even column remainder; [Allis 1988](https://tromp.github.io/c4/connect4_thesis.pdf)) | yes, by disjoint union; Allis's VICTOR selects a **compatible set** of rule instances by searching for an independent set in a "solutions × problems" graph (§9.2–9.3 of the thesis) | exact cover / max-SAT over candidate pairs; generic once lines (groups) are read off the terminal test |
| **Potential criteria** (Erdős–Selfridge) | none beyond the rule "move to minimise Σ 2^-(cells left) over live opponent lines" | one inequality: if the potential is < 1/2 (uniform n-sets: fewer than 2^(n-1) sets), Breaker draws ([Erdős & Selfridge 1973](https://www.sciencedirect.com/science/article/pii/0097316573900058)) | additive over lines | parameter-free; evaluate on each frontier position. Usually too weak except in sparse late positions |
| **Parity / zugzwang** (Nim; Connect-4 odd/even threats; Dots and Boxes long-chain rule) | an invariant f(state) that the holder can always restore, e.g. nim-sum 0 ([Bouton 1901](https://www.jstor.org/stable/1967631)), "White odd threat ⇒ White controls zugzwang" (Allis §8), long-chain parity ([Berlekamp, *The Dots-and-Boxes Game*](https://en.wikipedia.org/wiki/Dots_and_boxes)) | induction: every opponent move breaks the invariant, and some reply restores it | yes, via game sums (nimbers) | search for XOR / parity functions over generic features (SAT over parity constraints); verify by induction on solved positions, then exhaustively on the closure |
| **Forcing chains** (threat-space search; λ-search; dependency-based search; VCF in Gomoku; ladders) | a threat sequence where each defender reply is forced | restricted AND-OR search with the defender limited to the forced replies, plus a check that the defender has no counter-threat ([Allis et al., Go-Moku solved](https://aaai.org/papers/0001-fs93-02-001-go-moku-solved-by-new-search-techniques/); [Thomsen λ-search](http://www.t-t.dk/publications/lambda_icga.pdf)) | sequentially; dependency-based search combines independent threats ([Allis 1994 thesis](https://cris.maastrichtuniversity.nl/files/588468/guid-36b5cf0a-cf06-4602-afdb-1af04d65c23b-ASSET3.0.pdf)) | these are proofs produced by search, not induced rules. Our `forced_win(k)` predicate already is one |
| **Virtual connections** (Hex H-search) | (endpoints, carrier set) that a player can connect even moving second | deduction: AND rule (two VCs in series through an empty cell or own stone), OR rule (semi-connections whose carriers have an empty common intersection) ([Anshelevich 2002](https://philpapers.org/rec/ANSAHA-2)) | **yes, by design**: small proven setups compose into large ones; sound but incomplete | closure computation plus stored patterns (edge templates; [Henderson & Hayward, 4-3-2 template](https://webdocs.cs.ualberta.ca/~hayward/papers/probe432.pdf)); new templates found by search |
| **Inferior and dead cells** (Hex ICE; Go Benson; Othello stable discs; Kalah "over half captured") | static facts that bound the value: a cell is dead or captured; a block is unconditionally alive; a disc can never flip | static or local proof ([Benson 1976](https://webdocs.cs.ualberta.ca/~games/go/seminar/2002/020717/benson.pdf); [dead-cell analysis](https://webdocs.cs.ualberta.ca/~hayward/papers/bergeParis.pdf)); ICE uses 273 patterns (per [Henderson et al. 2009](https://www.ijcai.org/Proceedings/09/Papers/091.pdf), via search summary **(unverified count)**) | as bounds and fill-in; they shrink the game, then the rest is searched | generic for "irreversible facts": search for monotone predicates whose truth never flips |
| **CGT decomposition** (Winning Ways; Go endgames; Nimstring) | a local region plus its canonical value (number, nimber, temperature) | local exhaustive search, then the sum theory gives optimal play in the sum ([Müller, decomposition search](https://www.ijcai.org/Proceedings/99-1/Papers/083.pdf): Go endgames with solution lengths over 60 moves; [Berlekamp & Wolfe 1994](https://books.google.com/books/about/Mathematical_Go_Endgames.html?id=U3m1AAAACAAJ)) | **yes, exactly**, but only when regions are independent | requires a generic independence test (no shared cells or liberties); values are computed, not learned |
| **Priority maps** (WeakC4) | a static per-cell priority map plus win/block | exhaustive closure verification (§2.2) | no explicit algebra, but diagrams are **shared** across siblings (522 entries / 481 diagrams) and seed one another | GA, hill-climbing, SAT-CEGIS, exact reachability-guarded SAT (§2.5) |
| **Endgame playbooks with a progress measure** (chess KRK, Advice Languages) | ordered rules plus a well-founded measure ("room" shrinks) | proof of correctness and termination: by hand in Bratko's AL1 ([Bratko 1978](https://chessprogramming.org/Ivan_Bratko)), by SAT in [Maliković & Janičić 2013](https://argo.matf.bg.ac.rs/publications/2013/2013-icga-krk-sat.pdf) | sequential phases | the SAT paper states that "fully automated approaches still cannot produce endgame strategies that are as understandable as human derived strategies". GP ([Hauptman & Sipper 2005](https://link.springer.com/chapter/10.1007/978-3-540-31989-4_11)) and ILP ([Bain & Muggleton, KRK](https://www.chessprogramming.org/Michael_Bain)) produce strong but uncertified players |

**Composition in Hex, quantified.** Hayward, Arneson and Henderson write Yang's hand-made 7x7 centre-opening win (about 40 patterns, about six pages) as an **and/or "autotree"**. And-nodes are *combinatorial sums of independent sub-strategies*; or-nodes merge opponent replies. In this notation the strategy is about 700 lines, with 1,480 and-nodes, 2,339 or-nodes, 3,514 leaves and 25,574 implicit root-to-leaf paths. It verifies in under 1 s. They warn that the number of paths is exponential: Gardner's n×(n−1) pairing strategy has 2^(n(n−1)/2) paths, which is 2^91 at n=14. Compositional checks are therefore needed beyond tiny boards ([Automatic Strategy Verification for Hex](https://webdocs.cs.ualberta.ca/~hayward/papers/verify.pdf)). This is the clearest published example of a **compact certified strategy language with an explicit composition operator**.

**Why our claimeven + baseinverse certificates found almost nothing.** These two rules certify that **Black (second player) holds**, and only when Black controls zugzwang. Allis's §8 shows that White needs an odd threat to take control. Allis used all **nine rules** (claimeven, baseinverse, vertical, aftereven, lowinverse, highinverse, baseclaim, before, specialbefore), combined through the independent-set search. Even then, VICTOR could not settle 1.d1 on 7x6 without conspiracy-number search and a book of about 500,000 positions ([Allis 1988](https://tromp.github.io/c4/connect4_thesis.pdf), abstract and §10.3). On first-player-win trees, a two-rule hold-certificate can only prune Red's *non-winning* alternatives, which is what our 106 / 21,175 result reflects. A priority map contains the claimeven/claimodd follow-up (dave-zyx) and is the better first vehicle.

---

## 4. Discovery methods

| method | what it outputs | certified? | fit for us |
|---|---|---|---|
| SAT/ILP synthesis of decision lists ([Yu, Ignatiev, Stuckey & Le Bodic, JAIR 2021](https://www.jair.org/index.php/jair/article/download/12719/26747/29092); our `rule_search.py`) | minimum perfect list on a **fixed case set** | only on the cases | needs to become **closed-loop**: the policy decides which positions must be covered |
| CEGIS ([Solar-Lezama, sketching](https://people.csail.mit.edu/asolar/papers/thesis.pdf)) and weakc4lab auto-complete | candidate → counterexample → clause | yes (FOUND is exhaustively verified) | good completer; weak clauses (about 120 literals of 224 variables) make it slow from scratch |
| **reachability-guarded exact SAT** (`dsat`) | diagram or UNSAT-in-language | FOUND verified; UNSAT sound for the language | **best fit**: decides existence per frontier node |
| genetic / hill-climbing (2swap GA; dave-zyx archive walk; [GP-EndChess](https://link.springer.com/chapter/10.1007/978-3-540-31989-4_11)) | candidates | only after verification | cheap seed generator; reuse cousins |
| ILP (MIGO, [Muggleton & Hocquette](https://www.researchgate.net/publication/332654748_Machine_Discovery_of_Comprehensible_Strategies_for_Simple_Games_Using_Meta-interpretive_Learning); Popper, [Cropper & Morel 2021](https://arxiv.org/pdf/2005.02259)) | logic programs (e.g. `win_k` minimax predicates for tic-tac-toe and Hexapawn) | MIGO's `win_k` is minimax-grounded | Popper's generate–test–constrain is CEGIS in ILP form; worth borrowing for predicate invention. Note: [Inductive GGP](https://link.springer.com/article/10.1007/s10994-019-05843-w) learns game *rules*, not strategies, so it is out of scope |
| generalised policies ([Bonet, Francès & Geffner, AAAI 2019](https://arxiv.org/pdf/1811.07231), max-SAT; [Francès, Bonet & Geffner, AAAI 2021](https://arxiv.org/pdf/2101.00692); sketches, [Drexler, Seipp & Geffner 2022](https://ojs.aaai.org/index.php/ICAPS/article/view/19786); FOND, [Hofmann & Geffner 2024](https://arxiv.org/abs/2404.02499)) | rules "C → E" over features from a description-logic grammar, with **feature complexity as the cost** | verified on the training instances, by construction | the grammar is the model for a generic predicate language beyond cell maps. Applying FOND policies to adversarial games is my extrapolation |
| distillation from nets (PIRL, [Verma et al. 2018](https://arxiv.org/pdf/1804.02477); VIPER, [Bastani et al. 2018](https://arxiv.org/abs/1805.08328)) | programmatic or decision-tree policies imitating a net | verification is post hoc | candidate generator only (the net as a proposal oracle) |
| proof producers (PNS, [Allis et al. 1994](https://research.tilburguniversity.edu/en/publications/proof-number-search-2/); DFPN, [Pawlewicz & Hayward](https://webdocs.cs.ualberta.ca/~hayward/papers/pawlhayw.pdf); TSS / dependency-based search) | explicit AND-OR proof trees | yes | the **table baseline**; "excised trees" (Hayward et al.) compress proofs by dropping opponent moves |
| succinct strategy representations (decision-tree strategies, [Brázdil et al. TACAS 2018](https://arxiv.org/pdf/1802.00758); BDDs, [Böck 2025](https://arxiv.org/abs/2507.05267)) | compressed strong solutions | yes | the BDD is a whole strong solution (89.6 GB), the wrong regime. Decision-tree strategies show that structure beats bit-level compression |

---

## 5. Per game: known setup families and where they stop

| game | known compact certified setups | largest proven-optimal play with them | where they stop |
|---|---|---|---|
| **Connect-4** | Allis's 9 rules (Black holds; White with an odd threat); WeakC4 priority maps | a **complete first-player win** in about 11 KB (§2) | the trunk (plies ≤ about 8) must be stored; some roots are UNSAT in the language (§6) |
| **Kalah** | value bounds only (more than half the seeds captured ends it); endgame databases. Irving et al. note that "the often-used heuristic of capture difference … might not work very well" ([Solving Kalah](https://naml.us/paper/irving2000_kalah.pdf)) | solved by search: (6,4) in 35 s, (6,5) in 4.7 h on 2000 hardware; (6,6) by Carstensen 2011 **(unverified, [Wikipedia](https://en.wikipedia.org/wiki/Kalah))** | no strategy-level setups known. Sowing makes cell-priority maps meaningless; needs a predicate language (extra turn, capture, seeds-to-pit) |
| **Othello 6x6** | stable discs (a certified lower bound); parity of empty regions (heuristic, uncertified) | Feinstein 1993: second player wins 20–16, about 40 billion positions, 2 weeks **(unverified; secondary sources only, e.g. ["Perfect Analysis in miniature Othello", ICAROB](http://alife-robotics.co.jp/members2015/icarob/data/papers/OS/OS6-3.pdf))**; 8x8 is a draw ([Takizawa 2023](https://arxiv.org/abs/2310.19387)) | flips make the state non-monotone: no published compact winning strategy |
| **Hex (small)** | H-search VCs; templates; inferior-cell analysis; pairing (Gardner) | 7x7 all openings ([Hayward et al. 2003](https://webdocs.cs.ualberta.ca/~hayward/papers/solving7x7hex.pdf)); 8x8 ([Henderson et al. 2009](https://www.ijcai.org/Proceedings/09/Papers/091.pdf)); 9x9 all openings and one 10x10 opening ([Pawlewicz & Hayward](https://webdocs.cs.ualberta.ca/~hayward/papers/pawlhayw.pdf)); hand strategies for 7x7–9x9 centre openings by Yang | H-search is incomplete; search over VCs is needed beyond about 9x9. **Best-composing family of all** |
| **Dots and Boxes** | long-chain parity rule; Nimstring / CGT; optimal play in "simple loony endgames" by a human-learnable algorithm ([Buzzard & Ciere](https://arxiv.org/abs/1305.2156)) | 4x5 boxes is a tie ([Barker & Korf 2012](https://cdn.aaai.org/ojs/8144/8144-13-11671-1-2-20201228.pdf)); 5x5 "solved" claim **(unverified, [BGG thread](https://boardgamegeek.com/thread/1105496/5x5-dots-and-boxes-solved))** | rules cover the endgame once chains have formed, not the opening |
| **Chess** | rule of the square, opposition; Bratko's KRK strategy (proven correct, not move-optimal) | 7-piece tablebases ([Syzygy](https://chessprogramming.org/Syzygy_Bases)) | rules give *wins*, not DTM-optimal play; beyond tiny endgames only tables exist |
| **Go** | Benson unconditional life; CGT endgames; ladders (first-order λ-search) | small boards up to 30 points (5x6, 4x7) by search ([van der Werf & Winands 2009](https://www.researchgate.net/publication/220174568_Solving_Go_for_Rectangular_Boards)); life-and-death by DFPN ([Kishimoto & Müller 2005](https://cdn.aaai.org/AAAI/2005/AAAI05-218.pdf)) | ko, seki and whole-board fights; decomposition needs safe boundaries |

The pattern across the ladder: **certified compact strategies exist where moves are irreversible placements on a fixed geometry** (Connect-4, Hex, Gomoku, Dots and Boxes endgames). Elsewhere the compact certified objects are value bounds that shrink the search, not policies.

---

## 6. Pilot: do uncurated frontier positions admit a diagram? (*measured*)

**Sample.** Red-to-move, Red-win positions at ply N. Red picks a uniformly random winning move: from the repo's `books/c4_labels.json.gz` up to 8 stones (positions missing from the book were resampled), and from an exact solver beyond. Yellow picks a uniformly random legal move that does not hand Red an immediate win. There were 30 positions per ply.

**Steps.**
- (a) An all-tie "blank" diagram, which tests whether win/block tactics alone carry the position.
- (b) Pure claimeven and pure claimodd diagrams.
- (c) **Library reuse**: each of WeakC4's 481 diagrams and their mirrors (962) is verified exhaustively.
- (d) For positions with no library hit, `dsat` built from the published source with CaDiCaL 3.0.1, hybrid mode, with a 90–150 s cap.

| ply (stones) | tactics only | pure claimeven/odd | library hit | dsat on the rest | admits a diagram (lower bound) |
|---|---|---|---|---|---|
| 10 | 0/30 | 0/30 | 11/30 (37%) | not run | ≥ 37% |
| 12 | 0/30 | 0/30 | 10/30 (33%) | 3 tried: 1 UNSAT, 2 timeout | ≥ 33%; ≥ 1 proven impossible |
| 14 | 0/30 | 0/30 | 12/30 (40%) | 6 tried: 1 FOUND, 1 UNSAT, 4 timeout | ≥ 43%; ≥ 1 proven impossible |
| 16 | 0/30 | 0/30 | 25/30 (83%) | **all 5: 4 FOUND, 1 UNSAT** | **29/30 = 97%**; 1 proven impossible |

**Readings.**
- Coverage rises sharply with depth. **A shared library of discovered diagrams already covers a third of random frontier positions at plies 10–14**, before any new search. Library hits included very general diagrams: diagrams 0–2 won many unrelated ply-16 positions.
- Neither win/block alone nor pure claimeven covered anything. So hits are not vacuous, but this must be re-checked on the real frontier.
- UNSAT occurs at every depth tried. Some winning positions have **no** diagram in the 10-level language, so the trunk must branch there.
- At ply 12–14, `dsat` mostly hit the time cap under this machine's load. FOUND diagrams took 9–79 s wall at ply 16. The lab reports native times of 8–75 s at ply-4 roots on an unloaded machine.
- The FOUND diagrams look like WeakC4's, e.g. `4514234351213452` → `["0071777","0025052","009Y666","YYRRR02","YRYYR86","RRYRY75"]`.

---

## 7. Size arithmetic for the hybrid

| item | bits | source |
|---|---|---|
| net (20.6 K params x 32) | 659,712 | [game-state-compression.md](game-state-compression.md) |
| own-strategy table through ply 9 | 4,089 at 3 b/entry; ≈ 1,560 rank-coded | same |
| Red-to-move nodes of a heuristic own tree at ply 10 / 12 | 954 / 3,736 | earlier scratch measurement (`trees.json`), not re-run |
| one fresh diagram | ≈ 70 (32 empty cells x 2.23) | *measured* (§2.4) |
| one reused diagram | ≈ log2(library size) ≈ 9–10 | — |
| subtree a diagram replaces | median 4,038 Red decisions (≥ 4.4 K bits at 1.09 b/entry) | *measured* (WeakC4 leaves) |

If every one of 954 ply-10 frontier nodes needed its own fresh diagram, the rules would cost about 67 K bits. Table plus rules would then be about **70 K bits, roughly 9x smaller than the net**, with every claim certified. Uncovered nodes push the table two plies deeper, at about 4x more nodes each time, so the **coverage fraction p at the frontier decides the outcome**. WeakC4 shows that choosing trunk moves for steadiness raises p from about 35% (random winning move, §6) to about 85% (curated, §2.3).

---

## 8. Recommended first experiment

**Language (generic).** An ordered list: `wins(m)`, `blocks(m)`, then levels 0…L−1 (L = 10). Level k fires when **exactly one** legal move m has `cell(m) ∈ S_k`. The cell sets S_k over the empty cells are the discovered content. In `playbook.py` terms this adds two things:
- a generic `cell = i` predicate (the action's target square, available in any placement game);
- an `∃!` firing mode beside the current `∃`.

The tactical prefix uses only `step` and `winner`. Nothing is hand-written; the "unique at level" semantics is a language choice, which the constraint allows.

**Frontier.** Use the exact Red-to-move, Red-win nodes of our opening table at plies 10 and 12 (and 14 for the trend). Use both (i) our current table's move choice and (ii) a re-chosen trunk (below).

**Per frontier node f:**
1. **Library reuse.** Try every diagram found so far, and its mirror, under the exhaustive checker. The cost is per failed candidate and usually tiny, because failures stop at the first counterexample.
2. **Existence.** Port `dsat` or call it. Use a reachability-guarded SAT encoding with an exact oracle supplying winning-move sets, hybrid cut at root + 12 to 16, a state budget of at most 1.5 M (about 1 GB), and a time cap. Record FOUND / UNSAT / TIMEOUT. TIMEOUT counts as *not covered* (fail closed).
3. **Minimise.** Turn non-load-bearing cells to `9` (the lab's "Simplify"), then entropy-code the diagram.

**Trunk co-design (the MDL step).** Bottom-up DP over the table:

cost(n) = min( diag(n) if FOUND, min over winning moves m of [ code(m) + Σ over Yellow replies y of cost(child(n, m, y)) ] )

Here `code(m)` is about 1 bit under rank coding. This yields the MDL-optimal weak solution for this language and depth, which is what WeakC4 does by hand and partly by search ("I wasn't able to search for minimal branches at a depth any higher than about 8").

**Report.**
- per ply: % FOUND / UNSAT / TIMEOUT; bits per diagram; Red-decision subtree size per diagram; library hit rate and reuse factor;
- totals: table-only vs table + rules vs table + net;
- for scale: WeakC4 at about 35–45 K bits for the whole game.

**What the checker must verify** (mirroring WeakC4's nine checks):
- the root is Red-to-move, non-terminal and Red-winning;
- at every position reached with Red following the policy and Yellow playing *every* legal move:
  - the policy resolves to exactly one legal move (fall-through = failure);
  - no Yellow four; a full board = failure (a draw is not a win);
  - the game ends in a Red four;
- memoisation on board keys, so that transpositions and mirror handling are sound;
- at table nodes: every Yellow reply is covered by a table entry, a diagram or an immediate win; no unreachable entries.

**Compute bound.** Verification cost is the size of the policy-closed subtree (WeakC4: median 4,038, maximum 89,796 Red positions; 522 leaves in 121 s CPU in Python). Cap it at, say, 10^6 positions and reject when exceeded; the lab uses a 5 M node budget. For search, the SAT size is bounded by the hybrid cut and state budget, and the deadline is enforced inside the solver (CaDiCaL terminator), since one `solve()` call is otherwise uninterruptible.

**What could make it vacuous, and the guard for each:**

| risk | guard |
|---|---|
| tactics-only positions, where win/block carries everything | run the all-tie diagram first and report that class separately (0/120 in the pilot) |
| blunder-heavy frontiers (uniform Yellow) inflating coverage | report coverage weighted by subtree size and by depth-to-win; separate positions with `wins_in_5` true |
| selection bias from re-chosen trunk moves | report both the fixed-table and the co-designed frontier |
| verifier bugs | an independent second checker (the lab demands that two implementations agree); mutation tests: draws accepted, ties not skipped, full-board-not-failure, missing Yellow reply, mirror mix-up |
| unsound UNSAT | use only the guarded encoding; never read UNSAT as a game value, only as "not in this language at L levels" |
| oracle leakage | the oracle may guide search, but the **verifier must not consult it**; the certificate is the closed subtree |
| bit accounting | cells under stones are free; count the language's fixed prefix once; count library indices; count the frontier flag per node |

**Beyond Connect-4.**
- **Hex** (the cell language transfers directly; add the "and" composition of disjoint sub-strategies from the Hex autotree work).
- **Kalah, Othello, chess.** Keep the loop but switch the level predicates to generic one-ply lookahead features: `wins`, `blocks`, `gives_win`, `makes_threat`, `forced_win_k`, plus action ids. This is the Bonet–Geffner feature-grammar route, with the feature cost entering the MDL score. Expect low coverage there; the known certified objects are mostly bounds (§5).

---

## Reproduction

All in the session scratchpad: `…/scratchpad/weakc4/` and `…/scratchpad/wlab/`.

- **WeakC4 at `a770c05`.** `print_statistics.py`; `validate_solution.py --jobs 1` (173 s); `leafstats.py` (per-leaf empty cells, levels, entropy, Red positions visited); `representations/webclient/render.py --report` (1,480 nodes, 1,009 leaves).
- **dsat.** Built from `Waffle3z/weakc4lab/native` and CaDiCaL `rel-3.0.1`: `g++ -std=c++20 -O3 -DC4_W=7 -DC4_H=6 -Icadical/src dsat.cpp libcadical.a`.
- **Pilot.** `pilot/sample.cpp` (sampler), `pilot/libhit2.py` (tactics + library reuse), `pilot/base.py` (claimeven/claimodd), and dsat logs `pilot/d14a.log`, `d14b.log`, `d12.log`. Run single-process, `nice 19`, under 1 GB.

## Discrepancies and unverified items

- **WeakC4 explanation page vs repo HEAD.**
  - The page says "under 2,600 nodes" and "about two-thirds" leaves; HEAD renders 1,480 nodes, 1,009 of them leaves (68%).
  - The page says levels `0`–`d` and 4 bits/cell; the validator accepts `0`–`9` only.
  - The page credits a GA plus SAT; the repo history also credits dave-zyx's hill-climbing and an exact SAT engine.
- **Graph sizes are not comparable.** dave-zyx's "425 nodes" and 2swap's "3754 / ~8500" are counted with different mirror and DAG conventions.
- **The 11.2 KB protobuf size is my estimate** from `format.proto`; the page says "about 12 kilobytes".
- **The WeakC4 page's claim that checking the tree is faster than solving with Fhourstones** was not reproduced. My Python validation took 173 s on a heavily loaded machine.
- **Unverified game results.** Kalah(6,6) (Carstensen 2011), Othello 6x6 (Feinstein 1993, a newsletter report) and 5x5 Dots and Boxes are known from secondary sources only. The ICE pattern count (273) comes from a search summary.
- **The weakc4lab sentence** "Everything past ply 12 is already solved" is quoted; my reading of it is an interpretation.
