# Game-state compression for solving and learning finite board games

Research note, 2026-10-05. Question: should the hybrid (net + table, scored in description-length bits) key its table by the **move sequence** that reached a position (Connect-4: 3 bits a column; Kalah: the hole sown), and grow a "rainbow table" of these sequences as exploration fills gaps?

All counts below marked *measured* were produced by throwaway scripts that use the repo's `games/connect4.py` and `games/kalah.py` (see [Reproduction](#reproduction)). Everything else is cited with a URL. Anything I could not confirm at the source is marked **(unverified)**.

---

## 1. Verdict

1. **A sequence-keyed table is never smaller than a position-keyed one, and the gap grows fast.** Transpositions mean many sequences reach the same position. In Connect-4 the measured ratio of sequences to positions is 1.4x at ply 3, 31x at ply 8 and 404x at ply 11, and it grows by about 2.3–2.5x a ply. Kalah has almost no transpositions early (1.05x at move 9 in Kalah(6,4)), but in Kalah(4,3) the ratio still reaches 31x by move 17.
2. **The hybrid's best table needs no keys at all, so key encoding barely matters for it.** Inside a first-player strategy tree, the full table stored in canonical walk order costs 3 bits an entry. Measured on a perfect Connect-4 strategy through ply 9 (1,363 decision positions), that comes to **4,089 bits**. The same tree keyed by explicit move sequences costs **57,630 bits** (14x more). Walking the sequence tree with implicit keys costs **6,714 bits** (1.6x more, because the strategy tree has 2,238 sequences for 1,363 positions). Every one of these is 10–420x smaller than the 659,712-bit net.
3. **The real lever is how each entry is coded, not how it is keyed.** Store each table entry as the **rank of the chosen move under a free predictor** (win, then block, then centre-first), and entropy-code that rank. Measured: **1.09 bits an entry** for Connect-4 (79% of entries are rank 0) and **0.96 bits an entry** for Kalah(4,3). That gives about 1,560 bits for the Connect-4 tree, which is 420x smaller than the net. This is the same move-ranking trick Lichess uses to store chess games (Huffman-coded move ranks; [lichess-org/compression](https://github.com/lichess-org/compression)).
4. **Brute-force coverage is feasible for Connect-4 and Kalah(6,4), and nowhere above them.** All of Connect-4 at 2 bits a position under a perfect index takes 1.13 TB, and a BDD has already done it in 89.6 GB ([Böck 2025](https://arxiv.org/abs/2507.05267)). Sequence-keyed, it would need about 10^21 entries. Checkers (5×10^20 positions), chess (~4.8×10^44) and Go (~2.08×10^170) are out of reach by 8 to 158 orders of magnitude.
5. **For a learner, a compact state code is only a storage and lookup format; it is a poor network input.** A move sequence makes the net learn which hundreds of sequences are the same board, and that is exactly the invariance a board encoding gives for free. A perfect index or a hash has no locality, so a net cannot generalise from it. Sequence models *can* recover the board ([Othello-GPT, Li et al. 2023](https://arxiv.org/abs/2210.13382)), but that spends capacity and data. It does not save bits.
6. **"Rainbow table" is the wrong analogy.** A rainbow table inverts a one-way function by storing chain endpoints and recomputing chain interiors ([Oechslin 2003](https://infoscience.epfl.ch/bitstreams/37cd827e-c3a6-4ba3-b68f-8ab7669b4449/download)). Game solving has no one-way function to invert. The useful idea in the proposal is a partial table that grows as exploration proceeds and is backed by search. That idea already exists in position-keyed form as transposition tables, opening books, proof trees and partial-information endgame databases ([Björnsson, Schaeffer & Sturtevant 2005](https://link.springer.com/chapter/10.1007/11922155_2)). Keying such a table by sequence only makes it larger.

**Recommendation.** Do not build sequence-keyed tables. Keep the canonical-walk table, and change the pricing in `harness/description_length.py` in two ways. First, a full table costs about 3 bits an entry, not `ceil(log2 P) + 3` = 14. Second, try rank-coding the entries under a zero-parameter predictor. Section 7 proposes two experiments.

---

## 2. Literature

### Connect-4 counts and encodings

| Source | Result | URL |
|---|---|---|
| Tromp, OEIS A212693 | Legal positions after n plies: 1, 7, 49, 238, 1120, 4263, 16422, 54859, 184275, 558186, 1662623, 4568683, … Sum over n = 0..42 is **4,531,985,219,092**. | https://oeis.org/A212693 |
| Tromp, *Connect Four Playground* | 8-ply database of exact results (about 40,000 CPU-hours at CWI). 67,557 "unfinished and unforced" 8-ply positions released as an ML dataset. The playing applet reads the 8-ply database "from a 12Kbyte compressed file". The bitboard code shows how to "encode a position in 49 bits" (7 columns × (6+1) bits). | https://tromp.github.io/c4/c4.html |
| Repo | `harness/solver.py: canonical_key` uses the Pons key `position + mask` (49 bits), mirror-reduced. | local |
| Allis 1988, *A Knowledge-based Approach of Connect-Four* (VU Amsterdam, report IR-163) | Knowledge-based solve (program VICTOR: nine strategic rules plus conspiracy-number search): first player wins. Allis's upper-bound estimate of positions was 70,728,639,995,483 (quoted by Edelkamp & Kissmann). WeakC4 says Allis used an opening book of 500,000 nodes **(not checked in the thesis itself)**. | http://tromp.github.io/c4/connect4_thesis.pdf |
| Edelkamp & Kissmann 2008, *Symbolic Classification of General Two-Player Games* (KI 2008) | Exact count 4,531,985,219,092 by BDD breadth-first search. Explicit storage at 85 bits a state would need "nearly 43.8 terabytes" (43.8 TiB). The BDD of **all reachable states uses 84,088,763 nodes**. | https://fai.cs.uni-saarland.de/kissmann/publications/ki08-ggpsolver.pdf |
| Edelkamp & Kissmann, AAAI 2011 | The reachable set has polynomial-size BDDs under some variable order. The termination (win) predicate needs exponentially many nodes under every order. | https://mlanthology.org/aaai/2011/edelkamp2011aaai-complexity/ |
| Dietzfelbinger & Edelkamp, KI 2009 | Minimum perfect hash (rank and unrank) computed directly from the BDD of the reachable set, in O(n) time per query. | https://link.springer.com/chapter/10.1007/978-3-642-04617-9_5 |
| Böck 2025, *Strongly Solving 7×6 Connect-Four on Consumer Grade Hardware* | Strong solution (win/draw/loss for every reachable position) stored as BDDs in **89.6 GB**. Built in 47 h on one core with 128 GB RAM. Draw is not stored because draw = ¬win ∧ ¬loss. The paper's text puts ply 27 as the largest layer at about 1.1 billion nodes and 10.2 GB **(read from the HTML; not cross-checked)**. | https://arxiv.org/abs/2507.05267 |
| 2swap, *WeakC4* | A search-free weak solution: an opening tree of **"under 2,600 nodes"** whose leaves are "steady-state" rule diagrams, which "fits in about 12 kilobytes". The brief quoted ~25 KB and a search snippet said ~22 KB; the live page says ~12 KB, so the size may have shrunk across versions. | https://2swap.github.io/WeakC4/explanation/ |
| Takizawa 2024/2026, *Semi-Strongly Solved* | Certifies only the positions reachable when at least one side plays optimally. The abstract says this is "9,074x smaller than a published strong baseline" for Connect-4 **(abstract only; counting conventions not checked)**. Conceptually this is the hybrid's strategy-tree idea. | https://arxiv.org/abs/2411.01029 |
| Considine 2024, *Compressed Game Solving* | Move generation applied directly to compressed sets of position strings. In Breakthrough, sets of n positions took about n^0.5 to n^0.7 space. | https://arxiv.org/abs/2411.07273 |

### Ranking and perfect indexing in other solved games

| Game | Result | URL |
|---|---|---|
| Kalah, Irving, Donkers & Uiterwijk 2000 (ICGA J. 23(3)) | State-space sizes are in Table 2: **Kalah(6,4) = 1.31×10^13**, Kalah(4,3) = 77,134,200. Full game graph of Kalah(4,3) = **4,604,996** positions; ≥94% of the configuration space is unreachable. The authors' "conservative estimate" of the Kalah(6,4) graph is about **1.3×10^12**. Game-tree complexity of Kalah(6,4) is about 6.0×10^18. Endgame databases are indexed by **combinatorial ranking** over the active counters: Size = C(n+2m, 2m) − 2·C(n+m, m) + 1, "no unused space", **4 bits an entry** (for example, 30 counters means 11,054,221,305 positions in 5,271 MB). Kalah(6,4) solved in 35 s on an 800 MHz Athlon. | https://naml.us/paper/irving2000_kalah.pdf |
| Awari, Romein & Bal 2002/03 | All **889,063,398,406** positions scored. The largest database holds 204 billion entries in 178 GB **(numbers from the abstract and search snippets; full paper not read)**. | https://www.semanticscholar.org/paper/Awari-is-Solved-Romein-Bal/9651f4a7fd03be889d1e8a47407471ca38d68381 |
| Nine Men's Morris, Gasser 1996 | After symmetry and reachability reductions there are 7,673,759,269 states. A perfect hash maps them into **9,074,932,579 indices (85% dense)**, at 1 byte a state. All databases except 8-3, 9-3 and 9-4 come to about 9 GB. An 18-ply α-β search from the opening joins the databases and shows the game is a draw. | https://www.cs.brandeis.edu/~storer/JimPuzzles/GAMES/NineMensMorris/INFO/GasserArticle.pdf |
| Checkers, Schaeffer et al. 2007 (Science) | About 5×10^20 positions. The ≤10-piece databases hold **3.9×10^13 positions in 237 GB, "an average of 154 positions per byte"** (0.05 bits a position). The stored proof tree has about 10^7 positions, each backed by a search of about 10^7. The paper says a complete strong database "would require 5×10^20 data entries. Even an excellent compression algorithm might only reduce this to 10^18 bytes". | https://www.cs.mcgill.ca/~dprecup/courses/AI/Materials/checkers_is_solved.pdf |
| Chess tablebases | Nalimov 6-piece: 1.2 TB. Syzygy 6-piece: 67.8 GiB WDL + 81.4 GiB DTZ. Syzygy 7-piece: **8.5 TiB WDL + 8.3 TiB DTZ = 16.7 TiB (18.4 TB)**. Lomonosov 7-piece (DTM, 2012): about **140 TB**. 7-piece positions: 423,836,835,667,331. 8-piece: about 3.8×10^16 positions, estimated at ~2 PB in Syzygy format. | https://chessprogramming.org/Syzygy_Bases , https://en.wikipedia.org/wiki/Endgame_tablebase |
| Chess position count, Tromp | **(4.79 ± 0.04)×10^44** legal positions, estimated by sampling ranked positions. Tromp's ranking needs about 153 bits a position (upper bound about 2^152.4). | https://github.com/tromp/ChessPositionRanking , https://tromp.github.io/chess/chess.html |
| Go, Tromp & Farnebäck | Legal 19×19 positions ≈ **2.08×10^170** (about 1.2% of 3^361). | https://tromp.github.io/go/gostate.pdf |
| Game complexities (table) | Connect-4: state space 10^12–10^13, game tree 10^21. Kalah: 10^13 / 10^18. Checkers: 10^20 / 10^40. Chess: 10^44 / 10^123. Go: 10^170 / 10^505. | https://en.wikipedia.org/wiki/Game_complexity |

### Move-sequence encodings and opening books

| Source | Result | URL |
|---|---|---|
| Polyglot book format (chess) | Books are **position-keyed**: each 16-byte entry is an 8-byte Zobrist key, a 2-byte move, a 2-byte weight and 4 bytes of learning data. Transpositions merge automatically. | http://hardy.uhasselt.be/Toga/book_format.html |
| Lichess game compression | Games are stored as move sequences: each move is coded as its rank in a heuristically ordered legal-move list, then Huffman-coded. The repo describes the scheme as ~4.4 bits a move across the Lichess dataset **(figure from a search summary; not confirmed in the repo text)**. | https://github.com/lichess-org/compression |
| Buffett, *Compressing chess moves even further* | 3.7 bits a move using arithmetic coding over ranked legal moves (measured on opening lines). | https://lichess.org/@/marcusbuffett/blog/compressing-chess-moves-even-further-to-37-bits-per-move/YgugQc42 |
| Li et al. 2023, Othello-GPT (ICLR) | A transformer trained only on move sequences builds an internal board representation that probes can read out. Sequences are learnable, but the model has to reconstruct the state itself. | https://arxiv.org/abs/2210.13382 |
| Oechslin 2003, rainbow tables (CRYPTO) | A time–memory trade-off for inverting hashes. Chains of hash and reduction functions, with only the chain endpoints stored. | https://infoscience.epfl.ch/bitstreams/37cd827e-c3a6-4ba3-b68f-8ab7669b4449/download |
| Björnsson, Schaeffer & Sturtevant 2005, *Partial Information Endgame Databases* | Retrograde analysis with unknown values present, for building only the parts of a database that matter. This is the principled version of "fill gaps as you go". | https://link.springer.com/chapter/10.1007/11922155_2 |

I found no literature that keys solved-game tables by move sequence. Opening books, transposition tables and tablebases all key by position, and the measured transposition factors below explain why.

---

## 3. Bits per position, by encoding (Connect-4)

| Encoding | Bits | Notes |
|---|---|---|
| (a) Move sequence, 3 bits a ply | **3n**; up to 126 at n = 42 | Plus a length: either a 0-terminator (3 more bits) or one fixed-width file per ply. The information floor is log2(7) = 2.81 bits a move. |
| (b) Tromp/Pons bitboard key | **49** | 7 × (6+1) bits; `position + mask` is unique. |
| 2 bits a cell + side to move | 85 | Edelkamp & Kissmann's explicit encoding. |
| Ternary over 42 cells | 67 (42·log2 3 = 66.6) | |
| (c) Perfect index over all reachable positions | **43** (log2 4.53×10^12 = 42.04) | Rank and unrank via a BDD (Dietzfelbinger & Edelkamp). |
| (c') Perfect index within ply n | log2 A212693(n) | At most about 37 bits: the largest per-ply count shown is 1.4×10^11 (ply 26). |
| Mirror-canonical | −1 bit | Measured mirror classes ≈ positions / 2. |

Measured, by ply (bits = log2 of the count):

| ply n | 3n (sequence key) | log2 #sequences | log2 #positions (per-ply index) | log2 #mirror classes |
|---|---|---|---|---|
| 4 | 12 | 11.2 | 10.1 | 9.1 |
| 6 | 18 | 16.8 | 14.0 | 13.0 |
| 8 | 24 | 22.4 | 17.5 | 16.5 |
| 9 | 27 | 25.2 | 19.1 | 18.1 |
| 10 | 30 | 28.0 | 20.7 | 19.7 |
| 11 | 33 | 30.8 | 22.1 | 21.1 |

A sequence key is shorter than the 49-bit bitboard only up to ply 16, and shorter than a global perfect index only up to ply 14. It is always longer than a per-ply index from ply 3 on. **Shorter keys are not the issue anyway: a sequence table needs many more keys**, as the next section shows.

---

## 4. Measured: distinct move sequences vs distinct positions

Definitions. The sequence count at ply n is the number of legal move sequences of length n with no earlier game end (the last move may end the game). The position count at ply n is the number of distinct `state_key` values reached after n moves, including positions where the game has just ended, which is the A212693 convention. Sequences are counted by dynamic programming over positions: each position carries the number of paths that reach it. "Live" excludes ended games.

### Connect-4, plies 0–11 (*measured*, 174 s, peak RSS ~0.94 GB)

| ply | sequences | live sequences | positions | live positions | mirror classes | **seq / pos** | = A212693? |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 1 | 1 | 1.00 | yes |
| 1 | 7 | 7 | 7 | 7 | 4 | 1.00 | yes |
| 2 | 49 | 49 | 49 | 49 | 25 | 1.00 | yes |
| 3 | 343 | 343 | 238 | 238 | 121 | 1.44 | yes |
| 4 | 2,401 | 2,401 | 1,120 | 1,120 | 568 | 2.14 | yes |
| 5 | 16,807 | 16,807 | 4,263 | 4,263 | 2,144 | 3.94 | yes |
| 6 | 117,649 | 117,649 | 16,422 | 16,422 | 8,231 | 7.16 | yes |
| 7 | 823,536 | 810,504 | 54,859 | 54,131 | 27,473 | 15.0 | yes |
| 8 | 5,673,234 | 5,628,804 | 184,275 | 182,383 | 92,244 | 30.8 | yes |
| 9 | 39,394,572 | 38,307,690 | 558,186 | 538,774 | 279,241 | 70.6 | yes |
| 10 | 268,031,646 | 263,770,588 | 1,662,623 | 1,618,398 | 831,581 | 161 | yes |
| 11 | 1,844,590,828 | 1,777,308,076 | 4,568,683 | 4,295,422 | 2,284,733 | **404** | yes |

The position counts match Tromp's A212693 exactly at every ply from 0 to 11, which confirms the repo's game. Summed over plies 0–11 there are 2.16×10^9 sequences and 7.05×10^6 positions, a ratio of **306x**. At ply 7 there are 823,536 sequences rather than 7^7 = 823,543 because the 7 single-column fills have no 7th move in that column.

### Kalah(4,3), moves 0–17 (*measured*; stopped at the 1.5 GB guard)

| moves | sequences | positions | seq / pos | cumulative distinct positions |
|---|---|---|---|---|
| 4 | 158 | 158 | 1.00 | 228 |
| 6 | 1,507 | 1,485 | 1.01 | 2,199 |
| 8 | 14,011 | 13,275 | 1.06 | 19,847 |
| 10 | 118,427 | 101,422 | 1.17 | 153,625 |
| 12 | 902,040 | 600,196 | 1.50 | 889,390 |
| 13 | 2,402,079 | 1,201,835 | 2.00 | 1,723,975 |
| 14 | 6,246,770 | 1,943,773 | 3.21 | 2,697,414 |
| 15 | 15,873,265 | 2,543,201 | 6.24 | 3,472,842 |
| 16 | 39,386,256 | 2,886,160 | 13.6 | 3,961,170 |
| 17 | 95,122,460 | 3,033,975 | **31.4** | 4,249,212 |

The cumulative distinct count reaches 4,249,212 by move 17, approaching the paper's complete game graph of **4,604,996**. In Kalah the same position can also be reached after different *numbers* of moves, because extra turns shift the count. A position-keyed table merges those cases as well: across all depths, the sum of per-depth position counts is far larger than the cumulative distinct count. The paper's Table 1 estimate of game-tree size for Kalah(4,3), w^d ≈ 6.1×10^7, is below the exact 9.5×10^7 sequences of length 17 alone.

### Kalah(6,4), moves 0–9 (*measured*; stopped at the 2.5M-frontier cap, 15 s)

| moves | sequences | positions | seq / pos |
|---|---|---|---|
| 3 | 185 | 185 | 1.00 |
| 5 | 4,685 | 4,673 | 1.00 |
| 7 | 113,959 | 112,597 | 1.01 |
| 8 | 559,885 | 544,445 | 1.03 |
| 9 | 2,743,126 | 2,603,419 | **1.05** |

Kalah transposes far less than Connect-4. Sowing changes many holes and stores, so different orders rarely converge, and Connect-4's commuting column drops have no equivalent. The ratio still turns upward once captures and stores dominate, as Kalah(4,3) shows from move 12 on.

---

## 5. Worst-case storage to cover a whole game (value only, 2 bits)

| Game | Positions | Perfect index, value array (no keys) | Explicit position keys | Sequence-keyed | Best known in practice |
|---|---|---|---|---|---|
| Connect-4 | 4.53×10^12 | **1.13 TB** (0.90 TB at log2 3 bits) | 49-bit key + 2 bits: 28.9 TB | about 10^21 sequences × ~110 bits ≈ **10^22 bytes** | **89.6 GB** BDD, i.e. 0.158 bits a position (Böck). The weak solution needs only about 12 KB (WeakC4). |
| Kalah(6,4) | 1.31×10^13 state space; ~1.3×10^12 reachable (paper's estimate) | 3.3 TB by combinatorial ranking over the state space; 325 GB if a perfect hash over reachable positions existed | 44-bit rank + 2 bits: ~75 TB | about 6×10^18 sequences (Table 1) × ~93 bits ≈ 7×10^19 bytes | Weakly solved in 35 s with ≤20-counter endgame databases (107 MB at 4 bits an entry). |
| Checkers | 5×10^20 | 1.25×10^20 bytes (125 EB) | — | 10^40 game tree | Weakly solved: 237 GB of ≤10-piece databases (0.05 bits a position) plus a ~10^7-node proof tree. Schaeffer: a strong database would need about 10^18 bytes even compressed. |
| Chess | ~4.8×10^44 | 1.2×10^44 bytes (148–153-bit index) | — | 10^123 | Only up to 7 pieces: 4.2×10^14 positions in 16.7 TiB (0.35 bits a position WDL+DTZ, 0.18 for WDL alone). 8-piece: about 2 PB. |
| Go 19×19 | ~2.08×10^170 | ~5×10^169 bytes (566-bit index) | — | 10^505 | None. |

Plainly: covering the whole game by brute force is feasible for Connect-4 (it has been done) and for Kalah(6,4) at TB scale (never needed, because the game was weakly solved instead). **It is not feasible for checkers, chess or Go**: checkers is about 10^8 times beyond Connect-4's table, chess about 10^32 times and Go about 10^158 times. Checkers was solved weakly, by a proof tree plus partial databases, and that is the general pattern for any game with more than about 10^14 positions. Sequence keying makes every row worse, by the game-tree/state-space ratio: 10^8–10^9 for Connect-4 and 10^20 or more for checkers.

---

## 6. Usefulness for the hybrid

### 6.1 Description length of a perfect first-player strategy table (*measured*)

The tree is a perfect Connect-4 strategy for player 0 through ply 9. Player 0 plays the first value-optimal move in "tactical-centre" order (win now, then block, then centre-first), and the opponent's replies are all enumerated. Exact values come from the repo's `books/c4_labels.json.gz`; 9 positions not in that book were solved with `harness/native_solver`. This is a different perfect strategy from the H3 net's tree (1,758 positions), but the ratios carry over.

| ply | P0 decision positions | P0 decision sequences |
|---|---|---|
| 0 | 1 | 1 |
| 2 | 7 | 7 |
| 4 | 47 | 49 |
| 6 | 260 | 343 |
| 8 | 1,048 | 1,838 |
| **total** | **1,363** | **2,238 (1.64x)** |

Per-ply growth is 7 → 6.7 → 5.5 → 4.0x for each player-0 ply. A strategy tree transposes much less than the full game (1.64x against 31x at ply 8), because player 0's moves are fixed.

| Table encoding for these 1,363 decisions | bits | net (659,712 bits) ÷ this |
|---|---|---|
| Rank-coded under the tactical-centre predictor (entropy 1.086 bits an entry, plus ~80 bits for the code table) | **~1,560** | 423x |
| Canonical walk, arithmetic-coded over legal moves (mean log2 legal = 2.795) | 3,810 | 173x |
| **Canonical walk, fixed 3 bits an entry** | **4,089** | 161x |
| Sequence trie (implicit keys, one 3-bit edge per sequence) | 6,714 | 98x |
| Explicit per-ply perfect index (ceil log2 A212693(p)) + 3 | 27,412 | 24x |
| Current `description_length` pricing for a full table (ceil(log2 1363) + 3 = 14 bits) | 19,082 | 35x |
| Explicit sequence keys (3·ply) + 3 | 57,630 | 11x |
| Explicit 49-bit bitboard key + 3 | 70,876 | 9x |
| Net, 20,616 params × 32 bits (× 8 bits = 164,928) | 659,712 | 1x |

Rank histogram under tactical-centre: 0 → 1,095, 1 → 143, 2 → 56, 3 → 39, 4 → 14, 5 → 8, 6 → 8. Under plain centre-first the entropy is 1.125 bits.

Kalah(4,3), the **whole game** (*measured*, exact via `Kalah.optimal_actions`): the perfect player-0 strategy tree has **124 decision positions and no transpositions** (sequences = positions at every depth). That is 248 bits at 2 bits an entry, 162 bits arithmetic-coded (mean log2 legal = 1.31), and **~119 bits rank-coded** (0.956 bits an entry; rank 0 → 94, 1 → 25, 2 → 5, under extra-move > capture > other). The full database for the same game is 4,604,996 × 2 bits = 1.15 MB.

**Answers.**
- *Is a sequence-keyed table ever cheaper than the canonical walk?* No. With implicit keys the best case is a tie, which happens only when the tree has no transpositions; Kalah(4,3) is such a case. Otherwise the cost scales with the number of sequences (≥ positions). With explicit keys it is 14x worse here.
- *When does the net start paying for itself?* Not until the 3-bit walk table outgrows 659,712 bits, which means about 220K entries. Growth of 4–6x for each player-0 ply puts that around **ply 14–16** (extrapolated; wins and forced lines may slow the growth). A rank-coded table or a net quantised to 8 bits moves the crossover by about one player-0 ply in either direction.
- *Exceptions.* For the sparse "net + exceptions" case, a bitmap over the walk (P + 3E bits) or an entropy-coded bitmap (P·H(E/P) + 3E) beats the 14-bit index once more than about 9% of positions are exceptions. For H3 (232 exceptions over 1,758 positions) that is 3,248 bits indexed, 2,454 bits as a bitmap and ~1,690 bits entropy-coded. The saving is irrelevant next to the 660K-bit net.

### 6.2 Compact encodings as net input

- The net's description length is its **parameters**, not its input bits, so a 49-bit or 43-bit input saves nothing.
- **Move sequence as input:** the rules are Markov, so history adds no information. The net would have to learn that up to about 400 sequences (ply 11) are the same board, which is the invariance the board planes give for free. Othello-GPT shows a sequence model can learn the board, which proves the work is learnable but not that it is cheap.
- **Perfect index or hash as input:** nearby indices are unrelated boards, so there is nothing to generalise over. These codes are useful only as storage keys.
- Where compact codes help: table keys, canonical walk orders (implicit keys), mirror reduction (already used: `canonical_input`, `canonical_key`), and **move-rank coding of table entries**, where a cheap predictor removes most of the entropy.

### 6.3 What a sequence "rainbow table" of partial knowledge would buy

Compared with a position-keyed transposition table, opening book or tablebase, it buys nothing, and it costs a transposition factor (×1.4 at ply 3 to ×404 at ply 11 in Connect-4) in entries, lookups and exact solves. Every transposed copy is a separate entry that needs its own solve or label, or a value copied from another sequence, which requires recognising the position anyway. The one real property of a sequence key, that it is computable without game state, does not help: Zobrist or Pons keys update in O(1) per move. The good parts of the idea already exist under other names: growing coverage with exploration (transposition tables, persistent books, partial-information databases) and recompute-instead-of-store (checkers' proof tree over databases, and an opening book plus search).

---

## 7. Experiments worth running

**E1 (Connect-4): price tables by rank coding and find the real net/table crossover.**
Extend the perfect strategy tree to plies 10, 12 and 14. Plies 0–8 are already covered by `books/c4_labels.json.gz`; deeper plies need `native_solver` (ply-10+ solves take seconds, ply-0–2 solves take minutes or more). At each depth, measure decision positions, sequences, and the rank entropy under (i) tactical-centre (0 params) and (ii) the trained net's own move order. Score these hybrids:
- full walk table at 3 bits;
- rank-coded table;
- net + rank-coded exceptions, where the net is the predictor and "exceptions" are entries with rank > 0.

Success criterion: the first depth at which a certified net-based hybrid beats the rank-coded table alone. Today's extrapolation says ply 14–16 at 32 bits a weight. If the table still wins at ply 14, the net is not compressing anything at that scale.

**E2 (Kalah): strategy-tree size against the game graph, across shapes.**
Kalah(4,3) gives 124 decisions against 4.6M positions, about 4×10^4 times fewer. Repeat for Kalah(4,4–6), (5,n) and (6,1–3) with the exact solver. Abort Kalah(6,4) if the memoised solver goes above about 1.5 GB. Record decisions, sequences and rank entropy. This checks whether the hybrid's strategy-tree table stays tiny when there are no transpositions, and gives a second game for the description-length ranking. It also answers the sequence-key question for Kalah directly: if sequences equal positions in the tree, the key is free but the table is no smaller.

Not worth running: building any sequence-keyed table, or trying sequence or index codes as network inputs.

---

## Reproduction

The scripts live in the session scratchpad, not the repo. Run them from `examples/boardgames` with `PYTHONPATH=. .venv/bin/python <script>`. All are single-process with RSS guards at about 1.5–1.8 GB.

| Script | What it does |
|---|---|
| `count_seq_vs_pos.py c4 out.json 11` | Connect-4 sequences and positions by ply (174 s to ply 11). |
| `count_seq_vs_pos.py kalah43 out.json` and `kalah64` | Kalah sequences, positions and cumulative distinct positions. |
| `c4_perfect_tree.py solve out.json` | Perfect player-0 Connect-4 tree through ply 9 from `books/c4_labels.json.gz`, plus `native_solver` for 9 positions (31 s). |
| `perfect_trees.py kalah out.json 4 3` | Perfect player-0 Kalah(4,3) tree to game end (5 s). |

Scratchpad: `/private/tmp/claude-501/-Users-cloud-Documents-Work-thefactory-tools/da5a4b0e-014f-4c03-b74e-234e8b96709f/scratchpad/`. Raw outputs: `c4_final.json`, `k43.json`, `k64.json`, `c4pt.json`, `pt_k43.json`.
