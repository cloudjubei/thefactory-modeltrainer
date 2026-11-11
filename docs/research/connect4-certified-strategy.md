# A certified first-player strategy for Connect-4 in 106,044 bits

Write-up, 2026-10-29, updated 2026-11-10 (T8). Question: how small can a complete, certified description of a winning
first-player strategy for Connect-4 (7×6) be, if a generic process finds it? Each claim cited as hNNN is a registered claim in
`examples/boardgames/hypotheses.json`, with its proof test and evidence file. A claim marked *after the data* was written
once the run had finished; every other headline claim was registered before its run started, except h373, which was
registered while X4 ran: X4's rows were unfinished and U2's search had never been run, but the register counts it as after
the data.

**How to read this result.** The strategy was found with an exact Connect-4 solver in the search: the opening table,
the builder's winning moves and the local search's score all read it. Only the rule language and the checks that the
strategy wins are solver-free. So this is what the representation can hold when the winning moves are known, not what a
process without a solver can find. The project's aim is the second, with a solver used only to grade (§7).

---

## 1. Result

1. **A certified winning strategy for the first player, from the empty board, in one file of 106,044 bits (13,256 bytes)**
   (T8; h395, h396, h397). The file reads back from its bytes alone to exactly the strategy written, and that strategy,
   played by the run against every reply, wins all 3,372 leaves' lines. The decoder is given the game's rules, the map
   rule's settings and the board's mirror; none of those, nor the decoder itself, is in the 106,044. By part:

   | Part | Bits | Share | In T1's file | In the count |
   |---|---|---|---|---|
   | 1,008 maps: which cells hold a level (27,995) and the levels of their 18,444 cells (43,495) | 71,491 | 67.4% | 89,289 | 99,613 |
   | 1,625 exceptions on 129 leaves: their flags (13,728), moves (2,478) and which leaves have any (692) | 16,898 | 15.9% | 21,051 | 21,815 |
   | The tree: 4,052 positions, 680 moves, 3,372 leaves, their references (13,331) and readings | 17,655 | 16.6% | 19,663 | 41,324 |
   | The coder's ending and rounding | 0 | | 2 | |
   | **Total** | **106,044** | | **130,005** | **162,752** |

   The file is the whole game below, which every earlier size in this write-up counted rather than wrote. Six steps
   took the count to this file. None built anything new, each was registered before its file existed, and each file
   reads back to the game of the step before:

   | Step | What changed | File (bits) | Of the step before | Where the saving was (after the data) |
   |---|---|---|---|---|
   | T1 | The game written as one file, as it was built | 130,005 | 0.7988 (of the count) | The tree: 21,661 of the 32,747 bits saved (h382) |
   | T2 | Each map given the lowest levels that make the same moves; maps that agree merged | 118,197 | 0.9092 | The levels: 11,519 of the 11,808 bits saved (h385) |
   | T3 | Each level coded under what the decoder has read of its map | 115,599 | 0.9780 | The levels only, 5.6% fewer (h388) |
   | T4 | Each cell's flag also coded under what the map said of the two cells below | 112,644 | 0.9744 | The cell flags only, 9.5% fewer (h391) |
   | T5 | Each exception flag coded under the exceptions its leaf's walk has read | 109,034 | 0.9680 | The exception flags only, 20.2% fewer (h394) |
   | T8 | The position read too: references ranked by board likeness, exception flags by depth | 106,044 | 0.9726 | The references, 15.5% fewer, and the exception flags, 3.8% (h397) |

   The file is 0.652 of the count and 0.816 of T1's file. Four more coding ideas (T6, T7, T9, T10) were studied in
   scratch, found under 1% of the file each and were not built (§5). What the count pays and leaves out is in §2.

   The game is G3's whole game (280,640 bits; h324, h326) with 50 of its 148 ply-6 subtrees replaced
   by strategies built from ply 6 (E1, O1, E3, V1, V2; §6), 76 of the 416 ply-8 strategies rebuilt from every
   map that game holds (G4; §6), 20 of the 50 ply-6 strategies rebuilt the same way (Z1; §6), 9 ply-6 and 48
   ply-8 strategies rebuilt with every map also readable on the mirrored board (X2, X3; §6), 9 more ply-6 strategies rebuilt
   from every map the game then held, read both ways (Z2; §6), and 3 ply-8 strategies from the
   control run with that reading off (X4; §6). U2's search chose among all of those pieces at once. It plays every reply the opponent can make, and every piece passed
   an independent walk confirming that it wins every line. The assembled game, by how it was built, as counted:

   | Part | Bits | Share |
   |---|---|---|
   | Opening: 56 table moves to ply 6, 98 more at ply 6, and 233 positions the empty map already wins | 849 | 0.5% |
   | Library: the 801 shared maps that some piece plays, each paid once | 70,435 | 43.3% |
   | Strategies at the 416 ply-8 positions still used (76 of them G4's rebuilds, 48 X3's, 3 X4's) | 49,692 | 30.5% |
   | Strategies built from the 50 ply-6 positions (20 of them Z1's rebuilds, 9 X2's, 9 Z2's) | 41,776 | 25.7% |
   | **Total** | **162,752** | |

2. **Where the bits went in G3, the last whole game built in one run:**

   | Part | Bits | Share |
   |---|---|---|
   | Opening: 204 table moves from the empty board to ply 8, and 259 positions the empty map already wins | 1,075 | 0.4% |
   | Library: the 1,207 shared maps that some position plays, each paid once | 102,815 | 36.6% |
   | The 671 frontier positions' own strategies | 176,750 | 63.0% |
   | **Total** | **280,640** | |

   Inside the 176,750 bits: 709 maps of each position's own take 63,799, 3,359 exceptions take 43,902, and the tree
   (1,003 table moves, 6,330 leaves, the leaves' references) takes the remaining 69,049. Maps are 59% of the whole.
3. **Against the reference points:**

   | Description | Bits | vs this result |
   |---|---|---|
   | The repo's policy net (not certified) | ~660,000 | 6.2x larger |
   | WeakC4, as shipped ([2swap](https://2swap.github.io/WeakC4/explanation/)) | ≈ 90,000 | 1.2x smaller |
   | WeakC4, its diagrams entropy-coded | ≈ 35,000–45,000 | 2.4–3.0x smaller |

   WeakC4 uses a Connect-4-specific language (see §6, Y1). Every rule here is generic: nothing in it knows about
   gravity, columns or threats beyond "a move that wins now" and "a move that lets the opponent win now". The mirror a
   leaf may read (§2) is a symmetry found by checking sampled play, not assumed.
4. **The heavy tail sets the size.** The median frontier position takes 18 bits. The heaviest 10% (67 positions) hold
   108,834 of the 176,750 strategy bits, and the largest single position takes 5,691.

The evidence for G3 is `examples/boardgames/evidence/c49_g3.json.gz`, and `harness/floor_g3.py` (`g3_size`)
recomputes its total from it. The assembled game is recomputed by `tests/test_c49_z2_evidence.py`
(`harness/floor_z2.py`: U2's local search over every certified piece, from U2's game and from every position's smallest
piece) from G3 and the E1, O1, E3, V1, V2, G4, Z1, X2, X3, X4 and Z2 evidence files. Each file's evidence,
script and proof test are listed under Reproduction.

---

## 2. What the description is

**Opening.** From the empty board to ply 8, a table of 204 first-player moves (4 bits each), each a winning move by the
label cache. A further 259 positions above ply 8 need no move, because the empty map already wins them (1 bit each).
The opening leaves 671 first-player positions at ply 8 that the empty map does not win: the **frontier**.

**Each frontier position** gets a strategy tree:
- A **move node** is a table move (3 bits): the first player's move, followed by every reply.
- A **leaf** hands the rest of the game to a **priority map**: up to 8 levels (3 bits each) assigned to some of the
  empty cells.

**How a map plays a position (the rule, L5):**
1. Win now if possible.
2. Otherwise, a move is *safe* if it doesn't let the opponent win on the next move. If only one move is safe, play it.
3. Otherwise, the first level holding any safe move decides, and plays its lowest such move. Safe moves on cells the
   map leaves without a level count as a last level.

**Reading a map mirrored.** A leaf built by X2 may play a library map on the board's mirror image, with the move
mirrored back. The rule breaks ties by the lowest move, so this is not the same as flipping the map's cells (h364). The
reference says which reading a leaf plays, and a map read both ways is paid once.

**Exceptions.** A leaf may also carry exceptions: positions where the map's move does not win, each with the move to
play instead. They are coded in the order a decoder replays the leaf, with flags only where the rules leave a choice
(h183).

**Coding the rest:**
- **Map bits:** a map costs its levelled cells among the cells empty where it was found.
- **References:** a leaf names its map by a move-to-front rank in walk order, the library maps it uses being named once
  (first_use coding; h312).
- **Library:** maps shared across positions form a library, and each library map some leaf plays is paid once for the
  whole game. In X2's rebuilds the library's reference list holds each map twice, once per reading.

**The count and the file.** Every size in §3 and §4 is a count: it adds up what the parts would take, and no file of
that length was ever written. A map's count pays one bit for each cell empty where the map was found, a position the
assembled game never states, and the count pays nothing for telling a piece from a position the empty map wins or for
saying which list a piece's references point into. T1 writes the game instead (`harness/strategy_file.py`,
`harness/arithmetic_coding.py`): one walk from the empty board, in the same order for writing and reading, with adaptive
arithmetic coding that uses only what a decoder knows at that point. A position reached by two pieces is written once,
and each map is trimmed to the cells empty at some leaf that plays it, since a leaf reads no other cell. Five more
steps change only how that game is written:
- **T2, lowest levels** (`harness/map_orders.py`). A map matters only by the order it puts the safe moves' cells in
  where the rule has a choice. Each map gets the lowest levels that keep that order at every position its leaves reach,
  and maps that agree are merged. The searches that found the maps keep any levels that work, spread over all 8; the
  lowest levels crowd the bottom and take fewer bits.
- **T3, levels in context** (`harness/level_context.py`). Each level is coded under what the decoder has already read
  of its map: the highest level so far, the cell's row and whether the cell below holds a level. The row and the cell
  below come from the board's shape, which the mirror already uses; nothing in it is about how Connect-4 is played.
- **T4, cell flags in context** (`harness/cell_context.py`). Each cell's flag is coded under the cell, whether it is
  empty where the map is first read, and for the two cells below it whether each is empty and holds a level.
- **T5, exception flags in context** (`harness/exception_context.py`). Each exception flag is coded under how many
  exceptions the leaf's walk has read so far, up to two, and how many flags since the last one, on a log scale. It reads
  nothing of the board.
- **T8, the position read** (`harness/position_walk.py`). This is strategy_file's walk copied once, with two things
  read from the board. A map read again is named by its place among the known maps, ranked by how many cells of the
  leaf's board agree with the best of the boards each was read at, either way round, ties in move-to-front order. Each
  exception flag is also coded under the plies below its leaf, on a log scale.

**Certification.** Every leaf's map is verified over every continuation by a solver-free walk, then certified a
second time independently. `check` re-walks the whole strategy against every reply with no oracle, and the oracle
confirms that every table move in the tree wins. All 671 positions passed all four checks. A mirrored leaf is checked
and certified on the real board, with real replies, so neither rests on the symmetry.

---

## 3. How it was found

A **builder** works depth-first from each frontier position:
1. **Make a leaf where a map wins.** The empty map is tried first, then the maps already held (the build's own and the
   library's), then a fresh **local search** for a map.
2. **Keep a map with exceptions if it is small enough.** A map plus exceptions is kept when it is at least 30x smaller
   than the leaf's positions written as a table.
3. **Otherwise make a move node.** It plays the winning move whose replies are most often already covered, and the
   build carries on below it.

Each build has a 2-hour cap.

After a build is complete, two **improvement** passes run while time remains:
- **Improve:** split the costliest leaves with exceptions, keeping each split only if the whole strategy shrinks.
- **Lookahead:** judge a split only after its own leaves have been improved.

**The local search** scores a candidate map by the size of the whole leaf it makes: map bits plus the exceptions it
needs on every line. Before that change, the search scored a map by its first failing line, and found nothing at
ply 10.

**Every whole game, in order.** F1 to G3 are whole-game runs. J1 to Z2 assemble the game from certified pieces. T1 to
T8 write Z2's game as a file:

| Run | What changed | Whole game (bits) | Ratio to the previous run | Claims |
|---|---|---|---|---|
| F1 | Rule L2: the first level holding any safe move decides | 548,424 | — | h271, h273 |
| F2 | Rule L5: unlevelled safe moves count as a last level | 455,959 | 0.83 | h288, h289 |
| F2, references by first_use | Recoded only; no move changed | 398,195 | 0.87 | h312 |
| G2 | Every position rebuilt from every map F2 found, all within reach | 293,843 | 0.74 | h320, h321 |
| G3 | Once more, from every map G2's whole game holds | 280,640 | 0.955 | h324, h325 |
| J1 | G3 with 30 ply-6 subtrees rebuilt from ply 6 (E1, O1, E3), assembled | 203,774 | 0.73 | h349 |
| V1 | Every winning move built at the 22 heaviest ply-6 positions with a choice, re-assembled | 197,650 | 0.970 | h351 |
| V2 | Every unbuilt winning move in the 9 heaviest transposition groups, free builds included, by group | 192,591 | 0.974 | h355, h356 |
| G4 | The 409 ply-8 positions still taken from G3, rebuilt from every map the assembled game holds; 76 kept | 184,556 | 0.958 | h357–h359 |
| U1 | The same rebuilds chosen by local search, one rebuild or every rebuild playing one map at a time; 66 kept | 184,328 | 0.9988 | h360 |
| Z1 | The 51 ply-6 builds the game takes, rebuilt from every map it holds, chosen by the same search; 27 taken | 180,199 | 0.978 | h361–h363 |
| X2 | The same 51 rebuilt from the same maps, every map also readable on the mirrored board; 7 taken | 175,425 | 0.974 | h364–h367 |
| X3 | The 409 ply-8 strategies rebuilt from the same maps read both ways; 71 taken | 166,623 | 0.950 | h368–h370 |
| U2 | No new builds: every certified piece an option at its position, X4's control rebuilds included; 49 positions change | 165,023 | 0.990 | h373, h375 |
| Z2 | The 50 ply-6 builds of U2's game, rebuilt with both readings from every map it holds, chosen by U2's search; 9 taken | 162,752 | 0.986 | h376–h378 |
| T1 | No new builds: Z2's game written as one file, read back from its bytes and played against every reply | 130,005 (file) | 0.799 | h379, h380, h382 |
| T2 | No new builds: T1's file with each map given the lowest levels that keep its moves, maps that agree merged | 118,197 (file) | 0.909 | h383–h385 |
| T3 | No new builds: T2's game written again, each level coded under what its map has said so far | 115,599 (file) | 0.978 | h386–h388 |
| T4 | No new builds: T3's game written again, each cell flag also coded under the two cells below it | 112,644 (file) | 0.974 | h389–h391 |
| T5 | No new builds: T4's game written again, each exception flag coded under the exceptions its walk has read | 109,034 (file) | 0.968 | h392–h394 |
| T8 | No new builds: T5's game written again, references ranked by board likeness, exception flags by depth too | 106,044 (file) | 0.973 | h395–h397 |

The whole-game passes stop at G3 by a rule set before it ran: pass again only while a pass gains a tenth. G4 and Z1 are
library passes over the assembled game instead, each judged against its own 97/100 bar: G4 cleared it, Z1 fell short.
X2 changes the language rather than the library and fell short of the same bar. X3 carries X2's change to the ply-8
strategies and cleared it, though the rebuilds with no mirrored leaf clear it without the others (h370). X4, the control
for that pass, puts the second reading's own worth at ply 8 at 2,577 bits (h372, h374). U2 builds nothing and cleared its
199/200 bar.

The total compute was about 216 CPU-hours for F2, 219 for G2 and 151 for G3, on one 10-core laptop.

---

## 4. What made it smaller, in order of effect

| Change | Effect | Claims |
|---|---|---|
| Score local search by the whole leaf's size, so a map may carry exceptions | Leaves at ply 10 for the first time; a median 39x under the leaf's table | h172–h174 |
| Anytime improvement: complete first, then split the costliest leaves while time remains | 23% of the plain build's bits on the heavy positions | h236, h237 |
| Lookahead in improvement | A further 21% cut | h244, h245 |
| Rule L2: the first level holding any safe move decides, with its lowest move | A fifth of the exact rule's bits; 90 maps instead of 586 | h258, h259 |
| Rule L5: unlevelled safe moves count as a last level | 0.84 on 32 heavy positions; 0.83 on the whole game | h286, h289 |
| References coded by first_use | 100,888 → 43,124 bits for F2's references (12.7% of F2) | h312 |
| Library passes: rebuild every position from every map the last pass found | 0.74, then 0.955; 0.958 and 0.978 on the assembled game | h321, h325, h358, h359, h362, h363 |
| The move at ply 6 chosen by what lies below: the builder's pick, then every winning move built | 0.69 of G3, assembled | h340, h344, h347, h349, h351, h356 |
| Library maps also read on the mirrored board | 0.974 on the ply-6 builds; 0.950 on the ply-8 strategies, 0.984 from the rebuilds playing a mirrored leaf | h364, h366, h367, h369, h370 |

---

## 5. Dead ends

None of these changes shrank the strategy. Each line is a registered claim, so they can't quietly be tried again as
if new.

| Tried | Outcome | Claims |
|---|---|---|
| Longer SAT searches (6x budget), or 14 levels instead of 8 | No leaves at ply 10: slower, not stronger | h162, h163 |
| Local search by the old score, then patching with ≤ 100 exceptions | No leaf | h166, h168, h170 |
| Warm-starting SAT from local search's best map | Nothing in 30 minutes; the same as SAT started cold | h175–h177 |
| A fixed acceptance threshold raised from 30x to 100x | Cuts both ways: one position 32,544 → 6,207 bits, another 3,877 → 18,093 | h230–h232 |
| Choosing leaf or split by size during the build | Better where it finished; doesn't fit 2 hours | h233–h235 |
| Re-searching the costliest leaves at the end | 0.56% | h247 |
| Coding each map as edits of the previous one | 18% larger | h248 |
| Maps of 4 levels at 2 bits instead of 8 at 3 | 3.7x larger; 23 of 32 complete | h255–h257 |
| Safety two plies deep (no forced win in two) | 4.6% larger | h263–h265 |
| Preferring forcing moves | Starves the search: 4 of 32 complete | h279–h281 |
| Cheaper exact codes for the maps | At most 3.2% | h283, h284 |
| Longer leaf searches (120 s instead of 30 s) | 1.30x larger: more time admits leaves with thousands of exceptions | h291–h293 |
| A stricter acceptance bar (100x) | 1.34x larger | h294–h296 |
| A lower acceptance bar (10x) | 0.92, the gain from two positions | h308, h309, h318 |
| Ties broken by tactics (win in two, then not losing in two) | 0.94 on 32 positions; 0.93 on the whole game | h297–h301, h310, h311, h323 |
| Maps that switch to second levels part-way | Barely used; 1.03x | h302–h307 |
| Exception flags split by context, or moves coded tactics first | 11% larger; 1.9% | h315, h316 |
| Ties broken by a 19,996-bit learned net | Larger than tactics: 68,693 against 55,809 bits on 28 positions | h317, h327–h329 |

**Studied in scratch and not built.** These were never registered or proven. Each figure is a simulation of the coder
on G3's file, or for P1 a stand-in for the file's cost.

| Studied | Outcome |
|---|---|
| P1: pieces picked by their file cost instead of the count | U2's search moved 18 of 819 positions; the gain was inside the stand-in's own error |
| T6: references coded in context from what a wrapper of the coder sees | 2.5% of the references at best, about 0.4% of the file |
| T7: levels under richer contexts, as binary choices, or with another seed and step | No context beat T3's; 0.6% of the levels at best |
| T9: each new map coded under what the most alike known map says for the cell | Cell flags 0.4% fewer; levels worse |
| T10: the kind flags, moves and exception moves under more context | 968 bits together, 0.5% of the file |

**Two lessons about method:**
- **A 32-position sample can't project the whole game.** Its bootstrap 95% high fell below the true mean 11% of the
  time; the frontier's tail is too heavy (h272, h276).
- **A default that keeps the win more often doesn't make a smaller strategy.** Tactics misses 200 positions in 4,523
  and the net 62, against 308 for "lowest". Neither shrank the heavy tail (h297, h301, h317, h329).

---

## 6. Each step after G3, as measured

One entry per step, oldest first. What is still open is in §7.

- **R2, measured (closed):** the builder priced each map reference at the index's full width (10 bits a leaf with
  G1's 767-map lists), while the judges code references by first_use, where a repeated map costs 1–3 bits. Rebuilding
  G1's 16 heaviest positions with the builder counting the judges' way finished all 16 (h330) but took 82,251 bits, a
  tenth more than G1's 74,556 (h331 refuted). Cheaper references made splitting pay: 2,173 leaves against 1,513 and
  449 moves against 283, which cost more than the exceptions saved (14,639 bits against 19,101; h341). The full-width
  price was holding the tree back, so the builder keeps it.
- **Where WeakC4's edge comes from (Y1, measured):** its current language is ours in all but name. Our "exact" rule
  reads 1,003 of its 1,009 leaf diagrams as they are (h332). Priced in our code, its solution takes 123,881 bits, 0.44
  of this result (h333, h334). Its tree is the difference: 251 table moves and 1,009 leaves from ply 4, against our
  1,207 table moves, 6,330 leaves and 3,359 exceptions. The rest is its code: one diagram per mirror pair, entropy-coded.
  The levers that follow are generic: an opening chosen together with its maps so leaves come earlier, and maps shared
  across mirror images.
- **Mirror sharing after the fact (X1, measured):** no stored map mirrors another (h335 refuted), and only 28 of 671
  frontier positions pair with their mirror image, saving 4.4% (h336 inconclusive). Mirroring can pay only during the
  build.
- **The opening's move choice (E1, measured):** G3's 8 heaviest ply-6 subtrees, rebuilt from ply 6 with the builder
  free to pick the move there, all finish (h339) in 34,976 bits against G3's 47,038 (h340). No leaf moved earlier. The
  saving is the move: on three positions the builder played another winning move than the opening's lowest column,
  and those subtrees fell from 18,156 bits to 2,840; the five that kept the opening's move came out about even (h342).
  The opening fixes its moves before any map is found, so choosing them by what lies below is the open lever.
- **The builder's move across the opening (O1, measured):** the builder's own move rule (the move whose replies the
  most known maps cover), run with G3's library at every ply-6 position, picks another move than the opening on 34 of
  148. The 31 beyond E1's all finish (h343) in 20,704 bits against G3's 51,312, 0.40 (h344); 28 shrink, 3 grow, and 5
  become a single leaf at ply 6 (h345). With E1, about 42,700 of the 280,640 bits, not yet assembled into a whole game.
- **Every winning move against the free pick (E3, measured):** at E1's 7 heavy positions with more than one winning
  move, the 13 other winning moves were each built from ply 6. All finish (h346), and keeping the cheapest move at
  each position takes 17,484 bits against the pick's 28,712, 0.61 (h347). The pick was cheapest only where it had
  differed from the opening's move (#115, #130, #145). At the other four another move saved 1,839-3,696 bits each:
  two picks were ties on covered replies, and at two the move covering the most replies lost (h348). The moves at
  one position differ up to 13-fold, so counting covered replies is a weak guide; building every move is the reliable
  rule, at the cost of one build per winning move.
- **The whole game assembled (J1, after the data):** G3 with the ply-6 builds swapped in wherever the whole game
  shrinks, each ply-8 position and library map paid once, takes 203,774 bits against G3's 280,640 (h349). The saving,
  76,866 bits, is more than the per-run comparisons added up to (about 54,000): a swapped-in subtree also frees ply-8
  positions shared with other subtrees and the library maps only they played (1,207 maps fall to 891). 30 of the 39
  positions with a build take one.
- **Every winning move, first batch (V1, measured):** at the 22 ply-6 positions with more than one winning move under
  which G3 pays at least 1,500 bits, the 63 winning moves not yet built were each built from ply 6. All finish (h350),
  and re-assembled the whole game takes 197,650 bits, 0.970 of J1's (h351, supported at its 97/100 bar with 10 bits to
  spare). Only 8 of the 63 are taken (h352, after the data). At 4 of the 5 positions taking none, every ply-8 position
  below is also reached from another ply-6 position that keeps G3's piece, so G3's strategies there stay and a build
  saves at most the 4-bit move: #13's move 3 takes 66 bits against G3's 1,536 and is not taken. Transpositions join
  the 148 ply-6 positions into groups of up to 10 (197 of 820 ply-8 positions are shared), and a chooser that swaps
  at most two positions at a time cannot use such builds.
- **The whole game assembled by group (J2, after the data):** choosing jointly within each group of ply-6 positions
  sharing ply-8 positions — every subset of a group's positions with builds swapped at once — finds nothing more,
  197,650 again (h353). Every group holding an untaken build also holds positions with no build at all, six of them
  with a single winning move, which no batch of "every winning move where there is a choice" builds. So a batch is
  chosen by group, single-move positions included.
- **Every winning move by group (V2, measured):** in the 9 groups on which the whole game still paid at least 4,000
  bits, the 67 builds no earlier root covered — each unbuilt winning move, and a free build (no move forced at the
  root) at each single-move position never built. All finish (h354), and the whole game takes 192,591 bits, 0.974 of
  197,650: short of the 97/100 bar by 871 bits, so h355 is inconclusive (h356, after the data). 16 builds are taken;
  one group (#91 to #114) gives 1,920 of the 5,059 bits saved. The free builds at the heavy single-move positions save
  10–11% (#2, #68) or nothing (#96's is larger than G3's). Every winning move is now built in all but one of the 11
  groups still paying at least 3,000 bits (29% of the game), so further batches of every winning move reach only the
  lighter groups (51 builds in 8 groups at 2,000 bits). The library's 834 maps, 72,637 bits, are now the largest part.
- **A library pass over the assembled game (G4, measured):** the 409 ply-8 positions the assembled game still takes
  from G3, rebuilt with G3's builder from every map that game holds (G3's 2,030-map library, then those strategies'
  own maps, then the own maps of the 51 builds taken: 2,530). All finish (h357), in 55.2 hours against G3's 67.0 for
  the same positions, and the whole game takes 184,556 bits, 0.958 of 192,591, inside the 97/100 bar (h358). Alone 86
  rebuilds are smaller than G3's strategy, and the 409 take 55,928 bits against 61,427; but put in all at once they
  grow the game to 195,394, for the new maps they pay. Sending back one at a time the rebuild whose return shrinks the
  game most, 333 go back and 76 stay (22,674 bits against G3's 29,270); the ply-6 builds chosen again keep the same
  51 positions and change one build, #44's (h359, after the data). The library is 816 maps, 71,109 bits, 83 of them
  new.
- **Choosing among the rebuilds (U1, after the data):** a local search over which rebuilds stay — one rebuild, or every
  rebuild playing one map, in or out at a time, the move that shrinks the game most first — stops at 61 rebuilds and
  184,410 bits from none kept, 66 and 184,328 from all kept. Choosing the ply-6 builds afresh would take 184,386, so
  V2's choice stays. The whole game takes 184,328 bits, 228 fewer than G4's (h360): the chooser was worth 0.12%.
- **The ply-6 builds rebuilt (Z1, measured):** the 51 builds the game takes, rebuilt at the same positions with the
  same root moves and the builder they were built with, from every map the game holds (G4's 2,530-map library, then
  the 68 own maps of the 66 rebuilds U1 keeps: 2,598). All finish (h361), and the whole game takes 180,199 bits, 0.9776
  of 184,328: short of the 97/100 bar and inside the 99/100 one, so h362 reads inconclusive. Alone 36 rebuilds are
  smaller than their builds, and the 51 take 51,065 bits against 57,682; but all taken they grow the game to 186,419,
  for the maps they pay. U1's search stops at the same 27 from none taken and from all taken (21,606 bits against
  28,215; h363, after the data). The library grows to 837 maps, 73,329 bits, and is the largest part at 40.7%.
- **Maps read on the mirrored board (X2, measured):** X1 left mirroring to the build. With no build, the 2,598 maps with
  their cells flipped cover none of the 343 move nodes of Z1's rebuilds, because the rule breaks ties by the lowest
  move; read on the mirrored board they cover 18 (h364, after the data). So the same 51 were rebuilt from the same maps
  with both readings available, plain tried first. All finish (h365), and the whole game takes 175,425 bits, 0.9735 of
  180,199: 632 bits short of the 97/100 bar and inside the 99/100 one, so h366 reads inconclusive. The 51 play 38
  mirrored leaves of 1,773, in 10 rebuilds. Those 10 take 20,260 bits against Z1's 25,085 for the same positions, and
  the 41 with none take 26,778 against 25,980. U1's search keeps the same 7 from none taken and from all taken, and 22
  of Z1's 27 stay. Two of the 7 read mirrored, #2 and #4, saving 3,311 and 645 bits. The library falls to 832 maps,
  72,989 bits, and takes 41.6% (h367, after the data).
- **The ply-8 strategies rebuilt with both readings (X3, measured):** The 409 positions G4 rebuilt, with G4's builder and
  2-hour cap and the mirror on, from X2's maps. All finish (h368), and the whole game takes 166,623 bits, 0.9498 of
  175,425, past the 97/100 bar, so h369 is supported. The verdict does not credit the mirror alone. The 409 play 32
  mirrored leaves of 2,134, in 18 rebuilds, which take 6,677 bits against 9,488 for the pieces they stand in for. With
  only those 18 as candidates the game takes 172,678 (0.984), short of the bar; with only the 391 that play no mirrored
  leaf it takes 169,962 (0.969), past it. Those 391 were still built with both readings available, and most of the
  pieces they stand in for come from smaller libraries, so this run cannot say what the same rebuilds would do with the
  mirror off. Of the 71 taken, 13 read mirrored and save 2,831 bits, and 58 do not and save 3,083; the library falls
  2,888 bits to 797 maps, 70,101 bits (h370, after the data). U1's search no longer agrees from its two starts: 71 taken
  from all, 54 and 167,046 bits from none, so 166,623 is a local best. All 409 taken the game takes 176,641. Four builds
  ran past the cap, which ends improvement and leaves a build complete; three of them are taken.
- **The mirror's own worth at ply 8 (X4, measured):** The 18 rebuilds that played a mirrored leaf, rebuilt from the same
  list with the mirror off. All finish (h371). Put in X3's place, with U1's search run again, they give a game of 169,200
  bits, 2,577 more than 166,623 (1.0155), past the 1% bar, so h372 is supported. The 18 take 9,279 bits against 6,677: 15
  are larger, 3 smaller, and the game takes 8 of them against X3's 13 (h374, after the data). This holds the other 391 as
  X3 built them, taken to be what the mirror off would have built.
- **Every certified piece chosen at once (U2, measured):** Each earlier pass chose only among its own rebuilds. U2 lets
  every position take any piece any run certified, and moves one position or every position sharing a map. From X3's game
  its search stops at 165,196; from every position's smallest piece it stops at 165,023, 0.9904 of 166,623, past the 199/200
  bar (h373). No new builds. 49 positions change, all in play: 25 of X3's 71 rebuilds leave (19 for G4's piece, 5 for G3's,
  1 for X4's), 11 of G3's strategies go to G4's, and 7 of the 51 builds move, one to no build. Three of X4's mirror-off
  rebuilds are taken. The library grows to 807 maps, 70,917 bits, while the builds fall to 43,653 (h375, after the data). The
  two starts disagree, so 165,023 is again a local best.
- **The ply-6 builds rebuilt once more (Z2, measured):** The 50 builds U2's game takes, rebuilt with both readings from X2's
  maps plus the 163 own maps of the pieces the game plays. All finish, in 47.2 build-hours (h376). With each one more option
  for U2's search the game takes 162,752 bits from U2's game and 163,104 from every position's smallest piece: 0.9862 of
  165,023, past the 99/100 bar (h377). 21 of the 50 are smaller than the piece the game took at their position and 29
  larger. The search takes 9, eight of the smaller and #118, which is larger, and moves 4 ply-8 strategies to G4's piece. The
  builds fall to 41,776 and the library to 70,435 (801 maps, 7 of them beyond X2's list), and the strategies rise to 49,692
  (h378, after the data). The two starts disagree, so 162,752 is again a local best.
- **The game written as a file (T1, measured):** The count's map cost rests on positions the game never states (§2). So
  Z2's game was written as one file, in a code using only what a decoder knows, with both claims registered before any
  file of that game existed. It reads back with 0 differences and every line of what it reads wins, over 3,372 leaves
  (h379). It takes 130,005 bits, under the bar of 9/10 of 162,752 (h380). The levels are
  the largest part, 57,615 bits, 2.92 a cell of the 3 a level can take; the cells 31,674, the exception flags 17,884 and
  the references 15,339 follow. 514 of the 1,052 maps are read once and 616 of the 3,372 leaves play the empty map
  (h382, after the data). Before it ran, the same code had written G3's game in 225,678 bits against 280,640 counted, a dry
  run seen when the bar was set. The pieces were chosen against the count, not the file, so the search has never seen
  what a piece costs in the file.
- **The maps given their lowest levels (T2, measured):** Under the rule a map matters only by the order it puts the safe
  moves' cells in where the rule has a choice. T2 walked every line each of T1's leaves plays, collected those orders,
  gave each map the lowest levels that keep them, merged maps whose orders agree when that costs fewer cells, and wrote
  the game again, built so that every move is T1's. The file reads back with 0 differences and every line wins, over 3,372 leaves
  (h383), in 118,197 bits, under the bar of 19/20 of 130,005 (h384). The maps fall from 89,289 bits to 77,045 (1,008
  maps of 18,444 cells), almost all of it in the levels, 57,615 to 46,096, 2.50 bits a cell: 5,432 cells sit on level 0
  and 73 on level 7. No exception became redundant, and the tree takes 436 bits more (h385, after the data). Before it
  ran, the same steps had taken G3's file from 225,678 bits to 204,454, a dry run seen when the bar was set. The pieces
  were still chosen against the count.
- **The levels coded in context (T3, measured):** After T2 a map's levels run up from 0 without a gap, so what a map
  has said narrows what comes next. Of 17 contexts simulated with the coder on G3's T2 file, the highest level so far
  with the cell's row and whether the cell below holds a level coded G3's levels in 72,644 bits against 76,981. Written
  with it, Z2's game reads back to T2's game with 0 differences and every line wins, over 3,372 leaves (h386), in
  115,599 bits, under the bar of 99/100 of 118,197 (h387). Only the levels change, 46,096 to 43,495 bits, 2.36 a cell:
  5.6% fewer, as on G3 (h388, after the data).
- **The cell flags coded in context (T4, measured):** A map levels the cells its safe moves would fill, and a column
  fills from the bottom. Of 29 contexts simulated on G3's file, the cell and its emptiness with, for the two cells below,
  whether each is empty and holds a level coded G3's flags in 46,828 bits against 51,994; the row alone, a count, the
  cell to the left, or three or more cells below did worse. Written with it, Z2's game reads back to T3's game with 0
  differences and every line wins, over 3,372 leaves (h389), in 112,644 bits, under the bar of 99/100 of 115,599 (h390).
  Only the flags change, 30,949 to 27,995 bits, 9.5% fewer, near G3's 9.9% (h391, after the data).
- **The exception flags coded in context (T5, measured):** A leaf's walk asks at each of its own positions with a
  choice whether the rule is overruled; on G3's file 3,157 of 1,895,096 flags were. Exceptions come together: once a
  line leaves the rule it meets more. Of 65 contexts simulated on G3's file, how many exceptions the walk had read, up
  to two, and how many flags since the last one, on a log scale, coded G3's exception flags in 27,285 bits against
  33,698; the level of the rule's move, the safe moves or the stones did less. Written with it, Z2's game reads back
  to T4's game with 0 differences and every line wins, over 3,372 leaves (h392), in 109,034 bits, under the bar of
  99/100 of 112,644 (h393). Only the exception flags change, 17,884 to 14,273 bits, 20.2% fewer, near G3's 19.0%
  (h394, after the data).
- **Reading the position (T8, measured):** On G3's file, ranking the known maps by board likeness took the references
  from 28,842 bits to 24,779, and the plies below the leaf took the exception flags from 27,285 to 26,306. Written by
  that walk, Z2's game reads back to T5's game with 0 differences and every line wins, over 3,372 leaves (h395), in
  106,044 bits, under the bar of 99/100 of 109,034 (h396). Only those two parts change: the references 15,775 to
  13,331 bits, 15.5% fewer, the exception flags 14,273 to 13,728, 3.8% fewer, near G3's 14.1% and 3.6% (h397, after the
  data).

---

## 7. Open

The file's code looks near its floor for this game: the last four coding studies each found under 1% of the file
(§5). A smaller file now needs a smaller game or a different language. Nothing in this section has been measured.

- **Finding it without a solver.** This comes first. The verifying walk is a proof that needs no solver, so a process
  can prove positions from the bottom up and use each proof in place of the solver above it. Whether the builder can
  find maps this way, and how much larger the result is, has not been measured.

- **Where the gap to WeakC4 lies (scratch, never proven).** Written through T8's coder under its own rule setting
  ("exact" ties, ten levels), WeakC4's game takes 39,728 bits against our 106,044; the count had priced it at 123,881.
  Of the gap, 57% is maps (385 against 1,008), 25% exceptions (45 against 1,625) and 4% the tree's own bits. Under our
  setting ("lowest" ties) 247 of its 385 diagrams cannot be written. WeakC4 stays a yardstick; nothing of it is used.
- **Fewer, earlier leaves.** The game has 680 moves and 3,372 leaves; WeakC4 has 251 and 1,009 (Y1, §6). Maps are 67% of
  the file, at 71 bits a map, and a leaf that moves up the tree removes the maps below it. The levers are generic: an
  opening chosen together with its maps, and builds started above ply 6.
- **Builds priced by the file.** Every map was searched for, and every piece chosen, against the count. T2 showed that
  the levels the searches keep are not the cheapest. A search that scores a map by what it costs in the file has not
  been tried; P1's stand-in covered only the choice among pieces already built.
- **Another library pass.** Z2, the last, gained 1.4%, and each pass has gained less than the one before.
- **A second game.** The process is meant to be generic. It has been run on one game.
- **Decoding time.** T8's decoder ranks every known map at every leaf, so writing the file and reading it back takes
  51 minutes against T5's 14. This costs time, not bits.

---

## Reproduction

Each file's evidence is in `examples/boardgames/evidence/`. The replay of every line is the run's, recorded in that
file.

| Step | Evidence | Written and played by | Proof test, and what it does |
|---|---|---|---|
| T1 | `c49_t1.json.gz` | `scripts/c4_strategy_file.py` | `tests/test_c49_t1_evidence.py`: decodes the file from its bytes, assembles the game again, compares the two and encodes the game back to the same bytes |
| T2 | `c49_t2.json.gz` | `scripts/c4_map_orders.py` | `tests/test_c49_t2_evidence.py`: decodes T1's file, rebuilds the maps from the orders stored in T2's, encodes the game to the same bytes and decodes them back to it |
| T3 | `c49_t3.json.gz` | `scripts/c4_level_context.py` | `tests/test_c49_t3_evidence.py`: decodes T2's file, writes its game with `harness/level_context.py` to the same bytes and decodes them back to T2's game |
| T4 | `c49_t4.json.gz` | `scripts/c4_cell_context.py` | `tests/test_c49_t4_evidence.py`: the same from T3's file with `harness/cell_context.py` |
| T5 | `c49_t5.json.gz` | `scripts/c4_exception_context.py` | `tests/test_c49_t5_evidence.py`: the same from T4's file with `harness/exception_context.py` |
| T8 | `c49_t8.json.gz` | `scripts/c4_position_walk.py` | `tests/test_c49_t8_evidence.py`: the same from T5's file with `harness/position_walk.py` |

From `examples/boardgames/`:

```
PYTHONPATH=. .venv/bin/python -c "
from harness.evidence import load_evidence
from harness.floor_g3 import SPEC, g3_size
e, g2, f2 = (load_evidence(f'evidence/c49_{n}.json.gz') for n in ('g3', 'g2', 'f2'))
print(g3_size(e, g2, f2, SPEC))"
PYTHONPATH=. .venv/bin/python scripts/hypotheses.py list
```

Builds: `scripts/c4_strategy_s2.py --spec harness.floor_g3` with `scripts/c4_library_file.py` to write its library.
Evidence (gitignored) is backed up to the private `thefactory-modeltrainer-evidence` repo.
