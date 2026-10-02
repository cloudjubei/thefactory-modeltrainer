# thefactory-modeltrainer — implementation plan

**Remaining work only.** Architecture: `docs/architecture.md`. Contract: `docs/model-training-standard.md`.
- **What was tried and how it came out:**
  - the claims register `examples/boardgames/hypotheses.json`;
  - the generated `docs/process-knowledge.md`;
  - the history in `docs/experiment-journal.md`.
- **Raw evidence:** `examples/boardgames/evidence/`, backed up to the private repo `thefactory-modeltrainer-evidence`.

## North star

> **★ Overarching rule:** build everything as **reusable, chat-reachable modeltrainer tools**: a harness capability +
> a `.factory/trainer.json` activity + a chat tool (`ModelTrainerTools` / backend `trainerTools`). Each is generic, so
> it serves other trainings too. A script is a prototype; work is done only when it is promoted and the script is
> deleted.

1. **The best generic pipeline for creating any model** (propose → run → judge → explore), with an evaluation
   engine that filters false positives: leakage control, pre-registered gates, multiplicity correction, adversarial
   verification.
2. **One fool-proof HYBRID process for any finite game.** It produces an agent that plays perfectly, plus **the most
   compact written description of a perfect strategy**: an ordered list of logical rules, a small net for what the
   rules do not cover, and a small exception table. All of it is certified, with evidence that the process is the
   best of those tried. Two tracks run in parallel and feed each other through one verifier:
   - **L** — the self-play learner;
   - **R** — written rules: discovered, extracted from nets, or known.
3. **Publish the evidence** (§D) once the corpus clears its pre-defined bar.

## Repo split

| Repo | Owns |
| --- | --- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`. |
| **thefactory-tools** | Generic infra: `ComputeRunner` seam (+ `RemoteComputeRunner`, `ContentAddressedDataCache`, pairing); work-item engine. |
| **thefactory-backend** | Activity registration + composition; app-view serving; PIN-pairing + runner WS channel. |
| **BlackSwan** (the trading repo) | Its `TrainerManifest` + additive `trainer/` CLI conformance. No Overseer code. |

## Rules for every experiment

- **Pre-register before the data.** A claim in the register, with:
  - a proof test (and an undecidable proof where a gate can fail);
  - the data file it will read;
  - pinned judge files;
  - measurable SUPPORTED / REFUTED / INCONCLUSIVE bars.

  A claim written after the data says so ("descriptive, registered after the data").
- **Measurement integrity:**
  - seeds have roles: a choosing seed never reports;
  - every rate carries n, seed and a CI;
  - a selected maximum is re-measured on a held-out seed;
  - comparisons are paired, final-vs-final;
  - power the claim before making it;
  - one training run is one sample.
- **Fingerprinted code is frozen while a run is active.** Runs refuse to write evidence if their training or
  measurement fingerprint changes. Edit in a scratch copy and copy back after the run exits, merging
  `hypotheses.json` and `evidence/manifest.json` rather than overwriting them.
  - Never mutation-test a module that a running job's workers import.
- **Every guard is mutation-tested through a tracked spec** (`tests/mutations/*.json`, run with
  `scripts/mutate.py --spec`), never by hand.
- **Capture everything:**
  - pilots and smoke runs that inform a decision → `evidence/` + a registered claim;
  - bugs and wrong turns → a dead-end claim proven by the regression test;
  - abandoned trials without a proof → `knowledge_map.json` `unproven_trials`;
  - every new claim → filed in `knowledge_map.json`, then `scripts/knowledge_digest.py`;
  - after new evidence → `scripts/backup_evidence.py --push`.
- **Licences:** no AGPL code or dependencies.

---

## 1. Board games — in flight

**Where the solver-free Connect-4 process stands.**
- Its nets are wrong in the opening, and their own search agrees with the wrong moves (h91, h92).
- More search does not help: 200 → 20,000 sims leaves opening labels at 84–87%, and the first moves at 33–60%
  (h96; h93 inconclusive).
- So **self-search cannot supply the opening**. The open question is which other source of truth can, without the
  solver.
- **Written certificates (option (a), the user's choice) do not reach the opening either.**
  - The claimeven/baseinverse pairing (`harness/c4_certificates.py`) is sound: no contradictions (h98).
  - It flags none of the search's 144 wrong opening moves (h99 refuted).
  - Through ply 8 of the nets' trees it certifies no position, and adds 0.5% of non-winning moves beyond the
    immediate-win tactic (h100). It is an endgame tool.
  - So certificate labels are **not** wired into training.

**1.1 Value-aware stop — T11 recalibration on tic-tac-toe (running, h101).**
- The stop now also counts a raw move as agreeing when its search Q is within δ = 0.1 of the label's top move
  (`train_alphazero(stop_value_delta=…)`; the share reading is kept beside it).
- T11: T9's process, relabel search weakened 200 → 50 sims, seeds 371–380
  (`scripts/small_floor.py --spec floor_value_stop`).
- **Gate:** if h101 is refuted (any false stop), the value stop is unsafe and T12 does not launch.

**1.2 T12 — solver-free Connect-4 with the value-aware stop (pre-register and launch once h101 holds).**
- T10's process plus δ = 0.1; seeds 381–390; cap 60; certify the stopped nets through 10 plies
  (`harness/floor_c4_value.py`, `scripts/c4_solver_free.py --value-delta 0.1`).
- Cost: T10 took ~2.7 h per seed with 8 relabel workers, so ≤ ~27 h for 10 seeds, less where a seed stops.
- Prediction at registration: **refuted.** The opening labels are wrong (h91, h96) and nothing here fixes them. The
  run's value is the false-stop count: does the value stop fire on C4, and on a wrong net?

**1.3 The opening's source of truth — still open (decision for the user after T12).**
- **(a2) Stronger certificates:** Allis's remaining rules (aftereven, lowinverse, highinverse, before), and a
  proof-number search with certificates at the leaves, as in VICTOR. A bounded search, declared like the playbook's
  lookahead.
- **(b) An opening exception table** in the hybrid: a small solver-built table for the first plies (WeakC4 uses
  ~2,600 opening nodes), counted in the description length. The process stays solver-free beyond it.
- **(c) A stronger value signal:** much longer self-play or larger nets. Costly, and the literature says it can work
  for Connect-4 given enough compute.

## 2. Board games — next

**2.1 Depth-12 oracle frontier** (`scripts/strategy_frontier.py`; nets saved; independent certification).
- conv-64 (~53K params) first. ~3–5 days on CPU.
- Optional first: an MPS device option for refits, with a test that MPS and CPU fits agree.
- **Success:** certified through 12 plies, giving the depth 8 → 10 → 12 curve. Then report plainly whether a
  whole-game first-player certificate (~10^9 positions) is feasible on this machine.
- Optional: sharpen the depth-10 bracket (8,008, 20,616] with conv-20/24.

**2.2 Solver-free Connect-4 at depth 12**, once 1.2's fix holds at depth 10.

## 3. Written rules (track R) — runs beside self-play

`harness/playbook.py` holds:
- the rule language — `IF ∃m: L1(m) ∧ … THEN play m`, ordered decision lists;
- the generic tactical predicates and tic-tac-toe geometry;
- `evaluate` (correct only when every recommended move is optimal) and `own_play_positions`.

Every rule is graded **position-sound** (optimal at any legal position) or **strategy-sound** (optimal on its own
play's positions only). Only position-sound rules may label arbitrary positions or override a net. Measure each
rule standalone and in decision-list order, by ply.

- **3.1 E2b — a richer predicate language for tic-tac-toe.**
  - Why: with the 11 current predicates no position-sound playbook exists up to 60 rules (h94 refuted, h95). The
    obstruction is across positions, so add predicates, not rules.
  - Candidates: layered MIGO-style predicates `wins_in_k` / `loses_in_k` (k = 1, 2, 3, bounded lookahead),
    `blocks_fork_by_threat`, line counts. Each predicate's lookahead cost is declared, so a rule cannot hide a solver.
  - Method: `harness/rule_search.py` (SAT; python-sat installed) and `scripts/minimal_playbook.py`.
  - **Success (pre-register):** a proven-minimal position-sound playbook with the richest predicate costing at most
    k = 3 plies of lookahead. Report the size against Newell–Simon's 9 rules.
  - Then the strategy-sound list per side.
- **3.2 E4 — extraction from a net.** Fit the rule language to the 938-param net's moves (VIPER-style,
  mistake-cost weighting), then verify with the solver. **Success:** as precise as E2's list.
- **3.3 Connect-4 rules.**
  - Move rules: win, block, never play under their threat, double threats via threat-space search.
  - **Track C — value certificates:** the pairing (claimeven + baseinverse) is built and sound (h98–h100); next are
    Allis's other rules with their compatibility check (see 1.3 (a2)).
  - Measure on three sets: a ply-matched sample, positions our agents visit, and the rules' own play.
- **3.4 E3 — hybrid on Connect-4.** The playbook in front of a smaller certified net at depth 8. **Success:** at
  most half of 1,576 params, with the rules' compute declared and bounded and reported beside the params.
- **3.5 Feeding self-play (R → L).** One pre-registered experiment each for labels, features and search pruning.
  A rule counts as feeding only if a fresh self-play agent measurably improves with it.

## 4. The hybrid process — definition of done

- **"Best", measurably:**
  - a certificate for every claim;
  - the smallest total description in bits (rules + net params × bits + table), each part reported; WeakC4's ~25 KB
    first-player Connect-4 strategy is the outside benchmark;
  - bounded compute per move;
  - minimality proven relative to the declared language or net family;
  - the best process wins on every ladder game with **zero game-specific process edits**.
- **Fool-proof requirements (each a test):**
  - claims pre-registered with pinned judges;
  - no game names in process code;
  - every game proven to terminate (a progress measure, checked on random play);
  - the verifier independent of what it checks;
  - fingerprinted, bit-reproducible runs;
  - a process fix re-runs every earlier game.
- **Game ladder (finite, with an exact oracle):**
  1. tic-tac-toe;
  2. Connect-4;
  3. **Kalah**, small variants first: game module, solver, termination test (seeds in pits or their distance to the
     owner's store falls every move);
  4. **Othello 6×6**: add a board-size parameter to `games/othello.py` (fixed at 8);
  5. **small Hex or Dots and Boxes**.
- **Order:**
  - finish Connect-4 (sections 1–3);
  - the hybrid on tic-tac-toe and Connect-4 (smallest certified total description vs pure net and pure rules);
  - Kalah → Othello 6×6 → Hex / Dots and Boxes;
  - freeze the process and write the "any finite game" runbook (inputs, certificates issued, stated limits);
  - promote it to chat-reachable tools (★ rule).

## 5. Other tracks (carried over unchanged — re-check the current state before starting)

- **Trading (BlackSwan)** — probe first, build second.
  - **Remaining probes:**
    - B1: on-chain flows and the funding-basis term structure (only if a cheap proxy motivates the mine);
    - B2: FOMC decision days, token unlocks / listings, whale moves, exchange outages;
    - B3: LLM-read Reddit crowd sentiment (Arctic Shift);
    - B4: silver/copper world models were inconclusive as partial models (industrial drivers missing).
  - **B5 (trigger: trading exhausted AND "safer AI-written code" is the stated mission):** one pre-registered
    defect-risk decision probe.
  - **B6, optional:** live handoff of the autopilot champion; Jupyter notebooks (scope kernel + security); a
    runner-channel WebSocket; remote git repoRefs.
  - Runtime step: set BlackSwan's overseer project `metadata.hasApp=true`, `metadata.appDir="app"`.
- **Engine / evaluation.**
  - Game suite: skull · flip7 · skull_king · for_sale, then terra_mystica · poker (CFR) · catan · altered.
  - A neural self-play core as a `model_name` lever; personas + league play for the luck games.
  - The specialist-vs-generalist-finetuned hypothesis; a BoardGameArena live-play bridge.
  - A unified "find the best model" process with a Models view. Open decision: an explicit manifest
    `compoundCores` vs deriving it from a `*_warm_start` lever.
  - The S9 leakage tail: per-split signature + overlap detector; move the trading fidelity predicate into
    BlackSwan.
  - Owner ratifications of the conservative gate defaults.
- **Tools to build:**
  - the `disentangle` capability, after which `scale_up.py` / `gumbel_ab.py` are deleted;
  - a game feature-probe tool;
  - a process registry;
  - Game-interface extensions as sequencing demands them.
- **Publication (§D).**
  - Evidence export as a reproducible package, and a reproduce-and-refute workflow.
  - Paper 1: BlackSwan no-edge (battery + refutation table).
  - Paper 2: the methods paper (null where it must be null, true positive where it must be positive).
  - Define with the owner: N families, K refuted papers, X% byte-exact reproduction, M markets, P regimes.

## 6. Open questions and trigger-blocked items

- **The provenance of `harness/solver.py`** (possible AGPL derivation) — the user's call. The C solver is ported
  from it.
- **`with_extra_data` projection rung** — build once at least one asset-specific series is mined.
- **Host → iframe `data:updated` push channel** — the viewer is poll-only.
- **Remote artifact / checkpoint storage** — once remote runs and live handoff both exist.
- **GPU + sandbox profile for training images** — `--gpus` is wired but unexercised.
- **Judge/proposer model transport** (`ModelSelection` API vs CLI) — revisit once the CLI inference stage lands.
- **Model comparison view** — trigger: a Connect-4 net certified for the whole game. A read surface over the
  existing leaderboard records.
