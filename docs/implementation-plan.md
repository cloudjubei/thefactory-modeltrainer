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

**1.1 T10, the solver-free Connect-4 run (h75) — running.**
- Setup: seeds 361–370, cap 60 iterations, canon conv-32 (20,616 params), strategy tree to 10 plies, stop on full
  agreement, then solver certification through 10 plies.
- Seed 361 did not stop: disagreements fell 1,280 → 21 by iteration 13, then plateaued at 50–90 through iteration 60.
  ~3.5 h per seed, so ~35 h of training plus certification.
- On exit: verify h75; back up; update the digest.

**1.2 Diagnose the plateau** (alongside T10, low priority; pre-registered as descriptive).
- Method: certify seed 361's saved final net through 10 plies, and solve exactly the positions where the walk
  disagreed. Classify each disagreement: net wrong / search wrong / both.
- **Outcome decides the next step:**
  - **mostly search wrong** → redesign the stop signal, and recalibrate it on tic-tac-toe with a deliberately
    weakened search before using it on Connect-4. Candidates: stronger search at walk positions only; agreement
    only where the search is confident; a calibrated agreement-rate threshold;
  - **mostly net wrong** → capacity or coverage: conv-48/64 at depth 10, re-run as T10.

**1.3 After T10 exits.**
- Run the 6 pending mutation specs, so `tests/test_mutation_specs.py` is green again: `neural_strategy_tree`,
  `neural_parallel_relabel`, `neural_stop_on_agreement`, `strategy_tree`, `transfer_on_pass`,
  `transfer_forbid_native`.
- Add an automatic trial log to every driver: a start line, then an end or failure line per run, so aborted runs
  are recorded without anyone remembering to.

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

- **3.1 E2 — minimal rule lists for tic-tac-toe** (pre-register).
  - Method: the SAT encoding of Yu, Ignatiev, Stuckey & Le Bodic 2021, minimising literals.
  - Targets:
    - the smallest position-sound list over all 4,520 positions;
    - the smallest strategy-sound list for each side (Newell–Simon's 9 rules are the bound to beat).
  - **Success:** each list is proven minimal (UNSAT at k − 1) relative to the declared predicate language, and
    verified exhaustively.
- **3.2 E4 — extraction from a net.** Fit the rule language to the 938-param net's moves (VIPER-style,
  mistake-cost weighting), then verify with the solver. **Success:** as precise as E2's list.
- **3.3 Connect-4 rules.**
  - Move rules: win, block, never play under their threat, double threats via threat-space search.
  - **Track C — value certificates:** Allis's nine rules with their compatibility check.
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
