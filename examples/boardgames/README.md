# boardgames — a game-RL consumer and the §C evaluation driving case

A conformant modeltrainer consumer (the 4th, alongside `cartpole`, `tabular`, and BlackSwan) that trains and
evaluates agents for board games and reports the **§C rigorous-evaluation** metric battery, so the honesty
gates (seed-significance, best-of-N deflation, locked held-out TEST, proxy selection-regret, degeneracy) fire
on a live, ground-truth task. The outcome is a process that **produces model artifacts you then test live**.

## The contract (mirrors `examples/cartpole`)

- **Manifest** — `.factory/trainer.json`: objective `win_rate` (max), `hypothesisBenchmark` with a seed
  `quorum`, sweepable levers, and a `diagnostics` block that declares **every** §C gate.
- **CLI** — `python3 -m harness.run` honouring `--config-json` / `--summary-out` / `--calibrate` / `--evaluate`.
- **RunSummary** — objective + a metric battery (`win_rate` truth, `episode_reward` shaped PROXY, `draw_rate`,
  `first_player_seat_winrate` confound canary), a first-class **`cost`** block (wall/cpu time, compute, and
  documented energy/$ estimates + a step breakdown), a sampled **`sample_game`** replay (move log + ASCII
  frames) so you can follow a game, full **provenance** (gitCommit/configHash/seed/dataVersion), and a
  checkpoint reference.

A run evaluates one `model_name` against one `opponent` rung over `eval_games` games (seats alternated for
fairness). `opponent` is the **split axis**; the strongest rung (`mcts`) is the **locked held-out TEST** — the
incumbent is selected on the lower rungs and certified once on the test.

## Run it

The manifest runs `.venv/bin/python` (a concrete path, so the Overseer's runner finds it regardless of its
`PATH` — an Electron app launched from Finder has a minimal `PATH` where bare `python3` may not resolve). The
dependency-light cores (`random` / `heuristic` / `mcts`) need only the standard library, so the venv needs **no
packages** — create it once:

```sh
cd examples/boardgames
python3 -m venv .venv                 # once — no pip installs needed for the light cores
.venv/bin/python -m harness.run --calibrate --summary-out /tmp/bg-cal.json
.venv/bin/python -m harness.run --config-json configs/default.json --summary-out /tmp/bg-sum.json
# re-test a saved checkpoint against a chosen opponent:
.venv/bin/python -m harness.run --evaluate --config-json /tmp/eval.json --summary-out /tmp/bg-eval.json
```

(The neural core will add `torch` to this venv later.) `harness.load_policy` turns a checkpoint back into a
playable `act(game, state, rng)` — the seam a live test drives.

## Watch or play against a model

**In-app is the primary surface.** A completed run's detail in the Overseer viewer shows a **Game replay**
stepper (from the summary's `sample_game`) and, when the manifest declares a `play` command, a **Play against
this model** board. Interactive play routes each turn through the backend: the viewer → `playBoardGame` trainer
tool → the manifest's `play` command (`harness.play --serve`) → the run's checkpoint. The same tool is
chat-invocable (`playBoardGame` with `mode: 'move' | 'autoplay'`), so an agent can play or watch too. The
`--serve` mode below is that oracle; the interactive CLI is a convenience for local dev.

`harness.play` is also a direct human-facing surface over a trained checkpoint (it reuses `load_policy` and the
same replay a run samples, so it needs no packages either):

```sh
# PLAY against a saved model in the terminal (you drop columns; it answers):
.venv/bin/python -m harness.play --checkpoint checkpoints/<hash>.json --vs human
# ...or against an inline core without a checkpoint:
.venv/bin/python -m harness.play --model mcts --mcts-sims 200 --vs human

# WATCH the model play an opponent rung, move by move:
.venv/bin/python -m harness.play --checkpoint checkpoints/<hash>.json --vs mcts

# REPLAY the exact game a run sampled (from its summary's sample_game block):
.venv/bin/python -m harness.play --from-summary /tmp/bg-sum.json
```

`--seat 0|1` sets who moves first (the model takes `--seat`; in a human game you take the other seat).

## The cores (`model_name`)

- **`random`** — legal move at random. **`heuristic`** — win/block/centre rules.
- **`mcts`** — a real UCT tree search: selection → expansion → random rollout → back-up, with a
  **transposition table** (positions reached different ways share statistics) that **persists across the
  agent's moves in a game**. Strength scales with `mcts_sims`. It is SEARCH, not learning — no parameters, no
  memory across games.
- **`alphazero`** — a **learned** policy+value net (`harness/neural.py`, needs torch) trained by self-play,
  then used to *guide* the MCTS (PUCT priors + a value head instead of random rollouts). A training run
  produces a `.pt` weights file beside the checkpoint; the knowledge is in the weights and **generalises**
  across positions (what a transposition table cannot). Levers: `az_iterations`, `az_selfplay_games`,
  `az_sims`, `az_epochs`, `az_warm_start`.
  - **Warm-start + league** (`harness/champions.py`): a run does NOT relearn from zero. It **warm-starts from
    the best champion** on disk, trains with self-play MIXED with games against a **league** (strong mcts +
    heuristic + past champions), then is measured against a strong mcts + the champion and **promoted only if
    it beats the incumbent** (≥55%). Set `az_warm_start: 0` for a reproducible from-scratch run. Self-play is
    the main strength engine (balanced games); the league is a minority + the yardstick. Getting a genuinely
    strong model takes real budget + several compounding runs — a single tiny run stays weak.
  - **Champion autopilot** (in-app): the `train-champion` activity — a "Train champion" launcher in the
    viewer's Exploration tab, and the `startTrainerActivity` chat tool — automates the compounding: it keeps
    training warm-started generations, promoting the stronger net, until strength plateaus / a target vs the
    strong-mcts yardstick is hit / the generation budget is spent. This is the "keep looking for the best
    model" loop for a LEARNED core — distinct from the config-space Exploration autopilot (grid-search).

```sh
# TRAIN a learned model (needs torch — .venv/bin/pip install torch numpy):
.venv/bin/python -m harness.run --config-json az.json --summary-out /tmp/az.json
#   az.json: {"model_name":"alphazero","opponent":"mcts","eval_games":20,
#             "az_iterations":6,"az_selfplay_games":24,"az_sims":80,"az_epochs":6}
# then PLAY it (loads the trained net) exactly like any other checkpoint:
.venv/bin/python -m harness.play --checkpoint checkpoints/<hash>.json --vs human
```

## Architecture

```
harness/
  game.py       # the Game protocol (OBSERVATION-based + a state_key for transpositions)
  agents.py     # random / heuristic / real-UCT mcts cores + the extensible opponent REGISTRY
  neural.py     # the alphazero core: net + net-guided MCTS + self-play + LEAGUE training + weight I/O (torch)
  champions.py  # the champion store — warm-start from the best net + a pool of past champions (cross-run memory)
  selfplay.py   # match play + ladder evaluation + §C metrics + cost + sampled replay
  summary.py    # the RunSummary builder
  config.py     # the --config-json contract + provenance + cost constants
  play.py       # watch / replay / interactive-play surface (incl. the --serve RPC the in-app board drives)
  run.py        # CLI + training path + checkpoint/weights save + load_policy seam
  registry.py   # where a new game (and a luck game's personas) is wired in
games/
  connect4.py   # game #1 — perfect-information, deterministic reference
```

Adding a game is one file implementing `Game`. A new agent core is one class + a registry entry.

## Game roadmap

Tier 1 (now, simplest → hardest): **connect4** ✓ · skull · flip7 · skull_king · for_sale.
Tier 2 (frontier): terra_mystica (huge action space) · poker (bluffing — needs a CFR core) ·
catan (trading) · altered (deckbuild + play — a **two-model adversarial** harness).

## Forward milestones

- **Connect-4 SOLVED** (ACTIVE — the crisp end-state) — Connect 4 has ground truth (first player wins with
  perfect play). Today `alphazero` plateaus weak (champion stuck at gen12; later generations never promote), so
  reaching the known winning strategy is the priority. Build: a perfect-play **oracle** (`harness/solver.py`:
  bitboard negamax+αβ+transposition+opening book) as a persona + the top rating rung; near-perfect ladder rungs
  (tactical-rollout MCTS, depth-limited oracle); an alphazero setup that reaches it (ResNet + real budget +
  **oracle distillation** + oracle in the league); and a measurable SOLVED bar — `oracle_optimality_rate ≥ 0.99`
  (greedy move ∈ the solver's optimal set over a fixed benchmark) AND `wins_as_p1_vs_oracle == 1.0` (NOT
  win-rate vs oracle, which is a degenerate 0.5). Surfaced as a `health: solved` badge.
- **Unified "find the best model" process + Models-first view** (ACTIVE) — one process, not two launchers:
  Exploration (search configs) and Improve (warm-start champion ladder) fold into a single reducer
  (`screen-new → search → improve → converged`) that re-screens newly-added `model_name` choices on
  start/resume. The Exploration tab shows ONE surface at a time by state (a live run always wins). The primary
  view becomes **Models ranked by strength** (the leaderboard); a **Run becomes one training step in a model's
  history**, not a flat list — model identity spans runs via the champion lineage.
- **Model comparison** (DEFERRED — only after Connect-4 is solved) — a side-by-side of champions/architectures
  on the one gauntlet scale with the §C gates per model; a read surface over the leaderboard records. See
  `docs/implementation-plan.md` §E for why it waits (nothing certified to compare until a model reaches ground
  truth).
- **Neural self-play core** — ✓ SHIPPED: `alphazero` as a `model_name` lever (`harness/neural.py`). Trains a
  policy+value net by self-play, saves weights, plays through the same replay/play tooling. Next: a `ppo`
  variant; a larger/residual net; per-game net shapes (the current net is connect4-shaped).
- **Personas + league play** — the luck-based games (skull, flip7, skull_king) are tested against fixed
  protocol **personas**; once models beat them, new models train against a growing pool of **past champions +
  personas** (the AlphaGo/AlphaStar insight). The opponent registry already accepts personas and `champion:<ref>`
  checkpoints; wiring the champion pool onto the `opponent` axis (via `choicesFrom` over an on-disk champion
  catalogue) is the next step.
- **Specialist vs. generalist-finetuned** — a first-class §C hypothesis: does a game-specific model beat a
  generic game-playing model finetuned on the game? Judged with declared gates + seeds + held-out opponents.
  Needs a HuggingFace survey (decision-transformer / Gato-style / LLM-as-policy candidates); slots onto
  `model_name` as `specialist` vs `generalist_finetuned`.
- **BoardGameArena live test** — the final validation: a thin per-game bridge drives `load_policy` to play on
  BGA. Learned policies act from the observation alone (what a live venue exposes); search cores need a
  reconstructed state.

## Why the gates matter here (an honest first result)

A 36-run Connect 4 sweep (4 setups × 3 opponents × 3 seeds) crowns `mcts,160` as the incumbent and it beats
the locked `mcts` test rung — yet the champion verdict is **not steady**: with only 3 seeds the seed-stability,
seed-significance, and best-of-N gates all decline to certify the edge as more than noise. That is the
false-positive filter working: to earn a steady verdict you add seeds/games until the intervals tighten. The
process produces the model; the gates keep the claim honest.
