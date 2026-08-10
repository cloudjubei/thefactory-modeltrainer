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

```sh
cd examples/boardgames
python3 -m harness.run --calibrate --summary-out /tmp/bg-cal.json
python3 -m harness.run --config-json configs/default.json --summary-out /tmp/bg-sum.json
# re-test a saved checkpoint against a chosen opponent:
python3 -m harness.run --evaluate --config-json /tmp/eval.json --summary-out /tmp/bg-eval.json
```

The dependency-light cores (`random` / `heuristic` / `mcts`) need only the standard library. `harness.load_policy`
turns a checkpoint back into a playable `act(game, state, rng)` — the seam a live test drives.

## Architecture

```
harness/
  game.py       # the Game protocol — OBSERVATION-based, so hidden-info games fit the same harness
  agents.py     # random / heuristic / mcts cores + the extensible opponent REGISTRY (personas, champions)
  selfplay.py   # match play + ladder evaluation + §C metrics + cost + sampled replay
  summary.py    # the RunSummary builder
  config.py     # the --config-json contract + provenance + cost constants
  run.py        # CLI + checkpoint save + load_policy seam
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

- **Neural self-play core** — `ppo_selfplay` / `alphazero` as `model_name` levers (torch is available); the
  harness is unchanged.
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
