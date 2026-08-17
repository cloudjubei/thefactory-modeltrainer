"""The champion store — cumulative memory ACROSS alphazero runs, so training never restarts from zero.

The best net(s) for a game are kept on disk, separate from the run records: a run WARM-STARTS from the
strongest champion, trains, and PROMOTES its net only if it beats the incumbent (an AlphaGo-style gate). Past
champions are also handed back as a POOL of opponents for league self-play. Kept torch-free (weights are
written through a `saver` callback) so the light cores never import torch to read champion metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

CHAMPION_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "champions"


def _meta_path(game_name: str, store_dir: Path = CHAMPION_DIR) -> Path:
    return store_dir / f"{game_name}.json"


def _read_meta(game_name: str, store_dir: Path = CHAMPION_DIR) -> dict:
    p = _meta_path(game_name, store_dir)
    return json.loads(p.read_text()) if p.is_file() else {}


def champion_generation(game_name: str, store_dir: Path = CHAMPION_DIR) -> int:
    """How many champions have been crowned for this game (0 = none yet)."""
    return int(_read_meta(game_name, store_dir).get("generation", 0))


def champion_is_distilled(game_name: str, store_dir: Path = CHAMPION_DIR) -> bool:
    """Whether the current champion was trained WITH oracle distillation. A pre-distillation champion has a
    broken opening (edge-first), so warm-starting a distillation run from it inherits that flaw — the run should
    start FRESH instead. Once a distilled champion exists, warm-starting from it compounds safely."""
    return bool(_read_meta(game_name, store_dir).get("distilled", False))


def best_champion_path(game_name: str, store_dir: Path = CHAMPION_DIR) -> str | None:
    """Weights path of the current best champion, or None when the store is empty / the file is missing."""
    best = _read_meta(game_name, store_dir).get("best")
    return best if best and Path(best).is_file() else None


def champion_pool_paths(game_name: str, k: int = 3, store_dir: Path = CHAMPION_DIR) -> list[str]:
    """The most recent `k` champion weights (the league's past-self opponents), newest last."""
    history = [h for h in _read_meta(game_name, store_dir).get("history", []) if Path(h).is_file()]
    return history[-k:]


def promote_champion(
    saver: Callable[[str], None], game_name: str, store_dir: Path = CHAMPION_DIR, distilled: bool = False
) -> tuple[str, int]:
    """Crown a new champion: write its weights (via `saver(path)`) as generation N+1 and record it as `best`.
    `distilled` marks whether it was trained with oracle distillation (drives the warm-start decision). Returns
    (weights_path, generation)."""
    store_dir.mkdir(parents=True, exist_ok=True)
    gen = champion_generation(game_name, store_dir) + 1
    wpath = store_dir / f"{game_name}-gen{gen}.pt"
    saver(str(wpath))
    meta = _read_meta(game_name, store_dir)
    history = meta.get("history", [])
    history.append(str(wpath))
    meta.update({"generation": gen, "best": str(wpath), "history": history[-10:], "distilled": bool(distilled)})
    _meta_path(game_name, store_dir).write_text(json.dumps(meta, indent=2) + "\n")
    return str(wpath), gen
