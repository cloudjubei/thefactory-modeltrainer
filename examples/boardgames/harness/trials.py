"""The automatic TRIAL LOG: every experiment driver runs inside `trial`, which appends a start line and then an end
line (completed, or failed with its error) to the tracked trials.jsonl. A run that was killed outright never writes
its end line, so `read_trials` reports it as aborted — dead ends are recorded without anyone remembering to."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "trials.jsonl"


def _append(path: Path, entry: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def trial(name: str, argv: list, path: Path = LOG):
    """Log one run of the driver `name` with its arguments; re-raises whatever stops it."""
    run_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    _append(path, {"event": "start", "id": run_id, "name": name, "argv": list(argv), "at": _now()})
    try:
        yield
    except BaseException as e:
        _append(path, {"event": "end", "id": run_id, "status": "failed", "error": f"{type(e).__name__}: {e}",
                       "at": _now(), "seconds": round(time.time() - t0, 1)})
        raise
    _append(path, {"event": "end", "id": run_id, "status": "completed", "at": _now(),
                   "seconds": round(time.time() - t0, 1)})


def read_trials(path: Path = LOG) -> list:
    """Every logged run in start order: name, argv, start time and status (completed / failed / aborted)."""
    starts, ends = {}, {}
    for line in path.read_text().splitlines() if path.exists() else []:
        entry = json.loads(line)
        (starts if entry["event"] == "start" else ends)[entry["id"]] = entry
    return [{**s, "status": ends[i]["status"] if i in ends else "aborted",
             "error": ends.get(i, {}).get("error")} for i, s in starts.items()]


def logged(main, script_file: str, path: Path = LOG):
    """Run a driver's `main` inside `trial`, named after its script file, with the process's own arguments."""
    import sys

    with trial(Path(script_file).stem, sys.argv[1:], path):
        return main()
