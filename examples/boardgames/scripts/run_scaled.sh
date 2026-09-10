#!/bin/bash
# §C.17a LAUNCH *AND* RESUME a batched scaled run — the same command for both, because a long run on a machine
# that sleeps, reboots or gets killed will be resumed far more often than it is launched.
#
# What this adds over calling harness.scaled_run directly:
#   1. STAMP ONCE. provenance.json records the training era at LAUNCH, from outside the fingerprinted modules
#      (§C.17a rule 2). A resume must NEVER re-stamp: overwriting the launch era with today's would erase the
#      very drift the stamp exists to expose.
#   2. PREFLIGHT. harness.resume refuses a run whose training code moved underneath it, and repairs the two
#      crash-window artefacts that would otherwise silently cost the run its ledger-readability (see resume.py).
#
# Usage:  scripts/run_scaled.sh othello_302K_s0
set -u
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MKL_NUM_THREADS=4
D=checkpoints/scaled_runs
NAME=${1:?usage: run_scaled.sh <run-name>}
CFG=$D/$NAME.json
L=$D/$NAME.log
mkdir -p "$D/$NAME"

.venv/bin/python - "$D/$NAME" "$CFG" <<'PY' || exit 1
import json, pathlib, sys
from harness.fingerprint import training_fingerprint, config_fingerprint

run_dir, cfg_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cfg = json.loads(cfg_path.read_text())
prov_path = run_dir / "provenance.json"
if prov_path.exists():
    print("provenance: already stamped (resume) —", json.loads(prov_path.read_text())["training_fingerprint"])
else:
    prov = {"training_fingerprint": training_fingerprint(cfg["game"]),
            "config_fingerprint": config_fingerprint(cfg), "game": cfg["game"], "request": cfg}
    prov_path.write_text(json.dumps(prov, indent=1))
    print("provenance: stamped", prov["training_fingerprint"], prov["config_fingerprint"])
PY

# --repair deletes checkpoints whose metrics row is missing, and a LIVE trainer looks exactly like that for the
# few seconds between save_net and the metrics append. Repairing under a running trainer would make this script
# the data-loss cause it exists to prevent, so refuse rather than race it.
if pgrep -f "harness.scaled_run .*$NAME.json" > /dev/null; then
  echo "REFUSING: a trainer for $NAME is already running (pid $(pgrep -f "harness.scaled_run .*$NAME.json" | tr '\n' ' '))" >&2
  exit 1
fi

.venv/bin/python -m harness.resume "$D/$NAME" --repair || exit 1

echo "=== START $NAME $(date) ===" >> "$L"
nice -n 10 /usr/bin/time -p .venv/bin/python -m harness.scaled_run \
  --config-json "$CFG" --summary-out "$D/$NAME/summary.json" >> "$L" 2>&1
echo "=== END $NAME $(date) ===" >> "$L"
touch "$D/${NAME}_DONE"
