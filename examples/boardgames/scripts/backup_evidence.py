"""Mirror the board-games evidence into the private backup repository (default: a clone of
thefactory-modeltrainer-evidence beside this repository), so the raw results do not live on one disk only.

What is mirrored, under <dest>/boardgames/:
  evidence/*.json.gz + evidence/manifest.json   every file checked against its manifest hash BEFORE it is copied, so
                                                a corrupted file is refused, never backed up
  books/c4_labels.json.gz                       the exact Connect-4 move values (days of solver time to rebuild)
  checkpoints/c49*/**.pt                        the saved Connect-4 nets the certificates refer to

Nothing is ever deleted from the backup. A file whose backed-up copy differs is replaced only when the local copy
passes its hash check. Without --push only the working copy of the backup repository changes; --push also records
and uploads it there.

    PYTHONPATH=. .venv/bin/python scripts/backup_evidence.py            # mirror and report
    PYTHONPATH=. .venv/bin/python scripts/backup_evidence.py --push     # mirror, record and upload
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT.parent.parent.parent / "thefactory-modeltrainer-evidence"


def _verified_evidence() -> list:
    manifest = json.loads((ROOT / "evidence" / "manifest.json").read_text())["files"]
    out, refused = [], []
    for path in sorted((ROOT / "evidence").glob("*.json.gz")):
        record = manifest.get(path.name)
        if record is None or hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() != record["sha256"]:
            refused.append(path.name)
        else:
            out.append(path)
    if refused:
        print(f"REFUSED (not the recorded content): {refused}")
    return out + [ROOT / "evidence" / "manifest.json"]


def _sources() -> list:
    files = _verified_evidence()
    labels = ROOT / "books" / "c4_labels.json.gz"
    if labels.exists():
        files.append(labels)
    files += sorted(p for d in (ROOT / "checkpoints").glob("c49*") for p in d.rglob("*.pt"))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest)
    if not (dest / ".git").exists():
        raise SystemExit(f"{dest} is not a clone of the backup repository")
    copied = 0
    for src in _sources():
        target = dest / "boardgames" / src.relative_to(ROOT)
        if target.exists() and target.read_bytes() == src.read_bytes():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied += 1
    print(f"{copied} file(s) mirrored into {dest / 'boardgames'}")
    if args.push:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        subprocess.run(["git", "add", "-A"], cwd=dest, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=dest).returncode != 0:
            subprocess.run(["git", "commit", "-m", f"boardgames evidence {stamp}"], cwd=dest, check=True)
        subprocess.run(["git", "push"], cwd=dest, check=True)


if __name__ == "__main__":
    main()
