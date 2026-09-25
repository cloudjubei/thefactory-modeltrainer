"""Stored evidence: how a run's results are written, read, and pinned.

Evidence files are the data every registered claim and pinned proof reads. They are too large for git (§C.46's nine
arms were 32 MB pretty-printed), so the evidence directory is gitignored and lives on the machine that produced it.
Leaving git loses the one thing git gave for free: a record of WHAT the data was, so a file regenerated or edited
under a frozen verdict would change the verdict's meaning silently. The manifest restores that record without the
bytes: a directory holding `manifest.json` is MANAGED, and the manifest (the one tracked file) pins the content hash
of every evidence file in it.

  E1 ONE FORMAT. Evidence is minified JSON, gzip-compressed with a zeroed header timestamp, so the same content
     always produces the same bytes. Any other suffix is refused rather than read some other way.
  E2 A MANAGED FILE IS READ ONLY IF IT IS THE CONTENT THE MANIFEST PINS. A file missing from the manifest, or whose
     content hash differs, is refused — a restored backup is proved to be the data the verdicts were drawn from.
  E3 A MANAGED FILE IS WRITTEN ONCE. Replacing one needs the hash of the version being replaced, so evidence is
     never overwritten by accident, and every intended replacement shows as a changed hash in the tracked manifest.
     The manifest is updated under an exclusive lock, because arms of one experiment run concurrently."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

SUFFIX = ".json.gz"
MANIFEST = "manifest.json"


def _encode(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def evidence_path(path) -> Path:
    """`path` as a Path, refused unless it is named as stored evidence (E1)."""
    path = Path(path)
    if not path.name.endswith(SUFFIX) or path.name == SUFFIX:
        raise ValueError(f"{str(path)!r}: evidence is stored as gzip JSON and must be named *{SUFFIX}")
    return path


def _manifest_path(path: Path) -> Path:
    return path.parent / MANIFEST


def _read_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text())["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"{str(path)!r} is not a readable evidence manifest: {e}") from e


@contextmanager
def _locked(manifest: Path):
    with open(manifest.with_name(manifest.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def manifest_entry(path) -> dict | None:
    """The manifest's record for a managed evidence file ({sha256, started}), or None when its directory is not
    managed or the manifest does not list it."""
    path = evidence_path(path)
    manifest = _manifest_path(path)
    if not manifest.exists():
        return None
    return _read_manifest(manifest).get(path.name)


def save_evidence(path, obj, replaces: str | None = None) -> str:
    """Write `obj` as evidence and return its content hash. In a managed directory the file is added to the
    manifest; a file the manifest already lists is replaced only when `replaces` is the hash it records (E3)."""
    path = evidence_path(path)
    raw = _encode(obj)
    sha = _sha(raw)
    blob = gzip.compress(raw, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_path(path)
    if not manifest.exists():
        _atomic_write(path, blob)
        return sha
    with _locked(manifest):
        files = _read_manifest(manifest)
        recorded = files.get(path.name)
        if recorded is None and replaces is not None:
            raise ValueError(f"{path.name}: `replaces` names a version, but the manifest records none")
        if recorded is not None and replaces != recorded["sha256"]:
            raise ValueError(f"{path.name} is already recorded ({recorded['sha256'][:12]}) — evidence is written "
                             f"once; pass the recorded hash as `replaces` to replace that exact version")
        if recorded is None and path.exists():
            raise ValueError(f"{path.name} exists but the manifest does not list it — refusing to overwrite data "
                             f"of unknown content")
        entry = {"sha256": sha}
        if isinstance(obj, dict) and obj.get("started"):
            entry["started"] = obj["started"]
        _atomic_write(path, blob)
        files[path.name] = entry
        _atomic_write(manifest, json.dumps({"files": dict(sorted(files.items()))}, indent=1).encode())
    return sha


def load_evidence(path):
    """Read evidence. A file in a managed directory is returned only if the manifest lists it and its content hash
    matches (E2)."""
    path = evidence_path(path)
    try:
        raw = gzip.decompress(path.read_bytes())
    except (OSError, EOFError, gzip.BadGzipFile) as e:
        raise ValueError(f"cannot read evidence {str(path)!r}: {e}") from e
    manifest = _manifest_path(path)
    if manifest.exists():
        recorded = _read_manifest(manifest).get(path.name)
        if recorded is None:
            raise ValueError(f"{path.name} is not in {manifest} — evidence of unrecorded content is not read")
        if _sha(raw) != recorded["sha256"]:
            raise ValueError(f"{path.name} does not match its manifest hash — it is not the data the recorded "
                             f"verdicts were drawn from")
    return json.loads(raw)
