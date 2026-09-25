"""Direct tests for harness/evidence.py — how stored evidence is written, read and pinned once it no longer lives
in git. The files leave the repository; the tracked manifest keeps the record of what they contain, so a restored
or regenerated file is read only if it is the data the verdicts were drawn from."""
from __future__ import annotations

import gzip
import json
import math
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from harness.evidence import MANIFEST, load_evidence, manifest_entry, save_evidence

SHAPES = (
    {"started": "2026-09-24T06:16:53+00:00", "seeds": [{"seed": 41, "passes": [[1, 2, 3]]}]},
    [0.1, 1e-300, -2.5, "é", None, True],
    {"nested": {"deep": [[[{}]]]}, "empty": []},
)


def _managed(tmp_path) -> Path:
    d = tmp_path / "evidence"
    d.mkdir()
    (d / MANIFEST).write_text(json.dumps({"files": {}}))
    return d


def _manifest(d: Path) -> dict:
    return json.loads((d / MANIFEST).read_text())["files"]


@pytest.mark.parametrize("obj", SHAPES)
def test_evidence_round_trips_exactly_whatever_its_shape(tmp_path, obj):
    save_evidence(tmp_path / "a.json.gz", obj)
    assert load_evidence(tmp_path / "a.json.gz") == obj


def test_non_finite_values_round_trip(tmp_path):
    save_evidence(tmp_path / "a.json.gz", {"x": [math.nan, math.inf]})
    x = load_evidence(tmp_path / "a.json.gz")["x"]
    assert math.isnan(x[0]) and x[1] == math.inf


def test_evidence_is_minified_gzip_and_the_same_content_gives_the_same_bytes_at_any_time(tmp_path, monkeypatch):
    import time

    obj = SHAPES[0]
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    save_evidence(tmp_path / "one" / "a.json.gz", obj)
    monkeypatch.setattr(time, "time", lambda: 86400.0 * 365 * 20)
    save_evidence(tmp_path / "two" / "a.json.gz", obj)
    one, two = (tmp_path / "one" / "a.json.gz").read_bytes(), (tmp_path / "two" / "a.json.gz").read_bytes()
    assert one == two
    raw = gzip.decompress(one)
    assert raw == json.dumps(obj, separators=(",", ":")).encode()


@pytest.mark.parametrize("name", ["a.json", "a.gz", ".json.gz", "a.json.gz.bak", "a.jsongz"])
def test_any_other_name_is_refused_for_writing_and_reading(tmp_path, name):
    with pytest.raises(ValueError, match="gzip JSON"):
        save_evidence(tmp_path / name, {"x": 1})
    with pytest.raises(ValueError, match="gzip JSON"):
        load_evidence(tmp_path / name)


def test_a_corrupt_file_is_refused_with_its_path(tmp_path):
    (tmp_path / "a.json.gz").write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="a.json.gz"):
        load_evidence(tmp_path / "a.json.gz")


@pytest.mark.parametrize("obj", SHAPES)
def test_saving_into_a_managed_directory_records_the_content_hash(tmp_path, obj):
    d = _managed(tmp_path)
    sha = save_evidence(d / "a.json.gz", obj)
    entry = _manifest(d)["a.json.gz"]
    assert entry["sha256"] == sha and manifest_entry(d / "a.json.gz") == entry
    assert entry.get("started") == (obj.get("started") if isinstance(obj, dict) else None)
    assert load_evidence(d / "a.json.gz") == obj


def test_a_managed_file_whose_content_changed_is_refused(tmp_path):
    d = _managed(tmp_path)
    save_evidence(d / "a.json.gz", {"seeds": [1, 2]})
    (d / "a.json.gz").write_bytes(gzip.compress(json.dumps({"seeds": [1, 3]}).encode()))
    with pytest.raises(ValueError, match="does not match its manifest hash"):
        load_evidence(d / "a.json.gz")


def test_a_managed_file_the_manifest_does_not_list_is_refused(tmp_path):
    d = _managed(tmp_path)
    save_evidence(tmp_path / "loose.json.gz", {"x": 1})
    shutil.copy(tmp_path / "loose.json.gz", d / "b.json.gz")
    with pytest.raises(ValueError, match="not in"):
        load_evidence(d / "b.json.gz")
    assert manifest_entry(d / "b.json.gz") is None


def test_a_managed_file_is_written_once_and_replaced_only_by_naming_the_version_replaced(tmp_path):
    d = _managed(tmp_path)
    first = save_evidence(d / "a.json.gz", {"v": 1})
    with pytest.raises(ValueError, match="written once"):
        save_evidence(d / "a.json.gz", {"v": 2})
    with pytest.raises(ValueError, match="written once"):
        save_evidence(d / "a.json.gz", {"v": 2}, replaces="0" * 64)
    assert load_evidence(d / "a.json.gz") == {"v": 1} and _manifest(d)["a.json.gz"]["sha256"] == first
    second = save_evidence(d / "a.json.gz", {"v": 2}, replaces=first)
    assert second != first and load_evidence(d / "a.json.gz") == {"v": 2}
    assert _manifest(d)["a.json.gz"]["sha256"] == second


def test_replacing_a_version_the_manifest_never_recorded_is_refused(tmp_path):
    d = _managed(tmp_path)
    with pytest.raises(ValueError, match="records none"):
        save_evidence(d / "a.json.gz", {"v": 1}, replaces="0" * 64)
    assert not (d / "a.json.gz").exists() and _manifest(d) == {}


def test_an_existing_unlisted_file_is_never_overwritten(tmp_path):
    d = _managed(tmp_path)
    (d / "a.json.gz").write_bytes(b"precious")
    with pytest.raises(ValueError, match="unknown content"):
        save_evidence(d / "a.json.gz", {"v": 1})
    assert (d / "a.json.gz").read_bytes() == b"precious"


def test_an_unmanaged_directory_has_no_manifest_rules(tmp_path):
    save_evidence(tmp_path / "a.json.gz", {"v": 1})
    save_evidence(tmp_path / "a.json.gz", {"v": 2})
    assert load_evidence(tmp_path / "a.json.gz") == {"v": 2}
    assert manifest_entry(tmp_path / "a.json.gz") is None and not (tmp_path / MANIFEST).exists()


def test_an_unreadable_manifest_is_refused_not_treated_as_empty(tmp_path):
    d = tmp_path / "evidence"
    d.mkdir()
    (d / MANIFEST).write_text("{}")
    with pytest.raises(ValueError, match="not a readable evidence manifest"):
        save_evidence(d / "a.json.gz", {"v": 1})


def _save_one(job):
    d, i = job
    return save_evidence(Path(d) / f"arm{i}.json.gz", {"arm": i, "rows": list(range(2000))})


def test_concurrent_writers_into_one_managed_directory_lose_no_manifest_entry(tmp_path):
    d = _managed(tmp_path)
    with ProcessPoolExecutor(max_workers=8) as ex:
        shas = list(ex.map(_save_one, [(str(d), i) for i in range(24)]))
    files = _manifest(d)
    assert sorted(files) == sorted(f"arm{i}.json.gz" for i in range(24))
    assert [files[f"arm{i}.json.gz"]["sha256"] for i in range(24)] == shas


REPO_EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"


def test_the_repository_evidence_directory_is_managed_and_every_file_present_is_the_recorded_content():
    files = json.loads((REPO_EVIDENCE / MANIFEST).read_text())["files"]
    assert files, "the manifest records no evidence"
    present = sorted(p.name for p in REPO_EVIDENCE.iterdir() if p.name.endswith(".json.gz"))
    stray = sorted(p.name for p in REPO_EVIDENCE.iterdir()
                   if not p.name.endswith(".json.gz") and p.name not in (MANIFEST, MANIFEST + ".lock"))
    assert not stray, f"evidence/ holds files that are not managed evidence: {stray}"
    assert set(present) <= set(files), f"unrecorded evidence: {sorted(set(present) - set(files))}"
    for name in present:
        load_evidence(REPO_EVIDENCE / name)


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=REPO_EVIDENCE.parent, capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("git is not installed")
    if r.returncode != 0 and "not a git repository" in r.stderr:
        pytest.skip("not a git checkout")
    return r


def test_git_tracks_no_evidence_data_and_never_ignores_the_manifest():
    tracked = _git("ls-files", "--", "evidence").stdout.split()
    assert set(tracked) <= {f"evidence/{MANIFEST}"}, f"evidence data is tracked by git: {tracked}"
    assert _git("check-ignore", "-q", "evidence/any_new_arm.json.gz").returncode == 0
    assert _git("check-ignore", "-q", f"evidence/{MANIFEST}").returncode == 1


@pytest.mark.parametrize("module", ["tests.test_ceiling_evidence", "tests.test_c45_evidence",
                                    "tests.test_c46_evidence", "tests.test_c47_floor_evidence",
                                    "tests.test_c47_stage0_evidence", "tests.test_c47_stage0v2_evidence"])
def test_the_evidence_proofs_skip_only_when_their_recorded_files_are_absent(module):
    """The pinned proofs skip on a checkout without the evidence. A skip condition that names the wrong path skips
    them EVERYWHERE, silently — so each module's files must be ones the manifest records, and on a machine holding
    them the proofs must run."""
    import importlib

    mod = importlib.import_module(module)
    recorded = json.loads((REPO_EVIDENCE / MANIFEST).read_text())["files"]
    assert mod.FILES and set(mod.FILES) <= set(recorded)
    present = all((REPO_EVIDENCE / f).exists() for f in mod.FILES)
    assert mod.pytestmark.args[0] is (not present)
    if not present:
        pytest.skip("the evidence is not on this machine")
