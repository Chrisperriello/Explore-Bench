"""Freezing room-map sets: manifest contents, write-once, cleanup and checking."""

import importlib.util
import json
from pathlib import Path

import pytest

from explore_bench.config import configuration

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_map_set.py"
spec = importlib.util.spec_from_file_location("freeze_map_set", SCRIPT)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


def run(root, *extra):
    return freeze.main(["--name", "unit", "--version", "1", "--root", str(root), "--allow-dirty",
                        "--smoke-steps", "0", *extra])


def test_freezes_splits_with_manifest(tmp_path):
    assert run(tmp_path, "--splits", "train=2", "test=2", "--extra-doors", "0", "2") == 0
    directory = tmp_path / "unit_v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert [item["split"] for item in manifest["maps"]] == ["train", "train", "test", "test"]
    # Extra doors cycle across all maps, so each split gets both layouts.
    assert [item["extra_doors_requested"] for item in manifest["maps"]] == [0, 2, 0, 2]
    assert len({item["sha256"] for item in manifest["maps"]}) == 4
    assert set(manifest["maps"][0]["tags"]) >= {"loops", "free_area_m2", "narrow_share"}
    assert manifest["grid"] == {"resolution_m": 0.1, "free": 254, "wall": 0, "outside": 205}
    assert freeze.check(directory)
    assert not list(tmp_path.glob(".unit_v1.*"))


def test_manifest_paths_load_as_experiment_maps(tmp_path):
    run(tmp_path, "--splits", "test=2")
    manifest = json.loads((tmp_path / "unit_v1/manifest.json").read_text())
    supplied = configuration()["maps"]
    supplied["test"] = manifest["splits"]["test"]
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({"maps": supplied}))
    assert configuration(config)["maps"]["test"] == manifest["splits"]["test"]


def test_sets_are_write_once(tmp_path):
    run(tmp_path, "--splits", "test=1")
    with pytest.raises(SystemExit, match="write-once"):
        run(tmp_path, "--splits", "test=1")


def test_failure_leaves_nothing_behind(tmp_path, monkeypatch):
    def broken(*args):
        raise RuntimeError("validation crashed")
    monkeypatch.setattr(freeze.rooms, "validate", broken)
    with pytest.raises(RuntimeError):
        run(tmp_path, "--splits", "test=1")
    assert list(tmp_path.iterdir()) == []


def test_rejected_seeds_are_recorded(tmp_path, monkeypatch):
    real = freeze.rooms.validate

    def reject_first(path, seed, smoke_steps):
        checks = real(path, seed, smoke_steps)
        return checks + [("unit rejection", seed != 1, "")]
    monkeypatch.setattr(freeze.rooms, "validate", reject_first)
    run(tmp_path, "--splits", "test=1")
    manifest = json.loads((tmp_path / "unit_v1/manifest.json").read_text())
    assert [(item["seed"], item["failed"]) for item in manifest["rejected"]] == [(1, ["unit rejection"])]
    assert manifest["maps"][0]["seed"] == 2 and manifest["attempts"] == 2
    assert not (tmp_path / "unit_v1/unit_v1_seed_1.pgm").exists()


def test_check_detects_changed_and_extra_files(tmp_path):
    run(tmp_path, "--splits", "test=1")
    directory = tmp_path / "unit_v1"
    image = next(directory.glob("*.pgm"))
    image.write_bytes(image.read_bytes()[:-1] + b"\x00")
    (directory / "stray.pgm").write_bytes(b"")
    assert not freeze.check(directory)


def test_rejects_bad_splits():
    with pytest.raises(ValueError):
        freeze.parse_splits(["test=0"])
    with pytest.raises(ValueError):
        freeze.parse_splits(["test=1", "test=2"])
    with pytest.raises(ValueError):
        freeze.parse_splits(["holdout=3"])


def test_courtyards_cycle_after_extra_doors(tmp_path):
    run(tmp_path, "--splits", "test=4", "--extra-doors", "0", "1", "--courtyards", "0", "1")
    maps = json.loads((tmp_path / "unit_v1/manifest.json").read_text())["maps"]
    assert [(item["extra_doors_requested"], item["courtyard"]) for item in maps] == [
        (0, False), (1, False), (0, True), (1, True)]
    # A courtyard is one large island, unlike loops round a wall piece.
    assert all(item["tags"]["largest_island_m2"] >= 16 for item in maps if item["courtyard"])
    assert all(item["tags"]["largest_island_m2"] < 16 for item in maps if not item["courtyard"])


def fake_archive(tmp_path, widths=(15, 14, 13)):
    import io
    import tarfile
    path = tmp_path / "json.tar.gz"
    with tarfile.open(str(path), "w:gz") as archive:
        for width in widths:
            data = json.dumps({"id": "house{}".format(width), "room_num": 1,
                               "verts": [[0, 0], [width, 0], [width, 10], [0, 10]]}).encode()
            member = tarfile.TarInfo("json/house{}.json".format(width))
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def test_houseexpo_set_commits_only_its_manifest_and_rebuilds(tmp_path, monkeypatch):
    archive = fake_archive(tmp_path)
    monkeypatch.setattr(freeze.houseexpo, "ARCHIVE_SHA256", freeze.sha256(archive))
    root = tmp_path / "sets"
    assert freeze.main(["--source", "houseexpo", "--name", "hx", "--version", "1", "--root", str(root),
                        "--allow-dirty", "--smoke-steps", "0", "--archive", str(archive),
                        "--splits", "test=2"]) == 0
    directory = root / "hx_v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["source"] == "houseexpo" and len(manifest["maps"]) == 2
    assert all(item["source_id"].startswith("house") for item in manifest["maps"])
    assert "*.pgm" in (directory / ".gitignore").read_text()
    # Another machine: no maps until rebuilt, then identical to the manifest.
    for path in directory.glob("*.pgm"):
        path.unlink()
    assert not freeze.check(directory)
    assert freeze.main(["--rebuild", str(directory), "--archive", str(archive)]) == 0
    assert freeze.check(directory)


def test_rebuild_refuses_a_different_archive(tmp_path, monkeypatch):
    archive = fake_archive(tmp_path)
    monkeypatch.setattr(freeze.houseexpo, "ARCHIVE_SHA256", freeze.sha256(archive))
    root = tmp_path / "sets"
    freeze.main(["--source", "houseexpo", "--name", "hx", "--version", "1", "--root", str(root),
                 "--allow-dirty", "--smoke-steps", "0", "--archive", str(archive), "--splits", "test=1"])
    (tmp_path / "other").mkdir()
    other = fake_archive(tmp_path / "other", widths=(12,))
    with pytest.raises(SystemExit, match="pinned"):
        freeze.main(["--rebuild", str(root / "hx_v1"), "--archive", str(other)])
