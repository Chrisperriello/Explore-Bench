"""Geometry and lifecycle checks for the temporary HouseExpo preview."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preview_houseexpo_maps.py"
spec = importlib.util.spec_from_file_location("houseexpo_preview", SCRIPT)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def rectangle(width=15, height=10):
    return {"id": "rectangle", "room_num": 1,
            "verts": [[0, 0], [width, 0], [width, height], [0, height]]}


def doorway(width):
    low, high = 5 - width / 2, 5 + width / 2
    return {"room_num": 2, "verts": [[0, 0], [7, 0], [7, low], [8, low],
            [8, 0], [15, 0], [15, 10], [8, 10], [8, high], [7, high], [7, 10], [0, 10]]}


def test_meter_scale_translation_and_enclosure():
    house = rectangle()
    grid = preview.rasterize(house)
    shifted = dict(house, verts=(np.asarray(house["verts"]) + [10000, -10000]).tolist())
    np.testing.assert_array_equal(preview.rasterize(shifted), grid)
    rows, columns = np.nonzero(grid == preview.FREE)
    assert rows.ptp() + 1 == 100  # 10 m, not scaled to fit the 25 m canvas.
    assert columns.ptp() + 1 == 150
    assert grid.dtype == np.uint8 and grid.shape == (250, 250)
    assert set(np.unique(grid)) == {0, 205, 254}
    assert (grid[[0, -1]] == preview.OUTSIDE).all()
    assert (grid[:, [0, -1]] == preview.OUTSIDE).all()


def test_thin_wall_survives_downsampling():
    # A 2 cm slit from the lower boundary leaves a 1 m doorway at the top.
    house = {"verts": [[0, 0], [7, 0], [7, 9], [7.02, 9], [7.02, 0],
                       [15, 0], [15, 10], [0, 10]]}
    free = preview.rasterize(house) == preview.FREE
    middle = free[125].astype(int)
    assert np.count_nonzero(np.diff(np.pad(middle, 1)) == 1) == 2
    assert preview.component_count(free) == 1


@pytest.mark.parametrize("verts", [[], [[0, 0], [1, 1]], [[0, 0], [1, 1], [np.nan, 2]],
                                  [[0, 0], [0, 1], [0, 2]]])
def test_invalid_geometry_is_rejected(verts):
    with pytest.raises(ValueError):
        preview.rasterize({"verts": verts})


def test_oversized_home_is_rejected_without_resizing():
    with pytest.raises(ValueError, match="does not fit"):
        preview.rasterize(rectangle(width=30))


@pytest.mark.parametrize("door_width, expected", [(0.4, False), (1.2, True)])
def test_clearance_distinguishes_connected_narrow_door(tmp_path, door_width, expected):
    path = tmp_path / "door.pgm"
    Image.fromarray(preview.rasterize(doorway(door_width))).save(path)
    checks, _ = preview.validate(path, 1, smoke_steps=0)
    verdicts = {name: passed for name, passed, _ in checks}
    assert verdicts["connected"]
    assert verdicts["clearance filter"] == expected


def test_directory_and_archive_sampling_match_without_extracting(tmp_path):
    source = tmp_path / "json"
    source.mkdir()
    archive_path = tmp_path / "json.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for index in range(8):
            name = "house_{}.json".format(index)
            payload = json.dumps(rectangle(width=12 + index)).encode()
            (source / name).write_bytes(payload)
            # Even a path-traversal member is only read; no archive is extracted.
            member = tarfile.TarInfo("../outside/" + name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    from_dir = list(preview.candidates(source, seed=12, limit=5))
    from_archive = list(preview.candidates(archive_path, seed=12, limit=5))
    assert from_dir == from_archive
    assert len(from_dir) == 5
    assert not (tmp_path.parent / "outside").exists()


@pytest.mark.parametrize("display_error", [False, True])
def test_real_benchmark_smoke_and_cleanup(tmp_path, monkeypatch, display_error):
    source = tmp_path / "json"
    source.mkdir()
    (source / "one.json").write_text(json.dumps(rectangle()))
    real_temporary_directory = preview.tempfile.TemporaryDirectory
    directories, shown = [], []

    def temporary_directory(prefix):
        context = real_temporary_directory(prefix=prefix, dir=tmp_path)
        directories.append(Path(context.name))
        return context

    def show(results, attempted, rejected, output):
        shown.extend(results)
        assert attempted == 1 and rejected == 0
        assert not list(directories[0].glob("*.pgm"))
        if display_error:
            raise RuntimeError("window failed")

    monkeypatch.setattr(preview.tempfile, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(preview, "show", show)
    args = ["--source", str(source), "--count", "1", "--smoke-steps", "3"]
    if display_error:
        with pytest.raises(RuntimeError, match="window failed"):
            preview.main(args)
    else:
        assert preview.main(args) == 0
    assert len(shown) == 1
    assert all(passed for _, passed, _ in shown[0]["checks"])
    assert "Cost smoke run" in [name for name, _, _ in shown[0]["checks"]]
    assert not directories[0].exists()


def test_incomplete_selection_returns_failure(tmp_path, monkeypatch, capsys):
    (tmp_path / "too_big.json").write_text(json.dumps(rectangle(width=30)))
    monkeypatch.setattr(preview, "show", lambda *args: pytest.fail("no valid maps to show"))
    assert preview.main(["--source", str(tmp_path), "--count", "1"]) == 1
    output = capsys.readouterr().out
    assert "REJECT too_big" in output
    assert "Accepted 0, rejected 1, checked 1" in output
