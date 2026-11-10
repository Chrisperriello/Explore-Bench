"""Shape tags on hand-built grids with known answers and on the benchmark maps."""

import numpy as np
import pytest

from explore_bench.config import DATASETS, ROOT
from explore_bench.map_tags import FREE, OUTSIDE, WALL, load, tags


def room(size=20, border=5, total=50):
    """A size x size free square with a 1-cell wall, surrounded by outside."""
    grid = np.full((total, total), OUTSIDE, dtype=np.uint8)
    grid[border - 1:border + size + 1, border - 1:border + size + 1] = WALL
    grid[border:border + size, border:border + size] = FREE
    return grid


def test_open_room():
    result = tags(room())
    assert result["free_area_m2"] == pytest.approx(4.0)
    assert result["free_components"] == 1
    assert result["loops"] == 0 and result["largest_island_m2"] == 0
    # Corner to corner on a 4-connected grid: 19 + 19 steps of 0.1 m.
    assert result["geodesic_diameter_m"] == pytest.approx(3.8)
    assert result["extent"] == "small"


def test_pillar_is_one_loop():
    grid = room()
    grid[13:17, 13:17] = WALL
    result = tags(grid)
    assert result["loops"] == 1
    assert result["largest_island_m2"] == pytest.approx(0.16)


def test_wall_from_the_outer_wall_is_not_a_loop():
    grid = room()
    grid[5:20, 15] = WALL
    assert tags(grid)["loops"] == 0


def test_diagonal_wall_cells_form_one_island():
    grid = room()
    for offset in range(4):
        grid[12 + offset, 12 + offset] = WALL
    assert tags(grid)["loops"] == 1


def test_narrow_corridor_is_entirely_narrow():
    grid = np.full((50, 50), WALL, dtype=np.uint8)
    grid[20:28, 5:45] = FREE
    result = tags(grid)
    assert result["narrow_share"] == 1.0
    assert tags(room(size=40, total=50))["narrow_share"] < 0.5


def test_separate_rooms_are_counted():
    grid = room()
    grid[5:25, 15] = WALL
    assert tags(grid)["free_components"] == 2


def test_rejects_unknown_values_and_empty_maps(tmp_path):
    from PIL import Image
    path = tmp_path / "bad.pgm"
    Image.fromarray(np.full((10, 10), 100, dtype=np.uint8)).save(path)
    with pytest.raises(ValueError):
        load(path)
    with pytest.raises(ValueError):
        tags(np.full((10, 10), WALL, dtype=np.uint8))


@pytest.mark.parametrize("name, loops", [
    ("square_loop", 1), ("loop_with_corridor_sym", 1), ("corridor_sym", 0),
    ("corridor_asym", 0), ("room_with_corridor", 0), ("room_1", 6)])
def test_benchmark_maps(name, loops):
    result = tags(load(ROOT / DATASETS / (name + ".pgm")))
    assert result["free_components"] == 1
    assert result["loops"] == loops
    assert result["extent"] == "full"
