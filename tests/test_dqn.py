"""DQN building blocks, checked stage by stage (docs/dqn/plan.md)."""

import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from explore_bench.config import configuration
from explore_bench.dqn import action_cell, action_index, qmap_cell
from explore_bench.worlds import TrainingWorld


@pytest.fixture
def spec(tmp_path):
    """Three tiny walled rooms, as in tests/test_mappo_pipeline.py."""
    maps = []
    for index in range(3):
        grid = np.full((12, 14), 254, dtype=np.uint8)
        grid[0, :] = grid[-1, :] = 0
        grid[:, 0] = grid[:, -1] = 0
        grid[5, 3 + index] = 0
        path = tmp_path / "map_{}.pgm".format(index)
        Image.fromarray(grid).save(str(path))
        maps.append(str(path))
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps({"name": "test", "maps": dict(zip(("train", "validation", "test"), [[m] for m in maps])),
        "sensor_range": 0.2, "max_steps": 4, "training_seeds": [1, 2],
        "communication": {"mode": "perfect", "candidate_count": 4, "tile_size": 4},
        "training": {"num_envs": 1, "rollout_length": 4, "total_steps": 8, "checkpoint_interval": 4,
                     "ppo_epochs": 1}}), encoding="utf-8")
    return configuration(path)


def test_action_index_round_trips():
    for action in range(64 * 64):
        assert action_index(*action_cell(action, 64), 64) == action
    with pytest.raises(ValueError):
        action_cell(64 * 64, 64)


def test_every_map_cell_lands_in_one_qmap_cell():
    # 250 rows over 64 Q-map rows: each Q-map row covers 3 or 4 map rows,
    # and no Q-map row is empty.
    rows = Counter(qmap_cell((row, 0), (250, 250), 64)[0] for row in range(250))
    assert sorted(rows) == list(range(64))
    assert set(rows.values()) <= {3, 4}
    assert qmap_cell((249, 249), (250, 250), 64) == (63, 63)


def test_training_episodes_record_their_method(spec):
    world = TrainingWorld(spec, 1, 0, method="dqn")
    assert world.ledger.common["method"] == "dqn"
    # Worlds pickled before the method argument existed fall back to MAPPO.
    old = TrainingWorld(spec, 1, 0)
    del old.__dict__["method"]
    old.reset()
    assert old.ledger.common["method"] == "mappo"
