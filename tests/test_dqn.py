"""DQN building blocks, checked stage by stage (docs/dqn/plan.md)."""

import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from explore_bench.config import DATASETS, configuration, task_for
from explore_bench.dqn import action_cell, action_index, frontier_qmap, masked_argmax, qmap_cell, request
from explore_bench.session import Session, frontier_candidates
from explore_bench.worlds import TrainingWorld
from onpolicy.envs.GridEnv.communication import FREE, UNKNOWN


def make_spec(tmp_path, **extra):
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
    path.write_text(json.dumps(dict({"name": "test", "maps": dict(zip(("train", "validation", "test"), [[m] for m in maps])),
        "sensor_range": 0.2, "max_steps": 4, "training_seeds": [1, 2],
        "communication": {"mode": "perfect", "candidate_count": 4, "tile_size": 4},
        "training": {"num_envs": 1, "rollout_length": 4, "total_steps": 8, "checkpoint_interval": 4,
                     "ppo_epochs": 1}}, **extra)), encoding="utf-8")
    return configuration(path)


@pytest.fixture
def spec(tmp_path):
    return make_spec(tmp_path)


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


def test_dqn_key_is_optional_and_validated(tmp_path):
    assert "dqn" not in configuration()
    assert make_spec(tmp_path, dqn={})["dqn"] == {}
    for bad in ({"typo": 1}, []):
        with pytest.raises(ValueError):
            make_spec(tmp_path, dqn=bad)


@pytest.mark.parametrize("navigation", ["belief", "ground_truth"])
def test_frontier_targets_are_exactly_the_goals_route_can_move_to(tmp_path, navigation):
    session = Session(make_spec(tmp_path, dqn={}, navigation=navigation))
    session.reset_task(task_for(session.spec, session.spec["maps"]["train"][0], 100001, "fixture"))
    belief = np.full((12, 14), UNKNOWN, np.uint8)
    belief[1:4, 1:4] = FREE    # region A, holding the robot
    belief[4:10, 4:11] = FREE  # region C: touches A only at the corner (3,3)-(4,4)
    belief[1:3, 11:13] = FREE  # island B: cut off by unknown cells
    session.communication_broker.belief_maps[0] = belief
    session.agent_pos[0] = [2, 2]
    # A and C's frontier cells touch diagonally, so they form one frontier
    # group; C is larger, so that group's target cell lies in C.
    candidates = frontier_candidates(belief)
    assert [tuple(int(v) for v in cell) for cell in candidates] == [(1, 11), (4, 6)]  # B's, then A+C's (in C)
    oracle = [tuple(int(v) for v in cell) for cell in candidates if len(session.route(0, cell)) > 1]
    targets = session.frontier_targets(0)
    assert targets == oracle
    # Belief routing: A* may not cut the corner into C, and B is cut off, so
    # nothing is reachable. Ground-truth routing: the true map joins them all.
    assert len(targets) == (0 if navigation == "belief" else 2)


def test_qmap_holds_one_reachable_target_per_legal_cell(tmp_path):
    room = DATASETS + "room_1.pgm"
    spec = make_spec(tmp_path, dqn={}, sensor_range=3.5, max_steps=1000)
    session = Session(spec)
    actor, _, _ = session.reset_task(task_for(spec, room, 100001, "fixture"))
    assert actor["frontier_mask"].shape == (2, 64, 64)
    assert actor["frontier_goal"].shape == (2, 64, 64, 2)
    mask, goal = frontier_qmap(session)
    for agent in range(2):
        targets = session.frontier_targets(agent)
        assert targets
        legal = {tuple(cell) for cell in np.argwhere(mask[agent] == 1)}
        assert legal == {qmap_cell(target, session.gt_map.shape, 64) for target in targets}
        for row, column in legal:
            chosen = tuple(int(v) for v in goal[agent, row, column])
            assert chosen in targets and qmap_cell(chosen, session.gt_map.shape, 64) == (row, column)
        assert np.all(goal[agent][mask[agent] == 0] == -1)
    # MAPPO configurations keep their observation exactly as before.
    plain = Session(make_spec(tmp_path, sensor_range=3.5, max_steps=1000))
    actor, _, _ = plain.reset_task(task_for(plain.spec, room, 100001, "fixture"))
    assert "frontier_mask" not in actor and "frontier_goal" not in actor


def test_a_robot_with_no_reachable_frontier_can_only_stay(tmp_path):
    session = Session(make_spec(tmp_path, dqn={}))
    session.reset_task(task_for(session.spec, session.spec["maps"]["train"][0], 100001, "fixture"))
    belief = np.full((12, 14), UNKNOWN, np.uint8)
    belief[1:4, 1:4] = FREE    # robot 0's region has no frontier of its own...
    belief[4:10, 4:11] = FREE  # ...the nearby one is only reachable by cutting a corner
    session.communication_broker.belief_maps[0] = belief
    session.agent_pos[0] = [2, 2]
    assert session.frontier_targets(0) == []
    mask, goal = frontier_qmap(session)
    here = qmap_cell((2, 2), (12, 14), 64)
    assert mask[0].sum() == 1 and mask[0][here] == 1
    assert tuple(goal[0][here]) == (2, 2)
    observation = {"frontier_mask": mask, "frontier_goal": goal}
    actions = masked_argmax(np.random.default_rng(0).normal(size=(2, 64, 64)), mask)
    assert action_cell(actions[0], 64) == here
    np.testing.assert_array_equal(request(actions, observation)["motion_goals"][0], (2, 2))


def test_masked_argmax_and_request_only_use_legal_actions():
    mask = np.zeros((2, 64, 64), np.float32)
    goal = np.full((2, 64, 64, 2), -1, np.int64)
    mask[0, 3, 4] = mask[0, 10, 10] = mask[1, 0, 0] = 1
    goal[0, 3, 4], goal[0, 10, 10], goal[1, 0, 0] = (13, 17), (40, 41), (1, 2)
    q = np.zeros((2, 64, 64))
    q[0, 3, 4], q[0, 10, 10], q[0, 50, 50] = 1.0, 2.0, 99.0  # 99 is illegal
    q[1] = -5.0
    actions = masked_argmax(q, mask)
    assert [action_cell(a, 64) for a in actions] == [(10, 10), (0, 0)]
    observation = {"frontier_mask": mask, "frontier_goal": goal}
    np.testing.assert_array_equal(request(actions, observation)["motion_goals"], [(40, 41), (1, 2)])
    with pytest.raises(ValueError):
        request([action_index(50, 50, 64), actions[1]], observation)
    with pytest.raises(ValueError):
        masked_argmax(q, np.zeros_like(mask))


def test_legal_actions_drive_a_real_episode(tmp_path):
    spec = make_spec(tmp_path, dqn={}, sensor_range=3.5, max_steps=40)
    session = Session(spec)
    actor, _, _ = session.reset_task(task_for(spec, DATASETS + "room_1.pgm", 100001, "fixture"))
    rng = np.random.default_rng(0)
    start = session.measurement()["team_coverage_cells"]
    while not (session.terminated or session.truncated):
        before = [list(p) for p in session.agent_pos]
        actions = masked_argmax(rng.normal(size=(2, 64, 64)), actor["frontier_mask"])
        sent = request(actions, actor)
        (actor, _, _), _, _, _, _ = session.step_request(sent)
        np.testing.assert_array_equal(session.last_goals, sent["motion_goals"])  # executed = chosen
        for agent in range(2):  # one cell per step, toward the chosen frontier
            assert max(abs(a - b) for a, b in zip(session.agent_pos[agent], before[agent])) <= 1
    assert session.measurement()["team_coverage_cells"] > start
