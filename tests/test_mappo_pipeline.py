"""Research-contract tests, independent metric oracles and real PPO smoke runs."""

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from explore_bench.artifacts import file_hash
from explore_bench.config import configuration, evaluation_tasks, task_for
from explore_bench.controllers import RandomController, WaitController
from explore_bench.evaluation import evaluate, prepare, read_tasks, validate
from explore_bench.mappo import ActorController, arguments
from explore_bench.scoring import checked_rows, score, select
from explore_bench.session import Session
from explore_bench.training import train
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.shared_buffer import SharedReplayBuffer


@pytest.fixture
def spec(tmp_path):
    maps = []
    for index in range(3):
        grid = np.full((12, 14), 254, dtype=np.uint8)
        grid[0, :] = grid[-1, :] = 0
        grid[:, 0] = grid[:, -1] = 0
        grid[5, 3 + index] = 0
        path = tmp_path / ("map_{}.pgm".format(index))
        Image.fromarray(grid).save(str(path))
        maps.append(str(path))
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps({"name": "test", "maps": dict(zip(("train", "validation", "test"), [[name] for name in maps])),
        "sensor_range": 0.2, "max_steps": 4, "training_seeds": [1, 2],
        "communication": {"mode": "perfect", "candidate_count": 4, "tile_size": 4},
        "training": {"num_envs": 1, "rollout_length": 4, "total_steps": 8, "checkpoint_interval": 4,
                     "ppo_epochs": 1}}), encoding="utf-8")
    return configuration(path)


def start(spec):
    env = Session(spec)
    task = task_for(spec, spec["maps"]["train"][0], 100001, "fixture")
    env.reset_task(task)
    return env


def test_wait_has_no_reward_and_does_not_end_team(spec):
    env = start(spec)
    initial = env.measurement()["team_coverage_cells"]
    _, rewards, terminated, truncated, measured = env.step_request(
        {"motion_goals": np.asarray(env.agent_pos), "comm_actions": [0, 0]})
    assert measured["team_coverage_cells"] == initial
    np.testing.assert_array_equal(rewards, 0)
    assert not terminated and not truncated
    assert env.communication_broker.current_step == 1
    for _ in range(3):
        _, rewards, terminated, truncated, _ = env.step_request(
            {"motion_goals": env.agent_pos, "comm_actions": [0, 0]})
    assert truncated and not terminated
    assert env.task_reward_total == 0


def test_moving_teammate_continues_and_distance_matches_path(spec):
    env = start(spec)
    # Give robot 1 a known legal adjacent cell without consulting hidden truth.
    belief = env.communication_broker.belief_maps[1]
    row, column = env.agent_pos[1]
    adjacent = [(row + dr, column + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if belief[row + dr, column + dc] == 254]
    assert adjacent
    stationary = list(env.agent_pos[0])
    env.step_request({"motion_goals": [stationary, adjacent[0]], "comm_actions": [0, 0]})
    assert env.agent_pos[0] == stationary
    assert tuple(env.agent_pos[1]) == adjacent[0]
    assert not env.terminated
    assert env.path_lengths_m[0] == 0
    assert env.path_lengths_m[1] == pytest.approx(spec["resolution"])


def test_shared_observation_identity_history_and_privilege_boundary(spec):
    env = start(spec)
    observation = env.observation()
    np.testing.assert_array_equal(observation[0]["vector"], np.eye(2))
    before = copy.deepcopy(observation[0])
    # Changing a cell that the robot has not sensed does not change its input.
    unknown = np.argwhere(env.built_map[0] == 205)[0]
    env.gt_map[tuple(unknown)] = 0
    after = env.observation()[0]
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert "agent_beliefs" not in after
    assert "agent_beliefs" in observation[1]


def test_sensing_origin_is_the_recorded_grid_pose(spec, monkeypatch):
    env = start(spec)
    seen = []
    original = env.sensors[0].sense

    def capture(position, yaw, grid, resolution):
        seen.append(list(position))
        return original(position, yaw, grid, resolution)

    monkeypatch.setattr(env.sensors[0], "sense", capture)
    env._sense_agent(0, [3, 8], 0.0)
    assert seen == [[3, 8]]


def test_physical_coverage_does_not_increase_from_belief_delivery(spec):
    env = start(spec)
    before = env.measurement()
    env.communication_broker.belief_maps[0][:] = 254
    after = env.measurement()
    assert after["team_coverage_cells"] == before["team_coverage_cells"]
    assert after["overlap_cells"] == before["overlap_cells"]


def test_measurement_matches_hand_counted_union_and_overlap():
    env = object.__new__(Session)
    env.num_agents = 2
    env.resolution = 0.5
    env.gt_map = np.asarray([[254, 254, 205], [254, 0, 254]])
    env.built_map = [np.asarray([[254, 254, 205], [205, 205, 205]]),
                     np.asarray([[205, 254, 205], [254, 0, 205]])]
    env.path_lengths_m = np.asarray([0.5, 1.0])
    from types import SimpleNamespace
    env.communication_broker = SimpleNamespace(metrics=lambda: {})
    measured = env.measurement()
    assert measured["total_explorable_cells"] == 5
    assert measured["team_coverage_cells"] == 4
    assert measured["team_coverage_ratio"] == pytest.approx(0.8)
    assert measured["overlap_cells"] == 1
    assert measured["coverage_std_area_m2"] == pytest.approx(0.125)
    assert measured["team_path_length_m"] == pytest.approx(1.5)


def test_success_bonus_is_only_for_a_real_coverage_crossing(spec):
    env = start(spec)
    initial = env.measurement()
    old_union = np.any(np.stack([grid != 205 for grid in env.built_map]), axis=0)
    row, column = env.agent_pos[1]
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        goal = (row + dr, column + dc)
        if env.communication_broker.belief_maps[1][goal] != 254:
            continue
        yaw = env.naive_local_planner([tuple(env.agent_pos[1]), goal])[-1][1]
        frame = env._sense_agent(1, goal, yaw)
        new_union = old_union | (frame.full_map != 205)
        expected_new = int((new_union & (env.gt_map != 205)).sum()) - initial["team_coverage_cells"]
        if expected_new:
            break
    else:
        pytest.fail("fixture has no revealing adjacent move")
    env.spec["target_coverage"] = (initial["team_coverage_cells"] + expected_new / 2) / initial["total_explorable_cells"]
    _, rewards, terminated, truncated, _ = env.step_request({"motion_goals": [env.agent_pos[0], goal], "comm_actions": [0, 0]})
    assert terminated and not truncated
    np.testing.assert_allclose(rewards, spec["reward_new_cell"] * expected_new + spec["reward_success"])
    with pytest.raises(ValueError, match="finished"):
        env.step_request({"motion_goals": env.agent_pos, "comm_actions": [0, 0]})


def test_bootstrap_distinguishes_terminal_and_external_cap(spec):
    # An external cutoff with reward 2 and V(final)=10 has target 11;
    # a genuine terminal transition has target 2. No reset-state value enters.
    import gym
    args = arguments(spec)
    args.episode_length = 1
    args.use_valuenorm = False
    args.gamma = 0.9
    env = Session(spec)
    buffer = SharedReplayBuffer(args, 2, env.observation_space[0], env.share_observation_space[0],
                                gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32))
    buffer.masks[1] = 0
    buffer.rewards[0, 0, 0, 0] = 2 + args.gamma * 10
    buffer.rewards[0, 0, 1, 0] = 2
    buffer.compute_returns(np.full((1, 2, 1), 999, dtype=np.float32))
    np.testing.assert_allclose(buffer.returns[0, 0, :, 0], [11, 2])


def test_box_action_joint_probability_has_one_storage_column(spec):
    import gym
    args = arguments(spec)
    env = Session(spec)
    space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
    buffer = SharedReplayBuffer(args, 2, env.observation_space[0], env.share_observation_space[0], space)
    assert buffer.action_log_probs.shape[-1] == 1
    layer = ACTLayer(space, 4, True, 0.01)
    features = torch.zeros(2, 4)
    action, old_log = layer(features)
    new_log, _ = layer.evaluate_actions(features, action)
    np.testing.assert_allclose(torch.exp(new_log - old_log).detach().numpy(), 1)
    # Two independent standard normals at zero: log density = -log(2*pi).
    zero_log, _ = layer.evaluate_actions(features, torch.zeros(2, 2))
    np.testing.assert_allclose(zero_log.detach().numpy(), -np.log(2 * np.pi), rtol=1e-6)
    advantages = torch.tensor([[2.0], [-1.0]])
    ratio = torch.exp(new_log - old_log)
    objective = -torch.min(ratio * advantages, ratio.clamp(0.9, 1.1) * advantages).mean()
    assert objective.item() == pytest.approx(-0.5)


def test_task_table_and_recording_are_complete_and_detect_tampering(spec, tmp_path):
    path = tmp_path / "tasks.json"
    prepare(spec, path)
    tasks = read_tasks(path, spec, "familiar")
    assert len(tasks) == 10
    output = tmp_path / "evaluation"
    evaluate(spec, tasks, output, WaitController())
    _, episodes, steps = checked_rows(output)
    assert len(episodes) == 10
    assert all(float(row["task_reward"]) == 0 for row in episodes)
    assert all(len(rows) == 5 for rows in steps.values())
    scored = score([output], tmp_path / "scores")
    assert all(row["training_seeds"] == 0 for row in scored)
    assert all(row["ci95_low"] == "" for row in scored)
    with pytest.raises(FileExistsError):
        evaluate(spec, tasks, output, WaitController())
    table = json.loads(path.read_text())
    table["tasks"]["familiar"][0]["seed"] += 1
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_tasks(path, spec, "familiar")


def test_real_training_resume_validation_selection_and_fresh_actor(spec, tmp_path):
    uninterrupted = tmp_path / "full"
    train(spec, uninterrupted, 1, allow_dirty=True)
    partial = tmp_path / "partial"
    record = train(spec, partial, 1, stop_after_updates=1, allow_dirty=True)
    assert record["status"] == "interrupted"
    checkpoint = partial / "checkpoints/step_000000000004"
    resumed = tmp_path / "resumed"
    train(spec, resumed, 1, resume=checkpoint, allow_dirty=True)
    final = resumed / "checkpoints/step_000000000008"
    reference = uninterrupted / "checkpoints/step_000000000008"
    full_weights = torch.load(reference / "actor.pt", weights_only=True)
    resumed_weights = torch.load(final / "actor.pt", weights_only=True)
    for key in full_weights:
        torch.testing.assert_close(full_weights[key], resumed_weights[key], rtol=0, atol=0)
    payload = torch.load(final / "training.pt", weights_only=False)
    assert set(payload["value_normalizer"]) == {"running_mean", "running_mean_sq", "debiasing_term"}
    assert payload["actor_optimizer"]["state"]
    task_file = tmp_path / "tasks.json"
    prepare(spec, task_file)
    validation = tmp_path / "validation"
    validate(resumed, task_file, validation)
    selection = tmp_path / "selection.json"
    choice = select(validation, selection)
    assert len(json.loads(selection.read_text())["candidates"]) == 3
    controller = ActorController(choice["checkpoint"], spec)
    before = {key: value.clone() for key, value in controller.actor.state_dict().items()}
    tasks = evaluation_tasks(spec, "unseen")
    evaluate(spec, tasks, tmp_path / "test", controller, 1, controller.checkpoint_hash)
    checked_rows(tmp_path / "test")
    for key, value in controller.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    # A fresh process loads only R_Actor; no R_MAPPO trainer is imported.
    process = subprocess.run([sys.executable, "-c",
        "import json,sys; from explore_bench.mappo import ActorController; "
        "spec=json.load(open(sys.argv[2]))['configuration']; "
        "c=ActorController(sys.argv[1],spec); "
        "assert 'onpolicy.algorithms.r_mappo.r_mappo' not in sys.modules; print(c.checkpoint_hash)",
        choice["checkpoint"], str(selection)], check=True, capture_output=True, text=True)
    assert controller.checkpoint_hash in process.stdout


def test_process_workers_train_and_snapshot(spec, tmp_path):
    spec["training"].update(num_envs=2, total_steps=8, checkpoint_interval=8)
    result = train(spec, tmp_path / "process", 1, allow_dirty=True)
    assert result["environment_steps"] == 8
    state = torch.load(tmp_path / "process/checkpoints/step_000000000008/training.pt", weights_only=False)
    assert len(state["worlds"]) == 2
    assert state["worlds"][0].session is not state["worlds"][1].session
