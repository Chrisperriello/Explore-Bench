import argparse
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GRID_SIMULATOR = REPOSITORY_ROOT / "grid_simulator"
sys.path.insert(0, str(GRID_SIMULATOR))

from collect_experiments import (  # noqa: E402
    CsvOutputs,
    HeadlessGridEnv,
    _communication_delta,
    _measurement,
    _run_episode,
)
from communication import FREE, UNKNOWN  # noqa: E402


class MeasurementTests(unittest.TestCase):
    def test_team_coverage_uses_union_of_private_sensor_maps(self):
        class ProbeEnvironment:
            pass

        env = ProbeEnvironment()
        env.num_agents = 2
        env.resolution = 0.5
        env.gt_map = np.full((2, 2), FREE, dtype=np.uint8)
        first = np.full((2, 2), UNKNOWN, dtype=np.uint8)
        second = np.full((2, 2), UNKNOWN, dtype=np.uint8)
        first[0, 0] = FREE
        first[0, 1] = FREE
        second[0, 1] = FREE
        second[1, 1] = FREE
        env.built_map = [first, second]
        env.agent_pos = [[0, 0], [1, 1]]
        env.agent_yaw = [0.0, 0.0]
        env.communication_metrics = lambda: {}

        result = _measurement(env, np.asarray([0.5, 1.0]))

        self.assertEqual(result["team_coverage_cells"], 3)
        self.assertEqual(result["agent_coverage_cells"], [2, 2])
        self.assertEqual(result["overlap_cells"], 1)
        self.assertAlmostEqual(result["team_coverage_ratio"], 0.75)
        self.assertAlmostEqual(result["overlap_area_m2"], 0.25)
        self.assertAlmostEqual(result["team_path_length_m"], 1.5)

    def test_ground_truth_unknown_cells_are_excluded_from_denominator(self):
        class ProbeEnvironment:
            pass

        env = ProbeEnvironment()
        env.num_agents = 1
        env.resolution = 1.0
        env.gt_map = np.asarray([[FREE, UNKNOWN], [FREE, FREE]], dtype=np.uint8)
        env.built_map = [np.asarray([[FREE, FREE], [UNKNOWN, UNKNOWN]], dtype=np.uint8)]
        env.agent_pos = [[0, 0]]
        env.agent_yaw = [0.0]
        env.communication_metrics = lambda: {}

        result = _measurement(env, np.asarray([0.0]))

        self.assertEqual(result["total_explorable_cells"], 3)
        self.assertEqual(result["team_coverage_cells"], 1)

    def test_communication_delta_uses_receiver_copy_latency(self):
        before = {
            "attempted_messages": 1,
            "transmitted_messages": 1,
            "delivered_messages": 1,
            "collision_messages": 0,
            "lost_messages": 0,
            "expired_messages": 0,
            "in_range_recipients": 1,
            "attempted_bits": 10,
            "delivered_bits": 10,
            "pose_messages": 0,
            "map_patch_messages": 1,
            "latency_total": 2,
        }
        after = dict(before)
        after.update(
            {
                "delivered_messages": 3,
                "in_range_recipients": 3,
                "delivered_bits": 30,
                "latency_total": 8,
            }
        )

        result = _communication_delta(before, after)

        self.assertEqual(result["delivered_messages"], 2)
        self.assertEqual(result["latency_total"], 6)
        self.assertEqual(result["mean_latency"], 3.0)


class HeadlessAdapterTests(unittest.TestCase):
    def test_plotting_is_a_noop_without_visualization(self):
        env = HeadlessGridEnv.__new__(HeadlessGridEnv)
        env.visualization = False

        self.assertIsNone(env.plot_map_with_path())


class FakeTraditionalEnvironment:
    """Small deterministic environment for collector tests, not simulation tests."""

    def __init__(
        self,
        resolution,
        sensor_range,
        num_agents,
        max_steps,
        map_name,
        visualization=False,
        sensor_configs=None,
        communication_config=None,
        communication_protocol="round_robin",
        seed=None,
    ):
        del sensor_range, map_name, visualization, sensor_configs
        del communication_config, communication_protocol
        self.resolution = resolution
        self.num_agents = num_agents
        self.max_steps = max_steps
        self._seed = seed
        self.communication_broker = None
        self.last_goals = None
        self.positions_before_goal = None
        self.last_requested_communication_actions = np.zeros(
            num_agents, dtype=np.int64
        )

    def seed(self, seed=None):
        self._seed = seed
        return [seed]

    def reset_for_traditional(self):
        self.num_step = 0
        self.gt_map = np.full((2, 2), FREE, dtype=np.uint8)
        self.built_map = [
            np.full((2, 2), UNKNOWN, dtype=np.uint8)
            for _ in range(self.num_agents)
        ]
        self.built_map[0][0, 0] = FREE
        self.built_map[1][1, 1] = FREE
        self.complete_map = np.full((2, 2), UNKNOWN, dtype=np.uint8)
        self.complete_map[0, 0] = FREE
        self.complete_map[1, 1] = FREE
        self.agent_pos = [[0, 0], [1, 1]]
        self.agent_yaw = [0.0, 0.0]
        self.latest_sensor_readings = [
            {
                "sensor_type": "four_beam",
                "max_range": 3.5,
                "beams": {},
            }
            for _ in range(self.num_agents)
        ]
        return np.stack(self.built_map), {}

    def communication_metrics(self):
        return {}

    def step_for_cost(self):
        self.positions_before_goal = np.asarray(self.agent_pos, dtype=np.int64)
        self.last_goals = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        self.agent_pos = [[0, 1], [1, 0]]
        self.built_map[0][0, 1] = FREE
        self.built_map[1][1, 0] = FREE
        self.complete_map = np.full((2, 2), FREE, dtype=np.uint8)
        self.num_step += 1
        return np.stack(self.built_map)

    def step_for_mmpf(self):
        return self.step_for_cost()


def fake_args(output_directory):
    return argparse.Namespace(
        method="cost",
        maps=["unused.pgm"],
        team_sizes=[2],
        seeds=[7],
        resolution=0.1,
        max_steps=5,
        no_progress_patience=2,
        coverage_thresholds=[0.75, 1.0],
        target_coverage=1.0,
        sensor_types=["four_beam"],
        sensor_ranges=[3.5],
        communication_mode="legacy",
        communication_protocol="round_robin",
        comm_tile_size=8,
        comm_candidate_count=8,
        comm_range_cells=40.0,
        comm_latency_min_steps=1,
        comm_latency_max_steps=None,
        comm_packet_loss=0.0,
        comm_cooldown_steps=3,
        comm_bucket_capacity_bits=314,
        comm_bucket_refill_bits=53,
        comm_episode_budget_bits=None,
        comm_ttl_steps=8,
        comm_timestamp_bits=16,
        comm_cost_per_patch=0.01,
        record="full",
        map_snapshots="none",
        output=str(output_directory),
        visualize=False,
        verbose=False,
    )


class EpisodeCollectionTests(unittest.TestCase):
    def test_finite_episode_writes_episode_step_and_agent_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            args = fake_args(run_directory)
            outputs = CsvOutputs(
                run_directory, args.record, args.coverage_thresholds
            )
            try:
                result = _run_episode(
                    args,
                    run_directory,
                    outputs,
                    1,
                    Path("unused.pgm"),
                    2,
                    7,
                    env_class=FakeTraditionalEnvironment,
                )
            finally:
                outputs.close()

            self.assertTrue(result["success"])
            self.assertEqual(result["steps_executed"], 1)
            self.assertEqual(result["termination_reason"], "target_coverage")
            self.assertEqual(result["final_coverage_ratio"], 1.0)
            self.assertEqual(result["coverage_75_step"], 1)
            self.assertEqual(result["coverage_100_step"], 1)

            with (run_directory / "episodes.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                episode_rows = list(csv.DictReader(handle))
            with (run_directory / "steps.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                step_rows = list(csv.DictReader(handle))
            with (run_directory / "agents.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                agent_rows = list(csv.DictReader(handle))

            self.assertEqual(len(episode_rows), 1)
            self.assertEqual(len(step_rows), 2)  # reset plus one decision
            self.assertEqual(len(agent_rows), 4)  # two agents at two steps

    def test_same_seed_and_fake_environment_are_reproducible(self):
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temporary:
                run_directory = Path(temporary)
                args = fake_args(run_directory)
                outputs = CsvOutputs(
                    run_directory, args.record, args.coverage_thresholds
                )
                try:
                    results.append(
                        _run_episode(
                            args,
                            run_directory,
                            outputs,
                            1,
                            Path("unused.pgm"),
                            2,
                            7,
                            env_class=FakeTraditionalEnvironment,
                        )
                    )
                finally:
                    outputs.close()

        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
