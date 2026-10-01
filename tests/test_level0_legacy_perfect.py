import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


GRID_SIMULATOR = Path(__file__).resolve().parents[1] / "grid_simulator"
sys.path.insert(0, str(GRID_SIMULATOR))

from communication import (  # noqa: E402
    FREE,
    OCCUPIED,
    PERFECT,
    UNKNOWN,
    CommunicationBroker,
    CommunicationConfig,
)
from GridEnv import GridEnv  # noqa: E402


def legacy_merge(private_maps):
    shape = private_maps[0].shape
    explored = np.zeros(shape, dtype=bool)
    obstacles = np.zeros(shape, dtype=bool)
    for private_map in private_maps:
        explored |= private_map != UNKNOWN
        obstacles |= private_map == OCCUPIED
    merged = np.full(shape, UNKNOWN, dtype=np.uint8)
    merged[explored] = FREE
    merged[obstacles] = OCCUPIED
    return merged


class LegacyPerfectTests(unittest.TestCase):
    def test_perfect_fusion_matches_legacy_merge_for_fixed_sensor_frames(self):
        private_maps = [
            np.full((16, 16), UNKNOWN, dtype=np.uint8),
            np.full((16, 16), UNKNOWN, dtype=np.uint8),
        ]
        private_maps[0][0:8, 0:8] = FREE
        private_maps[1][8:16, 8:16] = FREE
        broker = CommunicationBroker(
            CommunicationConfig(mode=PERFECT, candidate_count=2),
            2,
            seed=31,
        )
        broker.reset(
            16,
            16,
            5,
            private_maps,
            [[2, 2], [12, 12]],
            [0, 2],
        )

        for step in range(5):
            private_maps[0][step, 8] = FREE
            private_maps[1][step, 8] = (
                OCCUPIED if step % 2 == 0 else FREE
            )
            broker.advance(
                private_maps, [[2, 2], [12, 12]], [0, 2]
            )
            expected = legacy_merge(private_maps)
            for belief in broker.belief_maps:
                np.testing.assert_array_equal(belief, expected)

    def test_seeded_standalone_controls_first_diverge_at_navigation_map(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            map_path = Path(temporary_directory) / "validation.pgm"
            true_map = np.full((32, 32), FREE, dtype=np.uint8)
            true_map[[0, -1], :] = OCCUPIED
            true_map[:, [0, -1]] = OCCUPIED
            Image.fromarray(true_map).save(str(map_path))
            legacy = GridEnv(1.0, 3.5, 2, 10, str(map_path), seed=17)
            perfect = GridEnv(
                1.0,
                3.5,
                2,
                10,
                str(map_path),
                communication_config=CommunicationConfig(
                    mode=PERFECT, candidate_count=8
                ),
                seed=17,
            )

            legacy.reset_for_traditional()
            perfect.reset_for_traditional()

            self.assertEqual(legacy.agent_pos, perfect.agent_pos)
            self.assertEqual(legacy.agent_dir, perfect.agent_dir)
            for legacy_map, perfect_map in zip(
                legacy.built_map, perfect.built_map
            ):
                np.testing.assert_array_equal(legacy_map, perfect_map)
            np.testing.assert_array_equal(
                legacy.complete_map, perfect.complete_map
            )
            for belief in perfect.communication_broker.belief_maps:
                np.testing.assert_array_equal(legacy.complete_map, belief)
            self.assertEqual(
                np.count_nonzero(legacy.complete_map != UNKNOWN),
                np.count_nonzero(perfect.complete_map != UNKNOWN),
            )
            np.testing.assert_array_equal(
                legacy.get_goal_for_cost(), perfect.get_goal_for_cost()
            )

            for agent_id in range(2):
                legacy_navigation = legacy._navigation_map(agent_id)
                perfect_navigation = perfect._navigation_map(agent_id)
                unknown = legacy.complete_map == UNKNOWN
                unseen_free = unknown & (true_map == FREE)
                self.assertTrue(np.any(unseen_free))
                self.assertTrue(
                    np.all(legacy_navigation[unseen_free] == FREE)
                )
                self.assertTrue(
                    np.all(perfect_navigation[unseen_free] == OCCUPIED)
                )

    def test_perfect_planner_rejects_a_legacy_true_map_shortcut(self):
        true_map = np.full((9, 9), OCCUPIED, dtype=np.uint8)
        true_map[4, 1:8] = FREE
        true_map[2, 1:8] = FREE
        true_map[3, 1] = FREE
        true_map[3, 7] = FREE
        belief = np.full((9, 9), OCCUPIED, dtype=np.uint8)
        belief[4, 1] = FREE
        belief[4, 7] = FREE
        belief[4, 2:7] = UNKNOWN
        belief[2, 1:8] = FREE
        belief[3, 1] = FREE
        belief[3, 7] = FREE
        legacy = GridEnv(0.1, 3.0, 1, 20, "unused.pgm")
        perfect = GridEnv(
            0.1,
            3.0,
            1,
            20,
            "unused.pgm",
            communication_config=CommunicationConfig(
                mode=PERFECT,
                tile_size=3,
                candidate_count=1,
            ),
        )
        for env in (legacy, perfect):
            env.width = 9
            env.height = 9
            env.gt_map = np.array(true_map, copy=True)
            env.complete_map = np.array(belief, copy=True)
            env.built_map = [np.array(belief, copy=True)]
            env.agent_pos = [[4, 1]]
            env.agent_dir = [0]
        perfect._reset_communication()

        start = [4, 1]
        goal = [4, 7]
        legacy_path = legacy.Astar_global_planner(
            start, goal, legacy._navigation_map(0)
        )
        perfect_path = perfect.Astar_global_planner(
            start, goal, perfect._navigation_map(0)
        )
        unseen_shortcut = {(4, column) for column in range(2, 7)}

        self.assertTrue(unseen_shortcut.intersection(legacy_path))
        self.assertFalse(unseen_shortcut.intersection(perfect_path))

    def test_perfect_cost_step_handles_disconnected_known_regions(self):
        map_path = (
            Path(__file__).resolve().parents[1]
            / "onpolicy"
            / "onpolicy"
            / "envs"
            / "GridEnv"
            / "datasets"
            / "corner.pgm"
        )
        env = GridEnv(
            0.1,
            3.5,
            2,
            5,
            str(map_path),
            communication_config=CommunicationConfig(mode=PERFECT),
            seed=41,
        )
        env.plot_map_with_path = lambda: None
        env.reset_for_traditional()

        env.step_for_cost()

        self.assertEqual(env.num_step, 1)
        for belief in env.communication_broker.belief_maps:
            np.testing.assert_array_equal(belief, env.complete_map)


if __name__ == "__main__":
    unittest.main()
