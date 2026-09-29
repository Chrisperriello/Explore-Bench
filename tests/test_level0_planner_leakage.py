import unittest

import numpy as np

from onpolicy.envs.GridEnv.GridEnv import GridEnv
from onpolicy.envs.GridEnv.communication import (
    FREE,
    NONE,
    OCCUPIED,
    UNKNOWN,
    CommunicationConfig,
)


class PlannerLeakageTests(unittest.TestCase):
    def test_none_mode_planner_cannot_use_unseen_shortcut(self):
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
        config = CommunicationConfig(
            mode=NONE,
            tile_size=3,
            candidate_count=1,
        )
        env = GridEnv(0.1, 3.0, 1, 20, communication_config=config)
        env.gt_map = true_map
        env.communication_broker.reset(
            9, 9, 20, [belief], [[4, 1]], [0]
        )
        start = [4, 1]
        goal = [4, 7]

        true_path = env.Astar_global_planner(start, goal)
        private_path = env.Astar_global_planner(
            start, goal, env._navigation_map(0)
        )

        unseen_shortcut = {(4, column) for column in range(2, 7)}
        self.assertTrue(unseen_shortcut.intersection(true_path))
        self.assertFalse(unseen_shortcut.intersection(private_path))
        self.assertTrue(
            all(belief[row, column] != UNKNOWN for row, column in private_path)
        )


if __name__ == "__main__":
    unittest.main()
