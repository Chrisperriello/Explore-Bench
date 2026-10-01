import sys
import unittest
from pathlib import Path

import numpy as np


GRID_SIMULATOR = Path(__file__).resolve().parents[1] / "grid_simulator"
sys.path.insert(0, str(GRID_SIMULATOR))

from communication import FREE, UNKNOWN, CommunicationConfig  # noqa: E402
from GridEnv import GridEnv  # noqa: E402


class StandaloneCommunicationTests(unittest.TestCase):
    def test_round_robin_protocol_advances_shared_broker(self):
        config = CommunicationConfig(
            mode="shared_collision",
            candidate_count=2,
            radio_range_cells=float("inf"),
            cooldown_steps=0,
            bucket_capacity_bits=1000,
            bucket_refill_bits=0,
            episode_budget_bits=2000,
        )
        env = GridEnv(
            0.1,
            3.5,
            2,
            10,
            "unused.pgm",
            communication_config=config,
            communication_protocol="round_robin",
            seed=3,
        )
        first = np.full((16, 16), UNKNOWN, dtype=np.uint8)
        second = np.full((16, 16), UNKNOWN, dtype=np.uint8)
        first[8:16, 8:16] = FREE
        second[0:8, 0:8] = FREE
        env.width = 16
        env.height = 16
        env.built_map = [first, second]
        env.complete_map = np.array(first, copy=True)
        env.agent_pos = [[1, 1], [2, 2]]
        env.agent_dir = [0, 0]
        env.num_step = 1

        env._reset_communication()
        env._transmit_traditional_communication()
        env._advance_traditional_communication()
        env._transmit_traditional_communication()
        env._advance_traditional_communication()

        self.assertEqual(env.communication_broker.belief_maps[1][9, 9], FREE)
        self.assertEqual(env.communication_broker.belief_maps[0][1, 1], FREE)
        metrics = env.communication_metrics()
        self.assertEqual(metrics["attempted_messages"], 2)
        self.assertEqual(metrics["transmitted_messages"], 2)
        self.assertEqual(metrics["delivered_messages"], 2)
        self.assertEqual(metrics["collision_messages"], 0)


if __name__ == "__main__":
    unittest.main()
