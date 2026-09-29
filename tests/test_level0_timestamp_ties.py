import unittest

import numpy as np

from onpolicy.envs.GridEnv.communication import (
    FREE,
    MAP_PATCH,
    OCCUPIED,
    PARALLEL,
    SILENCE,
    UNKNOWN,
    CommunicationBroker,
    CommunicationConfig,
)


class TimestampTieTests(unittest.TestCase):
    def test_equal_timestamp_keeps_local_observation_deterministically(self):
        outcomes = []
        positions = [[2, 2], [4, 4]]
        headings = [0, 2]
        for seed in range(10):
            maps = [
                np.full((16, 16), UNKNOWN, dtype=np.uint8),
                np.full((16, 16), UNKNOWN, dtype=np.uint8),
            ]
            maps[0][0:8, 0:8] = FREE
            maps[0][3, 3] = OCCUPIED
            maps[1][3, 3] = FREE
            config = CommunicationConfig(
                mode=PARALLEL,
                candidate_count=2,
                radio_range_cells=float("inf"),
                latency_min_steps=0,
                latency_max_steps=0,
                cooldown_steps=0,
                bucket_capacity_bits=1000,
                bucket_refill_bits=0,
                episode_budget_bits=2000,
            )
            broker = CommunicationBroker(config, 2, seed=seed)
            broker.reset(16, 16, 20, maps, positions, headings)

            broker.transmit([MAP_PATCH, SILENCE], positions, headings)
            outcomes.append(int(broker.belief_maps[1][3, 3]))
            self.assertEqual(broker.belief_timestamps[1][3, 3], 0)

        self.assertEqual(outcomes, [FREE] * 10)


if __name__ == "__main__":
    unittest.main()
