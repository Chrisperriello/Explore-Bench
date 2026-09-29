import unittest

import numpy as np

from onpolicy.envs.GridEnv.communication import (
    FREE,
    MAP_PATCH,
    PARALLEL,
    SILENCE,
    UNKNOWN,
    CommunicationBroker,
    CommunicationConfig,
)
from onpolicy.envs.env_wrappers import InfoDummyVecEnv


class BrokerProbeEnv:
    def __init__(self):
        self.observation_space = None
        self.share_observation_space = None
        self.action_space = None
        self.positions = [[1, 1], [2, 2]]
        self.headings = [0, 0]
        self.private_maps = [
            np.full((8, 8), UNKNOWN, dtype=np.uint8),
            np.full((8, 8), UNKNOWN, dtype=np.uint8),
        ]
        self.private_maps[0][1, 1] = FREE
        config = CommunicationConfig(
            mode=PARALLEL,
            tile_size=8,
            candidate_count=1,
            radio_range_cells=float("inf"),
            latency_min_steps=1,
            latency_max_steps=1,
            cooldown_steps=0,
            bucket_capacity_bits=1000,
            bucket_refill_bits=0,
            episode_budget_bits=2000,
        )
        self.broker = CommunicationBroker(config, 2, seed=3)
        self.broker.reset(
            8,
            8,
            10,
            self.private_maps,
            self.positions,
            self.headings,
        )

    def step(self, actions):
        self.broker.transmit(actions, self.positions, self.headings)
        self.broker.advance(
            self.private_maps, self.positions, self.headings
        )
        observation = np.array(
            [np.count_nonzero(grid != UNKNOWN) for grid in self.broker.belief_maps]
        )
        rewards = np.zeros((2, 1), dtype=np.float32)
        dones = np.zeros(2, dtype=bool)
        return observation, rewards, dones, {}

    def reset(self):
        raise AssertionError("the isolation probe should not reset")

    def close(self):
        pass


class VectorIsolationTests(unittest.TestCase):
    def test_vector_environments_keep_broker_delivery_isolated(self):
        vector_env = InfoDummyVecEnv([BrokerProbeEnv, BrokerProbeEnv])
        try:
            vector_env.step(
                [
                    [MAP_PATCH, SILENCE],
                    [SILENCE, SILENCE],
                ]
            )
            first, second = vector_env.envs

            self.assertIsNot(first.broker, second.broker)
            self.assertEqual(first.broker.belief_maps[1][1, 1], FREE)
            self.assertEqual(second.broker.belief_maps[1][1, 1], UNKNOWN)
            self.assertEqual(first.broker.metrics()["delivered_messages"], 1)
            self.assertEqual(second.broker.metrics()["delivered_messages"], 0)
        finally:
            vector_env.close()


if __name__ == "__main__":
    unittest.main()
