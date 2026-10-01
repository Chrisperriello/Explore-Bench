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


class ResetProbeEnv:
    def __init__(self, horizons):
        self.observation_space = None
        self.share_observation_space = None
        self.action_space = None
        self.horizons = tuple(horizons)
        self.episode_index = 0
        self.reset_count = 0
        self.positions = [[1, 1], [2, 2]]
        self.headings = [0, 0]
        self.config = CommunicationConfig(
            mode=PARALLEL,
            tile_size=8,
            candidate_count=1,
            radio_range_cells=float("inf"),
            latency_min_steps=3,
            latency_max_steps=3,
            cooldown_steps=2,
            bucket_capacity_bits=500,
            bucket_refill_bits=0,
            episode_budget_bits=1000,
        )
        self.broker = CommunicationBroker(self.config, 2, seed=23)
        self._start_episode()

    @property
    def horizon(self):
        return self.horizons[min(self.episode_index, len(self.horizons) - 1)]

    def _start_episode(self):
        self.step_count = 0
        self.private_maps = [
            np.full((8, 8), UNKNOWN, dtype=np.uint8),
            np.full((8, 8), UNKNOWN, dtype=np.uint8),
        ]
        marker = 1 + self.episode_index
        self.private_maps[0][marker, marker] = FREE
        self.broker.reset(
            8,
            8,
            self.horizon,
            self.private_maps,
            self.positions,
            self.headings,
        )

    def _observation(self):
        return np.array(
            [np.count_nonzero(grid != UNKNOWN) for grid in self.broker.belief_maps]
        )

    def step(self, actions):
        self.broker.transmit(actions, self.positions, self.headings)
        self.broker.advance(self.private_maps, self.positions, self.headings)
        self.step_count += 1
        done = self.step_count >= self.horizon
        rewards = np.zeros((2, 1), dtype=np.float32)
        dones = np.full(2, done, dtype=bool)
        return self._observation(), rewards, dones, {
            "comm_metrics": self.broker.metrics()
        }

    def reset(self):
        self.episode_index += 1
        self.reset_count += 1
        self._start_episode()
        return self._observation(), {"comm_metrics": self.broker.metrics()}

    def close(self):
        pass


class EpisodeResetTests(unittest.TestCase):
    def test_asynchronous_vector_reset_discards_previous_episode_radio_state(self):
        vector_env = InfoDummyVecEnv(
            [lambda: ResetProbeEnv((2, 6)), lambda: ResetProbeEnv((5, 5))]
        )
        try:
            vector_env.step(
                [[SILENCE, SILENCE], [SILENCE, SILENCE]]
            )
            _, _, dones, infos = vector_env.step(
                [[MAP_PATCH, SILENCE], [MAP_PATCH, SILENCE]]
            )
            reset_env, continuing_env = vector_env.envs

            self.assertTrue(np.all(dones[0]))
            self.assertFalse(np.any(dones[1]))
            self.assertIn("terminal_info", infos[0])
            self.assertEqual(reset_env.reset_count, 1)
            self.assertEqual(continuing_env.reset_count, 0)
            self.assertEqual(reset_env.broker.current_step, 0)
            self.assertEqual(reset_env.broker.in_flight_count, 0)
            np.testing.assert_array_equal(
                reset_env.broker.tokens, np.full(2, 500)
            )
            np.testing.assert_array_equal(
                reset_env.broker.episode_budget, np.full(2, 1000)
            )
            np.testing.assert_array_equal(
                reset_env.broker.cooldowns, np.zeros(2)
            )
            np.testing.assert_array_equal(
                reset_env.broker.last_results, np.zeros(2)
            )
            self.assertTrue(np.all(reset_env.broker.peer_pose_steps == -1))
            self.assertTrue(
                all(
                    np.all(grid == UNKNOWN)
                    for grid in reset_env.broker.last_broadcast_maps
                )
            )
            self.assertTrue(
                all(value == 0 for value in reset_env.broker.metrics().values())
            )
            self.assertEqual(reset_env.broker.belief_maps[1][1, 1], UNKNOWN)

            self.assertEqual(continuing_env.broker.current_step, 2)
            self.assertEqual(continuing_env.broker.in_flight_count, 1)
            self.assertEqual(
                continuing_env.broker.metrics()["attempted_messages"], 1
            )

            for _ in range(2):
                vector_env.step(
                    [[SILENCE, SILENCE], [SILENCE, SILENCE]]
                )

            self.assertEqual(reset_env.broker.belief_maps[1][1, 1], UNKNOWN)
            self.assertEqual(reset_env.broker.metrics()["delivered_messages"], 0)
            self.assertEqual(
                continuing_env.broker.belief_maps[1][1, 1], FREE
            )
            self.assertEqual(
                continuing_env.broker.metrics()["delivered_messages"], 1
            )
        finally:
            vector_env.close()


if __name__ == "__main__":
    unittest.main()
