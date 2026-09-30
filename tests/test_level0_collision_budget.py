import unittest

import numpy as np

from onpolicy.envs.GridEnv.communication import (
    MAP_PATCH,
    POSE,
    SHARED_COLLISION,
    UNKNOWN,
    CommunicationBroker,
    CommunicationConfig,
)


class CollisionBudgetTests(unittest.TestCase):
    def test_repeated_collisions_exhaust_budget_and_force_silence(self):
        agent_count = 3
        maps = [
            np.full((16, 16), UNKNOWN, dtype=np.uint8)
            for _ in range(agent_count)
        ]
        positions = [[2, 2] for _ in range(agent_count)]
        headings = [0 for _ in range(agent_count)]
        probe = CommunicationBroker(
            CommunicationConfig(
                mode=SHARED_COLLISION,
                candidate_count=2,
                cooldown_steps=0,
                bucket_capacity_bits=1000,
                bucket_refill_bits=0,
                episode_budget_bits=1000,
            ),
            agent_count,
            seed=7,
        )
        probe.reset(16, 16, 3, maps, positions, headings)
        pose_bits = probe.message_bits(POSE)
        broker = CommunicationBroker(
            CommunicationConfig(
                mode=SHARED_COLLISION,
                candidate_count=2,
                cooldown_steps=0,
                bucket_capacity_bits=3 * pose_bits,
                bucket_refill_bits=0,
                episode_budget_bits=3 * pose_bits,
            ),
            agent_count,
            seed=7,
        )
        broker.reset(16, 16, 3, maps, positions, headings)

        for slot in range(3):
            _, attempted = broker.transmit(
                [POSE] * agent_count, positions, headings
            )
            np.testing.assert_array_equal(
                attempted, np.full(agent_count, pose_bits)
            )
            np.testing.assert_array_equal(
                broker.episode_budget,
                np.full(agent_count, (2 - slot) * pose_bits),
            )
            broker.advance(maps, positions, headings)

        for agent_id in range(agent_count):
            np.testing.assert_array_equal(
                broker.available_actions(agent_id),
                [1.0, 0.0, 0.0, 0.0],
            )
        _, attempted = broker.transmit(
            [POSE] * agent_count, positions, headings
        )
        self.assertTrue(np.all(attempted == 0))
        self.assertEqual(broker.metrics()["attempted_messages"], 9)
        self.assertEqual(broker.metrics()["collision_messages"], 9)

    def test_patch_collision_refills_tokens_but_not_episode_budget(self):
        agent_count = 3
        maps = [
            np.full((16, 16), UNKNOWN, dtype=np.uint8)
            for _ in range(agent_count)
        ]
        positions = [[2, 2] for _ in range(agent_count)]
        headings = [0 for _ in range(agent_count)]
        probe = CommunicationBroker(
            CommunicationConfig(
                mode=SHARED_COLLISION,
                candidate_count=2,
                cooldown_steps=0,
                bucket_capacity_bits=1000,
                bucket_refill_bits=0,
                episode_budget_bits=1000,
            ),
            agent_count,
            seed=13,
        )
        probe.reset(16, 16, 3, maps, positions, headings)
        patch_bits = probe.message_bits(MAP_PATCH)
        broker = CommunicationBroker(
            CommunicationConfig(
                mode=SHARED_COLLISION,
                candidate_count=2,
                cooldown_steps=0,
                bucket_capacity_bits=patch_bits,
                bucket_refill_bits=patch_bits,
                episode_budget_bits=2 * patch_bits,
            ),
            agent_count,
            seed=13,
        )
        broker.reset(16, 16, 3, maps, positions, headings)

        _, attempted = broker.transmit(
            [MAP_PATCH] * agent_count, positions, headings
        )
        np.testing.assert_array_equal(
            attempted, np.full(agent_count, patch_bits)
        )
        np.testing.assert_array_equal(broker.tokens, np.zeros(agent_count))
        for agent_id in range(agent_count):
            np.testing.assert_array_equal(
                broker.available_actions(agent_id),
                [1.0, 0.0, 0.0, 0.0],
            )

        broker.advance(maps, positions, headings)
        np.testing.assert_array_equal(
            broker.tokens, np.full(agent_count, patch_bits)
        )
        np.testing.assert_array_equal(
            broker.episode_budget, np.full(agent_count, patch_bits)
        )
        for agent_id in range(agent_count):
            self.assertTrue(
                np.all(broker.available_actions(agent_id)[MAP_PATCH:] == 1.0)
            )

        broker.transmit([MAP_PATCH] * agent_count, positions, headings)
        np.testing.assert_array_equal(broker.tokens, np.zeros(agent_count))
        np.testing.assert_array_equal(
            broker.episode_budget, np.zeros(agent_count)
        )
        broker.advance(maps, positions, headings)
        np.testing.assert_array_equal(
            broker.tokens, np.full(agent_count, patch_bits)
        )
        np.testing.assert_array_equal(
            broker.episode_budget, np.zeros(agent_count)
        )
        for agent_id in range(agent_count):
            np.testing.assert_array_equal(
                broker.available_actions(agent_id),
                [1.0, 0.0, 0.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
