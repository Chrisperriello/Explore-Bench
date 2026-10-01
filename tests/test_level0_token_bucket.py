import math
import unittest

import numpy as np

from onpolicy.envs.GridEnv.communication import (
    MAP_PATCH,
    PARALLEL,
    SILENCE,
    UNKNOWN,
    CommunicationBroker,
    CommunicationConfig,
)


class TokenBucketTests(unittest.TestCase):
    def test_silence_refills_tokens_incrementally_without_restoring_budget(self):
        maps = [
            np.full((16, 16), UNKNOWN, dtype=np.uint8),
            np.full((16, 16), UNKNOWN, dtype=np.uint8),
        ]
        positions = [[2, 2], [4, 4]]
        headings = [0, 0]
        probe = CommunicationBroker(
            CommunicationConfig(
                mode=PARALLEL,
                candidate_count=2,
                cooldown_steps=0,
                bucket_capacity_bits=1000,
                bucket_refill_bits=0,
                episode_budget_bits=1000,
            ),
            2,
            seed=29,
        )
        probe.reset(16, 16, 20, maps, positions, headings)
        patch_bits = probe.message_bits(MAP_PATCH)
        refill_bits = 17
        broker = CommunicationBroker(
            CommunicationConfig(
                mode=PARALLEL,
                candidate_count=2,
                cooldown_steps=0,
                bucket_capacity_bits=patch_bits,
                bucket_refill_bits=refill_bits,
                episode_budget_bits=3 * patch_bits,
            ),
            2,
            seed=29,
        )
        broker.reset(16, 16, 20, maps, positions, headings)

        broker.transmit([MAP_PATCH, SILENCE], positions, headings)
        self.assertEqual(broker.tokens[0], 0)
        remaining_budget = 2 * patch_bits
        self.assertEqual(broker.episode_budget[0], remaining_budget)

        refill_steps = int(math.ceil(patch_bits / float(refill_bits)))
        for step in range(1, refill_steps + 1):
            broker.transmit([SILENCE, SILENCE], positions, headings)
            broker.advance(maps, positions, headings)
            expected_tokens = min(patch_bits, step * refill_bits)
            self.assertEqual(broker.tokens[0], expected_tokens)
            self.assertEqual(broker.episode_budget[0], remaining_budget)
            patch_available = expected_tokens >= patch_bits
            self.assertEqual(
                bool(np.all(broker.available_actions(0)[MAP_PATCH:] == 1.0)),
                patch_available,
            )

        self.assertEqual(broker.tokens[0], patch_bits)
        self.assertEqual(broker.metrics()["attempted_messages"], 1)


if __name__ == "__main__":
    unittest.main()
