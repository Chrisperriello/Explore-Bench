import unittest

import torch

from onpolicy.algorithms.utils.act import ACTLayer


class Box:
    def __init__(self, size):
        self.shape = (size,)


class Discrete:
    def __init__(self, size):
        self.n = size


class Tuple:
    def __init__(self, spaces):
        self.spaces = spaces

    def __getitem__(self, index):
        return self.spaces[index]


class MixedActionTests(unittest.TestCase):
    def test_discrete_mask_is_applied_to_mixed_action(self):
        layer = ACTLayer(Tuple((Box(2), Discrete(4))), 3, False, 0.01)
        features = torch.zeros((2, 3), dtype=torch.float32)
        available = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        actions, log_probs = layer(
            features, available_actions=available, deterministic=True
        )

        self.assertEqual(tuple(actions.shape), (2, 3))
        self.assertEqual(tuple(log_probs.shape), (2, 1))
        self.assertEqual(actions[:, 2].tolist(), [1.0, 3.0])

    def test_evaluation_uses_one_joint_log_probability(self):
        layer = ACTLayer(Tuple((Box(2), Discrete(3))), 3, False, 0.01)
        features = torch.zeros((2, 3), dtype=torch.float32)
        available = torch.ones((2, 3), dtype=torch.float32)
        actions, _ = layer(
            features, available_actions=available, deterministic=True
        )

        log_probs, entropy = layer.evaluate_actions(
            features, actions, available_actions=available
        )

        self.assertEqual(tuple(log_probs.shape), (2, 1))
        self.assertEqual(entropy.ndim, 0)


if __name__ == "__main__":
    unittest.main()
