import unittest

from onpolicy.envs.GridEnv.GridEnv import GridEnv
from onpolicy.envs.GridEnv.communication import CommunicationConfig


class CommunicationSpaceTests(unittest.TestCase):
    def test_legacy_action_space_remains_continuous(self):
        env = GridEnv(0.1, 3.0, 2, 100)

        self.assertEqual(env.action_space[0].__class__.__name__, "Box")
        self.assertNotIn("comm_status", env.observation_space[0].spaces)

    def test_communication_adds_discrete_choice_and_critic_state(self):
        config = CommunicationConfig(candidate_count=4)
        env = GridEnv(0.1, 3.0, 2, 100, communication_config=config)

        self.assertEqual(env.action_space[0].__class__.__name__, "Tuple")
        self.assertEqual(env.action_space[0][1].n, 6)
        self.assertIn("comm_status", env.observation_space[0].spaces)
        self.assertIn("agent_beliefs", env.share_observation_space[0].spaces)


if __name__ == "__main__":
    unittest.main()
