"""Controller protocol: stateful inference returns semantic action requests.

DQN can implement this protocol without inheriting a MAPPO runner or trainer.
The session, not the neural network, interprets coordinates as frontier goals.
"""

import random
import numpy as np


class RandomController:
    name = "random"

    def reset(self, task):
        self.rng = random.Random(task.seed)

    def act(self, observation, state=None, available_actions=None, deterministic=True):
        count = len(observation["vector"])
        return {"coordinates": np.asarray([[self.rng.random(), self.rng.random()] for _ in range(count)]),
                "comm_actions": np.zeros(count, dtype=np.int64)}, None


class WaitController:
    name = "wait"

    def reset(self, task):
        self.positions = np.asarray(task.start_positions)

    def act(self, observation, state=None, available_actions=None, deterministic=True):
        return {"motion_goals": self.positions.copy(),
                "comm_actions": np.zeros(len(self.positions), dtype=np.int64)}, None


class ClassicalController:
    """Use the preserved Cost/MMPF goal selector, with a legal belief view.

    Only goal selection is reused. Movement, sensing, clock and recording are
    executed by Session, preventing the old transition semantics from leaking
    into this comparison. bind_view is supplied only by the trusted evaluator.
    """

    def __init__(self, method, bind_view):
        self.name, self.bind_view = method, bind_view

    def reset(self, task):
        pass

    def act(self, observation, state=None, available_actions=None, deterministic=True):
        import contextlib
        import io
        import sys
        from .config import ROOT
        directory = str(ROOT / "grid_simulator")
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from collect_experiments import HeadlessGridEnv
        view = self.bind_view()
        adapter = object.__new__(HeadlessGridEnv)
        adapter.__dict__.update(view)
        with contextlib.redirect_stdout(io.StringIO()):
            goals = getattr(adapter, "get_goal_for_" + self.name)()
        return {"motion_goals": goals, "comm_actions": np.zeros(len(goals), dtype=np.int64)}, None
