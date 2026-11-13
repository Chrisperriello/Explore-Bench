"""DQN for Level-0 exploration (in progress; plan and decisions in docs/dqn/).

The network outputs a Q-map: one Q-value per cell of a side x side grid laid
over the 250 x 250 belief map. Action k is Q-map cell divmod(k, side), in
row-major order.

Legal actions (decision D1 in docs/dqn/plan.md): a Q-map cell is legal for a
robot when it holds one of that robot's reachable frontier targets
(Session.frontier_targets). The action is sent as that target's exact grid
cell ("motion_goals"), so the action Q-learning credits is the one executed.
"""

import numpy as np

SIDE = 64  # matches the 64 x 64 observation channels (ObservationBuilder)


def action_cell(action, side):
    """Q-map (row, column) of flat action index ``action`` (row-major)."""
    if not 0 <= action < side * side:
        raise ValueError("action outside the Q-map")
    return divmod(int(action), side)


def action_index(row, column, side):
    """Flat action index of Q-map cell (row, column); inverse of action_cell."""
    return int(row) * side + int(column)


def qmap_cell(cell, shape, side):
    """Q-map (row, column) containing full-resolution grid ``cell``.

    Uses cell centres, (2r + 1) * side // (2 * height), which matches how
    cv2.resize places the 64 x 64 observation pixels over the full map, so
    Q-map cell (i, j) covers the same area as observation pixel (i, j).
    """
    return ((2 * int(cell[0]) + 1) * side // (2 * shape[0]),
            (2 * int(cell[1]) + 1) * side // (2 * shape[1]))


def frontier_qmap(session, side=SIDE):
    """Legal-action mask and goal table for every robot.

    Returns ``mask`` [agents, side, side] float32 (1 = legal) and ``goal``
    [agents, side, side, 2] int64, the grid cell a legal action sends the robot
    to (-1 elsewhere). When two targets fall in one Q-map cell, the one nearest
    the cell centre is kept (ties by cell), so each action names one place.

    A robot with no reachable frontier gets one legal action instead: "stay",
    its own Q-map cell with its own position as the goal (decision D1b). The
    mask is never empty, so the learning target's max over legal next actions
    is always defined, and staying is what MAPPO's decoding and the classical
    planners do in the same situation.
    """
    shape = session.gt_map.shape
    mask = np.zeros((session.num_agents, side, side), np.float32)
    goal = np.full((session.num_agents, side, side, 2), -1, np.int64)
    scale = np.asarray(shape, float) / side
    for agent in range(session.num_agents):
        best = {}
        targets = session.frontier_targets(agent)
        if not targets:
            here = tuple(int(value) for value in session.agent_pos[agent])
            mask[agent][qmap_cell(here, shape, side)] = 1
            goal[agent][qmap_cell(here, shape, side)] = here
            continue
        for target in targets:
            row, column = qmap_cell(target, shape, side)
            centre = (np.asarray([row, column]) + 0.5) * scale - 0.5
            key = (float(np.sum((np.asarray(target) - centre) ** 2)), target)
            if (row, column) not in best or key < best[(row, column)]:
                best[(row, column)] = key
        for (row, column), (_, target) in best.items():
            mask[agent, row, column] = 1
            goal[agent, row, column] = target
    return mask, goal


def masked_argmax(q_values, mask):
    """Best legal flat action per robot: q_values and mask are [agents, side, side]."""
    legal = np.asarray(mask).reshape(len(mask), -1) > 0
    if not np.all(legal.any(axis=1)):
        raise ValueError("a robot has no legal action")
    return np.where(legal, np.asarray(q_values, float).reshape(len(mask), -1), -np.inf).argmax(axis=1)


def request(actions, observation, side=SIDE):
    """Session request for flat actions [agents]; refuses an illegal action.

    Communication action 0 is SILENCE: DQN v1 never transmits, which is
    valid under the "perfect" and "none" modes only (docs/dqn/plan.md).
    """
    goals = []
    for agent, action in enumerate(actions):
        cell = action_cell(action, side)
        if observation["frontier_mask"][agent][cell] != 1:
            raise ValueError("illegal action {} for robot {}".format(action, agent))
        goals.append(observation["frontier_goal"][agent][cell])
    return {"motion_goals": np.asarray(goals, np.int64), "comm_actions": np.zeros(len(goals), np.int64)}
