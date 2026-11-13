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
    """
    shape = session.gt_map.shape
    mask = np.zeros((session.num_agents, side, side), np.float32)
    goal = np.full((session.num_agents, side, side, 2), -1, np.int64)
    scale = np.asarray(shape, float) / side
    for agent in range(session.num_agents):
        best = {}
        for target in session.frontier_targets(agent):
            row, column = qmap_cell(target, shape, side)
            centre = (np.asarray([row, column]) + 0.5) * scale - 0.5
            key = (float(np.sum((np.asarray(target) - centre) ** 2)), target)
            if (row, column) not in best or key < best[(row, column)]:
                best[(row, column)] = key
        for (row, column), (_, target) in best.items():
            mask[agent, row, column] = 1
            goal[agent, row, column] = target
    return mask, goal
