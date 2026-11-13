"""DQN for Level-0 exploration (in progress; plan and decisions in docs/dqn/).

The network outputs a Q-map: one Q-value per cell of a side x side grid laid
over the 250 x 250 belief map. Action k is Q-map cell divmod(k, side), in
row-major order. Which cells are legal and what request an action becomes
are decision D1 in docs/dqn/plan.md.
"""


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
