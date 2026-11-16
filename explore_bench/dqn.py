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
import torch
import torch.nn as nn
import torch.nn.functional as F

SIDE = 64  # matches the 64 x 64 observation channels (ObservationBuilder)


def _orthogonal(layer, gain):
    """Initialize a learned layer like MAPPO: orthogonal weights, zero bias."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class SpatialBlock(nn.Module):
    """Two 3 x 3 convolution/ReLU pairs that preserve the grid dimensions."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        gain = np.sqrt(2.0)
        self.layers = nn.Sequential(
            _orthogonal(nn.Conv2d(input_channels, output_channels, 3, padding=1), gain),
            nn.ReLU(),
            _orthogonal(nn.Conv2d(output_channels, output_channels, 3, padding=1), gain),
            nn.ReLU(),
        )

    def forward(self, values):
        return self.layers(values)


class QNetwork(nn.Module):
    """Shared encoder-decoder that returns one raw Q-value per 64 x 64 cell.

    ``maps`` is [batch, 8, 64, 64]. ``other`` is [batch, 5K + 8 + N], the
    complete non-map actor input prepared by :func:`actor_inputs`. The batch
    may contain different robots and replay samples; no operation mixes rows.
    """

    def __init__(self, non_map_size):
        super().__init__()
        if not isinstance(non_map_size, int) or non_map_size < 1:
            raise ValueError("non_map_size must be a positive integer")
        self.non_map_size = non_map_size
        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList([
            SpatialBlock(8, 32),
            SpatialBlock(32, 64),
            SpatialBlock(64, 128),
            SpatialBlock(128, 128),
            SpatialBlock(128, 128),
        ])
        self.summary_linear = _orthogonal(nn.Linear(non_map_size, 128), np.sqrt(2.0))
        self.summary_norm = nn.LayerNorm(128)
        self.fusion = SpatialBlock(256, 128)
        self.decoders = nn.ModuleList([
            SpatialBlock(256, 128),
            SpatialBlock(256, 128),
            SpatialBlock(192, 64),
            SpatialBlock(96, 32),
        ])
        self.q_output = _orthogonal(nn.Conv2d(32, 1, 1), 1.0)

    @staticmethod
    def _double(values):
        return F.interpolate(values, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, maps, other):
        if maps.ndim != 4 or tuple(maps.shape[1:]) != (8, SIDE, SIDE):
            raise ValueError("maps must have shape [batch, 8, 64, 64]")
        if other.ndim != 2 or other.shape != (len(maps), self.non_map_size):
            raise ValueError("other must have shape [batch, non_map_size]")
        if maps.device != other.device:
            raise ValueError("maps and other must be on the same device")
        maps = maps.to(dtype=torch.float32)
        other = other.to(dtype=torch.float32)

        skips = []
        encoded = maps
        for index, encoder in enumerate(self.encoders):
            encoded = encoder(encoded)
            skips.append(encoded)
            if index < len(self.encoders) - 1:
                encoded = self.pool(encoded)

        summary = self.summary_norm(F.relu(self.summary_linear(other)))
        summary = summary[:, :, None, None].expand(-1, -1, 4, 4)
        decoded = self.fusion(torch.cat((encoded, summary), dim=1))
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            decoded = self._double(decoded)
            decoded = decoder(torch.cat((decoded, skip), dim=1))
        return self.q_output(decoded).squeeze(1)


def actor_inputs(observation):
    """Separate complete MAPPO actor inputs into maps and non-map values.

    Input arrays have one leading batch dimension (robots during inference,
    replay samples during learning). Returns float32 arrays [batch, 8, 64, 64]
    and [batch, 5K + 8 + N], where K is the number of radio patch candidates
    and N the length of the robot identity vector.

    Ordering matches MAPPO's MIXBase: team maps before private maps, then
    flattened patch candidates, radio status, and identity in the other branch.
    Explicit keys keep the frontier mask, goal lookup, and privileged critic
    observations out of the network (decisions D2a and D2d).
    """
    team = np.asarray(observation["global_merge_obs"], dtype=np.float32)
    own = np.asarray(observation["global_obs"], dtype=np.float32)
    if team.ndim != 4 or team.shape[1:] != (4, SIDE, SIDE) or own.shape != team.shape:
        raise ValueError("actor maps must both have shape [batch, 4, 64, 64]")
    batch = len(team)
    if batch == 0:
        raise ValueError("actor input batch must not be empty")
    candidates = np.asarray(observation["comm_candidates"], dtype=np.float32)
    status = np.asarray(observation["comm_status"], dtype=np.float32)
    identity = np.asarray(observation["vector"], dtype=np.float32)
    if (candidates.ndim != 3 or candidates.shape[0] != batch or candidates.shape[2] != 5
            or status.shape != (batch, 8) or identity.ndim != 2 or len(identity) != batch):
        raise ValueError("actor non-map inputs must have shapes [batch, K, 5], [batch, 8], [batch, N]")
    return (np.concatenate((team, own), axis=1),
            np.concatenate((candidates.reshape(batch, -1), status, identity), axis=1))


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
