"""Deterministic Level-0 communication and per-agent belief management.

This module is a *logical communication abstraction*, not a radio stack.  It
does not serialize bytes, encrypt packets, modulate signals, simulate bit
errors, or model walls and signal strength.  Instead, it lets Level-0
experiments control the communication effects that matter to the research
question: range, decision-step latency, independent packet loss, shared-slot
collisions, cooldowns, rate limits, total episode budgets, and message expiry.

Delivery is all-or-nothing.  If a message passes the configured checks, the
complete pose or map-patch snapshot is copied into each eligible receiver's
belief.  ``message_bits`` supplies a logical size for accounting and reward
cost; those bits are not encoded or individually transmitted.

The broker is synchronous.  :meth:`CommunicationBroker.transmit` processes
the actions selected for one environment decision, and
:meth:`CommunicationBroker.advance` moves the logical clock, incorporates new
private sensor maps, and delivers due messages.  This makes runs reproducible
and gives communication time the same unit as policy decisions.
"""

import copy
import heapq
import math

import numpy as np


__all__ = [
    "COMMUNICATION_MODES",
    "FREE",
    "MAP_PATCH",
    "NONE",
    "OCCUPIED",
    "PARALLEL",
    "PERFECT",
    "POSE",
    "SHARED_COLLISION",
    "SILENCE",
    "UNKNOWN",
    "CommunicationBroker",
    "CommunicationConfig",
    "Message",
    "TileCandidate",
]


UNKNOWN = 205
FREE = 254
OCCUPIED = 0

PERFECT = "perfect"
NONE = "none"
PARALLEL = "parallel"
SHARED_COLLISION = "shared_collision"
COMMUNICATION_MODES = (PERFECT, NONE, PARALLEL, SHARED_COLLISION)

SILENCE = 0
POSE = 1
MAP_PATCH = 2

RESULT_SILENT = 0
RESULT_TRANSMITTED = 1
RESULT_COLLISION = 2
RESULT_BUDGET = 3
RESULT_COOLDOWN = 4
RESULT_COUNT = 5


def _positive_int(name, value):
    """Return ``value`` as an integer, rejecting zero and negative values."""
    value = int(value)
    if value <= 0:
        raise ValueError("{} must be positive".format(name))
    return value


def _nonnegative_int(name, value):
    """Return ``value`` as an integer, rejecting negative values."""
    value = int(value)
    if value < 0:
        raise ValueError("{} cannot be negative".format(name))
    return value


class CommunicationConfig:
    """Validated parameters for the logical Level-0 communication model.

    Args:
        mode: One of ``perfect``, ``none``, ``parallel``, or
            ``shared_collision``.  Perfect and none force policy silence.
        tile_size: Width and height, in cells, of every map-patch payload.
        candidate_count: Number of ranked map patches offered to the policy.
        radio_range_cells: Maximum Euclidean sender/receiver separation in
            grid cells.  ``math.inf`` represents unlimited range.
        latency_min_steps: Minimum decision-step delivery delay.
        latency_max_steps: Maximum delay, inclusive.  ``None`` makes latency
            fixed at ``latency_min_steps``.
        packet_loss: Independent loss probability for each in-range receiver.
            This models whole-message loss, not corrupted individual bits.
        cooldown_steps: Decisions a sender must wait after an attempt.
        bucket_capacity_bits: Maximum short-term token balance per sender.
        bucket_refill_bits: Tokens restored to each sender per advance.
        episode_budget_bits: Non-refilling per-sender allowance.  ``None``
            derives a budget of one patch per three episode decisions.
        ttl_steps: Maximum message age before an undelivered message expires.
        timestamp_bits: Logical timestamp width included in message size.
        cost_per_patch: Reward cost of attempting one patch-equivalent.

    Notes:
        These values describe a simulation treatment.  They are not physical
        measurements of a particular Wi-Fi, Crazyradio, or ROS transport.
    """

    def __init__(
        self,
        mode=SHARED_COLLISION,
        tile_size=8,
        candidate_count=8,
        radio_range_cells=40.0,
        latency_min_steps=1,
        latency_max_steps=None,
        packet_loss=0.0,
        cooldown_steps=3,
        bucket_capacity_bits=314,
        bucket_refill_bits=53,
        episode_budget_bits=None,
        ttl_steps=8,
        timestamp_bits=16,
        cost_per_patch=0.01,
    ):
        if mode not in COMMUNICATION_MODES:
            raise ValueError("unsupported communication mode: {}".format(mode))
        self.mode = mode
        self.tile_size = _positive_int("tile_size", tile_size)
        self.candidate_count = _positive_int("candidate_count", candidate_count)
        self.radio_range_cells = float(radio_range_cells)
        if math.isnan(self.radio_range_cells) or self.radio_range_cells <= 0:
            raise ValueError("radio_range_cells must be positive or infinity")
        self.latency_min_steps = _nonnegative_int(
            "latency_min_steps", latency_min_steps
        )
        if latency_max_steps is None:
            latency_max_steps = self.latency_min_steps
        self.latency_max_steps = _nonnegative_int(
            "latency_max_steps", latency_max_steps
        )
        if self.latency_max_steps < self.latency_min_steps:
            raise ValueError("latency_max_steps cannot be less than latency_min_steps")
        self.packet_loss = float(packet_loss)
        if not 0.0 <= self.packet_loss <= 1.0:
            raise ValueError("packet_loss must be between zero and one")
        self.cooldown_steps = int(cooldown_steps)
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps cannot be negative")
        self.bucket_capacity_bits = _positive_int(
            "bucket_capacity_bits", bucket_capacity_bits
        )
        self.bucket_refill_bits = int(bucket_refill_bits)
        if self.bucket_refill_bits < 0:
            raise ValueError("bucket_refill_bits cannot be negative")
        self.episode_budget_bits = (
            None
            if episode_budget_bits is None
            else _nonnegative_int("episode_budget_bits", episode_budget_bits)
        )
        self.ttl_steps = _positive_int("ttl_steps", ttl_steps)
        self.timestamp_bits = _positive_int("timestamp_bits", timestamp_bits)
        self.cost_per_patch = float(cost_per_patch)
        if not math.isfinite(self.cost_per_patch) or self.cost_per_patch < 0:
            raise ValueError("cost_per_patch must be finite and nonnegative")


class TileCandidate:
    """One deterministic map-patch option exposed to the policy.

    Attributes:
        row: Top-left patch row in the full-resolution belief map.
        column: Top-left patch column in the belief map.
        features: Five normalized values: row, column, known fraction,
            changed fraction, and frontier fraction.
    """

    def __init__(self, row, column, features):
        self.row = int(row)
        self.column = int(column)
        self.features = np.asarray(features, dtype=np.float32)


class Message:
    """Immutable-in-practice snapshot scheduled for logical delivery.

    ``payload`` contains either a copied pose or copied map cells from the send
    step.  Later sender observations therefore cannot change an in-flight
    message.  ``recipients`` is fixed at send time after range and packet-loss
    checks.

    Args:
        sender: Integer agent identifier.
        message_type: ``POSE`` or ``MAP_PATCH``.
        sent_step: Broker step at which the action was attempted.
        deliver_step: Broker step at which delivery becomes eligible.
        bit_count: Logical size charged to the sender and metrics.
        payload: Pose dictionary or map-patch dictionary.
        recipients: In-range receivers that did not lose this message.
    """

    def __init__(
        self,
        sender,
        message_type,
        sent_step,
        deliver_step,
        bit_count,
        payload,
        recipients,
    ):
        self.sender = int(sender)
        self.message_type = int(message_type)
        self.sent_step = int(sent_step)
        self.deliver_step = int(deliver_step)
        self.bit_count = int(bit_count)
        self.payload = payload
        self.recipients = tuple(int(recipient) for recipient in recipients)


class CommunicationBroker:
    """Synchronous all-or-nothing message broker and belief-state manager.

    One broker belongs to one environment instance.  Episode reset rebuilds
    its queues, resources, metrics, peer state, timestamps, and beliefs, which
    prevents information from leaking between vector environments or episodes.

    The class intentionally stops above the physical/network layers: it uses
    logical bit counts and whole-message outcomes rather than byte encoding,
    encryption, checksums, acknowledgements, or retransmission.
    """

    def __init__(self, config, num_agents, seed=None):
        """Create an uninitialized broker.

        Call :meth:`reset` after the environment map and initial private maps
        are available.

        Args:
            config: Validated :class:`CommunicationConfig`.
            num_agents: Number of senders/receivers in the environment.
            seed: Seed for packet-loss and variable-latency sampling.
        """
        if not isinstance(config, CommunicationConfig):
            raise TypeError("config must be a CommunicationConfig")
        self.config = config
        self.num_agents = _positive_int("num_agents", num_agents)
        self.seed(seed)
        self.width = None
        self.height = None

    def seed(self, seed=None):
        """Reset the broker-local random generator and return Gym-style seeds."""
        self._rng = np.random.RandomState(seed)
        self._seed = seed
        return [seed]

    def reset(self, width, height, max_steps, private_maps, positions, headings):
        """Start a clean communication episode.

        Args:
            width: First occupancy-grid dimension in cells.
            height: Second occupancy-grid dimension in cells.
            max_steps: Maximum number of environment decisions.
            private_maps: One initial sensor-derived occupancy grid per agent.
            positions: Initial integer grid positions.
            headings: Initial discrete headings.

        The method creates independent belief maps from ``private_maps``, fills
        token buckets, restores episode budgets, clears in-flight messages and
        metrics, and initializes peer poses as unknown.  Perfect mode then
        replaces the private beliefs with an immediate team merge.
        """
        self.width = _positive_int("width", width)
        self.height = _positive_int("height", height)
        self.max_steps = _positive_int("max_steps", max_steps)
        if len(private_maps) != self.num_agents:
            raise ValueError("private_maps must contain one map per agent")
        if len(positions) != self.num_agents or len(headings) != self.num_agents:
            raise ValueError("positions and headings must contain one value per agent")
        expected_shape = (self.width, self.height)
        for private_map in private_maps:
            if np.asarray(private_map).shape != expected_shape:
                raise ValueError(
                    "every private map must have shape {}".format(expected_shape)
                )
        tile_count = int(math.ceil(self.width / float(self.config.tile_size))) * int(
            math.ceil(self.height / float(self.config.tile_size))
        )
        if self.config.candidate_count > tile_count:
            raise ValueError(
                "candidate_count cannot exceed the number of map tiles"
            )
        self.current_step = 0
        self._sequence = 0
        self._in_flight = []
        self.tokens = np.full(
            self.num_agents, self.config.bucket_capacity_bits, dtype=np.int64
        )
        patch_bits = self.message_bits(MAP_PATCH)
        configured_budget = self.config.episode_budget_bits
        if configured_budget is None:
            configured_budget = int(math.ceil(max_steps / 3.0)) * patch_bits
        self.episode_budget = np.full(
            self.num_agents, configured_budget, dtype=np.int64
        )
        self.cooldowns = np.zeros(self.num_agents, dtype=np.int64)
        self.last_results = np.full(
            self.num_agents, RESULT_SILENT, dtype=np.int64
        )
        self.belief_maps = [
            np.array(grid, dtype=np.uint8, copy=True) for grid in private_maps
        ]
        self.belief_timestamps = []
        self.belief_local_timestamps = []
        for grid in self.belief_maps:
            timestamps = np.full(grid.shape, -1, dtype=np.int64)
            timestamps[grid != UNKNOWN] = 0
            self.belief_timestamps.append(timestamps)
            self.belief_local_timestamps.append(np.array(timestamps, copy=True))
        self.last_broadcast_maps = [
            np.full((self.width, self.height), UNKNOWN, dtype=np.uint8)
            for _ in range(self.num_agents)
        ]
        self.peer_positions = np.full(
            (self.num_agents, self.num_agents, 2), -1, dtype=np.int64
        )
        self.peer_headings = np.full(
            (self.num_agents, self.num_agents), -1, dtype=np.int64
        )
        self.peer_pose_steps = np.full(
            (self.num_agents, self.num_agents), -1, dtype=np.int64
        )
        self._metrics = {
            "attempted_messages": 0,
            "transmitted_messages": 0,
            "delivered_messages": 0,
            "collision_messages": 0,
            "lost_messages": 0,
            "expired_messages": 0,
            "in_range_recipients": 0,
            "attempted_bits": 0,
            "delivered_bits": 0,
            "pose_messages": 0,
            "map_patch_messages": 0,
            "latency_total": 0,
        }
        if self.config.mode == PERFECT:
            self._synchronize_perfect(private_maps, positions, headings)
        self.refresh_candidates()

    def message_bits(self, message_type):
        """Calculate the logical size of a pose or map-patch message.

        The calculation includes sender/type/timestamp metadata plus either
        pose coordinates and heading or a tile index and two bits per patch
        cell.  It deliberately excludes real packet headers, framing,
        encryption, checksums, and retransmission overhead.

        Args:
            message_type: ``POSE`` or ``MAP_PATCH``.

        Returns:
            Integer logical bit count used for budgets, metrics, and cost.
        """
        sender_bits = max(1, int(math.ceil(math.log(self.num_agents, 2))))
        header_bits = sender_bits + 2 + self.config.timestamp_bits
        coordinate_bits = int(math.ceil(math.log(self.width, 2))) + int(
            math.ceil(math.log(self.height, 2))
        )
        if message_type == POSE:
            return header_bits + coordinate_bits + 2
        if message_type == MAP_PATCH:
            tile_rows = int(math.ceil(self.width / float(self.config.tile_size)))
            tile_columns = int(
                math.ceil(self.height / float(self.config.tile_size))
            )
            tile_bits = max(
                1, int(math.ceil(math.log(tile_rows * tile_columns, 2)))
            )
            return header_bits + tile_bits + 2 * self.config.tile_size ** 2
        raise ValueError("unknown message type: {}".format(message_type))

    def refresh_candidates(self):
        """Rebuild every agent's deterministic top map-patch choices."""
        self.candidates = []
        for agent_id in range(self.num_agents):
            self.candidates.append(self._candidate_tiles(agent_id))

    def candidate_features(self, agent_id):
        """Return the selected agent's ``(candidate_count, 5)`` feature array."""
        return np.stack(
            [candidate.features for candidate in self.candidates[agent_id]], axis=0
        )

    def available_actions(self, agent_id):
        """Return a binary mask of communication actions legal right now.

        Silence is always valid.  Perfect/none modes force silence; otherwise
        cooldown and both resource pools decide whether pose and patch actions
        are affordable.  This mask is used by the categorical MAPPO head, but
        the broker repeats the checks in :meth:`transmit` for safety.
        """
        action_count = self.config.candidate_count + MAP_PATCH
        available = np.zeros(action_count, dtype=np.float32)
        available[SILENCE] = 1.0
        if self.config.mode in (PERFECT, NONE):
            return available
        if self.cooldowns[agent_id] > 0:
            return available
        pose_bits = self.message_bits(POSE)
        patch_bits = self.message_bits(MAP_PATCH)
        if self._has_budget(agent_id, pose_bits):
            available[POSE] = 1.0
        if self._has_budget(agent_id, patch_bits):
            available[MAP_PATCH:] = 1.0
        return available

    def status_vector(self, agent_id):
        """Return normalized local radio state for one decentralized actor.

        The eight values are token fraction, episode-budget fraction, cooldown
        fraction, and a five-way one-hot encoding of the previous send result.
        """
        result = np.zeros(RESULT_COUNT, dtype=np.float32)
        result[self.last_results[agent_id]] = 1.0
        episode_total = float(
            self.config.episode_budget_bits
            or int(math.ceil(self.max_steps / 3.0)) * self.message_bits(MAP_PATCH)
        )
        cooldown_denominator = float(max(1, self.config.cooldown_steps))
        return np.concatenate(
            [
                np.array(
                    [
                        self.tokens[agent_id]
                        / float(self.config.bucket_capacity_bits),
                        self.episode_budget[agent_id] / episode_total,
                        self.cooldowns[agent_id] / cooldown_denominator,
                    ],
                    dtype=np.float32,
                ),
                result,
            ]
        )

    def transmit(self, actions, positions, headings):
        """Process one simultaneous communication decision for all agents.

        Eligible non-silent attempts are charged immediately, even if they
        later collide or have no in-range receiver.  In ``shared_collision``
        mode, more than one eligible sender causes every such attempt to fail.
        In ``parallel`` mode, each eligible sender is enqueued independently.

        Args:
            actions: One categorical communication action per agent.
            positions: Current grid positions, used for range checks.
            headings: Current headings, copied into pose payloads.

        Returns:
            Tuple ``(penalties, attempted_bits)`` with one value per sender.
            Penalties are patch-normalized reward costs, not network fees.
        """
        if len(actions) != self.num_agents:
            raise ValueError("actions must contain one choice per agent")
        self.last_results[:] = RESULT_SILENT
        attempted_bits = np.zeros(self.num_agents, dtype=np.int64)
        penalties = np.zeros(self.num_agents, dtype=np.float32)
        valid = []
        if self.config.mode in (PERFECT, NONE):
            return penalties, attempted_bits
        for agent_id, raw_action in enumerate(actions):
            action = int(raw_action)
            if action == SILENCE:
                continue
            message_type = POSE if action == POSE else MAP_PATCH
            bits = self.message_bits(message_type)
            if self.cooldowns[agent_id] > 0:
                self.last_results[agent_id] = RESULT_COOLDOWN
                continue
            if not self._has_budget(agent_id, bits):
                self.last_results[agent_id] = RESULT_BUDGET
                continue
            if message_type == MAP_PATCH:
                candidate_index = action - MAP_PATCH
                if not 0 <= candidate_index < self.config.candidate_count:
                    self.last_results[agent_id] = RESULT_BUDGET
                    continue
            valid.append((agent_id, action, message_type, bits))
            attempted_bits[agent_id] = bits
            self.tokens[agent_id] -= bits
            self.episode_budget[agent_id] -= bits
            self.cooldowns[agent_id] = self.config.cooldown_steps
            penalties[agent_id] = (
                self.config.cost_per_patch
                * bits
                / float(self.message_bits(MAP_PATCH))
            )
            self._metrics["attempted_messages"] += 1
            self._metrics["attempted_bits"] += bits
        if self.config.mode == SHARED_COLLISION and len(valid) > 1:
            for agent_id, _, _, _ in valid:
                self.last_results[agent_id] = RESULT_COLLISION
                self._metrics["collision_messages"] += 1
            return penalties, attempted_bits
        for agent_id, action, message_type, bits in valid:
            self.last_results[agent_id] = RESULT_TRANSMITTED
            self._metrics["transmitted_messages"] += 1
            if message_type == POSE:
                self._metrics["pose_messages"] += 1
            else:
                self._metrics["map_patch_messages"] += 1
            self._enqueue(
                agent_id, action, message_type, bits, positions, headings
            )
        return penalties, attempted_bits

    def advance(self, private_maps, positions, headings):
        """Advance one logical slot, update local knowledge, and deliver due data.

        Cooldowns decrement and tokens refill first.  Current private sensor
        maps are then written into beliefs before due remote messages arrive,
        allowing equally recent local sensing to win timestamp ties.  Perfect
        mode finishes with an instantaneous team synchronization.

        Returns:
            List of messages delivered during this advance.  One message may
            contain multiple receiver deliveries in its metrics.
        """
        self.current_step += 1
        self.cooldowns = np.maximum(0, self.cooldowns - 1)
        self.tokens = np.minimum(
            self.config.bucket_capacity_bits,
            self.tokens + self.config.bucket_refill_bits,
        )
        self._update_private_maps(private_maps)
        delivered = []
        while self._in_flight and self._in_flight[0][0] <= self.current_step:
            _, _, message = heapq.heappop(self._in_flight)
            if self.current_step - message.sent_step > self.config.ttl_steps:
                self._metrics["expired_messages"] += len(message.recipients)
                continue
            self._deliver(message)
            delivered.append(message)
        if self.config.mode == PERFECT:
            self._synchronize_perfect(private_maps, positions, headings)
        self.refresh_candidates()
        return delivered

    def metrics(self):
        """Return a defensive copy of cumulative episode communication metrics."""
        result = copy.deepcopy(self._metrics)
        delivered = result["delivered_messages"]
        result["mean_latency"] = (
            result["latency_total"] / float(delivered) if delivered else 0.0
        )
        return result

    @property
    def in_flight_count(self):
        """Number of delayed messages currently waiting in the priority queue."""
        return len(self._in_flight)

    def peer_pose_age(self, receiver, sender):
        """Return age of the last delivered pose, or ``None`` if never received."""
        step = self.peer_pose_steps[receiver, sender]
        if step < 0:
            return None
        return self.current_step - step

    def _has_budget(self, agent_id, bits):
        """Check both the refilling token bucket and non-refilling allowance."""
        return self.tokens[agent_id] >= bits and self.episode_budget[agent_id] >= bits

    def _enqueue(self, sender, action, message_type, bits, positions, headings):
        """Freeze recipients, latency, and payload for one accepted attempt.

        Range and independent whole-message loss are evaluated once at send
        time.  Delivery is all-or-nothing for each receiver: this model does
        not simulate partial payloads or corrupted bits.
        """
        recipients = []
        sender_position = np.asarray(positions[sender], dtype=np.float32)
        for receiver in range(self.num_agents):
            if receiver == sender:
                continue
            receiver_position = np.asarray(positions[receiver], dtype=np.float32)
            if np.linalg.norm(sender_position - receiver_position) > self.config.radio_range_cells:
                continue
            self._metrics["in_range_recipients"] += 1
            if self._rng.random_sample() < self.config.packet_loss:
                self._metrics["lost_messages"] += 1
                continue
            recipients.append(receiver)
        latency = int(
            self._rng.randint(
                self.config.latency_min_steps,
                self.config.latency_max_steps + 1,
            )
        )
        if message_type == POSE:
            payload = {
                "position": tuple(int(value) for value in positions[sender]),
                "heading": int(headings[sender]),
            }
        else:
            candidate = self.candidates[sender][action - MAP_PATCH]
            row = candidate.row
            column = candidate.column
            size = self.config.tile_size
            patch = np.full((size, size), UNKNOWN, dtype=np.uint8)
            row_end = min(self.width, row + size)
            column_end = min(self.height, column + size)
            patch[: row_end - row, : column_end - column] = self.belief_maps[
                sender
            ][row:row_end, column:column_end]
            payload = {"row": row, "column": column, "cells": patch}
            self.last_broadcast_maps[sender][
                row:row_end, column:column_end
            ] = self.belief_maps[sender][row:row_end, column:column_end]
        message = Message(
            sender,
            message_type,
            self.current_step,
            self.current_step + latency,
            bits,
            payload,
            recipients,
        )
        if recipients:
            if latency == 0:
                self._deliver(message)
            else:
                heapq.heappush(
                    self._in_flight,
                    (message.deliver_step, self._sequence, message),
                )
                self._sequence += 1

    def _deliver(self, message):
        """Apply one complete pose or patch snapshot to every recipient.

        Pose delivery updates the receiver's last-known peer state.  Patch
        delivery applies only known cells that are newer than the receiver's
        value.  On equal timestamps, a local observation wins; remote/remote
        ties use the fixed order occupied > free > unknown so queue order does
        not determine the map.
        """
        for receiver in message.recipients:
            if message.message_type == POSE:
                self.peer_positions[receiver, message.sender] = message.payload[
                    "position"
                ]
                self.peer_headings[receiver, message.sender] = message.payload[
                    "heading"
                ]
                self.peer_pose_steps[receiver, message.sender] = message.sent_step
            else:
                row = message.payload["row"]
                column = message.payload["column"]
                cells = message.payload["cells"]
                row_end = min(self.width, row + cells.shape[0])
                column_end = min(self.height, column + cells.shape[1])
                source = cells[: row_end - row, : column_end - column]
                target = self.belief_maps[receiver][row:row_end, column:column_end]
                timestamps = self.belief_timestamps[receiver][
                    row:row_end, column:column_end
                ]
                local_timestamps = self.belief_local_timestamps[receiver][
                    row:row_end, column:column_end
                ]
                source_priority = np.where(
                    source == OCCUPIED, 2, np.where(source == FREE, 1, 0)
                )
                target_priority = np.where(
                    target == OCCUPIED, 2, np.where(target == FREE, 1, 0)
                )
                newer = message.sent_step > timestamps
                remote_tie = (message.sent_step == timestamps) & (
                    local_timestamps != message.sent_step
                )
                update = (source != UNKNOWN) & (
                    newer | (remote_tie & (source_priority > target_priority))
                )
                target[update] = source[update]
                timestamps[update] = message.sent_step
            self._metrics["delivered_messages"] += 1
            self._metrics["delivered_bits"] += message.bit_count
            self._metrics["latency_total"] += self.current_step - message.sent_step

    def _update_private_maps(self, private_maps):
        """Fuse each agent's current direct sensor observations into its belief."""
        for agent_id, private_map in enumerate(private_maps):
            private_map = np.asarray(private_map)
            known = private_map != UNKNOWN
            self.belief_maps[agent_id][known] = private_map[known]
            self.belief_timestamps[agent_id][known] = self.current_step
            self.belief_local_timestamps[agent_id][known] = self.current_step

    def _synchronize_perfect(self, private_maps, positions, headings):
        """Apply the instantaneous, unlimited-information upper-bound control."""
        merged = np.full((self.width, self.height), UNKNOWN, dtype=np.uint8)
        free = np.zeros((self.width, self.height), dtype=bool)
        occupied = np.zeros((self.width, self.height), dtype=bool)
        for private_map in private_maps:
            private_map = np.asarray(private_map)
            free |= private_map == FREE
            occupied |= private_map == OCCUPIED
        merged[free] = FREE
        merged[occupied] = OCCUPIED
        for receiver in range(self.num_agents):
            self.belief_maps[receiver] = np.array(merged, copy=True)
            self.belief_timestamps[receiver][merged != UNKNOWN] = self.current_step
            for sender in range(self.num_agents):
                if receiver == sender:
                    continue
                self.peer_positions[receiver, sender] = positions[sender]
                self.peer_headings[receiver, sender] = headings[sender]
                self.peer_pose_steps[receiver, sender] = self.current_step

    def _candidate_tiles(self, agent_id):
        """Rank and describe the top map tiles available to one sender.

        Tiles are ordered by newly known cells, then frontier cells, then
        row-major position.  The deterministic final tie-break makes a
        categorical action retain the same meaning across repeated runs.
        """
        size = self.config.tile_size
        belief = self.belief_maps[agent_id]
        previous = self.last_broadcast_maps[agent_id]
        tile_rows = int(math.ceil(self.width / float(size)))
        tile_columns = int(math.ceil(self.height / float(size)))
        frontier = self._frontier_mask(belief)
        scored = []
        for tile_row in range(tile_rows):
            row = tile_row * size
            row_end = min(self.width, row + size)
            for tile_column in range(tile_columns):
                column = tile_column * size
                column_end = min(self.height, column + size)
                tile = belief[row:row_end, column:column_end]
                old_tile = previous[row:row_end, column:column_end]
                known = tile != UNKNOWN
                changed = known & (tile != old_tile)
                frontier_count = int(
                    frontier[row:row_end, column:column_end].sum()
                )
                scored.append(
                    (
                        -int(changed.sum()),
                        -frontier_count,
                        tile_row,
                        tile_column,
                        int(known.sum()),
                    )
                )
        scored.sort()
        candidates = []
        cell_count = float(size * size)
        for changed_negative, frontier_negative, tile_row, tile_column, known_count in scored[
            : self.config.candidate_count
        ]:
            row = tile_row * size
            column = tile_column * size
            candidates.append(
                TileCandidate(
                    row,
                    column,
                    [
                        row / float(max(1, self.width - 1)),
                        column / float(max(1, self.height - 1)),
                        known_count / cell_count,
                        -changed_negative / cell_count,
                        -frontier_negative / cell_count,
                    ],
                )
            )
        return candidates

    @staticmethod
    def _frontier_mask(grid):
        """Return unknown cells sharing a four-neighbor edge with known free space."""
        free = grid == FREE
        unknown = grid == UNKNOWN
        adjacent_free = np.zeros(grid.shape, dtype=bool)
        adjacent_free[1:, :] |= free[:-1, :]
        adjacent_free[:-1, :] |= free[1:, :]
        adjacent_free[:, 1:] |= free[:, :-1]
        adjacent_free[:, :-1] |= free[:, 1:]
        return unknown & adjacent_free
