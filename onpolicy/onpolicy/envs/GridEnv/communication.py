"""Deterministic Level-0 communication and per-agent belief management."""

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
    value = int(value)
    if value <= 0:
        raise ValueError("{} must be positive".format(name))
    return value


def _nonnegative_int(name, value):
    value = int(value)
    if value < 0:
        raise ValueError("{} cannot be negative".format(name))
    return value


class CommunicationConfig:
    """Configuration for the logical Level-0 radio model."""

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
    def __init__(self, row, column, features):
        self.row = int(row)
        self.column = int(column)
        self.features = np.asarray(features, dtype=np.float32)


class Message:
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
    """A synchronous radio model advanced explicitly by the environment."""

    def __init__(self, config, num_agents, seed=None):
        if not isinstance(config, CommunicationConfig):
            raise TypeError("config must be a CommunicationConfig")
        self.config = config
        self.num_agents = _positive_int("num_agents", num_agents)
        self.seed(seed)
        self.width = None
        self.height = None

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)
        self._seed = seed
        return [seed]

    def reset(self, width, height, max_steps, private_maps, positions, headings):
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
        self.candidates = []
        for agent_id in range(self.num_agents):
            self.candidates.append(self._candidate_tiles(agent_id))

    def candidate_features(self, agent_id):
        return np.stack(
            [candidate.features for candidate in self.candidates[agent_id]], axis=0
        )

    def available_actions(self, agent_id):
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
        result = copy.deepcopy(self._metrics)
        delivered = result["delivered_messages"]
        result["mean_latency"] = (
            result["latency_total"] / float(delivered) if delivered else 0.0
        )
        return result

    @property
    def in_flight_count(self):
        return len(self._in_flight)

    def peer_pose_age(self, receiver, sender):
        step = self.peer_pose_steps[receiver, sender]
        if step < 0:
            return None
        return self.current_step - step

    def _has_budget(self, agent_id, bits):
        return self.tokens[agent_id] >= bits and self.episode_budget[agent_id] >= bits

    def _enqueue(self, sender, action, message_type, bits, positions, headings):
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
        for agent_id, private_map in enumerate(private_maps):
            private_map = np.asarray(private_map)
            known = private_map != UNKNOWN
            self.belief_maps[agent_id][known] = private_map[known]
            self.belief_timestamps[agent_id][known] = self.current_step
            self.belief_local_timestamps[agent_id][known] = self.current_step

    def _synchronize_perfect(self, private_maps, positions, headings):
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
        free = grid == FREE
        unknown = grid == UNKNOWN
        adjacent_free = np.zeros(grid.shape, dtype=bool)
        adjacent_free[1:, :] |= free[:-1, :]
        adjacent_free[:-1, :] |= free[1:, :]
        adjacent_free[:, 1:] |= free[:, :-1]
        adjacent_free[:, :-1] |= free[:, 1:]
        return unknown & adjacent_free
