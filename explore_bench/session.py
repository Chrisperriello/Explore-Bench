"""Fixed-clock Level-0 sessions using the canonical sensors and radio broker.

Historical GridEnv entry points are intentionally not used for transitions:
their full-route actions and early-stop reward are different semantics.
"""

import copy
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
from PIL import Image

from onpolicy.envs.GridEnv.Astar import AStar
from onpolicy.envs.GridEnv.GridEnv import GridEnv
from onpolicy.envs.GridEnv.communication import CommunicationConfig, FREE, UNKNOWN, OCCUPIED
from onpolicy.envs.GridEnv.observations import build_communication_observations
from onpolicy.envs.GridEnv.sensors import SensorConfig, normalize_yaw, yaw_to_cardinal

from .config import file_hash, map_path


class BoundedAStar(AStar):
    """The existing A* algorithm with bounded neighbors and no corner cutting."""

    def get_neighbor(self, position):
        rows, columns = self.map.shape
        result = []
        for dr, dc in self.u_set:
            row, column = position[0] + dr, position[1] + dc
            if not (0 <= row < rows and 0 <= column < columns):
                continue
            if dr and dc and (self.map[position[0] + dr, position[1]] != FREE
                              or self.map[position[0], position[1] + dc] != FREE):
                continue
            result.append((row, column))
        return result


def frontier_candidates(belief):
    """One deterministic free-cell representative per 8-connected frontier.

    A frontier is a known free cell adjacent (4-connected) to unknown space.
    Unlike the old selector, this permits cells learned from delivered maps.
    """
    unknown = belief == UNKNOWN
    adjacent = np.zeros_like(unknown)
    adjacent[1:] |= unknown[:-1]
    adjacent[:-1] |= unknown[1:]
    adjacent[:, 1:] |= unknown[:, :-1]
    adjacent[:, :-1] |= unknown[:, 1:]
    pending = set(map(tuple, np.argwhere(adjacent & (belief == FREE))))
    candidates = []
    while pending:
        first = min(pending)
        pending.remove(first)
        queue, group = deque([first]), [first]
        while queue:
            row, column = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (row + dr, column + dc)
                    if neighbor in pending:
                        pending.remove(neighbor)
                        queue.append(neighbor)
                        group.append(neighbor)
        center = np.mean(group, axis=0)
        candidates.append(min(group, key=lambda cell: (float(np.sum((cell - center) ** 2)), cell)))
    return sorted(candidates)


class ObservationBuilder:
    """Own episode histories; expose the shared, tested CTDE preprocessing."""

    def __init__(self, count, shape, cap, radio):
        self.num_agents = count
        self.full_w, self.full_h = shape
        self.input_w = self.input_h = 64
        self.augment = 255 // sum(range(1, count + 1))
        self.max_steps = cap
        self.all_args = SimpleNamespace(comm_candidate_count=radio.candidate_count,
                                        comm_ttl_steps=radio.ttl_steps)
        self.all_agent_pos_map = np.zeros((1, count, *shape), dtype=np.float32)
        self.all_merge_pos_map = np.zeros((1, *shape), dtype=np.float32)

    def build(self, info):
        _, actor, critic, available = build_communication_observations(self, [None], [info])
        return ({key: value[0] for key, value in actor.items()},
                {key: value[0] for key, value in critic.items()}, available[0])


class Session(GridEnv):
    """One explicitly reset episode; step requests are semantic grid goals."""

    def __init__(self, spec):
        self.spec = copy.deepcopy(spec)
        radio = CommunicationConfig(**spec["communication"])
        super().__init__(spec["resolution"], spec["sensor_range"], spec["num_agents"],
                         spec["max_steps"], visualization=False,
                         sensor_configs=[SensorConfig(spec["sensor_type"], spec["sensor_range"])] * spec["num_agents"],
                         communication_config=radio)
        self.task = None

    def _sense_agent(self, agent_id, position, yaw):
        # Research sessions already use integer grid positions. The legacy
        # grid->metres->grid round trip truncates negative/half-cell values
        # and can shift sensing onto a neighboring wall (especially odd maps).
        # Call the same canonical sensor directly at the executed grid pose.
        frame = self.sensors[agent_id].sense(position, yaw, self.gt_map, self.resolution)
        self.latest_sensor_readings[agent_id] = self.sensors[agent_id].public_reading(frame.measurement)
        return frame

    def reset_task(self, task):
        if file_hash(map_path(task.map_path)) != task.map_sha256:
            raise ValueError("task map hash changed")
        self.task = task
        self.seed(task.seed)
        self.gt_map = np.asarray(Image.open(map_path(task.map_path))).copy()
        self.width, self.height = self.gt_map.shape
        self.agent_pos = copy.deepcopy(task.start_positions)
        self.agent_yaw = list(task.start_headings_rad)
        if len(self.agent_pos) != self.num_agents or len(self.agent_yaw) != self.num_agents:
            raise ValueError("task agent count changed")
        for position in self.agent_pos:
            row, column = position
            if not (0 <= row < self.width and 0 <= column < self.height) or self.gt_map[row, column] != FREE:
                raise ValueError("invalid spawn")
        self.agent_dir = [yaw_to_cardinal(value) for value in self.agent_yaw]
        self.num_step = 0
        self.path_lengths_m = np.zeros(self.num_agents, dtype=np.float64)
        self.path_log = [[list(position)] for position in self.agent_pos]
        self.built_map = [self._sense_agent(i, position, self.agent_yaw[i]).full_map.copy()
                          for i, position in enumerate(self.agent_pos)]
        self.communication_broker.reset(self.width, self.height, self.max_steps,
                                         self.built_map, self.agent_pos, self.agent_dir)
        self.builder = ObservationBuilder(self.num_agents, self.gt_map.shape, self.max_steps,
                                          self.communication_config)
        self.initial_cells = self.measurement()["team_coverage_cells"]
        self.previous_cells = self.initial_cells
        self.task_reward_total = 0.0
        self.communication_cost_total = 0.0
        self.learning_reward_total = np.zeros(self.num_agents)
        self.last_goals = np.asarray(self.agent_pos).copy()
        self.last_communication_actions = np.zeros(self.num_agents, dtype=np.int64)
        self.last_paths = [[list(position)] for position in self.agent_pos]
        self.last_reward = {"task_reward": 0.0, "communication_cost": 0.0,
                            "learning_rewards": np.zeros(self.num_agents)}
        self.terminated = self.measurement()["team_coverage_ratio"] >= self.spec["target_coverage"]
        self.truncated = False
        return self.observation()

    def _info(self):
        known = np.stack([grid != UNKNOWN for grid in self.built_map])
        obstacles = np.stack([grid == OCCUPIED for grid in self.built_map])
        info = {"current_agent_pos": np.asarray(self.agent_pos),
                "explored_each_map": known, "obstacle_each_map": obstacles,
                "explored_all_map": np.any(known, axis=0),
                "obstacle_all_map": np.any(obstacles, axis=0)}
        self._add_communication_info(info)
        return info

    def observation(self):
        return self.builder.build(self._info())

    def route(self, agent, goal):
        start, goal = tuple(self.agent_pos[agent]), tuple(map(int, goal))
        belief = self._navigation_map(agent)
        if start == goal:
            return [start]
        if not (0 <= goal[0] < self.width and 0 <= goal[1] < self.height) or belief[goal] != FREE:
            return [start]
        try:
            path, _ = BoundedAStar(start, goal, belief, "euclidean").searching()
            return list(reversed(path))
        except KeyError:
            return [start]

    def decode_coordinates(self, coordinates):
        """Snap normalized coordinates to reachable belief-frontier goals."""
        coordinates = np.asarray(coordinates)
        if coordinates.shape != (self.num_agents, 2) or not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be a finite [agents, 2] array")
        goals = []
        for agent, coordinate in enumerate(coordinates):
            desired = np.clip(coordinate, 0, 1) * np.asarray([self.width - 1, self.height - 1])
            candidates = frontier_candidates(self.communication_broker.belief_maps[agent])
            ordered = sorted(candidates, key=lambda cell: (float(np.sum((cell - desired) ** 2)), cell))
            selected = self.agent_pos[agent]
            for candidate in ordered:
                if len(self.route(agent, candidate)) > 1:
                    selected = candidate
                    break
            goals.append(selected)
        return np.asarray(goals, dtype=np.int64)

    def step_request(self, request):
        if self.terminated or self.truncated:
            raise ValueError("cannot step a finished episode")
        goals = (self.decode_coordinates(request["coordinates"])
                 if "coordinates" in request else np.asarray(request["motion_goals"], dtype=np.int64))
        actions = np.asarray(request["comm_actions"], dtype=np.int64)
        if goals.shape != (self.num_agents, 2) or actions.shape != (self.num_agents,):
            raise ValueError("invalid semantic action shape")
        self.last_goals, self.last_communication_actions = goals.copy(), actions.copy()
        penalties, _ = self.communication_broker.transmit(actions, self.agent_pos, self.agent_dir)
        self.last_paths = []
        for agent, goal in enumerate(goals):
            path = self.route(agent, goal)[:2]
            self.last_paths.append([list(cell) for cell in path])
            if len(path) > 1:
                delta = np.asarray(path[1]) - path[0]
                self.path_lengths_m[agent] += np.linalg.norm(delta) * self.resolution
                pose = self.naive_local_planner(path)
                self.agent_pos[agent] = list(path[1])
                self.agent_yaw[agent] = normalize_yaw(pose[-1][1])
                self.agent_dir[agent] = yaw_to_cardinal(self.agent_yaw[agent])
                self.path_log[agent].append(list(path[1]))
            frame = self._sense_agent(agent, self.agent_pos[agent], self.agent_yaw[agent])
            observed = frame.full_map != UNKNOWN
            self.built_map[agent][observed] = frame.full_map[observed]
        self.num_step += 1
        self.communication_broker.advance(self.built_map, self.agent_pos, self.agent_dir)
        measurement = self.measurement()
        self.terminated = measurement["team_coverage_ratio"] >= self.spec["target_coverage"]
        self.truncated = self.num_step >= self.max_steps and not self.terminated
        new_cells = measurement["team_coverage_cells"] - self.previous_cells
        self.previous_cells = measurement["team_coverage_cells"]
        task_reward = self.spec["reward_new_cell"] * new_cells
        if self.terminated:
            task_reward += self.spec["reward_success"]
        rewards = task_reward - np.asarray(penalties, dtype=np.float32)
        self.task_reward_total += task_reward
        self.communication_cost_total += float(np.sum(penalties))
        self.learning_reward_total += rewards
        self.last_reward = {"task_reward": task_reward, "communication_cost": float(np.sum(penalties)),
                            "learning_rewards": rewards.copy()}
        return self.observation(), rewards[:, None], self.terminated, self.truncated, measurement

    def measurement(self):
        """Evaluator-only physical coverage and executed movement, independent of reward."""
        valid = self.gt_map != UNKNOWN
        known = np.stack([(grid != UNKNOWN) & valid for grid in self.built_map])
        agent_cells = known.reshape(self.num_agents, -1).sum(axis=1)
        union = int(np.any(known, axis=0).sum())
        total = int(valid.sum())
        ratios = agent_cells / float(total)
        overlap = int(agent_cells.sum() - union)
        return {"team_coverage_cells": union, "team_coverage_ratio": union / float(total),
                "total_explorable_cells": total, "agent_coverage_cells": agent_cells,
                "agent_coverage_ratios": ratios, "coverage_std_ratio": float(np.std(ratios)),
                "coverage_std_area_m2": float(np.std(agent_cells * self.resolution ** 2)),
                "overlap_cells": overlap, "overlap_ratio_total": overlap / float(total),
                "team_path_length_m": float(self.path_lengths_m.sum()),
                "maximum_agent_path_length_m": float(self.path_lengths_m.max()),
                "communication": self.communication_broker.metrics()}
