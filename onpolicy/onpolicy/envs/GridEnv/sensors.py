import copy
import math

import numpy as np


__all__ = [
    "UNKNOWN",
    "FREE",
    "OCCUPIED",
    "OMNIDIRECTIONAL",
    "FOUR_BEAM",
    "SensorConfig",
    "BeamReading",
    "FourBeamMeasurement",
    "OmnidirectionalMeasurement",
    "SensorFrame",
    "SensorModel",
    "OmnidirectionalSensor",
    "FourBeamSensor",
    "normalize_yaw",
    "yaw_to_cardinal",
    "create_sensor",
    "normalize_sensor_configs",
    "sensor_configs_from_values",
]


UNKNOWN = 205
FREE = 254
OCCUPIED = 0

OMNIDIRECTIONAL = "omnidirectional"
FOUR_BEAM = "four_beam"

BEAM_OFFSETS = (
    ("front", 0.0),
    ("back", math.pi),
    ("left", math.pi / 2.0),
    ("right", -math.pi / 2.0),
)


class SensorConfig:
    def __init__(self, sensor_type=OMNIDIRECTIONAL, max_range=3.0):
        if sensor_type not in (OMNIDIRECTIONAL, FOUR_BEAM):
            raise ValueError("unsupported sensor type: {}".format(sensor_type))
        max_range = float(max_range)
        if math.isnan(max_range) or max_range <= 0:
            raise ValueError("sensor max_range must be positive or infinity")

        self.sensor_type = sensor_type
        self.max_range = max_range

    def __repr__(self):
        return "SensorConfig(sensor_type={!r}, max_range={!r})".format(
            self.sensor_type, self.max_range
        )


class BeamReading:
    def __init__(self, name, angle, distance, hit, cells):
        self.name = name
        self.angle = float(angle)
        self.distance = float(distance)
        self.hit = bool(hit)
        self.cells = tuple(cells)

    def public(self):
        return {"distance": self.distance, "hit": self.hit}


class FourBeamMeasurement:
    def __init__(self, position, yaw, readings):
        self.position = tuple(position)
        self.yaw = float(yaw)
        self.readings = readings


class OmnidirectionalMeasurement:
    def __init__(self, mask_map, position, x_bounds, y_bounds, immediate_cells):
        self.mask_map = mask_map
        self.position = tuple(position)
        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.immediate_cells = tuple(immediate_cells)


class SensorFrame:
    def __init__(self, local_map, full_map, x_bounds, y_bounds, measurement):
        self.local_map = local_map
        self.full_map = full_map
        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.measurement = measurement


class SensorModel:
    def __init__(self, config):
        self.config = config

    def measure(self, position, yaw, ground_truth_map, resolution):
        raise NotImplementedError

    def project(self, measurement, map_shape):
        raise NotImplementedError

    def sense(self, position, yaw, ground_truth_map, resolution):
        measurement = self.measure(position, yaw, ground_truth_map, resolution)
        return self.project(measurement, ground_truth_map.shape)

    def public_reading(self, measurement):
        return {
            "sensor_type": self.config.sensor_type,
            "max_range": self.config.max_range,
        }


class OmnidirectionalSensor(SensorModel):
    """The original Level-0 scanner, kept behaviorally compatible."""

    def measure(self, position, yaw, ground_truth_map, resolution):
        del yaw
        height, width = ground_truth_map.shape
        if math.isinf(self.config.max_range):
            grid_range = max(height, width)
        else:
            grid_range = int(self.config.max_range / resolution)

        x, y = int(position[0]), int(position[1])
        x_min, y_min = max(0, x - grid_range), max(0, y - grid_range)
        x_max, y_max = min(x + grid_range, height), min(y + grid_range, width)
        init_map = ground_truth_map[x_min:x_max, y_min:y_max]
        mask_map = copy.deepcopy(init_map)
        map_origin = [x - x_min, y - y_min]

        for j in range(mask_map.shape[1]):
            laser_path = self._simulate_laser(0, j, map_origin)
            self._mask_behind_obstacle(
                laser_path, [0, j], init_map, mask_map, map_origin, axis=1
            )
        for j in range(mask_map.shape[1]):
            laser_path = self._simulate_laser(mask_map.shape[0] - 1, j, map_origin)
            self._mask_behind_obstacle(
                laser_path,
                [mask_map.shape[0] - 1, j],
                init_map,
                mask_map,
                map_origin,
                axis=1,
            )
        for i in range(mask_map.shape[0]):
            laser_path = self._simulate_laser(i, 0, map_origin)
            self._mask_behind_obstacle(
                laser_path, [i, 0], init_map, mask_map, map_origin, axis=0
            )
        for i in range(mask_map.shape[0]):
            laser_path = self._simulate_laser(i, mask_map.shape[1] - 1, map_origin)
            self._mask_behind_obstacle(
                laser_path,
                [i, mask_map.shape[1] - 1],
                init_map,
                mask_map,
                map_origin,
                axis=0,
            )

        immediate_cells = []
        for target_x in range(x - 1, x + 2):
            for target_y in range(y - 1, y + 2):
                immediate_cells.append(
                    (target_x, target_y, ground_truth_map[target_x, target_y])
                )
        return OmnidirectionalMeasurement(
            mask_map,
            position,
            (x_min, x_max),
            (y_min, y_max),
            immediate_cells,
        )

    def project(self, measurement, map_shape):
        full_map = np.full(map_shape, UNKNOWN, dtype=float)
        x_min, x_max = measurement.x_bounds
        y_min, y_max = measurement.y_bounds
        full_map[x_min:x_max, y_min:y_max] = measurement.mask_map

        # Preserve the legacy scanner's immediate 3x3 reveal.
        for target_x, target_y, value in measurement.immediate_cells:
            full_map[target_x, target_y] = value

        return SensorFrame(
            measurement.mask_map,
            full_map,
            measurement.x_bounds,
            measurement.y_bounds,
            measurement,
        )

    @staticmethod
    def _mask_behind_obstacle(
        laser_path, endpoint, init_map, mask_map, map_origin, axis
    ):
        laser_path.reverse()
        laser_path.append(endpoint)
        for idx, point in enumerate(laser_path[:-1]):
            next_point = laser_path[idx + 1]
            point_is_obstacle = init_map[point[0], point[1]] == OCCUPIED
            next_is_obstacle = init_map[next_point[0], next_point[1]] == OCCUPIED
            changes_row_or_column = point[axis] != next_point[axis]
            beside_origin = point[axis] in (
                map_origin[axis] - 1,
                map_origin[axis],
            )
            if point_is_obstacle and (
                not next_is_obstacle or changes_row_or_column or beside_origin
            ):
                for hidden_point in laser_path[idx + 1 :]:
                    mask_map[hidden_point[0], hidden_point[1]] = UNKNOWN
                break

    @classmethod
    def _simulate_laser(cls, i, j, map_origin):
        row_offset = i + 0.5 - map_origin[0]
        column_offset = j + 0.5 - map_origin[1]
        if row_offset < -1 and column_offset < -1:
            return cls._return_laser_path(i + 1, j + 1, map_origin)
        if row_offset > 1 and column_offset < -1:
            return cls._return_laser_path(i, j + 1, map_origin)
        if row_offset < -1 and column_offset > 1:
            return cls._return_laser_path(i + 1, j, map_origin)
        if row_offset > 1 and column_offset > 1:
            return cls._return_laser_path(i, j, map_origin)
        if i - map_origin[0] == -1 and j - map_origin[1] < -1:
            return cls._return_laser_path(i, j + 1, map_origin)
        if i - map_origin[0] == -1 and j - map_origin[1] > 1:
            return cls._return_laser_path(i, j, map_origin)
        if i - map_origin[0] == 0 and j - map_origin[1] < -1:
            return cls._return_laser_path(i + 1, j + 1, map_origin)
        if i - map_origin[0] == 0 and j - map_origin[1] > 1:
            return cls._return_laser_path(i + 1, j, map_origin)
        if i - map_origin[0] < -1 and j - map_origin[1] == -1:
            return cls._return_laser_path(i + 1, j, map_origin)
        if i - map_origin[0] > 1 and j - map_origin[1] == -1:
            return cls._return_laser_path(i, j, map_origin)
        if i - map_origin[0] < -1 and j - map_origin[1] == 0:
            return cls._return_laser_path(i + 1, j + 1, map_origin)
        if i - map_origin[0] > 1 and j - map_origin[1] == 0:
            return cls._return_laser_path(i, j + 1, map_origin)
        return []

    @staticmethod
    def _return_laser_path(i, j, map_origin):
        laser_path = []
        step_length = max(abs(i - map_origin[0]), abs(j - map_origin[1]))
        path_x = np.linspace(i, map_origin[0], int(step_length) + 2)
        path_y = np.linspace(j, map_origin[1], int(step_length) + 2)
        for step in range(1, int(step_length) + 1):
            laser_path.append(
                [int(math.floor(path_x[step])), int(math.floor(path_y[step]))]
            )
        return laser_path


class FourBeamSensor(SensorModel):
    def measure(self, position, yaw, ground_truth_map, resolution):
        yaw = normalize_yaw(yaw)
        readings = {}
        for name, offset in BEAM_OFFSETS:
            angle = normalize_yaw(yaw + offset)
            readings[name] = self._cast_ray(
                name, angle, position, ground_truth_map, resolution
            )
        return FourBeamMeasurement(position, yaw, readings)

    def project(self, measurement, map_shape):
        full_map = np.full(map_shape, UNKNOWN, dtype=float)
        x, y = measurement.position
        visible_cells = [(x, y, FREE)]
        for reading in measurement.readings.values():
            visible_cells.extend(reading.cells)

        for row, column, value in visible_cells:
            full_map[row, column] = value

        rows = [cell[0] for cell in visible_cells]
        columns = [cell[1] for cell in visible_cells]
        x_bounds = (min(rows), max(rows) + 1)
        y_bounds = (min(columns), max(columns) + 1)
        local_map = full_map[
            x_bounds[0] : x_bounds[1], y_bounds[0] : y_bounds[1]
        ]
        return SensorFrame(
            local_map, full_map, x_bounds, y_bounds, measurement
        )

    def public_reading(self, measurement):
        result = super().public_reading(measurement)
        result["beams"] = {
            name: reading.public()
            for name, reading in measurement.readings.items()
        }
        return result

    def _cast_ray(self, name, angle, position, ground_truth_map, resolution):
        row, column = int(position[0]), int(position[1])
        direction_row = -math.sin(angle)
        direction_column = math.cos(angle)
        step_row = 1 if direction_row > 0 else -1
        step_column = 1 if direction_column > 0 else -1

        t_delta_row = (
            abs(1.0 / direction_row) if abs(direction_row) > 1e-12 else math.inf
        )
        t_delta_column = (
            abs(1.0 / direction_column)
            if abs(direction_column) > 1e-12
            else math.inf
        )
        t_max_row = 0.5 * t_delta_row
        t_max_column = 0.5 * t_delta_column
        cells = []
        hit = False
        hit_distance = self.config.max_range

        while True:
            if math.isclose(t_max_row, t_max_column, rel_tol=0.0, abs_tol=1e-12):
                distance_cells = t_max_row
                row += step_row
                column += step_column
                t_max_row += t_delta_row
                t_max_column += t_delta_column
            elif t_max_row < t_max_column:
                distance_cells = t_max_row
                row += step_row
                t_max_row += t_delta_row
            else:
                distance_cells = t_max_column
                column += step_column
                t_max_column += t_delta_column

            distance = distance_cells * resolution
            beyond_range = distance > self.config.max_range and not math.isclose(
                distance,
                self.config.max_range,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            if beyond_range:
                break
            if not (
                0 <= row < ground_truth_map.shape[0]
                and 0 <= column < ground_truth_map.shape[1]
            ):
                break

            value = ground_truth_map[row, column]
            if value == UNKNOWN:
                break
            cells.append((row, column, int(value)))
            if value == OCCUPIED:
                hit = True
                hit_distance = distance
                break

        return BeamReading(name, angle, hit_distance, hit, cells)


def normalize_yaw(yaw):
    return float(yaw) % (2.0 * math.pi)


def yaw_to_cardinal(yaw):
    return int(round(normalize_yaw(yaw) / (math.pi / 2.0))) % 4


def create_sensor(config):
    if config.sensor_type == OMNIDIRECTIONAL:
        return OmnidirectionalSensor(config)
    if config.sensor_type == FOUR_BEAM:
        return FourBeamSensor(config)
    raise ValueError("unsupported sensor type: {}".format(config.sensor_type))


def normalize_sensor_configs(sensor_configs, num_agents, default_range):
    if sensor_configs is None:
        sensor_configs = [SensorConfig(OMNIDIRECTIONAL, default_range)]
    elif isinstance(sensor_configs, SensorConfig):
        sensor_configs = [sensor_configs]
    else:
        sensor_configs = list(sensor_configs)

    if len(sensor_configs) == 1:
        sensor_configs = sensor_configs * num_agents
    if len(sensor_configs) != num_agents:
        raise ValueError(
            "expected one sensor config or {} per-agent configs, got {}".format(
                num_agents, len(sensor_configs)
            )
        )
    if not all(isinstance(config, SensorConfig) for config in sensor_configs):
        raise TypeError("sensor_configs must contain SensorConfig instances")
    return sensor_configs


def sensor_configs_from_values(sensor_types, sensor_ranges, num_agents, default_range):
    sensor_types = list(sensor_types or [OMNIDIRECTIONAL])
    sensor_ranges = list(sensor_ranges or [default_range])

    if len(sensor_types) == 1:
        sensor_types *= num_agents
    if len(sensor_ranges) == 1:
        sensor_ranges *= num_agents
    if len(sensor_types) != num_agents or len(sensor_ranges) != num_agents:
        raise ValueError(
            "sensor type and range lists must each contain one value or one per agent"
        )
    return [
        SensorConfig(sensor_type, max_range)
        for sensor_type, max_range in zip(sensor_types, sensor_ranges)
    ]
