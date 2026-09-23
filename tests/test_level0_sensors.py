import hashlib
import math
import unittest

import numpy as np

from onpolicy.envs.GridEnv.sensors import (
    FOUR_BEAM,
    FREE,
    OCCUPIED,
    OMNIDIRECTIONAL,
    UNKNOWN,
    FourBeamSensor,
    OmnidirectionalSensor,
    SensorConfig,
    normalize_sensor_configs,
    sensor_configs_from_values,
    yaw_to_cardinal,
)


class SensorConfigTests(unittest.TestCase):
    def test_single_config_broadcasts_to_each_agent(self):
        config = SensorConfig(FOUR_BEAM, 4.0)
        configs = normalize_sensor_configs(config, 3, 2.0)

        self.assertEqual(configs, [config, config, config])

    def test_cli_values_can_be_per_agent(self):
        configs = sensor_configs_from_values(
            [OMNIDIRECTIONAL, FOUR_BEAM], [3.0, math.inf], 2, 3.0
        )

        self.assertEqual(configs[0].sensor_type, OMNIDIRECTIONAL)
        self.assertEqual(configs[0].max_range, 3.0)
        self.assertEqual(configs[1].sensor_type, FOUR_BEAM)
        self.assertTrue(math.isinf(configs[1].max_range))

    def test_invalid_configs_are_rejected(self):
        for value in (0, -1, math.nan):
            with self.assertRaises(ValueError):
                SensorConfig(FOUR_BEAM, value)
        with self.assertRaises(ValueError):
            SensorConfig("camera", 3.0)
        with self.assertRaises(ValueError):
            normalize_sensor_configs(
                [SensorConfig(), SensorConfig(), SensorConfig()], 2, 3.0
            )

    def test_exact_yaw_has_a_backward_compatible_cardinal_index(self):
        self.assertEqual(yaw_to_cardinal(0.0), 0)
        self.assertEqual(yaw_to_cardinal(math.pi / 2.0), 1)
        self.assertEqual(yaw_to_cardinal(math.pi), 2)
        self.assertEqual(yaw_to_cardinal(3.0 * math.pi / 2.0), 3)
        self.assertEqual(yaw_to_cardinal(2.0 * math.pi), 0)


class FourBeamSensorTests(unittest.TestCase):
    def setUp(self):
        self.origin = (5, 5)
        self.ground_truth = np.full((11, 11), FREE, dtype=np.uint8)
        self.ground_truth[5, 8] = OCCUPIED
        self.ground_truth[5, 3] = OCCUPIED
        self.ground_truth[2, 5] = OCCUPIED
        self.ground_truth[7, 5] = OCCUPIED

    def test_cardinal_beams_reveal_first_obstacle_and_report_distance(self):
        sensor = FourBeamSensor(SensorConfig(FOUR_BEAM, 10.0))
        frame = sensor.sense(self.origin, 0.0, self.ground_truth, 1.0)
        readings = frame.measurement.readings

        self.assertAlmostEqual(readings["front"].distance, 2.5)
        self.assertAlmostEqual(readings["back"].distance, 1.5)
        self.assertAlmostEqual(readings["left"].distance, 2.5)
        self.assertAlmostEqual(readings["right"].distance, 1.5)
        self.assertTrue(all(reading.hit for reading in readings.values()))
        self.assertEqual(frame.full_map[5, 8], OCCUPIED)
        self.assertEqual(frame.full_map[5, 9], UNKNOWN)
        self.assertEqual(frame.full_map[5, 5], FREE)
        self.assertEqual(frame.full_map[4, 4], UNKNOWN)

    def test_beams_rotate_with_exact_yaw(self):
        sensor = FourBeamSensor(SensorConfig(FOUR_BEAM, 10.0))
        measurement = sensor.measure(
            self.origin, math.pi / 2.0, self.ground_truth, 1.0
        )

        self.assertEqual(measurement.readings["front"].cells[-1][:2], (2, 5))
        self.assertEqual(measurement.readings["right"].cells[-1][:2], (5, 8))

    def test_diagonal_beam_uses_one_cell_wide_grid_traversal(self):
        ground_truth = np.full((9, 9), FREE, dtype=np.uint8)
        ground_truth[1, 7] = OCCUPIED
        sensor = FourBeamSensor(SensorConfig(FOUR_BEAM, 10.0))
        reading = sensor.measure(
            (4, 4), math.pi / 4.0, ground_truth, 1.0
        ).readings["front"]

        self.assertEqual(
            [cell[:2] for cell in reading.cells], [(3, 5), (2, 6), (1, 7)]
        )
        self.assertAlmostEqual(reading.distance, math.sqrt(0.5 ** 2 * 2) + 2 * math.sqrt(2))

    def test_finite_miss_reports_max_range(self):
        ground_truth = np.full((9, 9), FREE, dtype=np.uint8)
        sensor = FourBeamSensor(SensorConfig(FOUR_BEAM, 1.0))
        reading = sensor.measure((4, 4), 0.0, ground_truth, 1.0).readings[
            "front"
        ]

        self.assertFalse(reading.hit)
        self.assertEqual(reading.distance, 1.0)
        self.assertEqual([cell[:2] for cell in reading.cells], [(4, 5)])

    def test_obstacle_exactly_at_max_range_is_a_hit(self):
        sensor = FourBeamSensor(SensorConfig(FOUR_BEAM, 2.5))
        reading = sensor.measure(
            self.origin, 0.0, self.ground_truth, 1.0
        ).readings["front"]

        self.assertTrue(reading.hit)
        self.assertEqual(reading.distance, 2.5)

    def test_infinite_miss_reaches_boundary_and_reports_infinity(self):
        ground_truth = np.full((7, 7), FREE, dtype=np.uint8)
        sensor = FourBeamSensor(SensorConfig(FOUR_BEAM, math.inf))
        frame = sensor.sense((3, 3), 0.0, ground_truth, 1.0)
        front = frame.measurement.readings["front"]

        self.assertFalse(front.hit)
        self.assertTrue(math.isinf(front.distance))
        self.assertEqual([cell[:2] for cell in front.cells], [(3, 4), (3, 5), (3, 6)])
        self.assertEqual(frame.full_map[3, 6], FREE)

    def test_unknown_ground_truth_stops_without_revealing_unknown_cell(self):
        ground_truth = np.full((9, 9), FREE, dtype=np.uint8)
        ground_truth[4, 6] = UNKNOWN
        sensor = FourBeamSensor(SensorConfig(FOUR_BEAM, 4.0))
        frame = sensor.sense((4, 4), 0.0, ground_truth, 1.0)
        front = frame.measurement.readings["front"]

        self.assertFalse(front.hit)
        self.assertEqual([cell[:2] for cell in front.cells], [(4, 5)])
        self.assertEqual(frame.full_map[4, 6], UNKNOWN)


class OmnidirectionalSensorTests(unittest.TestCase):
    def test_legacy_scan_matches_pre_refactor_golden_output(self):
        ground_truth = np.full((40, 50), FREE, dtype=np.uint8)
        ground_truth[[0, -1], :] = OCCUPIED
        ground_truth[:, [0, -1]] = OCCUPIED
        ground_truth[10:31, 27] = OCCUPIED
        ground_truth[25, 8:35] = OCCUPIED
        sensor = OmnidirectionalSensor(SensorConfig(OMNIDIRECTIONAL, 1.2))

        frame = sensor.sense((20, 20), 0.0, ground_truth, 0.1)

        self.assertEqual(frame.x_bounds, (9, 31))
        self.assertEqual(frame.y_bounds, (9, 31))
        self.assertEqual(
            hashlib.sha256(frame.local_map.tobytes()).hexdigest(),
            "5d71815cd9b23051d95f9af8ea93de85b4a2f228d1386a34cc7bca229824fc21",
        )
        self.assertEqual(
            hashlib.sha256(frame.full_map.tobytes()).hexdigest(),
            "bb59c6a6a78a6760bce296ce47ed35f2b3bc630f35bb23d75de69419950debe8",
        )

    def test_finite_sensor_keeps_legacy_square_extent_and_encoding(self):
        ground_truth = np.full((11, 13), FREE, dtype=np.uint8)
        ground_truth[5, 7] = OCCUPIED
        sensor = OmnidirectionalSensor(SensorConfig(OMNIDIRECTIONAL, 3.0))
        frame = sensor.sense((5, 5), 0.0, ground_truth, 1.0)

        self.assertEqual(frame.x_bounds, (2, 8))
        self.assertEqual(frame.y_bounds, (2, 8))
        self.assertEqual(frame.local_map.shape, (6, 6))
        self.assertEqual(frame.full_map[5, 7], OCCUPIED)
        self.assertEqual(frame.full_map[5, 8], UNKNOWN)
        self.assertEqual(frame.full_map[1, 1], UNKNOWN)

    def test_infinite_sensor_uses_complete_map_extent(self):
        ground_truth = np.full((7, 9), FREE, dtype=np.uint8)
        sensor = OmnidirectionalSensor(SensorConfig(OMNIDIRECTIONAL, math.inf))
        frame = sensor.sense((3, 4), 0.0, ground_truth, 1.0)

        self.assertEqual(frame.x_bounds, (0, 7))
        self.assertEqual(frame.y_bounds, (0, 9))


if __name__ == "__main__":
    unittest.main()
