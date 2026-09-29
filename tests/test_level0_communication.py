import unittest

import numpy as np

from onpolicy.envs.GridEnv.communication import (
    FREE,
    MAP_PATCH,
    NONE,
    OCCUPIED,
    PARALLEL,
    PERFECT,
    POSE,
    SHARED_COLLISION,
    SILENCE,
    UNKNOWN,
    CommunicationBroker,
    CommunicationConfig,
)


class CommunicationConfigTests(unittest.TestCase):
    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            CommunicationConfig(mode="radio")
        with self.assertRaises(ValueError):
            CommunicationConfig(packet_loss=1.1)
        with self.assertRaises(ValueError):
            CommunicationConfig(latency_min_steps=3, latency_max_steps=2)
        with self.assertRaises(ValueError):
            CommunicationConfig(latency_min_steps=-1)
        with self.assertRaises(ValueError):
            CommunicationConfig(bucket_refill_bits=-1)
        with self.assertRaises(ValueError):
            CommunicationConfig(episode_budget_bits=-1)
        with self.assertRaises(ValueError):
            CommunicationConfig(radio_range_cells=float("nan"))
        with self.assertRaises(ValueError):
            CommunicationConfig(cost_per_patch=float("inf"))


class CommunicationBrokerTests(unittest.TestCase):
    def setUp(self):
        self.maps = [
            np.full((16, 16), UNKNOWN, dtype=np.uint8),
            np.full((16, 16), UNKNOWN, dtype=np.uint8),
        ]
        self.maps[0][0:8, 0:8] = FREE
        self.maps[0][3, 3] = OCCUPIED
        self.maps[1][8:16, 8:16] = FREE
        self.positions = [[2, 2], [4, 4]]
        self.headings = [0, 2]

    def make_broker(self, mode=PARALLEL, **kwargs):
        values = {
            "mode": mode,
            "candidate_count": 2,
            "radio_range_cells": 20,
            "cooldown_steps": 0,
            "bucket_capacity_bits": 1000,
            "bucket_refill_bits": 0,
            "episode_budget_bits": 2000,
        }
        values.update(kwargs)
        config = CommunicationConfig(**values)
        broker = CommunicationBroker(config, 2, seed=7)
        broker.reset(16, 16, 20, self.maps, self.positions, self.headings)
        return broker

    def assert_belief_maps_equal(self, first, second):
        self.assertEqual(len(first.belief_maps), len(second.belief_maps))
        for first_map, second_map in zip(first.belief_maps, second.belief_maps):
            np.testing.assert_array_equal(first_map, second_map)

    def test_logical_bit_accounting_is_exact(self):
        broker = self.make_broker()

        self.assertEqual(broker.message_bits(POSE), 29)
        self.assertEqual(broker.message_bits(MAP_PATCH), 149)

    def test_default_map_logical_bit_accounting(self):
        maps = [
            np.full((250, 250), UNKNOWN, dtype=np.uint8),
            np.full((250, 250), UNKNOWN, dtype=np.uint8),
        ]
        config = CommunicationConfig()
        broker = CommunicationBroker(config, 2, seed=7)
        broker.reset(250, 250, 100, maps, [[0, 0], [1, 1]], [0, 0])

        self.assertEqual(broker.message_bits(POSE), 37)
        self.assertEqual(broker.message_bits(MAP_PATCH), 157)

    def test_pose_arrives_after_one_decision(self):
        broker = self.make_broker()

        broker.transmit([POSE, SILENCE], self.positions, self.headings)
        self.assertEqual(broker.peer_pose_steps[1, 0], -1)
        broker.advance(self.maps, self.positions, self.headings)

        np.testing.assert_array_equal(broker.peer_positions[1, 0], [2, 2])
        self.assertEqual(broker.peer_headings[1, 0], 0)
        self.assertEqual(broker.peer_pose_age(1, 0), 1)

    def test_map_patch_updates_only_after_delivery(self):
        broker = self.make_broker()
        action = MAP_PATCH

        broker.transmit([action, SILENCE], self.positions, self.headings)
        self.assertEqual(broker.belief_maps[1][3, 3], UNKNOWN)
        broker.advance(self.maps, self.positions, self.headings)

        self.assertEqual(broker.belief_maps[1][3, 3], OCCUPIED)
        self.assertEqual(broker.belief_maps[1][2, 2], FREE)

    def test_map_patch_does_not_leak_cells_outside_selected_tile(self):
        broker = self.make_broker()
        before = np.array(broker.belief_maps[1], copy=True)

        broker.transmit([MAP_PATCH, SILENCE], self.positions, self.headings)
        broker.advance(self.maps, self.positions, self.headings)

        np.testing.assert_array_equal(
            broker.belief_maps[1][8:16, 0:8], before[8:16, 0:8]
        )

    def test_shared_channel_collision_charges_both_senders(self):
        broker = self.make_broker(mode=SHARED_COLLISION)
        starting_tokens = broker.tokens.copy()

        penalties, attempted = broker.transmit(
            [POSE, POSE], self.positions, self.headings
        )
        broker.advance(self.maps, self.positions, self.headings)

        self.assertTrue(np.all(attempted > 0))
        self.assertTrue(np.all(penalties > 0))
        self.assertTrue(np.all(broker.tokens < starting_tokens))
        self.assertEqual(broker.metrics()["collision_messages"], 2)
        self.assertTrue(np.all(broker.peer_pose_steps == -1))

    def test_out_of_range_broadcast_has_no_recipient(self):
        broker = self.make_broker(radio_range_cells=1)

        broker.transmit([POSE, SILENCE], self.positions, self.headings)
        broker.advance(self.maps, self.positions, self.headings)

        self.assertEqual(broker.metrics()["transmitted_messages"], 1)
        self.assertEqual(broker.metrics()["delivered_messages"], 0)

    def test_seeded_packet_loss_is_reproducible(self):
        first = self.make_broker(packet_loss=0.5)
        second = self.make_broker(packet_loss=0.5)

        first.transmit([POSE, SILENCE], self.positions, self.headings)
        second.transmit([POSE, SILENCE], self.positions, self.headings)
        first.advance(self.maps, self.positions, self.headings)
        second.advance(self.maps, self.positions, self.headings)

        self.assertEqual(first.metrics(), second.metrics())
        np.testing.assert_array_equal(first.peer_pose_steps, second.peer_pose_steps)

    def test_total_packet_loss_produces_the_same_maps_as_none(self):
        lossy = self.make_broker(packet_loss=1.0)
        none = self.make_broker(mode=NONE)

        for _ in range(3):
            lossy.transmit([MAP_PATCH, MAP_PATCH], self.positions, self.headings)
            none.transmit([MAP_PATCH, MAP_PATCH], self.positions, self.headings)
            lossy.advance(self.maps, self.positions, self.headings)
            none.advance(self.maps, self.positions, self.headings)

        self.assert_belief_maps_equal(lossy, none)
        self.assertEqual(lossy.metrics()["delivered_messages"], 0)
        self.assertGreater(lossy.metrics()["lost_messages"], 0)

    def test_ideal_channel_matches_perfect_after_known_tiles_are_sent(self):
        ideal = self.make_broker(
            latency_min_steps=0,
            latency_max_steps=0,
            packet_loss=0.0,
            radio_range_cells=float("inf"),
            bucket_capacity_bits=1000000,
            episode_budget_bits=1000000,
            cost_per_patch=0.0,
        )
        perfect = self.make_broker(mode=PERFECT)

        ideal.transmit([MAP_PATCH, MAP_PATCH], self.positions, self.headings)

        self.assert_belief_maps_equal(ideal, perfect)
        self.assertEqual(ideal.metrics()["mean_latency"], 0.0)
        self.assertEqual(ideal.metrics()["delivered_messages"], 2)

    def test_newer_local_observation_wins_over_delayed_patch(self):
        broker = self.make_broker(latency_min_steps=2, latency_max_steps=2)
        broker.transmit([MAP_PATCH, SILENCE], self.positions, self.headings)
        receiver_maps = [np.array(grid, copy=True) for grid in self.maps]
        receiver_maps[1][3, 3] = FREE

        broker.advance(receiver_maps, self.positions, self.headings)
        broker.advance(receiver_maps, self.positions, self.headings)

        self.assertEqual(broker.belief_maps[1][3, 3], FREE)

    def test_candidate_order_is_deterministic(self):
        first = self.make_broker()
        second = self.make_broker()

        np.testing.assert_array_equal(
            first.candidate_features(0), second.candidate_features(0)
        )

    def test_none_mode_never_shares_private_maps(self):
        broker = self.make_broker(mode=NONE)

        broker.transmit([MAP_PATCH, MAP_PATCH], self.positions, self.headings)
        broker.advance(self.maps, self.positions, self.headings)

        self.assertEqual(broker.belief_maps[1][3, 3], UNKNOWN)
        self.assertEqual(broker.metrics()["attempted_messages"], 0)

    def test_zero_budget_produces_the_same_maps_as_none(self):
        budgetless = self.make_broker(episode_budget_bits=0)
        none = self.make_broker(mode=NONE)

        for _ in range(3):
            budgetless.transmit(
                [MAP_PATCH, MAP_PATCH], self.positions, self.headings
            )
            none.transmit([MAP_PATCH, MAP_PATCH], self.positions, self.headings)
            budgetless.advance(self.maps, self.positions, self.headings)
            none.advance(self.maps, self.positions, self.headings)

        self.assert_belief_maps_equal(budgetless, none)
        self.assertEqual(budgetless.metrics()["attempted_messages"], 0)

    def test_episode_long_cooldown_blocks_sharing_after_first_step(self):
        cooldown = self.make_broker(cooldown_steps=21)
        cooldown.transmit(
            [MAP_PATCH, MAP_PATCH], self.positions, self.headings
        )
        cooldown.advance(self.maps, self.positions, self.headings)
        after_first_delivery = [
            np.array(grid, copy=True) for grid in cooldown.belief_maps
        ]
        none_after_first = self.make_broker(mode=NONE)
        none_after_first.reset(
            16,
            16,
            20,
            after_first_delivery,
            self.positions,
            self.headings,
        )
        later_maps = [np.array(grid, copy=True) for grid in self.maps]
        later_maps[0][1, 9] = FREE

        for _ in range(19):
            cooldown.transmit(
                [MAP_PATCH, MAP_PATCH], self.positions, self.headings
            )
            none_after_first.transmit(
                [MAP_PATCH, MAP_PATCH], self.positions, self.headings
            )
            cooldown.advance(later_maps, self.positions, self.headings)
            none_after_first.advance(later_maps, self.positions, self.headings)

        self.assert_belief_maps_equal(cooldown, none_after_first)
        self.assertEqual(cooldown.belief_maps[1][1, 9], UNKNOWN)
        self.assertEqual(cooldown.metrics()["attempted_messages"], 2)

    def test_perfect_mode_synchronizes_immediately(self):
        broker = self.make_broker(mode=PERFECT)

        self.assertEqual(broker.belief_maps[1][3, 3], OCCUPIED)
        self.assertEqual(broker.belief_maps[0][10, 10], FREE)
        np.testing.assert_array_equal(broker.peer_positions[1, 0], [2, 2])

    def test_perfect_mode_gives_obstacles_priority_on_conflict(self):
        self.maps[1][3, 3] = FREE

        broker = self.make_broker(mode=PERFECT)

        self.assertEqual(broker.belief_maps[0][3, 3], OCCUPIED)
        self.assertEqual(broker.belief_maps[1][3, 3], OCCUPIED)

    def test_latency_longer_than_ttl_never_delivers(self):
        delayed = self.make_broker(
            latency_min_steps=2, latency_max_steps=2, ttl_steps=1
        )
        none = self.make_broker(mode=NONE)

        delayed.transmit([MAP_PATCH, SILENCE], self.positions, self.headings)
        none.transmit([MAP_PATCH, SILENCE], self.positions, self.headings)
        for _ in range(3):
            delayed.advance(self.maps, self.positions, self.headings)
            none.advance(self.maps, self.positions, self.headings)

        self.assert_belief_maps_equal(delayed, none)
        self.assertEqual(delayed.in_flight_count, 0)
        self.assertEqual(delayed.metrics()["delivered_messages"], 0)
        self.assertEqual(delayed.metrics()["expired_messages"], 1)


if __name__ == "__main__":
    unittest.main()
