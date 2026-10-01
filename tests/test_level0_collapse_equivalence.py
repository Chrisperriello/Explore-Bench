import numpy as np
import pytest

from onpolicy.envs.GridEnv.communication import (
    FREE,
    MAP_PATCH,
    NONE,
    OCCUPIED,
    PARALLEL,
    SILENCE,
    UNKNOWN,
    CommunicationBroker,
    CommunicationConfig,
)


def make_maps():
    maps = [
        np.full((16, 16), UNKNOWN, dtype=np.uint8),
        np.full((16, 16), UNKNOWN, dtype=np.uint8),
    ]
    maps[0][0:8, 0:8] = FREE
    maps[0][3, 3] = OCCUPIED
    maps[1][8:16, 8:16] = FREE
    return maps


def make_broker(maps, mode=PARALLEL, **overrides):
    values = {
        "mode": mode,
        "candidate_count": 2,
        "radio_range_cells": float("inf"),
        "cooldown_steps": 0,
        "bucket_capacity_bits": 10000,
        "bucket_refill_bits": 0,
        "episode_budget_bits": 10000,
    }
    values.update(overrides)
    broker = CommunicationBroker(CommunicationConfig(**values), 2, seed=19)
    broker.reset(16, 16, 20, maps, [[2, 2], [4, 4]], [0, 2])
    return broker


def assert_beliefs_equal(first, second):
    for first_map, second_map in zip(first.belief_maps, second.belief_maps):
        np.testing.assert_array_equal(first_map, second_map)


@pytest.mark.parametrize(
    "overrides,metric_name,warm_up",
    [
        pytest.param(
            {"packet_loss": 1.0},
            "lost_messages",
            False,
            id="total-packet-loss",
        ),
        pytest.param(
            {
                "latency_min_steps": 2,
                "latency_max_steps": 2,
                "ttl_steps": 1,
            },
            "expired_messages",
            False,
            id="latency-beyond-ttl",
        ),
        pytest.param(
            {"episode_budget_bits": 0},
            "attempted_messages",
            False,
            id="zero-episode-budget",
        ),
        pytest.param(
            {"cooldown_steps": 21},
            "attempted_messages",
            True,
            id="episode-long-cooldown",
        ),
    ],
)
def test_constrained_channel_collapses_to_none_step_by_step(
    overrides, metric_name, warm_up
):
    maps = make_maps()
    positions = [[2, 2], [4, 4]]
    headings = [0, 2]
    constrained = make_broker(maps, **overrides)

    if warm_up:
        constrained.transmit([MAP_PATCH, SILENCE], positions, headings)
        constrained.advance(maps, positions, headings)
        reference_maps = [
            np.array(grid, copy=True) for grid in constrained.belief_maps
        ]
        none = make_broker(reference_maps, mode=NONE)
    else:
        none = make_broker(maps, mode=NONE)

    for step in range(5):
        private_maps = [np.array(grid, copy=True) for grid in maps]
        private_maps[0][1, 8 + step] = FREE
        constrained.transmit([MAP_PATCH, SILENCE], positions, headings)
        none.transmit([MAP_PATCH, SILENCE], positions, headings)
        constrained.advance(private_maps, positions, headings)
        none.advance(private_maps, positions, headings)
        assert_beliefs_equal(constrained, none)

    metric_value = constrained.metrics()[metric_name]
    if overrides.get("episode_budget_bits") == 0:
        assert metric_value == 0
    elif warm_up:
        assert metric_value == 1
    else:
        assert metric_value > 0
