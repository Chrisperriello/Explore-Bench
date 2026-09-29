"""Property tests for the Level-0 communication trust boundary.

The checker deliberately keeps its own knowledge and accounting ledgers.  An
implementation bug therefore cannot make the test pass merely by corrupting
both a broker value and a derived broker metric in the same way.
"""

from types import SimpleNamespace

import numpy as np
from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from onpolicy.envs.GridEnv.GridEnv import GridEnv
from onpolicy.envs.GridEnv.communication import (
    FREE,
    MAP_PATCH,
    NONE,
    OCCUPIED,
    PARALLEL,
    SHARED_COLLISION,
    UNKNOWN,
    CommunicationBroker,
    CommunicationConfig,
)
from onpolicy.runner.shared.grid_runner import GridRunner


MAP_SIZE = 8
AGENT_COUNT = 2


class BrokerInvariantChecker:
    """Independent oracle checked after every simulated broker step."""

    def __init__(self, broker, true_map, private_maps):
        self.broker = broker
        self.true_map = np.asarray(true_map)
        self.allowed_knowledge = [
            np.asarray(private_map) != UNKNOWN for private_map in private_maps
        ]
        self.initial_episode_budget = np.array(
            broker.episode_budget, dtype=np.int64, copy=True
        )
        self.sent_recipient_messages = 0
        self.delivered_recipient_messages = 0
        # Retain the objects: CPython may reuse id(message) after delivery.
        self._seen_messages = set()
        self._colliding_attempts = set()
        self.check_current_state()

    def observe_private_maps(self, private_maps):
        for agent_id, private_map in enumerate(private_maps):
            private_map = np.asarray(private_map)
            known = private_map != UNKNOWN
            np.testing.assert_array_equal(
                private_map[known], self.true_map[known],
                err_msg="a private observation disagrees with the true map",
            )
            self.allowed_knowledge[agent_id] |= known

    def observe_transmit(self, attempted_bits, collided_senders):
        attempted_bits = np.asarray(attempted_bits)
        assert np.all(attempted_bits >= 0)
        if collided_senders:
            for sender in collided_senders:
                self._colliding_attempts.add((self.broker.current_step, sender))

        for _, _, message in self.broker._in_flight:
            if message in self._seen_messages:
                continue
            self._seen_messages.add(message)
            self.sent_recipient_messages += len(message.recipients)

            latency = message.deliver_step - message.sent_step
            assert self.broker.config.latency_min_steps <= latency
            assert latency <= self.broker.config.latency_max_steps
            assert message.deliver_step > self.broker.current_step

            if message.message_type == MAP_PATCH:
                row = message.payload["row"]
                column = message.payload["column"]
                cells = message.payload["cells"]
                row_end = min(MAP_SIZE, row + cells.shape[0])
                column_end = min(MAP_SIZE, column + cells.shape[1])
                source = cells[: row_end - row, : column_end - column]
                known = source != UNKNOWN
                allowed = self.allowed_knowledge[message.sender][
                    row:row_end, column:column_end
                ]
                assert np.all(~known | allowed), (
                    "a sender included a cell it had never observed or received"
                )
                truth = self.true_map[row:row_end, column:column_end]
                np.testing.assert_array_equal(
                    source[known], truth[known],
                    err_msg="an enqueued patch disagrees with the true map",
                )
        # Check accounting both before refill and after advance so a refill
        # cannot conceal a transiently negative bucket.
        self.check_current_state()

    def observe_advance(self, delivered):
        for message in delivered:
            assert self.broker.current_step >= message.deliver_step, (
                "message arrived before its sampled latency elapsed"
            )
            assert (
                self.broker.current_step - message.sent_step
                <= self.broker.config.ttl_steps
            ), "message arrived after its TTL"
            assert (message.sent_step, message.sender) not in self._colliding_attempts, (
                "a colliding shared-channel message was delivered"
            )

            self.delivered_recipient_messages += len(message.recipients)
            if message.message_type != MAP_PATCH:
                continue
            row = message.payload["row"]
            column = message.payload["column"]
            cells = message.payload["cells"]
            row_end = min(MAP_SIZE, row + cells.shape[0])
            column_end = min(MAP_SIZE, column + cells.shape[1])
            known = cells[: row_end - row, : column_end - column] != UNKNOWN
            for receiver in message.recipients:
                target = self.allowed_knowledge[receiver][
                    row:row_end, column:column_end
                ]
                target |= known

        assert self.delivered_recipient_messages <= self.sent_recipient_messages
        self.check_current_state()

    def check_current_state(self):
        broker = self.broker
        metrics = broker.metrics()

        for agent_id, belief in enumerate(broker.belief_maps):
            known = belief != UNKNOWN
            np.testing.assert_array_equal(
                belief[known], self.true_map[known],
                err_msg="a known belief cell disagrees with the true map",
            )
            assert np.all(~known | self.allowed_knowledge[agent_id]), (
                "a drone knows a cell it neither observed nor received"
            )

        assert metrics["delivered_messages"] <= metrics["transmitted_messages"]
        assert metrics["transmitted_messages"] <= metrics["attempted_messages"]
        assert np.all(broker.tokens >= 0), "a token bucket went negative"
        assert np.all(broker.tokens <= broker.config.bucket_capacity_bits)
        assert np.all(broker.episode_budget >= 0)
        spent = self.initial_episode_budget - broker.episode_budget
        assert np.all(spent <= self.initial_episode_budget)
        assert metrics["attempted_bits"] == int(spent.sum())

        for deliver_step, _, message in broker._in_flight:
            assert deliver_step == message.deliver_step
            assert deliver_step > broker.current_step


class RandomBrokerMachine(RuleBasedStateMachine):
    """Random policies hammer randomized broker configurations for 4k+ steps."""

    @initialize(
        mode=st.sampled_from((NONE, PARALLEL, SHARED_COLLISION)),
        tile_size=st.integers(min_value=1, max_value=4),
        candidate_count=st.integers(min_value=1, max_value=4),
        latency_a=st.integers(min_value=1, max_value=5),
        latency_b=st.integers(min_value=1, max_value=5),
        ttl_steps=st.integers(min_value=1, max_value=6),
        packet_loss=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        cooldown_steps=st.integers(min_value=0, max_value=4),
        bucket_capacity=st.integers(min_value=32, max_value=512),
        bucket_refill=st.integers(min_value=0, max_value=180),
        episode_budget=st.integers(min_value=32, max_value=3000),
        radio_range=st.floats(
            min_value=1.0, max_value=16.0, allow_nan=False, allow_infinity=False
        ),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def initialize_broker(
        self,
        mode,
        tile_size,
        candidate_count,
        latency_a,
        latency_b,
        ttl_steps,
        packet_loss,
        cooldown_steps,
        bucket_capacity,
        bucket_refill,
        episode_budget,
        radio_range,
        seed,
    ):
        latency_min = min(latency_a, latency_b)
        latency_max = max(latency_a, latency_b)
        config = CommunicationConfig(
            mode=mode,
            tile_size=tile_size,
            candidate_count=candidate_count,
            radio_range_cells=radio_range,
            latency_min_steps=latency_min,
            latency_max_steps=latency_max,
            packet_loss=packet_loss,
            cooldown_steps=cooldown_steps,
            bucket_capacity_bits=bucket_capacity,
            bucket_refill_bits=bucket_refill,
            episode_budget_bits=episode_budget,
            ttl_steps=ttl_steps,
        )
        rows, columns = np.indices((MAP_SIZE, MAP_SIZE))
        self.true_map = np.where(
            (rows * 3 + columns * 5) % 7 == 0, OCCUPIED, FREE
        ).astype(np.uint8)
        self.private_maps = [
            np.full((MAP_SIZE, MAP_SIZE), UNKNOWN, dtype=np.uint8)
            for _ in range(AGENT_COUNT)
        ]
        self.private_maps[0][0, 0] = self.true_map[0, 0]
        self.private_maps[1][-1, -1] = self.true_map[-1, -1]
        self.positions = [[0, 0], [MAP_SIZE - 1, MAP_SIZE - 1]]
        self.headings = [0, 2]
        self.broker = CommunicationBroker(config, AGENT_COUNT, seed=seed)
        self.broker.reset(
            MAP_SIZE,
            MAP_SIZE,
            1000,
            self.private_maps,
            self.positions,
            self.headings,
        )
        self.checker = BrokerInvariantChecker(
            self.broker, self.true_map, self.private_maps
        )

    @rule(
        action_0=st.integers(min_value=0, max_value=30),
        action_1=st.integers(min_value=0, max_value=30),
        reveal_row_0=st.integers(min_value=0, max_value=MAP_SIZE - 1),
        reveal_column_0=st.integers(min_value=0, max_value=MAP_SIZE - 1),
        reveal_row_1=st.integers(min_value=0, max_value=MAP_SIZE - 1),
        reveal_column_1=st.integers(min_value=0, max_value=MAP_SIZE - 1),
        row_0=st.integers(min_value=0, max_value=MAP_SIZE - 1),
        column_0=st.integers(min_value=0, max_value=MAP_SIZE - 1),
        row_1=st.integers(min_value=0, max_value=MAP_SIZE - 1),
        column_1=st.integers(min_value=0, max_value=MAP_SIZE - 1),
        heading_0=st.integers(min_value=0, max_value=3),
        heading_1=st.integers(min_value=0, max_value=3),
    )
    def random_step(
        self,
        action_0,
        action_1,
        reveal_row_0,
        reveal_column_0,
        reveal_row_1,
        reveal_column_1,
        row_0,
        column_0,
        row_1,
        column_1,
        heading_0,
        heading_1,
    ):
        self.private_maps[0][reveal_row_0, reveal_column_0] = self.true_map[
            reveal_row_0, reveal_column_0
        ]
        self.private_maps[1][reveal_row_1, reveal_column_1] = self.true_map[
            reveal_row_1, reveal_column_1
        ]
        self.checker.observe_private_maps(self.private_maps)

        self.positions = [[row_0, column_0], [row_1, column_1]]
        self.headings = [heading_0, heading_1]
        action_count = self.broker.config.candidate_count + MAP_PATCH
        actions = [action_0 % action_count, action_1 % action_count]
        _, attempted_bits = self.broker.transmit(
            actions, self.positions, self.headings
        )
        collided_senders = set()
        if self.broker.config.mode == SHARED_COLLISION:
            attempted = np.flatnonzero(attempted_bits)
            if len(attempted) > 1:
                collided_senders = set(int(sender) for sender in attempted)
        self.checker.observe_transmit(attempted_bits, collided_senders)

        delivered = self.broker.advance(
            self.private_maps, self.positions, self.headings
        )
        self.checker.observe_advance(delivered)

    @invariant()
    def invariants_hold_after_every_rule(self):
        self.checker.check_current_state()


TestRandomBroker = RandomBrokerMachine.TestCase
TestRandomBroker.settings = settings(
    max_examples=100,
    stateful_step_count=40,
    deadline=None,
    database=None,
    suppress_health_check=(HealthCheck.too_slow,),
)


def _runner_for_observation_test():
    runner = object.__new__(GridRunner)
    runner.num_agents = AGENT_COUNT
    runner.full_w = MAP_SIZE
    runner.full_h = MAP_SIZE
    runner.input_w = MAP_SIZE
    runner.input_h = MAP_SIZE
    runner.max_steps = 100
    runner.augment = 85
    runner.all_args = SimpleNamespace(
        comm_candidate_count=2,
        comm_ttl_steps=6,
    )
    runner.all_agent_pos_map = np.zeros(
        (1, AGENT_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.float32
    )
    runner.all_merge_pos_map = np.zeros(
        (1, MAP_SIZE, MAP_SIZE), dtype=np.float32
    )
    runner.eval_all_agent_pos_map = np.zeros_like(runner.all_agent_pos_map)
    runner.eval_all_merge_pos_map = np.zeros_like(runner.all_merge_pos_map)
    return runner


def _observation_info(privileged_variant):
    own_map = np.full((MAP_SIZE, MAP_SIZE), UNKNOWN, dtype=np.uint8)
    own_map[0, 0] = FREE
    other_map = np.full_like(own_map, UNKNOWN)
    other_map[7, 7] = OCCUPIED if privileged_variant else FREE
    explored_each = np.zeros((AGENT_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    explored_each[0, 0, 0] = 1
    explored_each[1, 7, 7] = privileged_variant
    obstacle_each = np.zeros_like(explored_each)
    peer_positions = np.full((AGENT_COUNT, AGENT_COUNT, 2), -1, dtype=np.int64)
    peer_steps = np.full((AGENT_COUNT, AGENT_COUNT), -1, dtype=np.int64)
    peer_positions[0, 1] = [3, 3]
    peer_steps[0, 1] = 2
    comm_status = np.zeros((AGENT_COUNT, 8), dtype=np.float32)
    comm_status[:, 0] = 1.0
    comm_status[1, :3] = privileged_variant
    return {
        "current_agent_pos": np.array(
            [[0, 0], [7, 7] if privileged_variant else [6, 6]], dtype=np.int64
        ),
        "belief_each_map": np.array([own_map, other_map]),
        "peer_positions": peer_positions,
        "peer_pose_steps": peer_steps,
        "comm_candidate_features": np.zeros(
            (AGENT_COUNT, 2, 5), dtype=np.float32
        ),
        "comm_status": comm_status,
        "available_comm_actions": np.ones(
            (AGENT_COUNT, 4), dtype=np.float32
        ),
        "comm_in_flight_count": 2 * privileged_variant,
        "comm_step": 4,
        "explored_each_map": explored_each,
        "obstacle_each_map": obstacle_each,
        "explored_all_map": np.full(
            (MAP_SIZE, MAP_SIZE), privileged_variant, dtype=np.uint8
        ),
        "obstacle_all_map": np.eye(MAP_SIZE, dtype=np.uint8) * privileged_variant,
    }


def test_actor_observation_is_independent_of_critic_privileged_state():
    baseline = _runner_for_observation_test()._resize_communication_convert(
        [None], [_observation_info(0)]
    )
    perturbed = _runner_for_observation_test()._resize_communication_convert(
        [None], [_observation_info(1)]
    )
    baseline_raw, baseline_actor, baseline_critic, baseline_available = baseline
    changed_raw, changed_actor, changed_critic, changed_available = perturbed

    critic_only = {"agent_beliefs", "global_comm_state"}
    config = CommunicationConfig(candidate_count=2)
    env = GridEnv(0.1, 3.0, AGENT_COUNT, 100, communication_config=config)
    assert critic_only.isdisjoint(env.observation_space[0].spaces)
    assert critic_only <= set(env.share_observation_space[0].spaces)

    for key in baseline_actor:
        np.testing.assert_array_equal(
            baseline_actor[key][0, 0], changed_actor[key][0, 0],
            err_msg="critic-only state leaked into actor field {!r}".format(key),
        )
        np.testing.assert_array_equal(
            baseline_raw[key][0, 0], changed_raw[key][0, 0],
            err_msg="critic-only state leaked into raw actor field {!r}".format(key),
        )
    np.testing.assert_array_equal(
        baseline_available[0, 0], changed_available[0, 0]
    )
    assert any(
        not np.array_equal(baseline_critic[key], changed_critic[key])
        for key in critic_only
    ), "the test did not actually perturb critic-only state"
