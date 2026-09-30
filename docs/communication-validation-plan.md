# Communication Validation Reconciliation Plan

This document is the source of truth for the remaining Level-0 communication
validation work. It distinguishes an implemented behavior from a test that
proves that behavior and from an experiment that supports a thesis claim. A
check is not complete merely because the relevant code path exists.

## Status Definitions

- **Verified** means a focused automated test asserts the stated invariant.
- **Partial** means related tests exist, but the exact invariant or observation
  boundary is not asserted.
- **Planned** means the implementation may exist, but the required automated
  evidence does not.

## Current Coverage

| Requirement | Status | Existing evidence | Remaining work |
| --- | --- | --- | --- |
| Communication-disabled collapse | Partial | `test_total_packet_loss_produces_the_same_maps_as_none`, `test_zero_budget_produces_the_same_maps_as_none`, `test_episode_long_cooldown_blocks_sharing_after_first_step`, and `test_latency_longer_than_ttl_never_delivers` in [`test_level0_communication.py`](../tests/test_level0_communication.py) | Use one parametrized harness and compare every belief map with a `none` run after every step. |
| Episode-boundary reset | Planned | `CommunicationBroker.reset` clears broker state, and `InfoDummyVecEnv` resets completed members independently. | Queue a message before one vector member terminates and prove that no state or delivery crosses into its next episode. |
| Planner information leakage | Verified | [`test_level0_planner_leakage.py`](../tests/test_level0_planner_leakage.py) places an unseen free shortcut in the true map and proves the `none` planner avoids it. | Keep as a regression gate. |
| Equal-timestamp fusion | Verified | [`test_level0_timestamp_ties.py`](../tests/test_level0_timestamp_ties.py) covers local-versus-remote ties and conflicting remote senders in both sender assignments. | Revisit the declared priority only when noisy sensors are introduced. |
| Token-bucket recovery | Partial | `test_patch_collision_refills_tokens_but_not_episode_budget` in [`test_level0_collision_budget.py`](../tests/test_level0_collision_budget.py) proves recovery after a patch collision. | Prove incremental refill during silence at the configured rate, independently of episode-budget recovery. |
| Vector-environment isolation | Verified | [`test_level0_vector_isolation.py`](../tests/test_level0_vector_isolation.py) uses in-process `InfoDummyVecEnv` and inspects distinct broker instances. | Keep the in-process wrapper explicit. A subprocess test would not exercise shared Python state. |
| Shared-channel delivery | Partial | `test_round_robin_protocol_advances_shared_broker` in [`test_level0_standalone_communication.py`](../tests/test_level0_standalone_communication.py) changes the receiving belief and records zero collisions. | Assert nonzero transmitted and delivered metrics directly over enough slots for both agents to send. |
| Legacy versus perfect standalone cost | Planned | No automated comparison exists. | Separate map-fusion equivalence from end-to-end planner equivalence before interpreting a coverage difference. |

The four collapse cases are individually tested today, so describing them as
"not written" is no longer accurate. Their status remains partial because most
of those tests compare only the final state, while the intended invariant is
step-for-step equivalence with `none`.

## Work Package 1: Stepwise Collapse to None

Create a parametrized broker test with cases for 100% packet loss, latency
longer than TTL, zero episode budget, and an episode-long cooldown. Each case
must drive the constrained broker and a `none` broker with identical private
maps, positions, headings, and requested actions. Compare every agent belief
after every advance rather than only at the end.

The cooldown case has one intentional exception: its first eligible send may be
delivered. Capture that post-delivery state, initialize the `none` reference
from it, and require equality for every remaining step. Preserve case-specific
counter assertions so equality cannot pass for the wrong reason: loss must
increase `lost_messages`, expiry must increase `expired_messages`, and zero
budget must keep `attempted_messages` at zero.

## Work Package 2: Episode Reset Isolation

Use two probe environments inside `InfoDummyVecEnv`, with different episode
lengths so only one member resets at a time. Send a delayed patch immediately
before the shorter member terminates. The reset member must return to all of
these initial conditions:

- `current_step == 0` and an empty in-flight queue;
- full token buckets and restored per-episode budgets;
- zero cooldowns and silent last results;
- reset metrics, peer poses, timestamps, and broadcast history;
- beliefs derived only from the new episode's private maps.

Continue the vector environment beyond the abandoned message's former delivery
step and prove that it never appears. At the same time, confirm that the longer
member's queue, resources, metrics, and beliefs continue without being reset.
This is the meaningful asynchronous-reset test because both brokers inhabit the
same Python process.

## Work Package 3: Incremental Token Refill

Add a focused broker test without collisions. Spend one patch from a bucket
whose capacity is exactly one patch and whose refill is a smaller, non-dividing
number of bits. Request silence for subsequent decisions and assert after each
advance that:

```text
tokens(t) = min(capacity, tokens(0) + t * refill)
```

The episode budget must decrease only for the original attempted patch and must
remain unchanged during silent refill steps. The pose mask may reopen earlier
because a pose is smaller, while patch actions must remain masked until the
bucket again contains `patch_bits`. This test isolates rate limiting from the
permanent episode allowance tested by the collision-budget suite.

## Work Package 4: Explicit Shared-Channel Delivery

Extend the standalone round-robin test rather than creating a duplicate. Run
enough slots for each agent to become the selected sender and require:

```text
collision_messages == 0
transmitted_messages > 0
delivered_messages > 0
```

Also verify that each receiving belief gains remote information. These direct
metrics distinguish a functional shared channel from a run that merely avoids
collisions. The learned `gate_shared` smoke result is not a replacement for
this test because training policy choices, range, and costs can all suppress
useful delivery.

## Work Package 5: Legacy and Perfect Standalone Controls

Legacy reproduction and fair communication comparison answer different
questions. When standalone communication is omitted, `_navigation_map` retains
the original behavior and returns `gt_map`. With communication enabled,
including `perfect`, navigation uses the agent belief and treats unknown cells
as blocked. End-to-end coverage can therefore diverge even when the two merge
procedures agree exactly.

Validate these boundaries in two stages:

1. Hold positions and sensor frames fixed. After every sensing step, compare the
   legacy merged map with every `perfect` belief. This is the map-fusion
   equivalence test.
2. Run standalone cost planners with the same deterministic map, seed, spawn,
   sensor configuration, and step count. Compare selected goals, positions,
   private maps, merged maps, and coverage after every step, and report the
   first differing field.

If fusion matches but paths differ at `_navigation_map`, record legacy as an
omniscient compatibility baseline rather than weakening `perfect` with true-map
access. A fair channel comparison must put both conditions behind the same
belief-limited planner. If exact reproduction is also required, expose it as a
separately named legacy control; do not silently change either meaning.

## Documentation and Evidence Gate

Before a long experiment sweep:

1. Every row above must be **Verified**, or the thesis protocol must identify it
   explicitly as an unresolved validity risk.
2. Focused tests must pass independently before running the complete suite.
3. The full Python test suite must pass in the documented Python 3.8 virtual
   environment.
4. One short standalone cost run and one MAPPO smoke run must record exploration
   and communication metrics.
5. Result notes must distinguish attempted messages, collisions, transmitted
   messages, in-range recipients, and delivered receiver-copies.

Implementation should be split by invariant: collapse equivalence, episode
reset isolation, refill rate, shared delivery metrics, legacy/perfect fusion,
legacy/perfect end-to-end diagnostics, and documentation status. That history
makes each research claim reviewable without combining unrelated evidence.
