# DQN plan (step 3 of [research-plan.md](../research-plan.md))

Goal: a value-based learner compared with MAPPO under the same session,
observation, reward, clock and evaluation, so the comparison changes the
learning algorithm and the discrete action it needs, nothing else.

## What it builds on

- **Clock.** Every controller chooses a goal for each robot every step; the
  robot then moves one cell (0.1 m, or 0.14 m diagonally) along the A* path
  to that goal (`route(...)[:2]`, explore_bench/session.py:240) and senses.
  The goal is a heading for one step, not a commitment.
- **MAPPO's action.** A 2-D Gaussian per robot, squashed to [0, 1] by a
  sigmoid (`requests`, explore_bench/mappo.py:43), then snapped to the
  nearest reachable frontier of the robot's belief (`decode_coordinates`,
  session.py:210). It already chooses frontiers, indirectly.
- **Reused unchanged.** `Session.step_request` (no learning code), the
  controller protocol (explore_bench/controllers.py), `evaluate()` and
  scoring (keyed by controller name), `WorldBatch` training worlds.
- **MAPPO-only, needs a DQN counterpart.** The `training` block of
  explore_bench/config.py (PPO keys, unknown keys rejected),
  explore_bench/checkpoints.py (actor/critic contents), `validate()` in
  explore_bench/evaluation.py and the `evaluate` command in
  explore_bench/__main__.py (both build only `ActorController`).
- **Provenance side effect.** Every `.py` under explore_bench/ and
  onpolicy/onpolicy/ is hashed into each run (explore_bench/artifacts.py:58)
  and resuming requires equal hashes, so MAPPO training runs made before the
  DQN code cannot be resumed. All are complete; evaluating their checkpoints
  is unaffected.

## Stages

1. **Actions (done).** Q-map indexing, the legal-action mask, the stay
   action, and `masked_argmax` / `request` in explore_bench/dqn.py.
   Decisions D1, D1b. Communication action is always 0 (SILENCE): valid for
   the `perfect` and `none` modes only; `parallel` and `shared_collision`
   need a communication head later (research-plan step 5).
2. **Q-network.** Inputs and architecture. Decision D2.
3. **Replay buffer.** Size and storage; one 8 x 64 x 64 float32 observation
   is 128 KB. Decision D3.
4. **Learning loop.** Exploration schedule, target network, loss, and the
   truncation rule (the 1000-step cap is not a terminal state; same rule as
   explore_bench/training.py:109). Decision D4.
5. **Wiring.** Optional config keys (old digests unchanged, as with
   `navigation`), checkpoints, validate/select/evaluate.
6. **Sanity, then pilot.** Learn on the tiny test maps, then one Ada pilot.

## Decision log

| ID | Question | Choice | Why | Date |
|---|---|---|---|---|
| D0 | Score every frontier, or a fixed list? | Q-map over a 64 x 64 grid, illegal cells masked | A fixed output needs a fixed meaning; a location keeps one, frontier #k does not. Every frontier gets a score. | 2026-11-12 |
| D1 | Which cells are legal, where the mask comes from, what request is sent | Legal = Q-map cells holding a reachable frontier target; the session computes the mask from the full-resolution belief; the action is sent as that target's exact cell (`motion_goals`) | One action per real choice; the blurred 64 x 64 channels would allow cells with no frontier; Q-learning must credit the executed action, and snapping to a neighbour's frontier would break that | 2026-11-12 |
| D1b | What a robot does when it has no reachable frontier | A "stay" action (its own Q-map cell, goal = its position), legal only then | Keeps the learning target's max over legal next actions defined; waiting still earns the teammate's team reward, so treating it as worth 0 is wrong; MAPPO and the classical planners also stay; an always-legal stay would give DQN an option the others lack | 2026-11-12 |
| D2 | Q-network inputs, architecture, parameter sharing | open | | |

### D1 implementation notes

- `Session.frontier_targets` (explore_bench/session.py) returns exactly the
  candidates `route()` can move to, from one connected-component pass instead
  of one A* search per candidate. Tested against `route()` in both
  navigation modes, including the no-corner-cutting rule (tests/test_dqn.py).
- `frontier_qmap` (explore_bench/dqn.py) adds `frontier_mask` [2, 64, 64] and
  `frontier_goal` [2, 64, 64, 2] to the actor observation, only when the
  configuration has a `dqn` key.
- Probe, random controller, 300 steps on room_1, square_loop and
  rooms_v1_seed_1: 2.0-5.0 legal actions per robot and step (of 4,096); two
  targets shared a Q-map cell 2 times in 6,971 (the nearest-to-centre rule
  almost never matters); no empty masks; about 5 ms per step for the mask
  against 160-220 ms for the whole step.
