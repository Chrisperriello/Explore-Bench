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

1. **Actions.** Q-map indexing (done: `action_cell`, `qmap_cell` in
   explore_bench/dqn.py), the legal-action mask, and the request an action
   becomes. Decision D1.
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
| D1 | Which cells are legal, where the mask comes from, what request is sent | open | | |
