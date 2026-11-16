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
2. **Q-network (in progress).** Complete actor inputs and separate branches
   chosen; `actor_inputs` prepares the arrays. Wider network widths chosen
   (D2e), fixed resizing (D2f), and the two-convolution layer recipe chosen
   (D2g). `QNetwork` implements the complete architecture.
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
| D2a | Q-network inputs | Complete MAPPO actor inputs: `global_obs` and `global_merge_obs` (8 map channels total), `vector` (robot identity), `comm_candidates`, and `comm_status`. The legal mask is applied to the output only | Matches the information available to MAPPO; giving the mask as an input channel is a later ablation if DQN struggles. Original choice 2026-11-13; complete input list confirmed 2026-11-16 | 2026-11-16 |
| D2b | Q-network architecture | b1: fully convolutional encoder-decoder with skip connections, producing a 64 x 64 Q-map | The same learned scoring rule applies across locations; spatial detail is preserved while gathering wider context. This encourages transfer to new maps without guaranteeing it. | 2026-11-16 |
| D2c | One network per robot or shared | One network shared by all robots, each robot feeding its own observation | Same as MAPPO (`share_policy=True`); doubles the data; works for any team size | 2026-11-13 |
| D2d | Where the non-map inputs join the spatial features | Separate map and non-map branches; broadcast the learned non-map summary over the compressed map and concatenate before decoding | Keeps the initial map processing smaller while allowing location scores to depend on the complete actor inputs | 2026-11-16 |
| D2e | Network widths | Wider version: encoder channels 32, 64, 128, 128, 128 at grids 64, 32, 16, 8, 4 per side; non-map summary length 128; decoder follows the reversed scales with skips | Chris chose the wider of the two proposed capacities; any learning benefit over the small candidate remains unmeasured | 2026-11-16 |
| D2f | Spatial reduction/expansion | Fixed resizing: 2 x 2 max pooling to halve the grids; bilinear interpolation to double them | Keeps resizing rules fixed while convolutions learn features at each scale; skips carry finer spatial detail | 2026-11-16 |
| D2g | Convolution blocks, summary layer, output layer, and initialization | Two 3 x 3 convolution/ReLU pairs per spatial block; linear/ReLU/LayerNorm non-map branch; linear 1 x 1 Q output; MAPPO-style orthogonal initialization | Two convolutions provide more processing between resizes; the raw output represents Q-values rather than probabilities | 2026-11-16 |

### Q-network input audit (2026-11-16)

The original D2a description omitted two MAPPO actor inputs. In addition to
the eight map channels and robot identity, the actor consumes `comm_candidates`
[K, 5] (map-patch summaries) and `comm_status` [8] (radio state). Both are
present even in `perfect` and `none` modes. `MIXBase` processes these alongside
identity in its non-map branch (onpolicy/onpolicy/algorithms/utils/mix.py).
Constructing the default actor confirmed eight map channels and 50 non-map
values: 8 x 5 candidate values, 8 status values, and 2 identity values.
Chris chose the complete MAPPO actor inputs on 2026-11-16. The frontier mask
remains an output filter; `frontier_goal` translates selected actions into
requests. Neither DQN helper array is a network input. Receiving radio-related
inputs does not add a transmission action: v1 still requests SILENCE.

For the default actor, MAPPO concatenates `global_merge_obs` then `global_obs`
along the channel axis. Its separate non-map branch concatenates flattened
`comm_candidates`, `comm_status`, then `vector`. With K candidate patches and
N robots, the latter has 5K + 8 + N values per robot (50 for K=8, N=2).
These are local actor inputs, not the privileged critic observation.

Chris chose separate branches for D2d on 2026-11-16: process the non-map values
separately and broadcast their learned summary into the compressed map before
decoding. Concatenating the summary with the map features preserves spatial
weight sharing and keeps the full-resolution input at eight channels.

`actor_inputs` in explore_bench/dqn.py prepares these two arrays. Tests compare
both directly with MAPPO's own preprocessing, check that DQN helpers and
critic-only fields are ignored, and reject batch dimensions that could mix
different robots' input values. `QNetwork` consumes the arrays using the
architecture selected in D2b-D2g.

### Selected widths for D2e (2026-11-16)

Chris chose the wider candidate. A feature channel is a learned map of pattern
responses, distinct from the eight fixed observation channels. The widths
below belong to the implemented separate-branch encoder-decoder with skips.

| Encoder grid | Selected feature channels |
|---|---|
| 64 x 64 | 32 |
| 32 x 32 | 64 |
| 16 x 16 | 128 |
| 8 x 8 | 128 |
| 4 x 4 | 128 |

The non-map branch produces a 128-value learned summary. Repeat that summary
over the 4 x 4 grid and concatenate with the 128 map feature channels: the
combined representation has shape [batch, 256, 4, 4]. Concatenation stacks
channels rather than adding their values. Decode through the reversed grid
scales, with the corresponding encoder features as skips, ending in one score
per 64 x 64 location. The wider selection increases representational capacity
and computation; its learning benefit is not established by choosing it.

Chris chose fixed resizing for D2f on 2026-11-16. Halve each grid with 2 x 2
max pooling, and double it with bilinear interpolation; separate convolutions
learn features at each scale. Pooling keeps each patch's largest feature
response; interpolation forms weighted averages of nearby responses. Learned
resizing was the alternative; neither has been compared for DQN learning here.

### Selected layer recipe for D2g (2026-11-16)

Chris chose the proposed two-convolution recipe. It builds on the chosen
widths, separate branches, skip connections, and fixed resizing:

- Each spatial block uses two learned 3 x 3 convolutions, each followed by
  ReLU (rectified linear unit, `max(0, x)`). Each convolution has stride 1 and
  one zero-padding cell on every side, retaining its input grid dimensions.
  Two such filters give a 5 x 5 receptive field within that grid: the region
  that can affect one output feature. Like MAPPO's convolution stages, these
  blocks use ReLU without extra feature-normalization layers.
- The non-map branch uses one linear layer from 5K + 8 + N values to 128,
  followed by ReLU and LayerNorm(128), mirroring MAPPO's non-map branch at
  the chosen wider width. Layer normalization rescales each sample's hidden
  summary; it does not mix different robots' samples or normalize Q-values.
- After concatenating the two branches at 4 x 4, a spatial block combines
  the 256 channels into 128. At every decoder stage, double the grid with
  bilinear interpolation (`align_corners=False`), concatenate the matching
  encoder skip, and apply a spatial block.
- A final 1 x 1 convolution maps 32 features to one raw Q-value per location.
  It has no output activation or probability normalization: Q-values estimate
  discounted future learning reward and need not lie in [0, 1] or sum to 1.
- Start convolution and linear weights with MAPPO's orthogonal initialization
  and zero biases; use gain sqrt(2) for layers followed by ReLU and gain 1 for
  the linear Q output. Keep LayerNorm's default initial scale 1 and offset 0.
  Initialization sets starting weights before any learning takes place.

The resulting implemented shapes, per sample, are:

| Stage | Input channels | Output shape (channels x rows x columns) |
|---|---|---|
| Encoder 0 | 8 | 32 x 64 x 64 |
| Pool, encoder 1 | 32 | 64 x 32 x 32 |
| Pool, encoder 2 | 64 | 128 x 16 x 16 |
| Pool, encoder 3 | 128 | 128 x 8 x 8 |
| Pool, encoder 4 | 128 | 128 x 4 x 4 |
| Concatenate summary, fusion block | 128 + 128 | 128 x 4 x 4 |
| Resize, skip 3, decoder 3 | 128 + 128 | 128 x 8 x 8 |
| Resize, skip 2, decoder 2 | 128 + 128 | 128 x 16 x 16 |
| Resize, skip 1, decoder 1 | 128 + 64 | 64 x 32 x 32 |
| Resize, skip 0, decoder 0 | 64 + 32 | 32 x 64 x 64 |
| Linear Q output | 32 | 1 x 64 x 64 |

Two convolutions per block provide more local processing between resizes than
the one-convolution alternative, at the cost of another set of weights and
computation. This design choice does not establish that the extra capacity
improves learning.

For K=8 and N=2, `QNetwork` has 2,397,985 learned parameters, matching the
analytical count for the selected recipe. The one-convolution alternative
would have had 1,420,129. These counts include convolution/linear biases and
the summary LayerNorm scale and offset; they describe model size, not learning
results.

### D2 implementation notes

- `actor_inputs` prepares the two selected branches and matches MAPPO's input
  ordering. It excludes DQN helper arrays and privileged critic fields.
- `QNetwork` accepts [batch, 8, 64, 64] maps and [batch, 5K + 8 + N] non-map
  values, and returns raw Q-values [batch, 64, 64]. The batch may mix robots
  and replay samples; the network shares weights without mixing batch rows.
- Tests verify the concrete shape at every external boundary, the exact
  2,397,985 parameter count for K=8/N=2, finite forward and backward passes,
  both input branches receiving gradients, sample independence, and an
  unrestricted linear Q output. These are architecture checks, not evidence
  that the network has learned an exploration policy.

## Proposed extensions (not in v1)

- **Rotate in place (proposed by Chris, 2026-11-13), four-beam only.** The
  four beams point at heading + 0/90/180/270 degrees
  (`FourBeamSensor.measure`, onpolicy/onpolicy/envs/GridEnv/sensors.py), and
  heading follows the last move (`calculate_pose`, GridEnv.py:822), so a robot
  only gets axis-aligned or diagonal beams by driving. A "turn 45 degrees"
  action would sense new directions without moving. It would be a few extra
  non-map outputs beside the Q-map, plus a new request type in
  `Session.step_request`, which is a new environment semantics version.
  MAPPO, Cost and MMPF cannot rotate, so it is an arm of its own: DQN with and
  without rotation on four-beam, asking whether an explicit sensing action
  helps when sensing is poor. With 360 degree sensing rotation changes nothing.
- **More than two robots.** The shared network itself handles any count, but
  configuration is frozen to 2 (explore_bench/config.py:115) and two inputs
  depend on team size: the identity one-hot has one entry per robot, and
  position markers are scaled by 255 // (1 + ... + N) (session.py:80,
  observations.py). A network trained with 2 robots therefore cannot run 3
  as is; decide how to encode identity before scaling the team.

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
