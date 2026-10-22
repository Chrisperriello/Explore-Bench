# Current MAPPO: implementation review

Reviewed at historical base `a371838`, with read-only runtime probes on
2026-10-21. The 80-run classical audit validates provenance, regression and a
directional paper claim. It does not exercise the learned-policy training loop.

## Algorithm and inherited Explore-Bench behavior

MAPPO is PPO with a centralized value function for cooperative agents. A shared
actor is applied separately to robot observations. During training, the critic
can observe additional team state to reduce variance in the policy gradient.
Execution needs only the actor. Parameter sharing does not mean that actors may
read another robot's undisclosed observations.

The fork retains separate actor and critic networks and Adam optimizers:
[policy](../../onpolicy/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py).
Mixed map observations are encoded by a CNN and other features by an MLP:
[encoder](../../onpolicy/onpolicy/algorithms/utils/mix.py). A GRU is optional;
the new initial recipe explicitly chooses the feedforward path.

Explore-Bench applies a Gaussian policy to two latent coordinates. A sigmoid
maps them to the grid; a handwritten routine selects a nearby frontier, and
A* plus the local planner execute movement. The network therefore learns a
high-level exploration preference, not low-level controls or frontier detection.
In communication modes, a categorical head adds silence, pose, and bounded
map-patch choices. The categorical mask controls radio feasibility, while
continuous motion remains unmasked.

At collection time, the learner saves actions and their log probabilities.
GAE uses rewards and value estimates to calculate advantages. PPO compares the
updated action probability with its collection probability, clips the ratio,
and optimizes the actor. The critic regresses to returns using value
normalization and Huber loss. Several update epochs use each fresh rollout;
old rollouts are not an off-policy experience dataset. See
[PPO update](../../onpolicy/onpolicy/algorithms/r_mappo/r_mappo.py) and
[buffer](../../onpolicy/onpolicy/utils/shared_buffer.py).

The old shell script requests 40 parallel environments, 50 million transitions,
20-step rollouts, three PPO epochs and learning rates of 0.001. Several CLI
flags have reversed meanings (`store_false`), including recurrence and WandB.
The seed loop passes constant seed 1. The new CLI resolves a named JSON
experiment, rejects unknown options, and records all effective parameters.

## Confirmed defects and their disposition

| Finding | Evidence | New research behavior |
| --- | --- | --- |
| Reset coverage counted as first-action reward | A no-movement probe discovered zero cells, ended at 0.12635 coverage and gave each robot 101.092635 reward | Session initializes previous coverage to reset coverage |
| One waiting robot terminates team | `GridEnv.step` sets the joint termination flag when a robot goal equals its current position | Waiting does not terminate; target/cap decide the episode |
| Completion bonus applied to failure/timeout | Success, waiting and timeout share the old bonus block | Bonus applies only to first target crossing |
| Evaluation reads training history | Changing `all_agent_pos_map` changed `_resize_eval_convert` output | Legacy conversion corrected; shared builder owns session history |
| Evaluation duplicates robot identity | Both legacy evaluation vectors were `[1, 0]` | One-hot identity uses actual agent and environment indices |
| Gaussian joint log probability broadcast twice | Scalar log probability assigned into Box buffer with two columns | One scalar stored for Box and mixed actions |
| Unequal movement opportunity | Learned step assigned `pose[-1]`; Cost/MMPF assigned `pose[1]` | New session executes at most one route edge per decision |
| Configured episode cap ignored | Requested legacy cap 20 constructed cap 100 | New session uses resolved configuration |
| History cleared at update boundaries | Old runner initialized history every rollout | Histories reset only at episode boundaries |
| Incomplete checkpoints | Old save wrote actor/critic weights only, repeatedly overwriting | Immutable exports and full continuation state |
| GPU value normalizer registration | `.to(cuda)` after `Parameter` construction returns an unregistered Tensor | Create parameters on the destination device before registration |
| Weak evaluation accounting | `eval_episodes` unused by GridEnv; all eval points logged at zero | Fixed explicit task inventory and separate evaluator |
| Hidden entropy coefficients | Mixed head applied 0.0025/0.01 before the learner coefficient | Research recipe explicitly sets component weights to 1 |
| Path logging mixes robots | Learned movement appends all paths to robot 0 | Per-agent executed-edge trace and distance |
| Sensor origin can differ from recorded pose | In a shifted toy corridor, recorded pose `[15,21]` round-tripped to sensor origin `[14,20]`, on a wall | New session passes the integer pose directly to the unchanged canonical sensor |

The historical environment transition methods remain available for the old
audit. Corrections to reward, termination and movement are implemented in the
opt-in `Session`; they are explicitly new research semantics. Existing legacy
entry points are not the recommended training workflow.

## Literature and implementation correspondence

- [MAPPO paper](https://arxiv.org/html/2103.01955v4), sections 3 and 5:
  centralized value learning, value normalization, restrained data reuse,
  clipping and batch-size sensitivity. These support the design and starting
  hyperparameters, not a promise of optimal exploration results.
- [Official MAPPO implementation](https://github.com/marlbenchmark/on-policy):
  parameter sharing and separation of algorithm, runner, and environment code.
- [Explore-Bench paper](https://arxiv.org/html/2202.11931v1), VI-D:
  learned global-goal selection followed by frontier selection, trained in
  Level 0. The new fixed-edge experiment is not a numerical reproduction.
- [Stable-Baselines3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html):
  repeated runs and separate evaluation environments.
- [Stable-Baselines3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html):
  explicit checkpoint/evaluation events and counting environment transitions
  correctly with multiple workers.
- [Gymnasium time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/):
  the distinction between termination and external truncation changes value
  bootstrapping.
- [PyTorch checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html):
  training continuation requires optimizer state as well as network weights.
- [Statistical Precipice](https://arxiv.org/abs/2108.13264): report uncertainty
  from limited independent training runs rather than relying on point estimates.
- [DQN interface and algorithm](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html):
  discrete actions, persistent replay and a target network require a different
  learning backend from MAPPO.

The Stable-Baselines3/Gymnasium links are design references. This implementation
does not install their latest releases or migrate the pinned Python 3.8/Gym
runtime as part of this work.
