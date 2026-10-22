# MAPPO research decision record

Status: approved implementation specification, 2026-10-21. These are our
experiment choices, informed by the sources below; they are not settings
recovered from the original Explore-Bench experiment.

## Architecture and algorithm

| ID | Decision and reason | Alternative/tradeoff | Implementation and evidence |
| --- | --- | --- | --- |
| M01 | Retain the fork's MAPPO actor, critic, PPO update and rollout buffer; correct concrete defects | Replacing the MARL framework introduces a second algorithm-equivalence project | `mappo.py`, `training.py`; real-update/resume integration test |
| M02 | Share session, task, observation, recording, scoring and artifact services with future algorithms | A universal trainer would force DQN into on-policy assumptions | Controller protocol in `controllers.py`; scripted controllers exercise same evaluator |
| M03 | Shared actor, centralized critic; critic-only fields are excluded from actor dictionary | Independent policies need more parameters and change the comparison | Shared preprocessing in `observations.py`; privilege-invariance tests |
| M04 | Start feedforward with existing CNN and hidden 64 | Recurrence is useful for partial observability but adds sequence/state controls to the first experiment | Explicit `arguments()` overrides; checkpoints record resolved arguments |
| M05 | One joint Gaussian/mixed log probability per action | Per-dimension duplicates distort loss weighting | `SharedReplayBuffer`; analytical density/objective test |
| M06 | Apply explicit entropy coefficient once; research component weights are both 1 | Old hidden weights suppress motion entropy when adding a radio head | `ACTLayer`, `R_Actor`, resolved arguments |
| M07 | PPO 5 epochs, one minibatch, clipping 0.1, Adam learning rates 0.0005, normalized values/advantages | More epochs or minibatches may make multi-agent updates less stable; these still need validation | [MAPPO paper, section 5](https://arxiv.org/html/2103.01955v4); defaults in `config.py` |

## Task semantics and measurement

| ID | Decision and reason | Alternative/tradeoff | Implementation and evidence |
| --- | --- | --- | --- |
| M08 | One bounded A* edge per robot per decision; one broker advance | Whole-route decisions grant unequal motion and give radio latency a different meaning | `Session.step_request`; movement/distance test |
| M09 | Prevent out-of-grid A* neighbors and diagonal corner cutting | Original A* can index beyond map bounds or pass a diagonal between blocked cells | `BoundedAStar`; new semantics version explicitly records this change |
| M10 | Frontier representatives are known-free cells adjacent to unknown space, clustered deterministically | The old learned selector sometimes requires private-free cells even when a free cell was delivered by radio | `frontier_candidates`; works from delivered beliefs only |
| M11 | Reward only newly sensed team-union cells plus a true-success bonus | Reset-area and timeout bonuses create rewards unrelated to useful action | Session reward + independent scorer checks; wait/timeout tests |
| M12 | Waiting never stops a teammate; stop on 99% or external 1,000-decision cap | Early no-frontier termination can prevent waiting for a message or another robot's progress | `Session`; moving-teammate and timeout tests |
| M13 | Bootstrap on external truncation, cut the GAE trace and reset session | Treating every done as terminal biases values; using reset observation bootstraps from the wrong state | Training uses final observation; [Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) |
| M14 | Physical coverage measures the union of private sensing, masked by terrain validity | Received belief should not create exploration without movement/sensing | `measurement()` and independent accounting tests |
| M15 | Accumulate executed edges per robot and retain path traces | Endpoint displacement underestimates a winding full route; old path log mixes robots | Recorder; scorer crosschecks positions, distances and fixed-edge allowance |
| M16 | Record 90/98/99% crossings, but use 99% as training target | Original learned environment stops at 98%, making T99 unavailable | One configuration drives both training and evaluation; missing milestones stay missing |
| M17 | Report geometric overlap and area/ratio dispersion under explicit names | Paper and released code use different overlap and sigma conventions | Schema v2; old audit keeps its separate paper-equation diagnostics |

## Sampling, selection, statistics and operation

| ID | Decision and reason | Alternative/tradeoff | Implementation and evidence |
| --- | --- | --- | --- |
| M18 | Five independent training seeds; disjoint worker episode streams and fixed paired evaluation starts | More evaluation episodes cannot replace independent training runs | `TrainingWorld`, task tables, seed-level summaries; [RL evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) |
| M19 | Separate familiar-map and unseen-layout test suites | Same-map new-start testing alone cannot establish layout transfer | Frozen map groups and task tables; two held-out loop maps limit claim breadth |
| M20 | Pick checkpoints using validation success, AUC, capped T99, earliest step | Selecting on final tests or reward alone confounds performance claims | `select()` checks validation inventory; test CLI consumes selection artifact |
| M21 | Include step-zero untrained actor in validation inventory | Hiding a better untrained baseline could conceal training degradation | Selected training step is recorded; step 0 winning requires investigation before a learned-performance claim |
| M22 | Validate snapshots in a separate stage after training | Inline evaluation can alter RNG/session state and complicate reproducibility | Actor-only evaluator and fresh-process test; [SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) motivates separation of events |
| M23 | Immutable checkpoints with optimizers, normalizer, global RNG and complete worker sessions | Weights alone permit inference, but do not faithfully continue training | Resume test compares actor tensors bit-for-bit; [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) |
| M24 | Bootstrap independent training-seed summaries, 10,000 replicates and fixed analysis seed | Episode-level pooling produces artificially narrow uncertainty | `score()`; [Statistical Precipice](https://arxiv.org/abs/2108.13264) supports uncertainty reporting; exact bootstrap choice is ours |
| M25 | AUC carries successful final coverage to the cap; failures get capped T99=cap+1 | Successful-run-only times can make a method with many failures look efficient | Scorer also reports milestone success counts and conditional medians |
| M26 | Version new schemas/semantics and retain historical artifacts | Silently changing old pipeline semantics would invalidate the prior regression claim | Schema 2, `level0_fixed_edge_v1`, `level0_ctde_v1`; legacy test suite |
| M27 | Separate Slurm stages and retry to a new attempt | Automatic unbounded retries consume resources and can obscure failure history | USR1 requests safe checkpoint; explicit resume command and manifests |
| M28 | Defer DQN's action discretization and keep its learner/replay separate | Vanilla DQN cannot optimize continuous coordinates or the existing mixed action tuple | Controller returns semantic goal requests; [DQN documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) |
| M29 | Sense at the actual executed grid pose, with no floating-point round trip | Historical truncation can move the sensing origin onto a neighboring wall; the shifted toy layout exposed this | `Session._sense_agent`; sensor-origin regression test |
| M30 | Save and reload every resolved policy argument with the checkpoint | Reconstructing unspecified defaults from a later parser can silently change inference | Checkpoint manifest `policy_arguments`; actor-only loader |
| M31 | Run a predeclared controlled-corridor learning check before the paper-map pilot | A successful optimizer call alone does not demonstrate exploration improves | `sanity.py`; validation AUC gain >= 0.01 over untrained actor, separate selected-checkpoint test |

## What can be claimed after verification

- Tests can demonstrate reward/transition/recording correctness and exact CPU
  checkpoint continuation on the tested platform.
- A completed workflow can demonstrate fixed-task comparison with provenance.
- Learned performance, convergence and the sufficiency of the initial budget
  require the controlled learning check and measured pilot/full results.
- A claim of improvement over Cost/MMPF requires new controls using the same
  semantics, maps, starts, sensors, target and budget. The old 80-run audit is
  historical validation evidence, not the matched-clock comparison dataset.
- Perfect sharing is an information upper control within the new architecture;
  it does not equal the legacy oracle navigation contract.

## Remaining scientific limits

The simulator has simplified grid motion, no physical radio stack, no dynamic
inter-robot collision model, and a fixed high-level frontier interface. Diagonal
edges have greater distance than axial edges despite sharing one logical slot.
The policy receives resized 64×64 maps, which can lose fine geometry. Classical
goal selectors operate on full-resolution legal beliefs; report that input
representation difference when comparing methods. DQN may introduce another
action representation difference that must be declared.

Five seeds and two reserved loop maps are modest evidence. A confidence interval
may be wide or unstable. A successful engineering smoke is not evidence of a
trained exploration strategy. CUDA hardware performance and bitwise CPU/CUDA
equivalence require separate measurements. The initial recipe is not a claim
that published Explore-Bench numbers or MAPPO benchmark results were recovered.
