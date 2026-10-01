# Communication Experiment Protocol

This protocol is intended to make claims about constrained communication
identifiable and reproducible. Record the full command, commit, map split, seeds,
sensor configuration, and communication configuration for every run.

## Primary Questions

1. How much exploration performance is lost when perfect sharing is removed?
2. Can learned communication recover that performance under a fixed bit budget?
3. Does learned scheduling outperform fixed scheduling on a collision-prone
   shared channel?
4. How do range, latency, loss, budget, and communication price change the
   policy's behavior?

## Required Controls

Run these conditions with identical maps, spawns, sensors, training steps, and
seeds:

| Condition | Mode | Sender policy | Purpose |
| --- | --- | --- | --- |
| Legacy | omitted | implicit perfect merge | Reproduce prior code path |
| Perfect | `perfect` | forced silence | Information upper bound |
| Private | `none` | forced silence | No-communication lower control |
| Learned parallel | `parallel` | MAPPO | Value of content under no contention |
| Learned shared | `shared_collision` | MAPPO | Learned content and scheduling |
| Fixed round-robin | `shared_collision` | next eligible sender | Collision-free scheduler baseline |
| Fixed always | `shared_collision` | deterministic | Uncoordinated access stress test |
| Fixed none | `shared_collision` | deterministic | Same broker with no sends |

The fixed protocols are available to standalone cost and MMPF planners. For a
strict policy comparison, also evaluate the learned navigator with communication
actions forced by equivalent protocols or clearly label planner differences.

Legacy and `perfect` are not identical controls: legacy uses the old observation
contract, while `perfect` uses the new actor/critic architecture with immediate
belief synchronization. Their difference estimates architecture and retraining
effects unrelated to channel constraints.

For standalone planners there is an additional distinction. With communication
omitted, the legacy navigation helper retains its original ground-truth route
map. Communication-enabled modes use delivered beliefs and treat unknown cells
as blocked. Compare legacy and `perfect` map fusion under fixed trajectories
before interpreting an end-to-end coverage difference as a communication
effect. The staged comparison is specified in the
[communication validation plan](communication-validation-plan.md).

## Default Treatment

Use the documented defaults as the main treatment:

```text
mode=shared_collision, tile=8, candidates=8, range=40 cells,
latency=1, loss=0, cooldown=3, capacity=314 bits,
refill=53 bits/decision, ttl=8, cost=0.01/patch-equivalent
```

For a 100-step, two-agent, `250 x 250` episode, the derived per-agent budget is
`ceil(100 / 3) * 157 = 5,338` logical bits.

## Ablations

Change one factor at a time from the main treatment before testing interactions:

| Factor | Suggested values |
| --- | --- |
| Range | 20, 40, infinity cells |
| Latency | 1, uniform 1-3, fixed 4 decisions |
| Packet loss | 0, 0.10, 0.25 |
| Budget multiplier | 0.5, 1, 2 |
| Cooldown | 0, 1, 3, 5 decisions |
| Patch size | 4, 8, 16 cells |
| Communication cost | 0, 0.005, 0.01, 0.02 |
| Channel | parallel, shared collision |

Patch size changes both payload bits and the semantic granularity of an action,
so it is not a pure bandwidth ablation. State that coupling explicitly.

## Metrics

Report exploration and communication separately before combining them:

- success rate at the target coverage;
- decisions and path length to 90% and 98% coverage;
- area explored per decision;
- overlap and per-agent coverage dispersion;
- task reward before communication cost;
- total reward after communication cost;
- attempted and delivered messages and bits;
- delivery ratio: delivered receiver-copies divided by in-range recipient
  opportunities;
- collision, loss, and expiry counts;
- mean delivered latency;
- bits per newly explored cell;
- silence, pose, and patch action frequencies;
- token/budget exhaustion frequency.

Because a broadcast can have multiple receivers, distinguish sender messages
from delivered receiver-copies. The broker logs `in_range_recipients` as the
denominator for loss/expiry analysis. Do not divide delivered copies by
transmitted sender messages and label it a packet delivery probability.

## Statistical Procedure

- Preselect training and evaluation seeds before inspecting results.
- Use at least five independent training seeds for policy comparisons; increase
  this if confidence intervals remain wide.
- Evaluate each checkpoint on the same held-out map/spawn seeds.
- Report mean, standard deviation, and a 95% confidence interval across training
  seeds rather than treating evaluation episodes as independent training runs.
- Plot performance against delivered bits and attempted bits. Collisions make
  those two axes meaningfully different.
- Select checkpoints by a rule fixed in advance, such as final checkpoint or
  validation task reward, and apply it to every condition.
- Report all planned primary comparisons or apply a multiple-comparison
  correction; do not select only favorable maps or seeds.

## Hypotheses

Write hypotheses before running the sweep. Reasonable examples are:

- `perfect` will achieve the strongest exploration efficiency but use an
  unrealistic information assumption.
- learned parallel communication will outperform `none` when private fields of
  view diverge enough to make patches informative.
- learned shared-channel communication will produce fewer collisions and better
  explored-cells-per-attempted-bit than `always`.
- round-robin will be robust at high load but waste slots when the designated
  sender has little new information.
- increasing latency or reducing range will shift actions toward silence or
  pose messages because patches become stale or unreachable.

These are hypotheses, not implementation guarantees.

## Reproducibility Record

Save this metadata with each result set:

```text
git commit and dirty status
training and evaluation command lines
training seeds and evaluation seeds
map files and train/validation/test split
number of agents and sensor configurations
all --comm_* values
policy checkpoint selection rule
software environment and hardware
raw per-episode exploration and communication metrics
```

The broker uses a per-environment seeded random generator for loss and variable
latency. Deterministic candidate ordering prevents platform-dependent tile
selection from changing the action meaning.

## Sanity Checks Before a Long Run

1. Confirm the actor observation does not change when an unseen peer map changes
   without a delivered message.
2. Confirm `none` never changes a peer belief and `perfect` synchronizes it.
3. Confirm 100% packet loss produces the same belief maps as `none`.
4. Confirm an ideal channel matches `perfect` after all known tiles are sent.
5. Confirm latency longer than TTL never delivers a message.
6. Confirm a zero budget blocks every send and an episode-long cooldown blocks
   every send after the first slot.
7. Confirm a one-step-latency message is absent before and present after one
   broker advance.
8. Confirm two simultaneous valid sends collide only in `shared_collision`.
9. Confirm collision attempts reduce tokens, budget, and reward.
10. Confirm the planner consumes each actor's belief rather than the true merged
   map by placing an unseen free shortcut in the ground-truth map.
11. Confirm an equal-timestamp local observation wins deterministically over a
   conflicting delivered patch, and permuting two conflicting remote senders
   does not change the receiver's result.
12. Confirm patch collisions drain every sender's tokens and budget, token
   refill reopens the action mask, and episode-budget exhaustion never recovers.
13. Confirm a message sent in one vectorized environment never changes another
   environment's beliefs, queue, or metrics.
14. Confirm actor and critic tensors have the documented information boundary.
15. Run a short seeded job twice and compare communication metrics.
16. Send a delayed patch just before one in-process vector environment resets;
    confirm that no queue, resource, metric, peer state, or message crosses the
    episode boundary while another environment continues.
17. Run standalone round-robin on `shared_collision` and require zero
    collisions, nonzero transmissions, and nonzero deliveries.
18. Compare legacy merged maps with `perfect` beliefs under fixed sensor frames,
    then separately locate the first divergence in a seeded standalone cost run.

Detailed automated coverage is tracked in the
[communication validation plan](communication-validation-plan.md). Stepwise
collapse, episode reset isolation, incremental refill, planner leakage,
timestamp ties, in-process vector isolation, shared-channel delivery, and the
legacy/perfect boundary are verified by focused tests. The legacy/perfect test
locates the first difference at the navigation map before movement instead of
misattributing later coverage divergence to fusion. The latest headless
standalone cost smoke and 40-timestep MAPPO `shared_collision` smoke both pass;
repeat them for the exact commit and environment used by each experiment
campaign.
