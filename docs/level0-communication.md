# Level-0 Communication

Level-0 can model communication as a limited resource instead of giving every
robot the team's merged map and exact poses immediately. The feature is opt-in:
when `--communication_mode` is omitted, the legacy action and observation path
is retained.

## Model Boundary

The communication broker is a deterministic state machine advanced by
`GridEnv.step`. It is not an operating-system thread. Each environment decision
is one radio slot, while a priority queue stores messages that are waiting for
their delivery step. This design makes rollouts reproducible, works in vectorized
environments, and gives latency a precise relationship to policy decisions.

The implementation is shared by both Level-0 entry points:

- `onpolicy/onpolicy/envs/GridEnv/communication.py` is canonical.
- `grid_simulator/communication.py` loads that implementation for standalone
  cost and MMPF evaluation.

The ROS/Gazebo Level-1 communication path is intentionally unchanged. A future
adapter can translate the same semantic messages to a real transport without
changing their experimental meaning.

## Decision Timeline

For decision `t`, the environment performs these operations in order:

1. Each actor observes its private map, previously delivered belief, and local
   radio state.
2. Each actor selects a navigation goal and one communication action.
3. The broker validates the action, charges its logical bits and reward cost,
   resolves a shared-slot collision if applicable, and snapshots the payload.
4. Robots move and collect new sensor observations.
5. The broker advances to `t + 1`, incorporates each robot's new private
   observations, and delivers messages whose latency has elapsed.
6. The next actor observation is constructed.

A latency of zero delivers the send-time snapshot during the transmission phase.
A latency of one means that data selected at decision `t` first appears after
the broker advances to `t + 1`. A delayed map patch cannot overwrite a newer
local observation because map cells are fused by timestamp.

## Semantic Messages

The learned action is `Tuple(Box(2), Discrete(K + 2))`:

| Action | Meaning |
| --- | --- |
| `0` | Silence |
| `1` | Current pose and heading |
| `2..K+1` | One of the current top `K` map-patch candidates |

The default patch is `8 x 8` cells and `K = 8`. Candidates are ranked
deterministically by newly known cells, frontier cells, and row-major tile
position. The actor receives five normalized features per candidate: row,
column, known fraction, changed fraction, and frontier fraction.

The broker accounts for logical information, not Python object size. With two
agents, a `250 x 250` map, 8-bit coordinates, and a 16-bit timestamp:

```text
header = sender(1) + type(2) + timestamp(16)
pose   = header + row(8) + column(8) + heading(2) = 37 bits
patch  = header + tile index(10) + 64 cells * 2 bits = 157 bits
```

The 2-bit cell encoding is sufficient for unknown, free, and occupied. Message
sizes are recalculated for other map dimensions, agent counts, and tile sizes.
This is an abstract bandwidth model; it does not include network framing,
checksums, encryption, or retransmission headers.

## Channel Modes

- `perfect`: immediate, unlimited synchronization of maps and poses. Policy
  communication actions are masked to silence. This is the upper-bound control.
- `none`: beliefs remain private and actions are masked to silence. This is the
  no-communication control.
- `parallel`: every valid sender can transmit in the same decision slot.
- `shared_collision`: one valid sender can occupy a slot; two or more valid
  senders collide and none of those messages is delivered.

Range is evaluated at transmission time. Packet loss is sampled independently
per intended receiver from the environment seed. A collision or an out-of-range
broadcast still consumes sender resources because airtime was attempted.

## Resource Limits

The default radio configuration is:

| Parameter | Default |
| --- | ---: |
| Range | 40 cells |
| Latency | 1 decision |
| Packet loss | 0 |
| Cooldown | 3 decisions |
| Token capacity | 314 bits |
| Token refill | 53 bits per decision |
| Episode budget | `ceil(max_steps / 3) * patch_bits` |
| Time to live | 8 decisions |
| Cost | 0.01 per patch-equivalent |

The sender cost is proportional to message size:

```text
communication cost = 0.01 * attempted_bits / patch_bits
training reward = task reward - communication cost
```

Invalid actions caused by cooldown or exhausted resources are masked before
sampling and are rejected again by the broker. An explicit episode budget of
zero disables all transmissions and is useful as a no-communication control.
Silence always remains valid.

## Information Available to the Policy

Each actor receives only information available under decentralized execution:

- its explored and obstacle maps, current position, and position history;
- its belief map after local sensing and delivered patches;
- last delivered peer poses with freshness, but not true peer poses;
- candidate patch features;
- remaining tokens, episode budget, cooldown, and last radio result;
- its agent identity.

The centralized critic additionally receives the true team map and positions,
all agent belief maps, and global broker resource state. This is centralized
training with decentralized execution: privileged state can reduce value
estimation variance, but it never enters the actor network or navigation
planner.

## Training

From `onpolicy/onpolicy/scripts`:

```bash
python train/train_grid.py \
  --env_name GridEnv \
  --algorithm_name mappo \
  --experiment_name shared_radio \
  --num_agents 2 \
  --communication_mode shared_collision \
  --max_steps 100 \
  --local_step_num 1 \
  --use_wandb
```

Every default can be overridden with the `--comm_*` options exposed by
`train_grid.py`. Communication-enabled runs use one replay-buffer entry per
environment decision so communication latency and PPO time steps have the same
unit, and the environment terminates at `max_steps` so the episode budget resets
at the same boundary. The implementation currently requires the shared-policy, centralized
critic runner. Note that this repository's legacy `--use_wandb` flag disables
Weights & Biases logging despite its name.

## Standalone Baselines

The cost and MMPF planners use each robot's communication-derived belief instead
of the omniscient merged map. They support three fixed transmit protocols:

- `round_robin`: the next eligible robot sends its best patch;
- `always`: every robot attempts its best patch per slot;
- `none`: all robots remain silent.

Example:

```bash
cd grid_simulator
python GridEnv.py cost 2 \
  ../onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm \
  --communication_mode shared_collision \
  --communication_protocol round_robin \
  --seed 1
```

`always` is deliberately simple. Under `shared_collision` it exposes the cost
of uncoordinated access rather than serving as a competitive scheduler.

## Logged Metrics

Training logs per-step attempted and delivered messages/bits, collisions, packet
losses, expirations, mean delivered latency, and communication reward cost. It
also logs cumulative episode totals at termination. Existing exploration ratio,
completion step, and per-agent metrics remain available. Terminal information is
preserved across vector-environment resets so the last step is included.

## Compatibility and Limits

- Omitting `--communication_mode` preserves the legacy continuous action space
  and observation path.
- Communication modes require new training; old checkpoints do not have the
  discrete radio head or new observation encoders.
- The model represents broadcast semantics but no acknowledgements, routing,
  retransmission, interference capture, or byte-level serialization.
- Candidate generation is fixed and deterministic. The policy learns whether
  and which candidate to send, not arbitrary compression.
- Radio decisions occur once per high-level environment decision, even when the
  local planner traverses several cells during that decision.
