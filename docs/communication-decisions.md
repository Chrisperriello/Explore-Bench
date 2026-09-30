# Communication Design Decisions

This document records the reasoning behind the Level-0 communication model so
the implementation can be defended, criticized, and revised as a research
choice rather than mistaken for an incidental software detail.

## Research Question

The original Level-0 learning path supplied every actor with a globally merged
map and exact team positions at every decision. That tests cooperative
navigation under perfect shared state, but it cannot test when robots should
communicate, what they should send, or how a constrained channel changes
coordination.

The implemented question is narrower and measurable:

> Given local sensing and a limited broadcast channel, can a multi-robot policy
> trade communication cost for exploration performance by selecting silence,
> pose, or a bounded map patch?

## Decision 1: A Synchronous Broker, Not a Thread

The broker is called directly from `GridEnv.step` and owns a delivery priority
queue. One environment decision is one transmission slot.

Why:

- MAPPO rollouts already define a discrete causal clock.
- A thread would introduce scheduler-dependent ordering and difficult seeding.
- Explicit advancement makes latency, collision, and reward attribution
  reproducible.
- Vectorized environments can each own an isolated broker without shared state.

The isolation requirement is tested by transmitting in one member of a vector
environment and confirming that the other member's belief and metrics remain
unchanged.

Rejected alternative: a background thread or socket-like service. It would be
appropriate for wall-clock robotics integration, but it adds concurrency without
improving the Level-0 research abstraction.

The answer to "thread or queue?" is therefore: a synchronous broker with an
internal delayed-message queue. The queue models time in flight; it does not
serialize access by itself. Channel mode decides whether simultaneous sends are
parallel or collide.

## Decision 2: Broadcast With Explicit Channel Modes

Messages are broadcasts to every in-range receiver. Four modes separate distinct
experimental assumptions: perfect, none, parallel, and shared collision.

Why:

- Broadcast matches local map sharing without requiring a learned recipient ID.
- `perfect` reproduces an information upper bound.
- `none` isolates the value of communication.
- `parallel` isolates bandwidth limits from medium-access coordination.
- `shared_collision` makes scheduling a joint multi-agent problem.

Rejected alternative: one universal FIFO that permits exactly one queued sender.
That would silently resolve contention for the agents and remove collision
avoidance from the learned task. A future TDMA mode could explicitly implement
that assumption and be compared as a separate scheduler.

## Decision 3: Semantic Actions Instead of Arbitrary Bytes

An actor selects silence, pose, or one of eight deterministic map tiles.

Why:

- The action space remains small enough for categorical MAPPO exploration.
- Logical bit accounting is interpretable across implementations.
- A fixed candidate generator separates content proposal from send scheduling.
- Learned compression and protocol parsing are not necessary to answer the
  initial research question.

Rejected alternative: transmit the complete local map. At 2 bits per cell, a
`250 x 250` map is already 125,000 payload bits and would collapse the decision
to rare all-or-nothing synchronization.

Rejected alternative: choose arbitrary tile coordinates with continuous action
components. Those coordinates would complicate credit assignment and introduce
invalid or redundant actions. Ranked candidates preserve bounded choice while
remaining inspectable.

## Decision 4: Logical Bits as the Common Currency

Token bucket capacity, refill, episode budget, and reward cost all use logical
bits. Both successful transmissions and collisions consume them.

Why:

- One unit controls burst rate, long-run rate, total episode usage, and cost.
- Pose and patch messages can be compared fairly despite different sizes.
- Charging failed airtime prevents collision or range probing from being free.
- The model remains independent of Python serialization and process overhead.

This is not a claim about a specific radio's throughput. Thesis results must
describe these values as simulation parameters, not physical-layer measurements.
Repeated collisions must consume tokens and episode budget on the same schedule
as successful attempts. Tokens refill up to the configured capacity, so the
action mask can reopen after a temporary token shortage. The episode budget
never refills; once it cannot fund any message, silence remains the only choice
for the rest of the episode.

## Decision 5: Snapshot at Send, Latest Timestamp Wins

Payloads copy the sender state when the action is taken. A receiver applies a map
cell when the message timestamp is newer than its current cell timestamp. A
local observation wins an equal-timestamp tie against a remote claim. Two
equal-timestamp remote claims use the fixed priority occupied, then free, then
unknown, so sender and queue order cannot change the result.

Perfect mode has no receiver-local provenance during its instantaneous merge,
so occupied wins every simultaneous conflict, including one between private
maps. The constrained modes deliberately preserve an equally recent local
observation instead. Noiseless sensing should not create a real disagreement;
the distinction must be revisited or retained explicitly when sensor noise is
introduced.

Why:

- Latency should make information stale, not retroactively current.
- Local sensing is authoritative when it is newer than an incoming patch.
- Cell timestamps give deterministic conflict resolution without assuming map
  certainty probabilities that the existing simulator does not represent.

Rejected alternative: union every delivered map blindly. That could overwrite a
new local observation with an old remote claim and make added latency improve or
damage maps for accidental implementation reasons.

## Decision 6: Centralized Critic, Decentralized Actor

The actor is restricted to local and delivered information. The critic sees
ground-truth team state and all beliefs during training.

Why:

- This follows centralized training with decentralized execution used by MAPPO.
- The critic can estimate the global value without turning perfect state into an
  execution-time shortcut.
- Keeping actor and critic conversion explicit makes leakage auditable.

The principal validity check is not merely that a private map exists. The
navigation planner must also consume the actor's belief map. Both the learned
runner and standalone planners now do so. Unknown belief cells are treated as
blocked for route planning, even when the corresponding ground-truth cells form
a shorter free corridor.

## Decision 7: Communication Is Part of the Joint Action

The policy action is a two-dimensional continuous navigation output plus one
categorical communication output. Their log probabilities are summed into one
joint PPO log probability.

Why:

- Navigation and communication affect the same next state and reward.
- Joint optimization lets the policy learn context-dependent tradeoffs.
- Action masks prevent the categorical head from selecting physically invalid
  transmissions while preserving silence.

Rejected alternative: a separate heuristic communication process during learned
navigation. It is useful as a baseline but cannot answer whether scheduling can
be learned.

## Decision 8: Communication Cost Is Additive to Task Reward

The environment records the original task reward and subtracts a size-normalized
communication penalty from each sender.

Why:

- The original exploration objective remains identifiable in logs.
- A scalar cost exposes a clear Pareto tradeoff between exploration and airtime.
- Sender-local charging gives direct credit assignment.

The coefficient is not universal. Results should include a cost sweep or report
a Pareto frontier rather than claim one default value is optimal.

## Assumptions and Threats to Validity

- One radio slot equals one high-level policy decision, not one second.
- Euclidean range in grid cells ignores walls and radio propagation.
- Packet loss is independent between receivers and decisions.
- There are no acknowledgements, retries, routing, capture effects, or hidden
  terminals.
- Every patch cell uses 2 logical bits and metadata has fixed fields.
- The candidate generator is privileged code but uses only the sender's belief
  and broadcast history.
- Team reward can still reveal aggregate progress indirectly; it does not reveal
  another agent's map content, but it is a cooperative learning signal.
- The critic is privileged during training, so claims concern decentralized
  execution performance, not fully decentralized learning.
- Level-1 has not yet been given the same broker, so conclusions initially apply
  to Level-0 simulation.

These are controlled abstractions, not defects to hide. Each should appear in
the thesis methodology and limitations sections.

## Extension Criteria

A proposed change should answer all of these questions:

1. What physical or algorithmic assumption does it add?
2. Can it be expressed as a new mode without changing existing controls?
3. Is its randomness seeded per environment?
4. Does it preserve the decision timeline and prevent actor information leaks?
5. Are resource usage and failed attempts measurable?
6. Which baseline or ablation makes its effect identifiable?

Examples of defensible extensions are explicit TDMA, receiver addressing,
acknowledgements, learned compression, wall attenuation, and a ROS transport
adapter. They should not be folded into the default model without separate
experimental controls.
