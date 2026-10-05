# Level-0 Data Collection

This document defines the standalone Level-0 data-collection contract. It is
the source of truth for what the collector measures, which information it may
inspect, how episodes end, and why the collector is kept separate from the
exploration controllers.

## Scope

`grid_simulator/collect_experiments.py` runs the existing standalone `cost` or
`mmpf` controller for finite, reproducible episodes. It supports headless batch
execution, the interchangeable Level-0 sensors, every existing communication
mode and fixed communication protocol, and teams of arbitrary positive size.

The collector is initially intended for handwritten-controller experiments.
It does not train MAPPO and does not change the learned-policy runner. A later
learned-policy evaluator should emit the same episode, step, and agent schemas
instead of inventing another result format.

The existing interactive `grid_simulator/GridEnv.py` command is deliberately
unchanged. It remains useful for visually debugging one trajectory.

## Information Boundary

The collector is a privileged evaluator, not a component of a robot.

```text
ground truth ------------------------------> evaluator
                                                |
four-beam sensing -> private map -> controller  | results only
                         ^          |            v
                  delivered data    +-> A* -> environment
```

The controller continues to receive only the map allowed by its communication
condition. The evaluator may inspect the ground-truth map after a decision to
define the denominator of a coverage metric, but it never writes ground truth,
evaluation statistics, or another robot's private map into a controller belief.

This boundary is why the collector is implemented outside `GridEnv`. Adding
metrics to controller observations would invalidate the communication study.

## Design Decisions

Every material implementation choice is recorded below.

### 1. Separate script, unchanged simulation engine

The collector imports the existing standalone environment and calls the
existing sensor, frontier selector, A* planner, motion logic, and communication
broker. It does not copy those algorithms. This keeps experiment behavior tied
to the implementation being evaluated and leaves the interactive entry point
available.

Rejected alternative: embed CSV writing in `GridEnv`. That would couple
simulation state transitions to filesystem behavior and make vectorized or
learned-policy reuse harder.

### 2. Headless adapter instead of a UI dependency

The traditional step functions call `plot_map_with_path` unconditionally, even
when `visualization=False`. `HeadlessGridEnv` overrides only that plotting hook.
If `--visualize` is supplied, it delegates to the original plotter. Planning,
sensing, movement, map fusion, and radio behavior are inherited unchanged.

Visualization is restricted to one episode per command. Opening several UI
windows during a sweep is error-prone and contradicts headless collection.

### 3. Fresh environment for every episode

Every map/team-size/seed combination creates and resets a new environment. This
prevents paths, maps, radio queues, token buckets, cooldowns, metrics, and random
generator state from leaking between episodes.

### 4. Explicit legacy mode

`--communication-mode legacy` means no broker is constructed. It preserves the
original standalone behavior, including ground-truth navigation. It is a
compatibility control, not a fair limited-information comparison.

`--communication-mode none` constructs the broker but permits no messages. Its
planner navigates using private beliefs and is the appropriate no-communication
condition for the thesis comparison. The collector defaults to `none` so an
omitted flag cannot silently enable the omniscient legacy planner.

### 5. Physical sensing defines team coverage

For agent `i`, a valid cell is counted as physically observed when its private
`built_map[i]` is no longer `UNKNOWN`. Team coverage is the union of these
private known-cell masks. Delivered map patches are not counted as new physical
exploration at the receiver.

Ground truth is used only to identify the valid/explorable-cell denominator.
Cells marked unknown in the ground-truth PGM are excluded from both numerator
and denominator.

This definition makes communication affect coverage through behavior—better
goal selection and less duplicated travel—not by relabeling received cells as
new sensor observations.

### 6. Initial state is step zero

The state immediately after reset is recorded as step `0`. This retains the
coverage supplied by the starting four-beam observations and prevents the first
movement from receiving credit for cells sensed before any decision.

### 7. Coverage thresholds and termination target are separate

The collector records the first crossing of every value supplied through
`--coverage-thresholds`. The defaults are 90%, 98%, and 99%:

- 90% preserves Explore-Bench's topological-coverage measurement.
- 98% preserves the current Level-0 completion convention.
- 99% preserves the original paper's total-coverage convention.

`--target-coverage` controls episode termination. When omitted, it is the
largest recorded threshold. Recording the full step curve allows thresholds to
be recomputed later without rerunning an experiment.

### 8. Termination has an explicit precedence

After each decision, termination is checked in this order:

1. target coverage reached;
2. every agent's selected goal equals its current position;
3. no newly sensed team cell for `--no-progress-patience` decisions;
4. `--max-steps` reached.

The recorded reason is respectively `target_coverage`, `all_agents_stuck`,
`no_progress`, or `max_steps`.

Only **all** agents selecting their current cells triggers the stuck condition.
One idle agent must not stop teammates that still have reachable frontiers.
No-progress patience defaults to 50 decisions because an agent may traverse a
known corridor before reaching a new sensing location. Passing zero disables
the no-progress rule; the maximum-step bound always remains.

### 9. Distance is measured from executed grid positions

Per-agent path length is accumulated from Euclidean displacement between
successive executed grid positions, multiplied by map resolution. Team path
length is the sum of agent lengths; maximum agent path length is also retained
as a load indicator.

This is more robust than reading the existing `path_log` length and gives
meters for cardinal or diagonal motion. It measures the simplified Level-0
motion, not physical flight dynamics.

### 10. Collaboration measurements remain decomposable

For each step the collector records:

- each agent's independently sensed cell count and coverage ratio;
- standard deviation of those ratios;
- standard deviation of independently sensed area in square meters;
- overlap cells, defined as the sum of private known-cell counts minus their
  union;
- overlap cells divided by total explorable cells;
- overlap area in square meters.

Counts and ratios are both retained so later analysis can reproduce the paper's
presentation or use a different normalization without rerunning episodes.

### 11. Communication distinguishes senders from receiver-copies

The collector copies every cumulative broker counter and calculates per-step
deltas:

- attempted, transmitted, and delivered messages;
- collisions, losses, and expirations;
- in-range recipient opportunities;
- attempted and delivered bits;
- pose and map-patch sender messages;
- latency totals and mean delivered latency.

`delivered_messages` counts receiver-copies, while `transmitted_messages` counts
sender broadcasts. Delivery ratio therefore uses delivered receiver-copies
divided by in-range recipient opportunities.

The fixed protocol's requested per-agent action is recorded in `agents.csv` as
silence or map patch. The current fixed protocols do not request pose messages,
but pose counters are retained for schema compatibility with learned policies.

### 12. Communication cost is diagnostic for handwritten planners

The cost is derived with the broker's existing rule:

```text
cost_per_patch * attempted_bits / patch_bits
```

Cost is recorded cumulatively but is not fed into cost or MMPF goal selection.
It supports comparison with learned policies whose reward includes that cost.

### 13. Paired conditions use identical seeds

The seed controls standalone spawn selection and broker randomness. Reusing the
same map/team-size/seed list across methods creates paired evaluation episodes.
Repeating the exact same deterministic condition is a reproducibility check,
not an independent sample.

### 14. Sweep dimensions are explicit

One command takes one method, sensor/communication configuration, and output
recording policy. It may sweep maps, team sizes, and seeds. Separate commands
should be used for scientifically distinct treatments such as `none`,
`perfect`, and `shared_collision` so an output directory corresponds to one
named condition.

The Cartesian product is ordered map, team size, then seed. Episode identifiers
are sequential and do not encode assumptions about filenames.

### 15. Sensor lists follow existing validation rules

One sensor type/range is broadcast to the entire team. Alternatively, exactly
one entry per agent may be supplied. A per-agent list cannot be reused across
different team sizes unless its length is one, so the collector validates the
configuration for every requested team size before starting.

### 16. Three recording levels control data volume

- `episode` writes only `episodes.csv`.
- `steps` also writes team-level `steps.csv` and is the default.
- `full` additionally writes long-form per-agent `agents.csv`, including pose,
  selected goal, sensor beams, path length, and requested communication action.

Long-form agent rows keep one schema for teams of 2, 4, or 8 instead of adding
different columns for every team size.

### 17. Map arrays are optional artifacts

`--map-snapshots none` stores no arrays and is the default. `milestones` stores
the private maps, physical team union, positions, headings, and communication
beliefs at threshold crossings and episode end. `all` stores them at every
step. Arrays use compressed NPZ because CSV is inappropriate for grid data.

Map snapshots can dominate storage, so normal numerical sweeps should use
`none` or `milestones`.

### 18. Outputs are local, portable files

The collector does not require W&B or TensorBoard. Each invocation creates the
next non-destructive run directory:

```text
<output>/<experiment-name>/runN/
├── manifest.json
├── episodes.csv
├── steps.csv             # steps/full only
├── agents.csv            # full only
└── maps/                 # requested snapshots only
```

Existing run directories are never overwritten. Automatic `runN` allocation
matches the repository's local training convention. Experiment names are
restricted to one path component so they cannot accidentally escape the chosen
output directory.

### 19. The manifest is the reproducibility anchor

`manifest.json` contains:

- schema version and run status;
- complete parsed configuration and original command;
- absolute map paths and SHA-256 hashes;
- Git commit, branch, dirty state, and porcelain status;
- Python, platform, and NumPy versions;
- planned and completed episode counts;
- timestamps and any uncaught failure.

The manifest is first written with `running`, updated after every completed
episode, and finalized as `complete` or `failed`. CSV handles are flushed after
each episode.

### 20. Existing terminal noise is suppressed by default

The legacy planner prints frontier and threshold messages during each step.
Those messages are not structured evidence and become unreadable during a
sweep, so they are redirected by default. `--verbose` restores them. Collector
episode summaries are always printed once per completed episode.

## Output Schemas

### `episodes.csv`

One row represents one complete episode. It includes configuration identity,
start/final coverage, success, termination reason, total and maximum-agent path
length, JSON-encoded per-agent final values, collaboration measurements,
threshold crossing steps and path lengths, cumulative communication counts,
communication cost, and attempted/delivered bits per newly explored cell.

Threshold columns are generated from the configured thresholds. For example,
`0.98` produces `coverage_98_reached`, `coverage_98_step`, and
`coverage_98_team_path_m`.

### `steps.csv`

One row represents the post-decision team state, including step zero. It stores
coverage and new cells, path lengths, collaboration metrics, cumulative and
per-step radio counters, communication cost, and a termination reason on the
last row.

### `agents.csv`

One row represents one agent at one step. It stores grid pose, exact heading,
selected frontier goal, private coverage, cumulative path length, sensor model,
four named beam readings when available, and requested communication action.

## Commands

### No-communication cost pilot

```bash
source .venv/bin/activate

python grid_simulator/collect_experiments.py \
  --experiment-name cost_four_beam_none \
  --method cost \
  --maps onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm \
  --team-sizes 2 \
  --seeds 1 2 3 4 5 \
  --sensor-types four_beam \
  --sensor-ranges 3.5 \
  --communication-mode none \
  --max-steps 1000 \
  --coverage-thresholds 0.90 0.98 0.99 \
  --record steps \
  --output results
```

### Perfect-information upper bound

Use the same maps, team sizes, and seeds, changing only:

```bash
--experiment-name cost_four_beam_perfect \
--communication-mode perfect
```

### Constrained shared channel with fixed scheduling

```bash
python grid_simulator/collect_experiments.py \
  --experiment-name cost_four_beam_shared_round_robin \
  --method cost \
  --maps onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm \
  --team-sizes 2 \
  --seeds 1 2 3 4 5 \
  --sensor-types four_beam \
  --sensor-ranges 3.5 \
  --communication-mode shared_collision \
  --communication-protocol round_robin \
  --comm-range-cells 40 \
  --comm-latency-min-steps 1 \
  --comm-packet-loss 0 \
  --comm-cooldown-steps 3 \
  --comm-bucket-capacity-bits 314 \
  --comm-bucket-refill-bits 53 \
  --comm-ttl-steps 8 \
  --comm-cost-per-patch 0.01 \
  --max-steps 1000 \
  --record steps \
  --output results
```

### One visual debugging episode

```bash
python grid_simulator/collect_experiments.py \
  --experiment-name debug_cost \
  --method cost \
  --maps onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm \
  --team-sizes 2 \
  --seeds 1 \
  --sensor-types four_beam \
  --sensor-ranges 3.5 \
  --communication-mode none \
  --visualize \
  --verbose
```

## Recommended Experimental Progression

1. Run one two-agent, one-map, one-seed `none` episode visually.
2. Run the same episode twice headlessly and compare the CSV rows exactly.
3. Run a small paired pilot for `none`, `perfect`, and shared round-robin.
4. Inspect termination reasons and coverage curves before expanding the seed
   list.
5. Freeze maps, spawn seeds, thresholds, and radio parameters.
6. Run the same fixed evaluation suite for every handwritten treatment.
7. Add 4- and 8-agent conditions only after two-agent semantics are stable.
8. Make a learned-policy evaluator emit the same schemas, using at least five
   independent training seeds and the same held-out evaluation episodes.

## Known Limitations

- The adapter reads existing environment attributes because the traditional
  step functions do not return structured `info` or `done` values. Renaming
  those attributes could require collector updates.
- `all_agents_stuck` is reconstructed by observing the goals returned by the
  inherited frontier selector. It intentionally corrects the interactive
  loop's inability to terminate, but does not change controller decisions.
- Level-0 path length describes simplified grid movement, not Crazyflie flight
  distance or energy.
- Wall-clock runtime is excluded as a primary metric because it varies with
  hardware, UI, process count, and logging. It may be added later as a clearly
  labeled diagnostic.
- The collector does not provide checkpoint loading or learned-policy
  evaluation yet.
- Concurrent commands targeting the same experiment name could race while
  choosing `runN`; launch one collector process per experiment directory.

If the attribute-based adapter becomes difficult to maintain, the smallest
future simulator change should be to make traditional steps return
`(observation, metrics, done)` and guard plotting with `self.visualization`.
That change should be separately tested and must not alter planning semantics.
