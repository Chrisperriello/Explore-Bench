# Level-0 Pilot Analysis and Visualization

Everything in this directory is research-facing experiment orchestration,
analysis, or visualization. It is deliberately kept outside the simulator and
learning implementation directories. These scripts read collector artifacts
under `results/`; plotting never mutates experiment data.

## Contents

- `run_scoped_pilot.py`: prints a treatment command by default and runs it only
  with explicit `--execute`.
- `plot_scoped_pilot.py`: validates the paired treatments and builds the main
  summary figure and CSV.
- `plot_scoped_pilot_seeds.py`: creates one four-panel diagnostic per seed.
- `plot_seed_sandboxes.py`: shows each seed's map, starts, paths, and endpoints.
- `plot_four_beam_snapshot.py`: overlays one logged four-beam observation on the
  source map.
- `plot_initial_pipeline_proof.py`: retains the original two-condition proof
  figure for provenance; prefer `plot_scoped_pilot.py` for current results.

All commands below assume the repository root is the working directory.

## Protocol

This protocol turns the initial pipeline proof into a small, interpretable
experiment. It is a calibration pilot, not the thesis-scale 2/4/8-agent study.
It deliberately leaves the simulator and collector unchanged.

## Question

At a fixed budget of 200 decisions, does delivered map information change the
behavior of the two-agent cost baseline when both agents use four thin beams?

The pilot separates three information conditions:

| Condition | Mode | Range | Purpose |
| --- | --- | ---: | --- |
| Private | `none` | ignored | no-communication lower control |
| Finite radio | `shared_collision` + `round_robin` | 150 cells | constrained fixed-scheduler treatment |
| Perfect | `perfect` | unlimited by definition | information upper bound |

The completed 40-cell run remains an out-of-contact negative control. It is
not part of the primary three-condition comparison because all three seeds had
zero in-range recipient opportunities and zero delivered bits.

## Why 150 Cells

The initial no-communication trajectories measured robot separation rather
than guessing a useful range:

| Seed | Minimum | Median | Maximum | Steps within 150 cells |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 96.8 | 127.3 | 151.2 | 199 / 201 |
| 2 | 146.9 | 175.4 | 202.4 | 29 / 201 |
| 3 | 94.0 | 119.2 | 142.4 | 201 / 201 |

Thus 150 cells creates frequent contact in two seeds and intermittent contact
in one without making the channel mathematically unlimited. At 0.1 m/cell it
is a 15 m simulation range. This value is a pilot calibration, not a claimed
measurement of Crazyflie radio range. A thesis treatment must either justify a
physical value or report range as a controlled simulation parameter.

## Fixed Choices

- `corner.pgm`: preserves the completed pipeline proof and stresses incomplete
  corner cleanup.
- cost planner: isolates sensing and communication before learned navigation.
- two agents: smallest multi-robot condition and cheapest valid communication
  demonstration.
- seeds 1, 2, and 3: paired spawns across every condition.
- four-beam range 3.5 m: the intended Multi-Ranger abstraction.
- 200 decisions: fixed-budget pilot; do not call it time to completion.
- full recording: retains step curves, per-agent coverage, goals, positions,
  and named beam distances.
- no map snapshots: numerical claims do not require large array artifacts.
- round robin: prevents two-agent channel collisions and isolates the value of
  delivered content from learned or uncoordinated scheduling.
- one-step latency, no random packet loss, 3-step cooldown, and existing token
  defaults: limits confounds during the first effective-delivery comparison.
- physical team coverage: received patches do not count as sensing; radio can
  improve coverage only by changing subsequent robot motion.

The maximum range, map, seeds, step budget, and sensor configuration are held
constant in all newly generated manifests. Range is ignored in `none`, while
`perfect` is intentionally an unlimited-information upper bound.

## Safety: Commands Print by Default

`run_scoped_pilot.py` does not start a collector unless `--execute` is passed.
Preview each command first:

```bash
source .venv/bin/activate
python analysis/level0_pilot/run_scoped_pilot.py limited150
python analysis/level0_pilot/run_scoped_pilot.py perfect
```

The existing no-communication run is scientifically reusable because range is
inactive in `none` mode:

```text
results/pipeline_proof_cost_four_beam/run1
```

To produce fresh, uniformly named artifacts instead, preview and then execute
the `none` condition as well.

## Runs to Start Manually

The operator starts the CPU-heavy jobs explicitly:

```bash
python analysis/level0_pilot/run_scoped_pilot.py limited150 --execute
```

In another terminal, if desired:

```bash
python analysis/level0_pilot/run_scoped_pilot.py perfect --execute
```

Expected output directories are:

```text
results/scoped_pilot_cost_four_beam_shared_r150/run1
results/scoped_pilot_cost_four_beam_perfect/run1
```

Run-number allocation is non-destructive. If a directory already exists, use
the newly created `runN` in the analysis command.

## Acceptance Checks

Before interpreting exploration performance, require:

1. every manifest reports `status: complete` and 3/3 episodes;
2. maps, seeds, team size, method, sensors, resolution, and step cap match;
3. the finite-radio condition has nonzero in-range recipients and delivered
   bits; otherwise it remains an out-of-contact control;
4. attempted bits are greater than or equal to delivered bits;
5. no-communication episodes have zero attempted and delivered bits;
6. coverage is compared at the same 200-decision horizon;
7. all three paired seeds are shown, because three seeds are insufficient for
   a strong distributional claim.

## Plotting

After both new runs finish:

```bash
python analysis/level0_pilot/plot_scoped_pilot.py \
  --none results/pipeline_proof_cost_four_beam/run1 \
  --limited results/scoped_pilot_cost_four_beam_shared_r150/run1 \
  --perfect results/scoped_pilot_cost_four_beam_perfect/run1 \
  --output results/scoped_pilot_analysis
```

The plotter validates the matched configuration and seed set before writing:

```text
results/scoped_pilot_analysis/scoped_pilot_summary.png
results/scoped_pilot_analysis/scoped_pilot_summary.csv
```

The figure reports coverage curves, paired final coverage, the original
Explore-Bench overlap metric, and attempted versus delivered finite-radio
traffic. With only three seeds, it shows every seed rather than drawing a
misleading confidence interval.

To create a separate diagnostic figure for every seed:

```bash
python analysis/level0_pilot/plot_scoped_pilot_seeds.py \
  --none results/pipeline_proof_cost_four_beam/run1 \
  --limited results/scoped_pilot_cost_four_beam_shared_r150/run1 \
  --perfect results/scoped_pilot_cost_four_beam_perfect/run1
```

This writes `seed_001.png`, `seed_002.png`, and `seed_003.png` under
`results/scoped_pilot_analysis/seeds/`. Each image contains absolute coverage,
paired coverage differences, cumulative finite-radio traffic, and spatial
robot trajectories. Pass `--seeds 2` to render only seed 2 or `--show` to open
the figures interactively after saving them.

To create full-size sandbox/path figures for all recorded seeds:

```bash
python analysis/level0_pilot/plot_seed_sandboxes.py \
  --run results/pipeline_proof_cost_four_beam/run1
```

To show one logged four-beam observation on the source map:

```bash
python analysis/level0_pilot/plot_four_beam_snapshot.py \
  --run results/pipeline_proof_cost_four_beam/run1 \
  --seed 1 \
  --step 0
```

## Interpretation Boundaries

- A difference between `none` and `perfect` shows that shared information can
  matter to this planner and sensor configuration.
- A difference between `none` and finite radio estimates the effect of the
  fixed constrained channel under this pilot range and scheduler.
- A gap between finite radio and `perfect` shows remaining information or
  channel limitations; it does not by itself identify which limit caused it.
- No difference with nonzero deliveries is still a result: received patches
  may be redundant or may not change nearest-frontier choices.
- These runs validate the experimental mechanism. They do not support final
  claims across maps, team sizes, methods, or learned communication policies.
