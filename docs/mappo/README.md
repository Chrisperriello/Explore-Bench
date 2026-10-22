# Reproducible Level-0 MAPPO workflow

This is the research workflow approved on 2026-10-21. It preserves the prior
paper-audit collection and adds a versioned experiment with explicit tasks,
corrected rewards, an equal movement clock, immutable checkpoints, independent
evaluation, and shared CSV recording. It is also the infrastructure boundary
for a future DQN controller; DQN itself is not implemented here.

- [Implementation review](review.md): what the inherited MAPPO does and the
  concrete correctness problems found in it.
- [Decision record](decisions.md): choices, alternatives, primary sources,
  implementation links, tests, and the limits of defensible claims.
- [Change ledger](changes.md): every changed code file and why it changed.

## Environment and initial recipe

Use the repository's Python 3.8 environment, including its MAPPO dependencies:

```bash
./.conda-env/bin/python --version
./.conda-env/bin/python -c 'import torch, numpy; print(torch.__version__, numpy.__version__)'
# If the existing environment has only the classical-collector dependencies:
./.conda-env/bin/python -m pip install -r requirements-level0-mappo.txt
```

The local development interpreter is `.venv/bin/python`. All commands below
run from the repository root. Do not use Ada's system `python3`.

Before using the paper-map pilot, run the controlled generated-corridor check:

```bash
./.conda-env/bin/python -m explore_bench sanity \
  --output ../slurm/data/mappo/controlled_corridor_learning_01
```

It performs 8,192 transitions with one independent training seed, validates
periodic snapshots, selects using validation only, and tests the selected actor
on a reserved generated layout. Its predeclared engineering criterion is a
validation coverage-AUC gain of at least 0.01 over the untrained actor and a
nonzero selected training step. Failure returns a nonzero exit code. This is
not a performance claim on the Explore-Bench maps. Use `--allow-dirty` only for
local development checks.

`experiments/mappo_omni_3p5_v1.json` selects the versioned defaults in
`explore_bench/config.py`. Every run stores the fully resolved configuration;
the one-line input file is not the entire reproducibility record.

| Parameter | Initial recipe |
| --- | --- |
| Robots; resolution | 2; 0.1 m |
| Sensor; communication | Omnidirectional, 3.5 m; `perfect` |
| Target; cap | 99%; 1,000 joint decisions |
| Model | Shared feedforward actor, centralized critic, default CNN, hidden 64 |
| Training seeds | 1, 2, 3, 4, 5 |
| Workers; rollout | 8 processes; 128 decisions per worker |
| Budget | 1,024,000 environment transitions per training seed |
| PPO epochs; minibatches | 5; 1 |
| Actor/critic learning rates; clipping | 0.0005; 0.1 |
| Discount; GAE lambda | 0.99; 0.95 |
| Entropy coefficient; gradient clip | 0.01; 10 |
| Value normalization | Enabled and saved in training checkpoints |
| Checkpoints | Step 0, every 102,400 transitions, and final/interrupted update |

One joint decision advances each robot by at most one A* edge and advances the
radio broker once. Diagonal distance is `sqrt(2) * resolution`. This is a
logical simulation clock; it is not calibrated robot flight time.

Training reward per robot is `0.02 * newly sensed union cells`, plus a `0.1`
bonus only on first reaching 99%, minus that robot's attempted-message cost.
The task reward is shared; summing it over robots would double count the team
reward. Time penalties and intrinsic rewards are disabled.

Sensing uses the actual integer grid pose. The inherited conversion through
floating-point world coordinates can shift the sensor origin, and is retained
only by the historical environment.

## Frozen map and spawn design

| Role | Existing map files, omitting `.pgm` |
| --- | --- |
| Train | room_1, room_2, room_3, room1_modified, corridor, corridor_sym, corridor_asym, corner |
| Validate | room_with_corner, room_with_corridor |
| Unseen-layout test | square_loop, loop_with_corridor_sym |

Validation: seeds 10001–10010 on each validation map (20 episodes).
Familiar-map test: seeds 20001–20010 on each training map (80 episodes).
Unseen-layout test: seeds 30001–30010 on each reserved map (20 episodes).
Training episode seeds are sampled from `[100000, 2**32)` independently per
worker. Related room/corridor map variants stay within the training partition.

Task preparation records actual distinct free spawn cells, cardinal headings,
map byte hashes, task/configuration hashes and the environment version. A task
table is immutable. The readers verify the frozen design before evaluation.

The test supports claims about these specific maps and starts. Only two loop
maps are held out; it does not establish generalization to arbitrary buildings.

## Local/Ada commands

Use a unique collection ID; output directories are immutable. This example
retains the existing `../slurm` convention:

```bash
export EXPERIMENT="../slurm/data/mappo/mappo_omni_3p5_01"
export CONFIG="experiments/mappo_omni_3p5_v1.json"

./.conda-env/bin/python -m explore_bench prepare \
  --config "$CONFIG" --output "$EXPERIMENT/tasks.json"

# Short integration run; it deliberately records status=interrupted.
# --allow-dirty is for development only; commit before research collection.
./.conda-env/bin/python -m explore_bench train \
  --config "$CONFIG" --seed 1 --output "$EXPERIMENT/integration_seed_1" \
  --stop-after-updates 4

# A 100-update pilot is 102,400 environment transitions. Inspect throughput
# and validation before deciding to spend the full five-seed budget.
./.conda-env/bin/python -m explore_bench train \
  --config "$CONFIG" --seed 1 --output "$EXPERIMENT/pilot_seed_1" \
  --stop-after-updates 100

# Full run (repeat for seeds 1–5, or use the Slurm array below).
./.conda-env/bin/python -m explore_bench train \
  --config "$CONFIG" --seed 1 --output "$EXPERIMENT/train/seed_1"

./.conda-env/bin/python -m explore_bench validate \
  --training-run "$EXPERIMENT/train/seed_1" \
  --tasks "$EXPERIMENT/tasks.json" --output "$EXPERIMENT/validation/seed_1"

./.conda-env/bin/python -m explore_bench select \
  --validation "$EXPERIMENT/validation/seed_1" \
  --output "$EXPERIMENT/selections/seed_1.json"

./.conda-env/bin/python -m explore_bench evaluate \
  --selection "$EXPERIMENT/selections/seed_1.json" \
  --tasks "$EXPERIMENT/tasks.json" --output "$EXPERIMENT/evaluation/seed_1"

./.conda-env/bin/python -m explore_bench score \
  --inputs "$EXPERIMENT"/evaluation/seed_* --output "$EXPERIMENT/scores"
```

Validation loads each saved actor into an independent evaluator. It runs after
training as a separate stage, evaluating the periodic snapshots on identical
tasks. It does not interrupt the training sessions or change their random
streams. Pilot validation is allowed, but final selection requires a completed
training attempt. Selection ranks checkpoints by validation success rate,
coverage AUC, capped T99, then earliest step. Final test CLI calls require the
selection artifact rather than an arbitrary checkpoint argument.

Scripted controls use the same tasks, session clock and recorder:

```bash
./.conda-env/bin/python -m explore_bench evaluate \
  --controller random --config "$CONFIG" --tasks "$EXPERIMENT/tasks.json" \
  --output "$EXPERIMENT/random"
# Also supported: wait, cost, mmpf. Classical controls request silence.
```

Cost/MMPF reuse existing goal selection against delivered beliefs. The new
session executes those goals. These are new matched-clock controls, not the
old paper-control traces. Controller/model differences and sensor/channel
differences require separate experiments.

## Checkpoints and resuming

Every snapshot contains:

```text
checkpoints/step_000000102400/
  actor.pt       # inference state_dict; no trainer required
  training.pt    # model/optimizer/normalizer/RNG/session state
  manifest.json # configuration, semantic versions, hashes, seed, step
```

Only load `training.pt` from trusted local runs: it contains Python session
objects. Inference uses the smaller `actor.pt` state dictionary.

SIGUSR1, SIGTERM and SIGINT request a checkpoint after the current full PPO
rollout. A hard kill can lose work since the last completed checkpoint. A
truncated rollout is never labeled complete.

Signal-interrupted training exits with code 75 after saving; an explicitly
requested `--stop-after-updates` pilot exits successfully with an `interrupted`
manifest because its full research budget remains unfinished.

Resume to a **new** attempt directory using the same configuration and code:

```bash
./.conda-env/bin/python -m explore_bench train \
  --config "$CONFIG" --seed 1 \
  --resume "$EXPERIMENT/pilot_seed_1/checkpoints/step_000000102400" \
  --output "$EXPERIMENT/train/seed_1"
```

Validation follows checkpoint lineage back through interrupted attempts, so
earlier snapshots remain eligible. Exact continuation is tested on CPU on the
same software/hardware. Cross-device bitwise equivalence is not promised.

## Output and scoring contract

Training writes `episodes.csv`, `updates.csv`, resolved arguments, manifest and
checkpoints. Training episodes unfinished at an update remain in session state
and are recorded only when they end; partial episodes are retained in snapshots.

Validation/testing additionally write `steps.csv` and `agents.csv`, including
step zero. Schema version 2 includes training seed, actor hash, explicit task
identity and `level0_fixed_edge_v1` environment semantics. Maps/sensor/broker
parameters and source hashes are in the manifest.

Milestones record first crossings of 90%, 98%, 99%; missing crossings are blank
with `reached=false`. Distances sum actual executed edges. Coverage is the
physical union of sensor maps, masked by valid terrain, rather than received
belief cells. Geometric overlap subtracts the sensed union; it is distinct
from the original paper's total-terrain subtraction convention.

Coverage AUC uses trapezoidal integration over joint decisions, divided by the
episode cap. After early success, final coverage is carried forward to the cap.
Capped T99 assigns failures `cap + 1`; conditional successful-run T99 is also
reported with its sample count.

Scoring validates complete manifests, tasks, step/agent counts, movement and
milestone extraction. It writes per-training-seed statistics, aggregate scores,
coverage curves, an input inventory, and a Markdown report. Bootstrap intervals
resample independent training-seed summaries (10,000 replicates; fixed analysis
seed 20261021), not individual episodes. Scripted controllers do not receive a
fabricated training-seed confidence interval. One seed can test the workflow,
but cannot establish between-training-run uncertainty.

Old schema-v1 artifacts remain available to their existing analyzers. Do not
mix schema-v1 historical runs with schema-v2 scores or pool different sensor/
communication contracts into a single comparison.

## Verification and research readiness

Run:

```bash
MPLCONFIGDIR=/tmp/explore-mappo-tests .venv/bin/python -m pytest -q
bash -n hpc/jobs/mappo/submit.sh hpc/jobs/mappo/run_stage.sbatch
```

The new tests exercise real PPO updates, interruption/resumption, checkpoint
lineage, validation selection, actor-only fresh-process loading, fixed-task
evaluation, process-worker state and independent toy-map metric checks.
The test names in the decision record are the acceptance evidence. See
[verification results](verification.md) for recorded execution evidence.

A small engineering smoke proves the workflow executes. It is not a
convergence result. A controlled exploration learning sanity check and the
102,400-transition pilot must be reviewed before a five-seed full sweep.
No full HPC sweep is automatically launched by installing this implementation.
