# Paper-consistency diagnostic

This folder runs one small check to explain why the current Level-0 results do
not exactly match the Explore-Bench paper. It is not the full validation run.

The check holds the map, planner, robot count, seed, and 360-degree sensor type
fixed. It changes only sensor distance or map-sharing behavior:

| Treatment | Sensor distance | Map-sharing behavior | Question answered |
| --- | ---: | --- | --- |
| `legacy_3p5` | 3.5 m | legacy | Current reference |
| `legacy_7p0` | 7.0 m | legacy | Does the larger sensor explain the difference? |
| `perfect_3p5` | 3.5 m | perfect | Does the navigation/map-sharing rule explain it? |

`legacy` is the compatibility behavior already used by the main Level-0 run.
It gives planning more map knowledge than the robots have sensed. `perfect`
shares all sensed maps immediately, but it does not plan paths through unknown
space. The name `perfect` therefore does not mean that it will always be faster.

## What is submitted

- Five-room map: `room1_modified.pgm`
- Planners: Cost and MMPF
- Seeds: 1 through 5
- Robots: 2
- Sensor type: original omnidirectional sensor
- Coverage checkpoints: 90%, 98%, and 99%
- Stop: 99% coverage or 1,000 steps
- Three treatments x two planners x five seeds = 30 runs

The same task number represents the same planner and seed in all three
treatments. For example, task 0 is Cost with seed 1 in every array. This makes
the comparisons paired: each changed run is compared with the same random
starting condition.

## Run on Ada

From the Explore-Bench repository root:

```bash
git pull
bash hpc/jobs/paper_consistency/submit.sh 4 paper_consistency_01
```

The first value is the number of tasks allowed to run at once **in each of the
three arrays**. A value of 4 can therefore use at most 12 CPU cores total. Each
run itself uses one CPU core. Use a new collection name if you submit again.

The jobs use the repository environment at `.conda-env`. The batch file loads
Ada's `python/anaconda3` module, removes the bad `PYTHONHOME` value, and then
activates `.conda-env`.

## Monitor it

```bash
squeue -u "$USER"
python3 hpc/monitor/slurm_monitor.py --once
```

Slurm writes outside the Git repository:

```text
../slurm/
├── tasks/paper_consistency/paper_consistency_01/
├── data/paper_consistency/paper_consistency_01/
│   ├── legacy_3p5/
│   ├── legacy_7p0/
│   └── perfect_3p5/
└── logs/paper_consistency/paper_consistency_01/
```

Each treatment contains ten `task_XXXXXX` folders. A completed task has a
`task_status.json`, a `manifest.json`, and an `episodes.csv`. If a task is
submitted again without deleting its completed data, the runner detects the
complete files and skips the duplicate run.

## Reading the result

Compare each treatment with `legacy_3p5` separately for Cost and MMPF:

- If `legacy_7p0` moves toward the paper values, sensor distance is a likely
  cause.
- If `perfect_3p5` moves toward the paper values, map-sharing/navigation rules
  are a likely cause.
- If neither does, the next things to check are starting positions, the exact
  map version, and the paper's time-unit definition.

Do not call the collector's step count seconds. The paper's exact timing unit
and its repeated-run/seed procedure are not stated clearly enough for that.
