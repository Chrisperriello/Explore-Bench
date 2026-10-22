# MAPPO jobs on Ada

See the [full runbook](../../../docs/mappo/README.md) and
[decision record](../../../docs/mappo/decisions.md) before collecting results.
The code is an independently versioned experiment, not a rerun of Table II.

Run `source ./setup.sh --hpc` inside a compute allocation to create/repair
`.conda-env`, install both editable packages and run the regression suite.
See [fresh-machine setup](../../../docs/setup.md). Commit source changes first,
then start from the repository root. The default recipe needs
eight CPU workers and a CPU-only PyTorch training process. CUDA is optional;
install the CUDA build with `source ./setup.sh --hpc --cuda`, request a GPU from
Slurm explicitly and set `EXPLORE_MAPPO_DEVICE=cuda` only when the cluster
provides one. No GPU availability is assumed. Jobs use the absolute project
interpreter and clear `PYTHONHOME`/`PYTHONPATH`; they do not require a module or
shell activation to locate Python or either local package.

The submit wrapper makes a frozen task file and launches arrays over declared
training seeds (two concurrent seeds by default). It creates logs before
submission. Each stage has separate outputs and must finish successfully
before the next stage is submitted:

```bash
bash hpc/jobs/mappo/submit.sh train mappo_omni_3p5_01 experiments/mappo_omni_3p5_v1.json
# Inspect squeue/sacct and every train/seed_*/manifest.json before continuing.
bash hpc/jobs/mappo/submit.sh validate mappo_omni_3p5_01 experiments/mappo_omni_3p5_v1.json
bash hpc/jobs/mappo/submit.sh select mappo_omni_3p5_01 experiments/mappo_omni_3p5_v1.json
bash hpc/jobs/mappo/submit.sh evaluate mappo_omni_3p5_01 experiments/mappo_omni_3p5_v1.json
bash hpc/jobs/mappo/submit.sh score mappo_omni_3p5_01 experiments/mappo_omni_3p5_v1.json
```

Extra arguments after the JSON path are passed to `sbatch`, for example
`--time=24:00:00`, `--partition=ada`, or `--cpus-per-task=8`. Defaults are
12 hours and 16 GB. They are initial resource requests, not measured runtime
or memory guarantees. Start with the integration/pilot commands in the
runbook; use observed throughput to choose time limits before the full sweep.

```text
../slurm/data/mappo/<ID>/tasks.json
../slurm/data/mappo/<ID>/train/seed_<N>/
../slurm/data/mappo/<ID>/validation/seed_<N>/
../slurm/data/mappo/<ID>/selections/seed_<N>.json
../slurm/data/mappo/<ID>/evaluation/seed_<N>/
../slurm/data/mappo/<ID>/scores/
../slurm/logs/mappo/<ID>/<stage>/<array>_<task>.out/.err
```

Slurm sends USR1 three minutes before timeout. Python saves at the next safe
update boundary; a very slow rollout might exceed that grace period. Existing
completed snapshots are immutable. Use the runbook's manual resume command
with a fresh attempt directory, then pass that attempt to `validate` manually.
The standard array paths cover initial attempts; automatic retry/resubmission
is deliberately not enabled. Never interpret a stale `running` manifest as
proof that a scheduler job is still alive.

No stage deletes inputs. Preserve raw evaluation data, selection artifacts and
the chosen actor checkpoints; archive and verify before cleaning unused
snapshots. A resume snapshot needs more storage than an actor-only export.
