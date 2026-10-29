#!/bin/bash -l
#SBATCH --job-name=mappo-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00

# One array task = one scripted controller on one map's ten frozen tasks.
#
# Why this exists: the scripted controls (random, wait, cost, mmpf) are the
# comparison every MAPPO result is judged against, and MMPF is slow (about
# 4-27 decisions/minute by map). Run as one 100-episode process it needs about
# six days; split one job per map, the slowest shard needs about a day. This
# script is the per-shard worker; submit_baselines.sh writes the shard list
# and submits the array. Shards are pooled again by `explore_bench score`.
#
# Resources: the evaluator is a single Python process, so one CPU is enough.
# 8G and 4 days are generous ceilings measured against the 2026-10-26 MMPF
# shards (slowest map, corner, about 31 h), not tight estimates.
set -euo pipefail

# Slurm runs a copy of this file from its spool directory, so the repository
# cannot be found from $0; submit_baselines.sh passes it explicitly.
cd "${EXPLORE_BENCH_ROOT:?}"

# Same isolation as run_stage.sbatch: ignore module-provided Python paths and
# user site-packages so only the project environment is imported.
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1

# Limit numeric libraries to one thread. Without this, NumPy/OpenBLAS/OpenCV
# start one thread per node core; on 2026-10-26 hand-submitted --wrap jobs
# that lacked these settings ran about 5x slower and slowed neighbouring jobs.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/explore-mappo-${SLURM_JOB_ID}"
PYTHON="${EXPLORE_BENCH_ROOT}/.conda-env/bin/python"

# Array task N reads line N+1 ("controller suite map") of the shard list.
# Reading a frozen file, rather than computing the shard here, means every
# task of the array sees the same list even if the repository changes later.
read -r CONTROLLER SUITE MAP < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${EXPLORE_MAPPO_SHARDS:?}")

# exec hands the process to Python so Slurm signals reach it directly.
# --maps restricts the frozen suite to this map; the output directory name
# <suite>__<map> keeps shards distinct and readable. evaluate refuses to
# overwrite an existing directory, so a rerun needs the old one moved aside.
exec "$PYTHON" -m explore_bench evaluate --controller "$CONTROLLER" --config "${EXPLORE_MAPPO_CONFIG:?}" \
  --tasks "${EXPLORE_MAPPO_DIRECTORY:?}/tasks.json" --suite "$SUITE" --maps "$MAP" \
  --output "${EXPLORE_MAPPO_DIRECTORY}/baselines/${CONTROLLER}/${SUITE}__${MAP}"
