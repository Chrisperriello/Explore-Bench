#!/bin/bash -l
#SBATCH --job-name=level0-mappo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180

set -euo pipefail
cd "${EXPLORE_BENCH_ROOT:?}"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
PYTHON="${EXPLORE_BENCH_ROOT}/.conda-env/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo 'Missing project Python. Run bash setup.sh --hpc first.' >&2
    exit 1
fi
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/explore-mappo-${SLURM_JOB_ID}"
STAGE="${EXPLORE_MAPPO_STAGE:?}"
EXPERIMENT="${EXPLORE_MAPPO_DIRECTORY:?}"
CONFIG="${EXPLORE_MAPPO_CONFIG:?}"
SEED=$("$PYTHON" -c 'import sys; from explore_bench.config import configuration; print(configuration(sys.argv[1])["training_seeds"][int(sys.argv[2])])' "$CONFIG" "${SLURM_ARRAY_TASK_ID:-0}")
TASKS="${EXPERIMENT}/tasks.json"

# exec lets Slurm's pre-timeout USR1 reach Python directly. Python checkpoints
# after the current complete rollout; it never labels a partial run complete.
case "$STAGE" in
  train)
    exec "$PYTHON" -m explore_bench train --config "$CONFIG" --seed "$SEED" \
      --device "${EXPLORE_MAPPO_DEVICE:-cpu}" --output "${EXPERIMENT}/train/seed_${SEED}"
    ;;
  validate)
    exec "$PYTHON" -m explore_bench validate --training-run "${EXPERIMENT}/train/seed_${SEED}" \
      --tasks "$TASKS" --output "${EXPERIMENT}/validation/seed_${SEED}"
    ;;
  select)
    exec "$PYTHON" -m explore_bench select --validation "${EXPERIMENT}/validation/seed_${SEED}" \
      --output "${EXPERIMENT}/selections/seed_${SEED}.json"
    ;;
  evaluate)
    exec "$PYTHON" -m explore_bench evaluate --selection "${EXPERIMENT}/selections/seed_${SEED}.json" \
      --tasks "$TASKS" --output "${EXPERIMENT}/evaluation/seed_${SEED}"
    ;;
  score)
    exec "$PYTHON" -m explore_bench score --inputs "${EXPERIMENT}"/evaluation/seed_* \
      --output "${EXPERIMENT}/scores"
    ;;
  *) exit 2 ;;
esac
