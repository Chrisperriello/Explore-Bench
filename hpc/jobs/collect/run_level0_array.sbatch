#!/bin/bash -l
#SBATCH --job-name=level0-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Ada's Conda activation reads variables that may not exist, so unset-variable
# checking is enabled only after activation finishes.
set -eo pipefail

REPO_ROOT="${EXPLORE_BENCH_ROOT:?submit with submit_level0.sh}"
SLURM_ROOT="${EXPLORE_BENCH_SLURM_ROOT:?submit with submit_level0.sh}"
TASK_FILE="${LEVEL0_TASK_FILE:?submit with submit_level0.sh}"
COLLECTION_ID="${LEVEL0_COLLECTION_ID:?submit with submit_level0.sh}"
VARIANT="${LEVEL0_VARIANT:?submit with submit_level0.sh}"
SENSOR="${LEVEL0_SENSOR:?submit with submit_level0.sh}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?this file must run as a Slurm array}"

cd "${REPO_ROOT}"

module load python/anaconda3
unset PYTHONHOME
eval "$(conda shell.bash hook)"
conda activate "${REPO_ROOT}/.conda-env"
set -u

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting Level-0 task ${TASK_ID}"
echo "Collection: ${COLLECTION_ID}"
echo "Variant: ${VARIANT} (${SENSOR})"
echo "Slurm job: ${SLURM_JOB_ID}"

python hpc/tasks/run_level0_task.py \
  --task-file "${TASK_FILE}" \
  --task-id "${TASK_ID}" \
  --sensor "${SENSOR}" \
  --variant "${VARIANT}" \
  --output-root "${SLURM_ROOT}/data/level0/${COLLECTION_ID}" \
  --repository-root "${REPO_ROOT}"
