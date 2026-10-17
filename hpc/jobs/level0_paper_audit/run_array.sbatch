#!/bin/bash -l
#SBATCH --job-name=level0-paper-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -eo pipefail

REPO_ROOT="${EXPLORE_BENCH_ROOT:?submit with submit.sh}"
SLURM_ROOT="${EXPLORE_BENCH_SLURM_ROOT:?submit with submit.sh}"
TASK_FILE="${LEVEL0_AUDIT_TASK_FILE:?submit with submit.sh}"
COLLECTION_ID="${LEVEL0_AUDIT_COLLECTION_ID:?submit with submit.sh}"
TREATMENT="${LEVEL0_AUDIT_TREATMENT:?submit with submit.sh}"
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

python hpc/jobs/level0_paper_audit/run_task.py \
  --task-file "${TASK_FILE}" \
  --task-id "${TASK_ID}" \
  --treatment "${TREATMENT}" \
  --output-root "${SLURM_ROOT}/data/level0_paper_audit/${COLLECTION_ID}" \
  --repository-root "${REPO_ROOT}"
