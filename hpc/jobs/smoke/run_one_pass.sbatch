#!/bin/bash -l
#SBATCH --job-name=explore-one-pass
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Do not enable "unset variable" checking until after Conda activates. Ada's
# Conda activation scripts read ADDR2LINE before guaranteeing that it exists.
set -eo pipefail

REPO_ROOT="${EXPLORE_BENCH_ROOT:?submit with submit_one_pass.sh}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_ROOT="${EXPLORE_BENCH_SLURM_ROOT:?submit with submit_one_pass.sh}"
OUTPUT_ROOT="${SLURM_ROOT}/data/one_pass/job_${JOB_ID}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

module load python/anaconda3
unset PYTHONHOME
eval "$(conda shell.bash hook)"
conda activate "${REPO_ROOT}/.conda-env"
set -u

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

echo "Starting one headless Explore-Bench pass"
echo "Job: ${JOB_ID}"
echo "Output: ${OUTPUT_ROOT}"
python --version

python grid_simulator/collect_experiments.py \
  --experiment-name original_cost_corner \
  --method cost \
  --maps onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm \
  --team-sizes 2 \
  --seeds 1 \
  --resolution 0.1 \
  --sensor-types omnidirectional \
  --sensor-ranges 3.5 \
  --communication-mode legacy \
  --max-steps 1000 \
  --no-progress-patience 0 \
  --coverage-thresholds 0.90 0.98 0.99 \
  --target-coverage 0.99 \
  --record full \
  --map-snapshots none \
  --output "${OUTPUT_ROOT}"

RUN_DIRECTORY="${OUTPUT_ROOT}/original_cost_corner/run1"
grep -q '"status": "complete"' "${RUN_DIRECTORY}/manifest.json"

echo "One pass completed successfully"
wc -l \
  "${RUN_DIRECTORY}/episodes.csv" \
  "${RUN_DIRECTORY}/steps.csv" \
  "${RUN_DIRECTORY}/agents.csv"
