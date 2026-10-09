#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SLURM_ROOT="${REPO_ROOT}/../slurm"
CONFIG_FILE="${SCRIPT_DIR}/paper_consistency_suite.json"
PYTHON_BIN="${REPO_ROOT}/.conda-env/bin/python"
MAX_PARALLEL="${1:-4}"
COLLECTION_ID="${2:-paper_consistency_$(date -u +%Y%m%dT%H%M%SZ)}"
TREATMENTS=(legacy_3p5 legacy_7p0 perfect_3p5)

if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi
if [[ ! "${COLLECTION_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "COLLECTION_ID may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}; create the repository Conda environment first" >&2
  exit 2
fi

TASK_ROOT="${SLURM_ROOT}/tasks/paper_consistency/${COLLECTION_ID}"
DATA_ROOT="${SLURM_ROOT}/data/paper_consistency/${COLLECTION_ID}"
LOG_ROOT="${SLURM_ROOT}/logs/paper_consistency/${COLLECTION_ID}"

if [[ -e "${TASK_ROOT}" ]]; then
  echo "Collection ID already exists: ${COLLECTION_ID}" >&2
  echo "Use a new ID so submitted task tables are never changed" >&2
  exit 2
fi

mkdir -p "${TASK_ROOT}" "${DATA_ROOT}" "${LOG_ROOT}"
for treatment in "${TREATMENTS[@]}"; do
  mkdir -p "${DATA_ROOT}/${treatment}" "${LOG_ROOT}/${treatment}"
done

cp "${CONFIG_FILE}" "${TASK_ROOT}/paper_consistency_suite.json"
TASK_COUNT="$(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_tasks.py" \
    --config "${TASK_ROOT}/paper_consistency_suite.json" \
    --output-directory "${TASK_ROOT}" \
    --repository-root "${REPO_ROOT}"
)"

if [[ "${TASK_COUNT}" != "30" ]]; then
  echo "Expected 30 tasks but generated ${TASK_COUNT}" >&2
  exit 2
fi

echo "Collection: ${COLLECTION_ID}"
echo "Tasks: 30 total, 10 per treatment"
echo "Maximum simultaneous tasks per treatment: ${MAX_PARALLEL}"

for treatment in "${TREATMENTS[@]}"; do
  task_file="${TASK_ROOT}/${treatment}.csv"
  job_id="$(
    sbatch --parsable \
      --job-name="paper-${treatment}" \
      --array="0-9%${MAX_PARALLEL}" \
      --chdir="${REPO_ROOT}" \
      --export="ALL,EXPLORE_BENCH_ROOT=${REPO_ROOT},EXPLORE_BENCH_SLURM_ROOT=${SLURM_ROOT},PAPER_CHECK_TASK_FILE=${task_file},PAPER_CHECK_COLLECTION_ID=${COLLECTION_ID},PAPER_CHECK_TREATMENT=${treatment}" \
      --output="${LOG_ROOT}/${treatment}/%x-%A_%a.out" \
      --error="${LOG_ROOT}/${treatment}/%x-%A_%a.err" \
      "${SCRIPT_DIR}/run_array.sbatch"
  )"
  echo "${treatment} array job: ${job_id}"
done

echo "Task tables: ${TASK_ROOT}"
echo "Data: ${DATA_ROOT}"
echo "Logs: ${LOG_ROOT}"
