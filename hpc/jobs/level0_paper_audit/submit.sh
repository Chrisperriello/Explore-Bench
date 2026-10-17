#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SLURM_ROOT="${REPO_ROOT}/../slurm"
CONFIG_FILE="${SCRIPT_DIR}/level0_paper_audit_suite.json"
PYTHON_BIN="${REPO_ROOT}/.conda-env/bin/python"
MAX_PARALLEL="${1:-4}"
COLLECTION_ID="${2:-level0_paper_audit_$(date -u +%Y%m%dT%H%M%SZ)}"
TREATMENTS=(omnidirectional_3p5 four_beam_3p5 omnidirectional_7p0 four_beam_7p0)

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

TASK_ROOT="${SLURM_ROOT}/tasks/level0_paper_audit/${COLLECTION_ID}"
DATA_ROOT="${SLURM_ROOT}/data/level0_paper_audit/${COLLECTION_ID}"
LOG_ROOT="${SLURM_ROOT}/logs/level0_paper_audit/${COLLECTION_ID}"

if [[ -e "${TASK_ROOT}" ]]; then
  echo "Collection ID already exists: ${COLLECTION_ID}" >&2
  exit 2
fi

mkdir -p "${TASK_ROOT}" "${DATA_ROOT}" "${LOG_ROOT}"
for treatment in "${TREATMENTS[@]}"; do
  mkdir -p "${DATA_ROOT}/${treatment}" "${LOG_ROOT}/${treatment}"
done

cp "${CONFIG_FILE}" "${TASK_ROOT}/level0_paper_audit_suite.json"
TASK_COUNT="$(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_tasks.py" \
    --config "${TASK_ROOT}/level0_paper_audit_suite.json" \
    --output-directory "${TASK_ROOT}" \
    --repository-root "${REPO_ROOT}"
)"
if [[ "${TASK_COUNT}" != "80" ]]; then
  echo "Expected 80 tasks but generated ${TASK_COUNT}" >&2
  exit 2
fi

echo "Collection: ${COLLECTION_ID}"
echo "Tasks: 80 total, 20 per treatment"
echo "Maximum simultaneous tasks per treatment: ${MAX_PARALLEL}"

for treatment in "${TREATMENTS[@]}"; do
  task_file="${TASK_ROOT}/${treatment}.csv"
  job_id="$(
    sbatch --parsable \
      --job-name="audit-${treatment}" \
      --array="0-19%${MAX_PARALLEL}" \
      --chdir="${REPO_ROOT}" \
      --output="${LOG_ROOT}/${treatment}/%A_%a.out" \
      --error="${LOG_ROOT}/${treatment}/%A_%a.err" \
      --export="ALL,EXPLORE_BENCH_ROOT=${REPO_ROOT},EXPLORE_BENCH_SLURM_ROOT=${SLURM_ROOT},LEVEL0_AUDIT_TASK_FILE=${task_file},LEVEL0_AUDIT_COLLECTION_ID=${COLLECTION_ID},LEVEL0_AUDIT_TREATMENT=${treatment}" \
      "${SCRIPT_DIR}/run_array.sbatch"
  )"
  echo "${treatment}: ${job_id}"
done

echo "Tasks: ${TASK_ROOT}"
echo "Data: ${DATA_ROOT}"
echo "Logs: ${LOG_ROOT}"
