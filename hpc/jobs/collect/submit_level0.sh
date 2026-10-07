#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SLURM_ROOT="${REPO_ROOT}/../slurm"
CONFIG_FILE="${REPO_ROOT}/hpc/tasks/level0_suite.json"
PYTHON_BIN="${REPO_ROOT}/.conda-env/bin/python"
MAX_PARALLEL="${1:-16}"
COLLECTION_ID="${2:-level0_$(date -u +%Y%m%dT%H%M%SZ)}"

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

TASK_ROOT="${SLURM_ROOT}/tasks/level0/${COLLECTION_ID}"
TASK_FILE="${TASK_ROOT}/tasks.csv"
DATA_ROOT="${SLURM_ROOT}/data/level0/${COLLECTION_ID}"
LOG_ROOT="${SLURM_ROOT}/logs/level0/${COLLECTION_ID}"

if [[ -e "${TASK_ROOT}" ]]; then
  echo "Collection ID already exists: ${COLLECTION_ID}" >&2
  echo "Use a new ID so a submitted task table can never be changed" >&2
  exit 2
fi

mkdir -p \
  "${TASK_ROOT}" \
  "${DATA_ROOT}/original" \
  "${DATA_ROOT}/four_beam" \
  "${LOG_ROOT}/original" \
  "${LOG_ROOT}/four_beam"

cp "${CONFIG_FILE}" "${TASK_ROOT}/level0_suite.json"
TASK_COUNT="$(
  "${PYTHON_BIN}" "${REPO_ROOT}/hpc/tasks/generate_level0_tasks.py" \
    --config "${TASK_ROOT}/level0_suite.json" \
    --output "${TASK_FILE}" \
    --repository-root "${REPO_ROOT}"
)"
ARRAY_END=$((TASK_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}%${MAX_PARALLEL}"

ORIGINAL_JOB_ID="$(
  sbatch --parsable \
    --job-name=level0-original \
    --array="${ARRAY_SPEC}" \
    --chdir="${REPO_ROOT}" \
    --export="ALL,EXPLORE_BENCH_ROOT=${REPO_ROOT},EXPLORE_BENCH_SLURM_ROOT=${SLURM_ROOT},LEVEL0_TASK_FILE=${TASK_FILE},LEVEL0_COLLECTION_ID=${COLLECTION_ID},LEVEL0_VARIANT=original,LEVEL0_SENSOR=omnidirectional" \
    --output="${LOG_ROOT}/original/%x-%A_%a.out" \
    --error="${LOG_ROOT}/original/%x-%A_%a.err" \
    "${SCRIPT_DIR}/run_level0_array.sbatch"
)"

FOUR_BEAM_JOB_ID="$(
  sbatch --parsable \
    --job-name=level0-four-beam \
    --array="${ARRAY_SPEC}" \
    --chdir="${REPO_ROOT}" \
    --export="ALL,EXPLORE_BENCH_ROOT=${REPO_ROOT},EXPLORE_BENCH_SLURM_ROOT=${SLURM_ROOT},LEVEL0_TASK_FILE=${TASK_FILE},LEVEL0_COLLECTION_ID=${COLLECTION_ID},LEVEL0_VARIANT=four_beam,LEVEL0_SENSOR=four_beam" \
    --output="${LOG_ROOT}/four_beam/%x-%A_%a.out" \
    --error="${LOG_ROOT}/four_beam/%x-%A_%a.err" \
    "${SCRIPT_DIR}/run_level0_array.sbatch"
)"

echo "Collection: ${COLLECTION_ID}"
echo "Tasks per sensor: ${TASK_COUNT}"
echo "Maximum simultaneous tasks per array: ${MAX_PARALLEL}"
echo "Original array job: ${ORIGINAL_JOB_ID}"
echo "Four-beam array job: ${FOUR_BEAM_JOB_ID}"
echo "Task table: ${TASK_FILE}"
echo "Data: ${DATA_ROOT}"
echo "Logs: ${LOG_ROOT}"
