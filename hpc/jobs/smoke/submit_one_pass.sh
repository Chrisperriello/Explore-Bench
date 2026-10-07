#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SLURM_ROOT="${REPO_ROOT}/../slurm"
LOG_ROOT="${SLURM_ROOT}/logs"
DATA_ROOT="${SLURM_ROOT}/data"

mkdir -p "${LOG_ROOT}" "${DATA_ROOT}"

sbatch \
  --chdir="${REPO_ROOT}" \
  --output="${LOG_ROOT}/%x-%j.out" \
  --error="${LOG_ROOT}/%x-%j.err" \
  "${SCRIPT_DIR}/run_one_pass.sbatch"
