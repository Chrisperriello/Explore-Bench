#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT="${1:-${REPOSITORY_ROOT}/../slurm/combined/level0/level0_validation_01}"
OUTPUT="${2:-${REPOSITORY_ROOT}/results/level0_baseline_analysis}"
PYTHON="${REPOSITORY_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; activate the Python 3.8 project environment first." >&2
  exit 1
fi

MPLBACKEND=Agg "${PYTHON}" "${REPOSITORY_ROOT}/analysis/level0_baseline/analyze_level0.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}"

echo "Open ${OUTPUT}/report.html"
