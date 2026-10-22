#!/bin/bash
# Usage: bash submit.sh STAGE COLLECTION_ID CONFIG [extra sbatch arguments]
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
STAGE="${1:?provide train, validate, select, evaluate or score}"
COLLECTION="${2:?provide a new collection ID}"
CONFIG="${3:?provide experiment JSON}"
shift 3
case "$STAGE" in train|validate|select|evaluate|score) ;; *) exit 2 ;; esac
case "$COLLECTION" in *[!a-zA-Z0-9_-]*|"") echo "invalid collection ID" >&2; exit 2 ;; esac
REPOSITORY=$(git rev-parse --show-toplevel)
cd "$REPOSITORY"
PYTHON="${REPOSITORY}/.conda-env/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo 'Missing project Python. Run bash setup.sh --hpc inside a compute allocation first.' >&2
    exit 1
fi
CONFIG=$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$CONFIG")
ROOT="${REPOSITORY}/../slurm/data/mappo/${COLLECTION}"
LOGS="${REPOSITORY}/../slurm/logs/mappo/${COLLECTION}/${STAGE}"
mkdir -p "$ROOT" "$LOGS"
if [ ! -f "${ROOT}/tasks.json" ]; then
    "$PYTHON" -m explore_bench prepare --config "$CONFIG" --output "${ROOT}/tasks.json"
fi
# Check the input contract again even when reusing an existing task table.
"$PYTHON" -c 'import sys; from explore_bench.config import configuration; from explore_bench.evaluation import read_tasks; read_tasks(sys.argv[2], configuration(sys.argv[1]), "validation")' "$CONFIG" "${ROOT}/tasks.json"
COUNT=$("$PYTHON" -c 'import sys; from explore_bench.config import configuration; print(len(configuration(sys.argv[1])["training_seeds"]))' "$CONFIG")
export EXPLORE_BENCH_ROOT="$REPOSITORY"
export EXPLORE_MAPPO_DIRECTORY="$ROOT"
export EXPLORE_MAPPO_CONFIG="$CONFIG"
export EXPLORE_MAPPO_STAGE="$STAGE"
LAST=$((COUNT - 1))
if [ "$STAGE" = score ]; then LAST=0; fi
sbatch --array="0-${LAST}%${EXPLORE_MAPPO_CONCURRENCY:-2}" \
    --output="${LOGS}/%A_%a.out" --error="${LOGS}/%A_%a.err" \
    "$@" hpc/jobs/mappo/run_stage.sbatch
