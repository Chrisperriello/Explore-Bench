#!/bin/bash
# Usage: bash submit_baselines.sh COLLECTION_ID CONFIG [extra sbatch arguments]
#
# Runs the scripted controls (random, wait, cost, mmpf) on a collection's
# frozen familiar and unseen test tasks as one Slurm array, one task per
# controller x map (4 controllers x 10 maps = 40 tasks for the v1 maps).
#
# Why: every MAPPO claim needs these controls on the SAME tasks, sensor,
# communication and navigation contract. On 2026-10-26 they were launched by
# hand with `sbatch --wrap` and an out-of-repo helper, which missed the
# thread limits, needed manual time-limit fixes and took about a week for
# MMPF. This wrapper makes that a single, versioned, reproducible command.
#
# Typical use (see hpc/jobs/mappo/README.md):
#   bash hpc/jobs/mappo/submit_baselines.sh mappo_omni_3p5_legacy_01 \
#       experiments/mappo_omni_3p5_legacy_v1.json
# Extra arguments go straight to sbatch, e.g. --time=2-00:00:00.
# EXPLORE_MAPPO_CONCURRENCY caps simultaneous tasks (default 20).
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
COLLECTION="${1:?provide a collection ID}"
CONFIG="${2:?provide experiment JSON}"
shift 2
# Collection IDs become directory names; reject anything that could escape
# ../slurm/data/mappo or collide with shell syntax.
case "$COLLECTION" in *[!a-zA-Z0-9_-]*|"") echo "invalid collection ID" >&2; exit 2 ;; esac
REPOSITORY=$(git rev-parse --show-toplevel)
cd "$REPOSITORY"
PYTHON="${REPOSITORY}/.conda-env/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo 'Missing project Python. Run bash setup.sh --hpc inside a compute allocation first.' >&2
    exit 1
fi
# Absolute config path: array tasks run later from Slurm's own directory.
CONFIG=$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$CONFIG")
# Same layout as submit.sh, so baselines sit beside the MAPPO stages of the
# same collection and one `score` call can read both.
ROOT="${REPOSITORY}/../slurm/data/mappo/${COLLECTION}"
LOGS="${REPOSITORY}/../slurm/logs/mappo/${COLLECTION}/baselines"
mkdir -p "$ROOT" "$LOGS"
# Reuse the collection's task table if MAPPO training already created it;
# otherwise freeze one now. Either way MAPPO and the controls share it.
if [ ! -f "${ROOT}/tasks.json" ]; then
    "$PYTHON" -m explore_bench prepare --config "$CONFIG" --output "${ROOT}/tasks.json"
fi
# Check the table matches THIS config before queueing 40 jobs. A mismatched
# config (for example a four-beam JSON against an omni collection) would
# otherwise fail 40 times on the compute nodes.
"$PYTHON" -c 'import sys; from explore_bench.config import configuration; from explore_bench.evaluation import read_tasks; read_tasks(sys.argv[2], configuration(sys.argv[1]), "familiar")' "$CONFIG" "${ROOT}/tasks.json"
# The shard list: one "controller suite map" line per array task, written
# from the config's own map splits (familiar = train maps, unseen = test
# maps) so it can never drift from the frozen task design.
SHARDS="${LOGS}/shards.txt"
"$PYTHON" - "$CONFIG" > "$SHARDS" <<'PY'
import sys
from pathlib import Path
from explore_bench.config import configuration
spec = configuration(sys.argv[1])
for controller in ("random", "wait", "cost", "mmpf"):
    for suite, split in (("familiar", "train"), ("unseen", "test")):
        for name in spec["maps"][split]:
            print(controller, suite, Path(name).stem)
PY
COUNT=$(wc -l < "$SHARDS")
# Everything the worker needs travels as environment variables, because
# Slurm executes a copied script that cannot see this shell's variables
# otherwise (sbatch exports the submitting environment by default).
export EXPLORE_BENCH_ROOT="$REPOSITORY" EXPLORE_MAPPO_DIRECTORY="$ROOT"
export EXPLORE_MAPPO_CONFIG="$CONFIG" EXPLORE_MAPPO_SHARDS="$SHARDS"
sbatch --array="0-$((COUNT - 1))%${EXPLORE_MAPPO_CONCURRENCY:-20}" \
    --output="${LOGS}/%A_%a.out" --error="${LOGS}/%A_%a.err" \
    "$@" hpc/jobs/mappo/run_baseline_shard.sbatch
