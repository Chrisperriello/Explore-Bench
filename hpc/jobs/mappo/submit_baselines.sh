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
# EXPLORE_MAPPO_CONTROLLERS picks controllers (default "random wait cost
# mmpf"), e.g. "cost mmpf" to rerun only the frontier planners.
# EXPLORE_MAPPO_SUITES picks suites (default "familiar unseen"); "unseen"
# skips the train maps when only the test maps are new.
# EXPLORE_MAPPO_SHARD_BY=episode makes one task per episode instead of per
# map: four-beam MMPF took up to 39 h per 10-episode map (job 22702), so an
# idle cluster finishes about ten times sooner. Scoring pools either way.
#
# Re-testing trained MAPPO on new test maps (explore_bench.config.
# retest_configuration): add "mappo" to EXPLORE_MAPPO_CONTROLLERS and set
# EXPLORE_MAPPO_SELECTIONS to the training collection's selections/
# directory. Every seed_N.json there gets its own shards, written to
# <collection>/evaluation/seed_N/, e.g.
#   EXPLORE_MAPPO_CONTROLLERS=mappo EXPLORE_MAPPO_SUITES=unseen \
#   EXPLORE_MAPPO_SELECTIONS=../slurm/data/mappo/mappo_omni_3p5_full_01/selections \
#   bash hpc/jobs/mappo/submit_baselines.sh mappo_omni_3p5_newmaps_01 \
#       experiments/mappo_omni_3p5_newmaps_v1.json
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
SELECTIONS=""
if [ -n "${EXPLORE_MAPPO_SELECTIONS:-}" ]; then
    SELECTIONS=$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$EXPLORE_MAPPO_SELECTIONS")
fi
# The shard list: one "controller suite map episode training_seed" line per
# array task ("-" = all episodes of the map / not MAPPO), written from the
# config's own frozen task design so it can never drift from it. A new
# list never overwrites an older one, so earlier arrays keep their lines.
SHARDS="${LOGS}/shards_$(date +%Y%m%d_%H%M%S).txt"
"$PYTHON" - "$CONFIG" "${EXPLORE_MAPPO_SUITES:-familiar unseen}" "${EXPLORE_MAPPO_SHARD_BY:-map}" \
    "$SELECTIONS" ${EXPLORE_MAPPO_CONTROLLERS:-random wait cost mmpf} > "$SHARDS" <<'PY'
import json
import sys
from pathlib import Path
from explore_bench.config import configuration, evaluation_tasks, retest_configuration
spec = configuration(sys.argv[1])
suites, shard_by, selections, controllers = sys.argv[2].split(), sys.argv[3], sys.argv[4], sys.argv[5:]
if not controllers or set(controllers) - {"random", "wait", "cost", "mmpf", "mappo"}:
    raise SystemExit("EXPLORE_MAPPO_CONTROLLERS must name random, wait, cost, mmpf and/or mappo")
if not suites or set(suites) - {"familiar", "unseen"}:
    raise SystemExit("EXPLORE_MAPPO_SUITES must name familiar and/or unseen")
if shard_by not in ("map", "episode"):
    raise SystemExit("EXPLORE_MAPPO_SHARD_BY must be map or episode")
seeds = ["-"]
if "mappo" in controllers:
    if not selections:
        raise SystemExit("mappo needs EXPLORE_MAPPO_SELECTIONS (a selections/ directory)")
    records = sorted(Path(selections).glob("seed_*.json"))
    if not records:
        raise SystemExit("no seed_*.json selections in " + selections)
    # Fail here, not on 600 compute tasks, if the checkpoints cannot be
    # re-tested under this config.
    for record in records:
        retest_configuration(json.loads(record.read_text())["configuration"], spec)
    seeds = [record.stem.split("_", 1)[1] for record in records]
for controller in controllers:
    for training_seed in (seeds if controller == "mappo" else ["-"]):
        for suite in suites:
            tasks = evaluation_tasks(spec, suite)
            if shard_by == "map":
                stems = dict.fromkeys(Path(task.map_path).stem for task in tasks)
                lines = [(stem, "-") for stem in stems]
            else:
                lines = [(Path(task.map_path).stem, str(task.seed)) for task in tasks]
            for stem, episode in lines:
                print(controller, suite, stem, episode, training_seed)
PY
COUNT=$(wc -l < "$SHARDS")
# Everything the worker needs travels as environment variables, because
# Slurm executes a copied script that cannot see this shell's variables
# otherwise (sbatch exports the submitting environment by default).
export EXPLORE_BENCH_ROOT="$REPOSITORY" EXPLORE_MAPPO_DIRECTORY="$ROOT"
export EXPLORE_MAPPO_CONFIG="$CONFIG" EXPLORE_MAPPO_SHARDS="$SHARDS" EXPLORE_MAPPO_SELECTIONS="$SELECTIONS"
echo "${COUNT} shards listed in ${SHARDS}"
sbatch --array="0-$((COUNT - 1))%${EXPLORE_MAPPO_CONCURRENCY:-20}" \
    --output="${LOGS}/%A_%a.out" --error="${LOGS}/%A_%a.err" \
    "$@" hpc/jobs/mappo/run_baseline_shard.sbatch
