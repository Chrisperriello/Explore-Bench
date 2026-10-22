# Code change ledger and rationale

This ledger describes the implementation of the approved MAPPO plan. Git diff
provides line-level changes; this file records the purpose of each code file.
Decision IDs refer to [decisions.md](decisions.md).

## Existing code

| File | Change | Why |
| --- | --- | --- |
| `onpolicy/utils/shared_buffer.py` | Store one joint Box-action log probability, matching mixed actions | Prevent duplicated PPO loss terms; M05 |
| `onpolicy/algorithms/utils/act.py` | Expose mixed-head entropy weights, retain historical defaults for old callers | Research recipe applies its coefficient once; M06 |
| `onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py` | Read explicit component entropy weights from resolved arguments | Avoid hidden model behavior; M06 |
| `onpolicy/algorithms/r_mappo/r_mappo.py` | Report approximate KL and clipping fraction per update | Observe policy movement and excessive clipping; M07 |
| `onpolicy/utils/valuenorm.py` | Create non-trainable Parameters directly on their destination device | Ensure GPU normalizer state is registered and checkpointed; M23 |
| `onpolicy/runner/shared/grid_runner.py` | Correct evaluation history/identity; retain histories between PPO updates; log evaluation at actual step; delegate communication preprocessing | Fix confirmed defects and eliminate preprocessing drift; M03, M26 |
| `onpolicy/envs/GridEnv/observations.py` | Extract existing communication CTDE conversion into a function shared by runners and sessions | Preserve the tested actor/critic information boundary; M02–M03 |

Paths in this section are relative to the nested `onpolicy/` package directory.
The old `GridEnv.step`, classical collectors, sensors and broker implementations
are retained; the new session selects explicit corrected transition semantics.

## Shared experiment services

| File | Purpose | Why this is reusable |
| --- | --- | --- |
| `explore_bench/config.py` | Resolved configuration, map hashing, deterministic explicit tasks and frozen splits | No learner implementation required; M18–M19 |
| `explore_bench/artifacts.py` | Strict streaming tables, atomic metadata, source/version provenance, immutable output directories | Every controller uses the same artifact lifecycle; M23, M26 |
| `explore_bench/session.py` | Explicit task reset, shared observation histories, legal frontier decoding, bounded fixed-edge movement, corrected rewards and evaluator measurements | MAPPO/DQN/classical controllers execute the same transition; M08–M17 |
| `explore_bench/controllers.py` | Duck-typed stateful inference protocol and random/wait/classical adapters | DQN can plug in without a PPO runner; M02, M28 |
| `explore_bench/recording.py` | Episode ledger, first crossings, raw traces, reward and communication accounting | Evaluation data have the same meaning across methods; M14–M17 |
| `explore_bench/evaluation.py` | Immutable tasks, actor-only evaluation, checkpoint lineage and snapshot validation | No optimizer update in final testing; M20–M22 |
| `explore_bench/scoring.py` | Validate raw records, select using validation only, aggregate by training seed, generate scores/curves/report | Algorithm-independent comparison; M20, M24–M25 |

## MAPPO-specific orchestration

| File | Purpose | Why it stays algorithm-specific |
| --- | --- | --- |
| `explore_bench/mappo.py` | Explicit PPO argument adapter and actor-only inference export loader | Gaussian/categorical policy and neural architecture |
| `explore_bench/training.py` | Collect rollouts, bootstrap final timeout states, update PPO, record diagnostics and checkpoint | PPO uses on-policy returns/log probabilities; DQN uses replay/TD targets |
| `explore_bench/worlds.py` | Persistent independently seeded sessions and process transport, with checkpointable episode ledgers | The transport/session ownership can be reused, while the current reset schedule names MAPPO training episodes |
| `explore_bench/checkpoints.py` | Immutable actor exports and full MAPPO continuation payloads | Checkpoint lifecycle is reusable, payload contains PPO optimizer/critic/normalizer |
| `explore_bench/__main__.py` | Strict prepare/train/validate/select/evaluate/score commands | Stages have distinct authority and output directories |
| `explore_bench/sanity.py` | Generate an isolated corridor experiment and measure improvement over an untrained actor with a predeclared criterion | Learning verification does not consume the reserved research map suites |

## Configuration, operations, documentation and tests

| File/group | Change and reason |
| --- | --- |
| `experiments/mappo_omni_3p5_v1.json` | Names the ordinary sensing baseline; complete settings are resolved into every manifest |
| `experiments/mappo_four_beam_3p5_v1.json` | Changes sensor type only, supporting a controlled later extension |
| `hpc/jobs/mappo/submit.sh` | Prepare/verify tasks, make log directories, array over declared seeds, forward resource options |
| `hpc/jobs/mappo/run_stage.sbatch` | Activate the pinned Python environment, control CPU threading, run one stage, request pre-timeout checkpoint |
| `tests/test_mappo_pipeline.py` | Independent toy-map metric/reward checks plus real training/resume/validation/inference tests |
| `docs/mappo/*`, HPC README and root README | Research review, decisions/sources/tradeoffs, exact commands, schema/score semantics, limitations and handoff |

## Verification policy

Existing tests remain the historical regression check. New tests must exercise
observable behavior rather than reproduce implementation expressions. The
integration test compares uninterrupted and resumed actor tensors, validates
all saved snapshots, selects a checkpoint, runs held-out episodes, checks raw
CSV consistency, proves inference parameters did not change, and loads the
actor in a fresh process that has not imported the MAPPO trainer. A separate
test uses two actual worker processes and snapshots their independent sessions.

Only tested outcomes should be added to the final handoff. Full research
performance requires the controlled learning check, pilot and five-seed runs;
these are distinct from unit/integration tests.

## Follow-up: complete environment provisioning

The first Ada test attempt exposed a deployment gap: installing only the MAPPO
requirements did not install the nested `onpolicy` package. Ada's module also
set a conflicting `PYTHONHOME`. These failed before algorithm tests executed.

| File/group | Change and reason |
| --- | --- |
| `setup.sh` | Provision/reuse the correct project Python, isolate module variables, select CPU wheels, install the complete stack and editable packages, verify and run tests by default |
| `setup_env.sh` | Delegate to the complete installer and activate/clear the calling shell when sourced; retain old flags as compatibility aliases |
| `pyproject.toml` | Make the root experiment package installable for worker/CLI imports outside the checkout |
| `requirements-level0-mappo.txt` | Include the grid base, pin the logging runtime and use headless OpenCV without system display-library requirements |
| `scripts/check_setup.py` | Independently check package origins, dependencies, native libraries, tensor execution, CLI and resolved maps, and retain actual versions in a report |
| `tests/test_setup.py` | Cover fresh/reused environments, provider bootstrap, checksum rejection, module poisoning, failures/retries, recoverable rebuilding, activation and strict dependency checks |
| `hpc/jobs/mappo/{submit.sh,run_stage.sbatch}` | Use the project interpreter directly and clear Python module overrides; avoid depending on a particular shell activation/module state |
| `.gitignore` | Keep environments, backups, setup logs and editable/build metadata out of research commits |
| Root/HPC/MAPPO READMEs and `docs/setup.md` | Replace optional manual installation steps with one setup command, explicit prerequisites, recovery and recorded verification boundaries |

This does not change policy inputs, rewards, training budgets or scientific
selection gates. The isolated installer verification uses a new environment;
it does not mutate the preexisting development environment or HPC data.
