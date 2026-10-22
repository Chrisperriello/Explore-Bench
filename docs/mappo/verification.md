# MAPPO implementation verification

Verified locally on 2026-10-21. This records implementation and controlled
learning evidence, not paper-map convergence or a new academic reproduction.

## Automated checks

- Full pytest suite: **102 passed**, including the historical tests.
- New MAPPO contract tests cover real PPO updates, process workers, actor-only
  fresh-process loading, checkpoint lineage, independent metrics, reward and
  termination behavior, sensor origin, immutable outputs and frozen tasks.
- Interrupted/resumed CPU training produced actor tensors bit-for-bit equal to
  uninterrupted training on the same fixture.
- Inference left actor parameters unchanged and did not import the MAPPO trainer.
- Python compilation, CLI help, Bash syntax and Git whitespace checks passed.
- Cost and MMPF goal adapters each executed a real five-room-map transition
  through the shared session. This is a smoke, not a baseline performance run.

Software: Python 3.8.20, PyTorch 2.3.1, NumPy 1.23.5, OpenCV 4.8.1, arm64 CPU.
Fresh-process OpenMP tests needed execution outside the local filesystem
sandbox because shared-memory initialization was blocked. They then passed.
CUDA execution and actual Ada Slurm submission have not been tested here.

## Controlled learning experiment

Command:

```bash
.venv/bin/python -m explore_bench sanity \
  --output /tmp/explore-mappo-learning.NceM4s/corrected --allow-dirty
```

The generated corridor experiment used two process workers, 32-step rollouts,
8,192 environment transitions, five PPO epochs, 40-decision episodes and
training seed 7. Training took approximately 129.94 seconds on this machine
(excluding validation/testing). This is not a runtime estimate for the much
larger paper maps.

Predeclared engineering criterion: validation coverage AUC must improve by
at least 0.01 over the untrained actor, and the selected checkpoint must be
from a nonzero training step. The check **passed**, with improvement
**0.04912037**. The complete numeric record is
[controlled_learning_result.json](controlled_learning_result.json).

| Controller | Validation mean final coverage | Validation AUC | Validation T99 successes |
| --- | --- | --- | --- |
| Untrained deterministic MAPPO | 0.881481 | 0.742870 | 1/10 |
| Random-coordinate controller | 0.811111 | 0.648873 | 1/10 |
| Selected trained MAPPO | 0.974074 | 0.791991 | 1/10 |

All saved checkpoints were considered under the frozen selection rule:

| Environment transitions | Validation success rate | Validation AUC | Validation final coverage |
| --- | --- | --- | --- |
| 0 | 0.10 | 0.742870 | 0.881481 |
| 1024 | 0.10 | 0.791991 | 0.974074 |
| 2048 | 0.10 | 0.772083 | 0.988889 |
| 3072 | 0.10 | 0.775046 | 0.988889 |
| 4096 | 0.10 | 0.781590 | 0.988889 |
| 5120 | 0.10 | 0.788966 | 0.988889 |
| 6144 | 0.10 | 0.788966 | 0.988889 |
| 7168 | 0.10 | 0.788966 | 0.988889 |
| 8192 | 0.10 | 0.788966 | 0.988889 |

Selected checkpoint: **1,024 transitions**. Actor SHA-256:

```text
b9820b6533bb1a2a6817ffd1449ca87cdceac072678a8a7128bfa22b5c871967
```

The selected actor was then tested on ten frozen starts in the reserved shifted
corridor. Mean final coverage was **0.961728**, AUC **0.807207**, and T99 success
was **3/10**. No test metric influenced checkpoint selection. There is only one
independent training seed, so the scorer correctly leaves training-seed
confidence intervals blank.

Validation AUC improved, but 99% success remains low. This verifies that useful
learning can occur and that the pipeline records incomplete exploration
honestly; it does not establish convergence or robust transfer.

## Additional finding exposed by the check

An initial, separately retained development check improved validation AUC but
performed poorly on the shifted corridor. Inspection showed that inherited
grid-to-world-to-grid truncation shifted a recorded position from [15,21] to a
sensing origin of [14,20], on a wall. The new session now senses directly at
the executed grid position. The results above come from the corrected run.

The historical paper-audit environment keeps its old behavior. Therefore no
prior raw measurements were silently reinterpreted.

## Command-line stage verification

After the corrected training run, validation was repeated with the final CSV
hashing/audit implementation, followed by separate CLI processes for select,
evaluate and score. The selected step and reported test metrics matched.
The scorer verified output hashes, physical trace consistency, reward totals,
milestones, full task cohorts and coverage AUC.

Raw development artifacts remain under:

```text
/tmp/explore-mappo-learning.NceM4s/corrected/
/tmp/explore-mappo-learning.NceM4s/final_validation/
/tmp/explore-mappo-learning.NceM4s/final_selection.json
/tmp/explore-mappo-learning.NceM4s/final_test/
/tmp/explore-mappo-learning.NceM4s/final_scores/
```

Temporary files are not permanent research storage. The checked-in numeric
record, generator, resolved recipe and tests preserve the procedure; rerun it
on Ada into the persistent external Slurm data directory.

## Next research operation

Commit and synchronize the source, rerun the controlled check on Ada, then run
the 102,400-transition paper-map pilot. Inspect throughput, validation curves,
termination reasons and seed sensitivity before the five-seed full sweep.
No paper-map pilot, full research sweep, or DQN training was launched locally.
