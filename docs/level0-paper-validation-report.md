# Explore-Bench Level-0 paper-validation report

**Audit date:** 2026-10-20  
**Repository commit:** `72450b5bd201447b4dd7e1c2ad94c1702bd17552`  
**Paper scope:** Explore-Bench Table II, Level-0, five-room scenario  
**Decision:** **USE THE CLAIM-LEVEL CONTINUATION GATE**

## Executive conclusion

The data pipeline passed every integrity and regression check. The available
Explore-Bench implementation did not reproduce six of the eight exact
Level-0 Table-II values, and those discrepancies remain recorded below.
Because the missing original configuration will not be recovered, exact-cell
matching is no longer the continuation gate.

This result separates two claims:

1. **Pipeline validation passed.** The audit is paired, provenance-locked, and
   exactly reproduces the previous 3.5 m legacy controls.
2. **Exact-cell reproduction failed.** Passing pipeline validation does not
   make the observed numbers equivalent to those published in Table II.
3. **Claim-level reproduction is the attainable target.** Continuation now
   depends on a strict paired test of the paper's directional Level-0 claim
   that potential-field/MMPF requires fewer decisions than Cost.

There is no evidence that the audit collector corrupted the data. The project
may proceed after the generated claim-level gate passes, but it must describe
the result as reproduction of the Level-0 method-ranking claim—not as exact
reproduction of Table II.

## Published targets

Table II reports the following Level-0 results for Cost and potential field.
The released code lane named `mmpf` is treated as the potential-field lane.

| Method | Ttopo | Ttotal | sigma | overlap ro |
| --- | ---: | ---: | ---: | ---: |
| Cost | 134 | 172 | 0.09 | 0.10 |
| Field/MMPF | 100 | 125 | 0.01 | 0.04 |

The paper labels time in seconds, sigma in square metres, and overlap as a
percentage. The released Level-0 code instead prints path-log counts,
normalized per-agent coverage imbalance, and an unscaled overlap ratio at a
strict `>98%` boundary. The audit retained these interpretations separately.

## Frozen audit design

The audit ran 80 Level-0 episodes:

- Five-room map: `room1_modified.pgm`
- Map SHA-256:
  `64619f41ee6b70812282bd518b3032a39f22f2a0e7834f9dd94e2209d74ca6a1`
- Methods: Cost and MMPF
- Robots: 2
- Seeds: 1 through 10
- Decision limit: 1,000
- Coverage boundaries: 90%, 98%, and 99%
- Navigation/information behavior: released legacy mode
- Four paired treatments:
  - omnidirectional, 3.5 m
  - four-beam, 3.5 m
  - omnidirectional, 7.0 m
  - four-beam, 7.0 m

The 3.5 m omnidirectional treatment is the released-code control. The 7 m
treatments are sensitivity checks because the paper's explicit 7 m statement
describes the Gazebo robot. Four-beam is a new sensor extension, not a paper
reproduction treatment.

## Pipeline-integrity result

All integrity gates passed:

| Check | Result | Evidence |
| --- | --- | --- |
| Paired starts/headings | Pass | 20 method/seed blocks contain all four treatments with exact starts |
| Repository provenance | Pass | Commit `72450b5bd201447b4dd7e1c2ad94c1702bd17552` |
| Map provenance | Pass | One map hash across all 80 runs |
| Collector provenance | Pass | Collector hash `0374270cacbd11e2979ccee89493e6da9e634d5c963ca862f038ba2e8261abc6` |
| Legacy regression | Pass | New seeds 1–5 exactly match all ten prior 3.5 m Cost/MMPF controls |

The exact legacy regression is the strongest pipeline result: the independent
audit collector produced the same termination, coverage, thresholds, sigma,
and overlap data as the preserved pipeline for every prior control.

## Paper-control results

### Cost

| Metric | Paper | 3.5 m control median | Control range | Best audited alternative | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Ttopo | 134 | 402 | 316–576 | 260.5, 7 m decision count | Unresolved |
| Ttotal | 172 | 513 | 459–655 | 372.5, 7 m strict-98% decision count | Unresolved |
| sigma | 0.09 | 17.5725 m² | 1.630–37.385 m² | 0.05139, 7 m normalized ratio | Explained convention mismatch |
| overlap ro | 0.10 | 0.34490 | 0.10706–0.58618 | 0.25559, released strict-98% ratio | Unresolved |

### Field/MMPF

| Metric | Paper | 3.5 m control median | Control range | Best audited alternative | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Ttopo | 100 | 312 | 229–363 | 240, 7 m decision count | Unresolved |
| Ttotal | 125 | 363 | 287–528 | 341, 7 m strict-98% decision count | Unresolved |
| sigma | 0.01 | 10.5075 m² | 1.035–38.775 m² | 0.02585, normalized strict-98% ratio | Explained convention mismatch |
| overlap ro | 0.04 | 0.22910 | 0.03411–0.40595 | 0.21604, released strict-98% ratio | Unresolved |

The 3.5 m controls all reached 99% coverage, so missing threshold observations
do not explain their disagreement with the paper. The published time values
are below even the minimum observed 3.5 m decision counts. Increasing the
sensor range to 7 m moves some timing values toward the paper but does not make
them close enough to pass the predefined audit tolerances.

The sigma magnitude changes by orders of magnitude depending on whether the
standard deviation is computed over square metres or normalized coverage
ratios. This explains the unit/convention discrepancy, but it does not repair
the unresolved time and overlap results.

## Four-beam extension result

No four-beam run reached 99% coverage within 1,000 decisions.

| Sensor | Method | T99 successes | Median final coverage | Final range | Termination | Median gain in final 100 decisions |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Four-beam 3.5 m | Cost | 0/10 | 0.6387 | 0.3915–0.8482 | 10 max-steps | 0.0235 |
| Four-beam 3.5 m | MMPF | 0/10 | 0.3667 | 0.0910–0.8936 | 6 max-steps, 4 stuck | 0.0077 |
| Four-beam 7.0 m | Cost | 0/10 | 0.7824 | 0.4164–0.9391 | 8 max-steps, 2 stuck | 0.0111 |
| Four-beam 7.0 m | MMPF | 0/10 | 0.6022 | 0.2631–0.8868 | 10 max-steps | 0.0000 |

The result is not primarily caused by premature stuck detection: 34 of 40
four-beam runs consumed the full decision budget. Cost continues to make slow
progress in some runs, while MMPF is near or at a plateau. The unmodified
classical planners therefore do not provide a successful four-beam baseline at
this budget.

The 7 m omnidirectional sensitivity also shows a planner interaction: Cost
reached 99% in all ten runs, but MMPF reached it in only five. Longer sensing
range is not a monotonic substitute for the paper configuration.

## Exact-cell verdict

| Question | Answer |
| --- | --- |
| Did all 80 audit runs complete? | Yes |
| Is the audit paired and provenance-locked? | Yes |
| Does the new control reproduce the preserved legacy pipeline? | Yes, exactly |
| Does the released implementation reproduce Table II? | No |
| Does 7 m sensing explain the mismatch? | No |
| Do alternate 98%/99% boundaries explain it? | No |
| Is the sigma discrepancy attributable to a formula/unit mismatch? | Yes |
| Are timing and overlap discrepancies resolved? | No |
| Can the project claim exact Table-II reproduction? | No |

The exact-cell outcome remains unresolved, not a pipeline defect. It is
retained as a diagnostic and does not control the revised continuation gate.

## Approved claim-level continuation gate

The paper's Level-0 results show potential field outperforming Cost at both
`Ttopo` and `Ttotal`. The revised gate tests that directional claim on the
released 3.5 m omnidirectional control using seeds 1–10.

Both `Ttopo` (90% coverage) and `Ttotal` (99% coverage) must independently
meet every requirement:

- all ten Cost and all ten MMPF runs reach the threshold;
- Cost and MMPF use identical starts and headings for each seed;
- at least nine of ten paired seeds require fewer MMPF decisions;
- the median paired MMPF-minus-Cost difference is negative;
- the at-least-95% distribution-free median interval is entirely below zero; and
- the exact one-sided sign-test p-value is below 0.05.

The existing integrity and exact legacy-regression checks remain mandatory.
The four-beam and 7 m treatments are excluded from the claim gate.

### Interpretation boundary

A pass permits the statement:

> Under the released Level-0 3.5 m control and ten paired seeds, the pipeline
> reproduces the paper's directional claim that MMPF outperforms Cost at 90%
> and 99% coverage.

A pass does not permit the statements that the exact Table-II cells, Level 1,
Level 2, cross-level consistency, or the four-beam extension were reproduced.

### Apply the revised gate

No new simulation runs are needed. Re-run the analyzer on the completed 80-run
collection. It now writes `claim_reproduction.csv` and
`method_ranking_pairs.csv`, and prints `PASS` only if both claim metrics and all
pipeline checks pass.

Until that rerun reports `PASS — Level-0 method-ranking claim reproduced`, the
continuation decision is pending.

## Audit artifacts on Ada

Raw collection:

```text
/home/cperr23/slurm/data/level0_paper_audit/level0_paper_audit_01
```

Frozen task tables:

```text
/home/cperr23/slurm/tasks/level0_paper_audit/level0_paper_audit_01
```

Generated report and combined tables:

```text
/home/cperr23/slurm/reports/level0_paper_audit/level0_paper_audit_01
```

Important generated files:

- `audit_report.md`
- `audit_report.html`
- `pipeline_checks.csv`
- `paper_verdicts.csv`
- `claim_reproduction.csv`
- `method_ranking_pairs.csv`
- `metric_candidates.csv`
- `summary.csv`
- `paired_sensor_results.csv`
- `paired_sensor_summary.csv`
- `episodes_combined.csv`
- `steps_combined.csv`
- `run_inventory.csv`

These files are the evidence for both the exact-cell diagnostic and the revised
claim-level gate. Do not delete the raw collection until it has been archived
and independently verified.
