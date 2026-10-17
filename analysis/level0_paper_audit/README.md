# Level-0 paper-audit analysis

`analyze.py` validates a completed 80-run audit before producing any paper
claim. It requires 20 complete tasks in each treatment, exact method/seed/start
pairing, one clean repository commit, one map hash, one collector hash, and one
task table per treatment.

## Metric audit

The raw episode file keeps interpretations separate instead of silently
assuming that similarly named metrics are equivalent:

- Paper-declared efficiency: first 90% (`Ttopo`) and 99% (`Ttotal`) crossings,
  with decision count, wall time, and CPU time retained independently.
- Released-code efficiency: first strict `>90%` and `>98%` crossings and the
  historical agent-0 path-log count printed by the released Level-0 code.
- Paper sigma: standard deviation of per-robot explored area in square metres.
- Released sigma: standard deviation of normalized per-robot coverage ratios.
- Paper/released overlap: `sum(agent area) / total terrain area - 1`.
- Collector overlap: duplicated sensed cells divided by total terrain area.

For overlap, both raw-ratio and multiplied-by-100 candidates are reported
because Table II labels the column as percent while the released code prints an
unscaled ratio.

The audit's `mmpf` method is the released multi-robot potential-field lane
reported as `field` in Table II.

Wall and CPU time are reported but cannot by themselves support reproduction,
because they depend on the machine. The automated numerical criteria are:

- time/count median within 20% of the paper number;
- sigma or overlap median within 0.05 of the paper number;
- otherwise, an alternate documented convention or 7 m treatment must itself
  meet the applicable closeness threshold and reduce the primary numerical gap
  by at least 80% to classify it as explained.

These thresholds are explicit audit rules, not statistical confidence claims.
The per-seed tables remain available for review.

## Outputs

The output directory contains:

- `audit_report.md` and `audit_report.html`: overall result and evidence;
- `paper_verdicts.csv`: supported, explained, or unresolved per method/metric;
- `metric_candidates.csv`: every formula/range interpretation and its tolerance
  result, including non-comparable wall/CPU timing for inspection;
- `pipeline_checks.csv`: provenance, pairing, and legacy regression gates;
- `summary.csv`: descriptive statistics for every treatment and method;
- `paired_sensor_results.csv`: seed-paired omnidirectional/four-beam effects;
- `paired_sensor_summary.csv`: median paired sensor effects by method and range;
- `episodes_combined.csv`, `steps_combined.csv`, and `run_inventory.csv`: the
  auditable combined source tables.

The overall result is `PASS` only if every Table-II metric is supported or has
a documented numerical explanation and the old/new 3.5 m control regression
passes. An unexplained result is not called a pipeline defect. A failed exact
legacy regression is.
