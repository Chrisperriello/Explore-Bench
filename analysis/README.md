# Research Analysis

This directory contains experiment launch helpers, offline analysis, and data
visualization. It is intentionally separate from `grid_simulator/` and
`onpolicy/`, which contain the runtime and learning implementations.

Current study:

- [`level0_paper_audit/`](level0_paper_audit/README.md): strict five-room
  Table-II validation with paired four-beam and range sensitivity checks.
- [`paper_consistency/`](paper_consistency/README.md): three-treatment report
  separating sensor-distance effects from restricted-map planning effects.
- [`level0_baseline/`](level0_baseline/README.md): paired original-versus-
  four-beam Level-0 report, tables, figures, and advisor packet.
- [`level0_pilot/`](level0_pilot/README.md): scoped four-beam sensing and
  communication pilot, plotting tools, and presentation artifacts.
