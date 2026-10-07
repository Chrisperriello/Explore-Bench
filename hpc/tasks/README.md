# Task files

A task file is a plain CSV table in which each row describes one collector run:
one map, planner, and seed. Both sensor arrays read the same table. The sensor
is deliberately supplied by the array rather than stored in the row, which
prevents the original and four-beam task lists from drifting apart.

## Files

- `level0_suite.json`: editable source settings for the collection.
- `generate_level0_tasks.py`: expands the settings into the frozen CSV table.
- `run_level0_task.py`: reads one row and calls the existing headless collector.

The default settings contain the six named benchmark maps, cost and MMPF, two
robots, legacy behavior, and seeds 1 through 5. The paper does not report how
many runs or seeds it used, so five seeds are our explicit repeatable choice,
not a claim about the paper.

At submission, the JSON settings and generated table are copied to
`../slurm/tasks/level0/<collection-id>/`. That copy records exactly what the
two arrays used even if the repository settings change later.
