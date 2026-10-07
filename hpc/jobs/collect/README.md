# Collection jobs

This folder submits the matched Level-0 raw-data runs. A Slurm array starts many
copies of one job script, with each copy reading a different row from the same
task table.

## Files

- `submit_level0.sh`: creates a frozen task table and submits both arrays.
- `run_level0_array.sbatch`: activates the Ada environment and runs one task.

The submission script launches:

- `level0-original`, using the original omnidirectional sensor;
- `level0-four-beam`, using the four-beam sensor.

Both arrays use the same task index for the same map, method, and seed. One
array task requests one CPU. The `%16` default means Slurm may run at most 16
tasks from each array simultaneously; it does not give one task 16 CPUs.

## Submit

```bash
bash hpc/jobs/collect/submit_level0.sh
```

Optional arguments are the simultaneous-task limit and collection ID:

```bash
bash hpc/jobs/collect/submit_level0.sh 8 level0_validation_01
```

Every task writes under its own directory in
`../slurm/data/level0/<collection-id>/`. Resubmitting a failed task creates a
new collector `runN` directory, while an already complete task exits without
duplicating its data.
