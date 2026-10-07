# Terminal monitor

`slurm_monitor.py` is a text-based screen for checking Explore-Bench jobs from
an Ada login shell. It does not start, stop, or change jobs.

It shows:

- how many jobs are running, waiting, complete, or failed;
- job name, state, elapsed time, and allocated CPUs;
- estimated CPU use when Ada's `sstat` command provides CPU time;
- average or maximum memory when `sstat` provides it;
- collector progress from each `manifest.json` file;
- final coverage and termination reason after an episode is recorded;
- paths to the job's data and logs;
- the end of an error log for the selected job.

Start the interactive display from the repository root:

```bash
python3 hpc/monitor/slurm_monitor.py
```

Keys:

- up/down: select a job;
- `r`: refresh now;
- `q`: quit.

For a single printable report instead of the interactive screen:

```bash
python3 hpc/monitor/slurm_monitor.py --once
```

The CPU percentage is an estimate computed from Slurm's recorded CPU time,
elapsed time, and allocated CPU count. Some clusters update this slowly or do
not expose it while a job is running; the monitor displays `-` in that case.
