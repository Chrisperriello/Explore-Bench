# Terminal monitor

`slurm_monitor.py` is a text-based screen for checking Explore-Bench jobs from
an Ada login shell. It does not start, stop, or change jobs.

It shows:

- how many jobs are running, waiting, complete, or failed;
- job name, state, elapsed time, and allocated CPUs;
- estimated CPU use when Ada's `sstat` command provides CPU time;
- average or maximum memory when `sstat` provides it;
- collector progress from each manifest or array-task status file;
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

The monitor searches nested folders, so it displays both the one-pass smoke
job and the individual tasks from the paired Level-0 arrays.

## Browser dashboard

`slurm_dashboard.py` provides a rendered live view for one Level-0 collection.
It reads that collection's frozen task table and status files, and only requests
live CPU and memory information for its running Slurm jobs.

Start it on Ada:

```bash
python3 hpc/monitor/slurm_dashboard.py \
  --collection level0_validation_01
```

Keep that process running. In a second terminal on your computer, create an SSH
tunnel (replace `ada` with the hostname you normally use to connect):

```bash
ssh -N -L 8765:127.0.0.1:8765 cperr23@ada
```

Then open `http://127.0.0.1:8765` in your computer's browser. The server binds
only to Ada's loopback interface, so the page is available through your SSH
tunnel rather than being exposed publicly. It uses only Python's standard
library and refreshes every five seconds.
