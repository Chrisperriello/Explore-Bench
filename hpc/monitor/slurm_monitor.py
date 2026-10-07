#!/usr/bin/env python3
"""Display Slurm and Explore-Bench data health in a terminal."""

import argparse
import csv
import curses
import getpass
import json
import os
import re
import subprocess
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SLURM_ROOT = REPOSITORY_ROOT.parent / "slurm"
JOB_ID_PATTERN = re.compile(r"-(\d+(?:_\d+)?)\.(?:out|err)$")


def run_command(arguments):
    """Run one read-only cluster command and return output plus an error."""
    try:
        result = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
    except OSError as error:
        return "", str(error)
    if result.returncode:
        return result.stdout, result.stderr.strip() or "command failed"
    return result.stdout, ""


def split_rows(output, expected):
    """Split pipe-separated command output, ignoring incomplete rows."""
    rows = []
    for line in output.splitlines():
        values = line.strip().strip("|").split("|")
        if len(values) >= expected:
            rows.append(values[:expected])
    return rows


def query_queue(user):
    """Return jobs currently visible in the user's Slurm queue."""
    output, error = run_command(
        [
            "squeue",
            "--noheader",
            "--user",
            user,
            "--format=%i|%j|%T|%M|%C|%D|%R",
        ]
    )
    jobs = {}
    for values in split_rows(output, 7):
        job_id, name, state, elapsed, cpus, nodes, reason = values
        jobs[job_id] = {
            "job_id": job_id,
            "name": name,
            "state": state,
            "elapsed": elapsed,
            "cpus": cpus,
            "nodes": nodes,
            "reason": reason,
            "exit_code": "",
        }
    return jobs, error


def discover_log_job_ids(log_root):
    """Find job identifiers embedded in monitorable Slurm log filenames."""
    identifiers = set()
    if not log_root.is_dir():
        return identifiers
    for path in log_root.iterdir():
        match = JOB_ID_PATTERN.search(path.name)
        if match:
            identifiers.add(match.group(1))
    return identifiers


def query_accounting(job_ids):
    """Return recent final states for known jobs no longer in the queue."""
    if not job_ids:
        return {}, ""
    output, error = run_command(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(sorted(job_ids)),
            "--format=JobIDRaw,JobName,State,Elapsed,AllocCPUS,ExitCode,NodeList",
        ]
    )
    jobs = {}
    for values in split_rows(output, 7):
        job_id, name, state, elapsed, cpus, exit_code, nodes = values
        if "." in job_id:
            continue
        state = state.split()[0].rstrip("+")
        jobs[job_id] = {
            "job_id": job_id,
            "name": name,
            "state": state,
            "elapsed": elapsed,
            "cpus": cpus,
            "nodes": nodes,
            "reason": "",
            "exit_code": exit_code,
        }
    return jobs, error


def parse_seconds(value):
    """Convert Slurm's day-hour-minute-second text to seconds."""
    if not value or value in ("Unknown", "UNLIMITED"):
        return None
    try:
        days = 0
        clock = value
        if "-" in value:
            day_text, clock = value.split("-", 1)
            days = int(day_text)
        parts = [int(float(part)) for part in clock.split(":")]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        else:
            hours = 0
            minutes = 0
            seconds = parts[0]
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (TypeError, ValueError):
        return None


def query_usage(jobs):
    """Read live CPU-time and memory counters for active jobs."""
    running_ids = [
        job_id
        for job_id, job in jobs.items()
        if job["state"].upper() in ("RUNNING", "COMPLETING")
    ]
    if not running_ids:
        return {}, ""
    output, error = run_command(
        [
            "sstat",
            "--allsteps",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(running_ids),
            "--format=JobID,AveCPU,AveRSS,MaxRSS",
        ]
    )
    usage = {}
    for values in split_rows(output, 4):
        step_id, average_cpu, average_rss, maximum_rss = values
        job_id = step_id.split(".", 1)[0]
        current = usage.setdefault(
            job_id,
            {"average_cpu": "", "average_rss": "", "maximum_rss": ""},
        )
        if average_cpu:
            current["average_cpu"] = average_cpu
        if average_rss:
            current["average_rss"] = average_rss
        if maximum_rss:
            current["maximum_rss"] = maximum_rss
    return usage, error


def find_job_id(path):
    """Find a job_<id> directory above an artifact."""
    for parent in (path,) + tuple(path.parents):
        if parent.name.startswith("job_"):
            return parent.name[4:]
    return ""


def read_last_episode(path):
    """Read the final row of an episode CSV when present."""
    if not path.is_file():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1] if rows else {}
    except (OSError, csv.Error):
        return {}


def load_data_records(data_root):
    """Load the newest collector manifest associated with each job."""
    records = {}
    if not data_root.is_dir():
        return records
    for manifest_path in data_root.rglob("manifest.json"):
        job_id = find_job_id(manifest_path)
        if not job_id:
            continue
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            continue
        modified = manifest_path.stat().st_mtime
        if job_id in records and records[job_id]["modified"] >= modified:
            continue
        configuration = manifest.get("configuration", {})
        episode = read_last_episode(manifest_path.parent / "episodes.csv")
        records[job_id] = {
            "modified": modified,
            "manifest_path": str(manifest_path),
            "run_directory": str(manifest_path.parent),
            "status": str(manifest.get("status", "unknown")),
            "planned": int(manifest.get("episodes_planned", 0) or 0),
            "completed": int(manifest.get("episodes_completed", 0) or 0),
            "error": str(manifest.get("error", "") or ""),
            "method": str(configuration.get("method", "")),
            "map": Path(str(configuration.get("maps", [""])[0])).name
            if configuration.get("maps")
            else "",
            "sensor": ",".join(configuration.get("sensor_types", [])),
            "range": ",".join(
                str(value) for value in configuration.get("sensor_ranges", [])
            ),
            "seed": str(episode.get("seed", "")),
            "coverage": str(episode.get("final_coverage_ratio", "")),
            "steps": str(episode.get("steps_executed", "")),
            "termination": str(episode.get("termination_reason", "")),
        }
    return records


def log_paths(log_root, job_id):
    """Return matching stdout and stderr paths for a job."""
    result = {"stdout": "", "stderr": ""}
    if not log_root.is_dir():
        return result
    for path in log_root.glob("*-{}.*".format(job_id)):
        if path.suffix == ".out":
            result["stdout"] = str(path)
        elif path.suffix == ".err":
            result["stderr"] = str(path)
    return result


def tail(path, lines=5):
    """Read a small tail from a text file without failing the monitor."""
    if not path:
        return []
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            return handle.readlines()[-lines:]
    except OSError:
        return []


def cpu_estimate(job, usage):
    """Estimate CPU percent from cumulative CPU time and elapsed wall time."""
    cpu_seconds = parse_seconds(usage.get("average_cpu", ""))
    elapsed_seconds = parse_seconds(job.get("elapsed", ""))
    try:
        cpus = max(1, int(job.get("cpus", 1) or 1))
    except ValueError:
        cpus = 1
    if cpu_seconds is None or not elapsed_seconds:
        return "-"
    return "{:.0f}%".format(100.0 * cpu_seconds / (elapsed_seconds * cpus))


def classify(job, record, error_tail):
    """Return one simple health label for a job and its data."""
    state = job.get("state", "").upper()
    data_status = record.get("status", "").lower()
    error_text = "".join(error_tail).lower()
    if data_status == "failed" or state in (
        "FAILED",
        "CANCELLED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
    ):
        return "FAILED"
    if "traceback" in error_text or "fatal python error" in error_text:
        return "ERROR"
    if data_status == "complete":
        return "COMPLETE"
    if state == "RUNNING" and data_status == "running":
        return "HEALTHY"
    if state == "RUNNING":
        return "STARTING"
    if state in ("PENDING", "CONFIGURING"):
        return "WAITING"
    return state or data_status.upper() or "UNKNOWN"


def collect_snapshot(slurm_root, user):
    """Collect one combined view of Slurm state, logs, and result files."""
    log_root = slurm_root / "logs"
    data_root = slurm_root / "data"
    queue, queue_error = query_queue(user)
    records = load_data_records(data_root)
    known_ids = set(records) | set(queue) | discover_log_job_ids(log_root)
    accounting, accounting_error = query_accounting(known_ids - set(queue))
    jobs = dict(accounting)
    jobs.update(queue)
    usage, usage_error = query_usage(jobs)

    rows = []
    for job_id in sorted(known_ids | set(jobs), key=lambda value: value.zfill(20)):
        job = jobs.get(
            job_id,
            {
                "job_id": job_id,
                "name": "",
                "state": "",
                "elapsed": "",
                "cpus": "",
                "nodes": "",
                "reason": "",
                "exit_code": "",
            },
        )
        record = records.get(job_id, {})
        logs = log_paths(log_root, job_id)
        error_tail = tail(logs["stderr"])
        job_usage = usage.get(job_id, {})
        rows.append(
            {
                "job": job,
                "record": record,
                "logs": logs,
                "error_tail": error_tail,
                "cpu": cpu_estimate(job, job_usage),
                "memory": job_usage.get("maximum_rss")
                or job_usage.get("average_rss")
                or "-",
                "health": classify(job, record, error_tail),
            }
        )
    errors = [error for error in (queue_error, accounting_error, usage_error) if error]
    return rows, errors


def coverage_text(value):
    """Format a recorded coverage ratio as a percentage."""
    try:
        return "{:.1f}%".format(float(value) * 100.0)
    except (TypeError, ValueError):
        return "-"


def summary(rows):
    """Count the main job states displayed by the monitor."""
    counts = {"running": 0, "waiting": 0, "complete": 0, "failed": 0}
    for row in rows:
        health = row["health"]
        if health in ("HEALTHY", "STARTING"):
            counts["running"] += 1
        elif health == "WAITING":
            counts["waiting"] += 1
        elif health == "COMPLETE":
            counts["complete"] += 1
        elif health in ("FAILED", "ERROR"):
            counts["failed"] += 1
    return counts


def plain_report(rows, errors, slurm_root):
    """Print one noninteractive monitor snapshot."""
    counts = summary(rows)
    print("Slurm root: {}".format(slurm_root))
    print(
        "Running: {running}  Waiting: {waiting}  Complete: {complete}  "
        "Failed: {failed}".format(**counts)
    )
    print(
        "{:<12} {:<20} {:<11} {:<10} {:>7} {:>9} {:>9} {:<10}".format(
            "JOB", "NAME", "STATE", "TIME", "CPU", "MEM", "COVER", "HEALTH"
        )
    )
    for row in rows:
        job = row["job"]
        record = row["record"]
        print(
            "{:<12} {:<20} {:<11} {:<10} {:>7} {:>9} {:>9} {:<10}".format(
                job.get("job_id", "")[:12],
                job.get("name", "")[:20],
                job.get("state", "")[:11],
                job.get("elapsed", "")[:10],
                row["cpu"][:7],
                row["memory"][:9],
                coverage_text(record.get("coverage"))[:9],
                row["health"][:10],
            )
        )
    for error in errors:
        print("Warning: {}".format(error))


def add_line(screen, row, text, width, attributes=0):
    """Draw one clipped line while tolerating small terminal windows."""
    try:
        screen.addnstr(row, 0, text, max(0, width - 1), attributes)
    except curses.error:
        pass


def interactive(screen, arguments):
    """Refresh and draw the keyboard-driven terminal screen."""
    curses.curs_set(0)
    screen.nodelay(True)
    selected = 0
    last_refresh = 0.0
    rows = []
    errors = []
    while True:
        now = time.time()
        if now - last_refresh >= arguments.refresh:
            rows, errors = collect_snapshot(arguments.root, arguments.user)
            selected = min(selected, max(0, len(rows) - 1))
            last_refresh = now

        screen.erase()
        height, width = screen.getmaxyx()
        counts = summary(rows)
        add_line(
            screen,
            0,
            "Explore-Bench Slurm monitor | running {running} | waiting {waiting} | "
            "complete {complete} | failed {failed} | q quit | r refresh".format(
                **counts
            ),
            width,
            curses.A_BOLD,
        )
        add_line(screen, 1, "Data and logs: {}".format(arguments.root), width)
        header = "{:<11} {:<18} {:<10} {:<9} {:>6} {:>8} {:>8} {:<9}".format(
            "JOB", "NAME", "STATE", "TIME", "CPU", "MEM", "COVER", "HEALTH"
        )
        add_line(screen, 3, header, width, curses.A_UNDERLINE)

        table_height = max(1, min(len(rows), max(1, height // 2 - 5)))
        start = max(0, selected - table_height + 1)
        visible = rows[start : start + table_height]
        for offset, row in enumerate(visible):
            job = row["job"]
            record = row["record"]
            line = "{:<11} {:<18} {:<10} {:<9} {:>6} {:>8} {:>8} {:<9}".format(
                job.get("job_id", "")[:11],
                job.get("name", "")[:18],
                job.get("state", "")[:10],
                job.get("elapsed", "")[:9],
                row["cpu"][:6],
                row["memory"][:8],
                coverage_text(record.get("coverage"))[:8],
                row["health"][:9],
            )
            attributes = curses.A_REVERSE if start + offset == selected else 0
            add_line(screen, 4 + offset, line, width, attributes)

        detail_row = 5 + table_height
        if rows:
            chosen = rows[selected]
            job = chosen["job"]
            record = chosen["record"]
            add_line(screen, detail_row, "Selected job details", width, curses.A_BOLD)
            details = [
                "Job {} | node/reason {} | CPUs {} | exit {}".format(
                    job.get("job_id", ""),
                    job.get("reason") or job.get("nodes", ""),
                    job.get("cpus", ""),
                    job.get("exit_code", ""),
                ),
                "Data {} of {} | method {} | map {} | sensor {} @ {}".format(
                    record.get("completed", 0),
                    record.get("planned", 0),
                    record.get("method", "-"),
                    record.get("map", "-"),
                    record.get("sensor", "-"),
                    record.get("range", "-"),
                ),
                "Seed {} | steps {} | final coverage {} | stopped by {}".format(
                    record.get("seed", "-"),
                    record.get("steps", "-"),
                    coverage_text(record.get("coverage")),
                    record.get("termination", "-"),
                ),
                "Run: {}".format(record.get("run_directory", "not created yet")),
                "Out: {}".format(chosen["logs"].get("stdout") or "not found"),
                "Err: {}".format(chosen["logs"].get("stderr") or "not found"),
            ]
            for offset, detail in enumerate(details, 1):
                add_line(screen, detail_row + offset, detail, width)
            error_row = detail_row + len(details) + 2
            if chosen["error_tail"]:
                add_line(screen, error_row, "Error log tail:", width, curses.A_BOLD)
                for offset, line in enumerate(chosen["error_tail"], 1):
                    add_line(screen, error_row + offset, line.rstrip(), width)
        else:
            add_line(screen, detail_row, "No jobs or result files found.", width)

        if errors:
            add_line(screen, height - 1, "Warning: {}".format(errors[0]), width)
        screen.refresh()

        key = screen.getch()
        if key in (ord("q"), ord("Q")):
            return
        if key in (curses.KEY_DOWN, ord("j")) and rows:
            selected = min(len(rows) - 1, selected + 1)
        elif key in (curses.KEY_UP, ord("k")) and rows:
            selected = max(0, selected - 1)
        elif key in (ord("r"), ord("R")):
            last_refresh = 0.0
        time.sleep(0.1)


def build_parser():
    """Create monitor command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_SLURM_ROOT,
        help="directory containing logs/ and data/ (default: ../slurm)",
    )
    parser.add_argument("--user", default=getpass.getuser())
    parser.add_argument("--refresh", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main():
    """Print one report or start the interactive terminal display."""
    arguments = build_parser().parse_args()
    arguments.root = arguments.root.expanduser().resolve()
    if arguments.refresh <= 0:
        raise SystemExit("--refresh must be positive")
    if arguments.once:
        rows, errors = collect_snapshot(arguments.root, arguments.user)
        plain_report(rows, errors, arguments.root)
        return
    curses.wrapper(interactive, arguments)


if __name__ == "__main__":
    main()
