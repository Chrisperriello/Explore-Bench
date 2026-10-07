import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HPC_MONITOR = REPOSITORY_ROOT / "hpc" / "monitor"
sys.path.insert(0, str(HPC_MONITOR))

from slurm_dashboard import build_snapshot  # noqa: E402


class Level0DashboardTests(unittest.TestCase):
    def test_snapshot_counts_each_task_once_across_both_sensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / "tasks" / "level0" / "demo"
            data_root = root / "data" / "level0" / "demo"
            task_root.mkdir(parents=True)
            tasks = [
                {
                    "task_id": "0",
                    "method": "cost",
                    "map_name": "loop",
                    "seed": "1",
                },
                {
                    "task_id": "1",
                    "method": "mmpf",
                    "map_name": "corner",
                    "seed": "2",
                },
            ]
            with (task_root / "tasks.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("task_id", "method", "map_name", "seed")
                )
                writer.writeheader()
                writer.writerows(tasks)

            original_task = data_root / "original" / "task_000000"
            original_run = original_task / "cost__loop__seed_0001" / "run1"
            original_run.mkdir(parents=True)
            (original_run / "episodes.csv").write_text(
                "final_coverage_ratio,steps_executed,termination_reason\n"
                "0.991,125,target_coverage\n",
                encoding="utf-8",
            )
            (original_task / "task_status.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "completed_at": "2026-01-01T00:02:00+00:00",
                        "run_directory": str(original_run),
                        "slurm_array_job_id": "111",
                        "slurm_array_task_id": "0",
                        "task": {"task_id": "0"},
                    }
                ),
                encoding="utf-8",
            )

            four_running = data_root / "four_beam" / "task_000000"
            four_failed = data_root / "four_beam" / "task_000001"
            four_running.mkdir(parents=True)
            four_failed.mkdir(parents=True)
            (four_running / "task_status.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "slurm_array_job_id": "222",
                        "slurm_array_task_id": "0",
                        "task": {"task_id": "0"},
                    }
                ),
                encoding="utf-8",
            )
            (four_failed / "task_status.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "slurm_array_job_id": "222",
                        "slurm_array_task_id": "1",
                        "task": {"task_id": "1"},
                    }
                ),
                encoding="utf-8",
            )

            def queue_provider(user):
                self.assertEqual(user, "tester")
                return (
                    {
                        "222_0": {
                            "job_id": "222_0",
                            "state": "RUNNING",
                            "elapsed": "00:01:00",
                            "cpus": "1",
                            "nodes": "ada02",
                            "reason": "",
                        },
                        "999_0": {
                            "job_id": "999_0",
                            "state": "RUNNING",
                            "elapsed": "00:01:00",
                            "cpus": "1",
                            "nodes": "other",
                            "reason": "",
                        },
                    },
                    "",
                )

            def usage_provider(jobs):
                self.assertEqual(set(jobs), {"222_0"})
                return (
                    {
                        "222_0": {
                            "average_cpu": "00:00:30",
                            "average_rss": "100M",
                            "maximum_rss": "120M",
                        }
                    },
                    "",
                )

            snapshot = build_snapshot(
                root,
                "demo",
                "tester",
                queue_provider=queue_provider,
                usage_provider=usage_provider,
            )

        self.assertEqual(
            snapshot["totals"],
            {
                "complete": 1,
                "running": 1,
                "waiting": 1,
                "failed": 1,
                "total": 4,
            },
        )
        self.assertEqual(snapshot["groups"]["original"]["array_job_id"], "111")
        self.assertEqual(snapshot["groups"]["four_beam"]["array_job_id"], "222")
        completed = next(row for row in snapshot["rows"] if row["health"] == "complete")
        running = next(row for row in snapshot["rows"] if row["health"] == "running")
        self.assertEqual(completed["coverage"], "0.991")
        self.assertEqual(completed["elapsed"], "00:02:00")
        self.assertEqual(running["cpu"], "50%")
        self.assertEqual(running["memory"], "120M")


if __name__ == "__main__":
    unittest.main()
