import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HPC_TASKS = REPOSITORY_ROOT / "hpc" / "tasks"
HPC_MONITOR = REPOSITORY_ROOT / "hpc" / "monitor"
sys.path.insert(0, str(HPC_TASKS))
sys.path.insert(0, str(HPC_MONITOR))

from generate_level0_tasks import build_rows, load_suite, write_rows  # noqa: E402
from run_level0_task import collector_command, complete_runs  # noqa: E402
from slurm_monitor import discover_log_job_ids, load_data_records, log_paths  # noqa: E402


class Level0TaskTableTests(unittest.TestCase):
    def setUp(self):
        self.config_path = HPC_TASKS / "level0_suite.json"
        self.suite = load_suite(self.config_path, REPOSITORY_ROOT)

    def test_default_suite_creates_sixty_matched_configurations(self):
        rows = build_rows(self.suite)

        self.assertEqual(len(rows), 60)
        self.assertEqual([row["task_id"] for row in rows], list(range(60)))
        self.assertEqual(rows[0]["task_key"], "cost__loop__seed_0001")
        self.assertEqual(rows[-1]["task_key"], "mmpf__comb2__seed_0005")
        self.assertNotIn("sensor", rows[0])

    def test_written_table_round_trips_all_rows(self):
        rows = build_rows(self.suite)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tasks.csv"
            write_rows(output, rows)
            with output.open(encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle))

        self.assertEqual(len(loaded), 60)
        self.assertEqual(loaded[17]["task_id"], "17")
        self.assertEqual(loaded[17]["coverage_thresholds"], "0.9;0.98;0.99")


class Level0TaskRunnerTests(unittest.TestCase):
    def test_sensor_is_the_only_sensor_specific_command_setting(self):
        task = build_rows(load_suite(HPC_TASKS / "level0_suite.json", REPOSITORY_ROOT))[0]
        with tempfile.TemporaryDirectory() as directory:
            task_directory = Path(directory)
            original = collector_command(
                REPOSITORY_ROOT, task, "omnidirectional", task_directory
            )
            four_beam = collector_command(
                REPOSITORY_ROOT, task, "four_beam", task_directory
            )

        differing = [
            (left, right)
            for left, right in zip(original, four_beam)
            if left != right
        ]
        self.assertEqual(differing, [("omnidirectional", "four_beam")])

    def test_only_complete_single_episode_runs_are_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            task_directory = Path(directory)
            run_directory = task_directory / "example" / "run1"
            run_directory.mkdir(parents=True)
            (run_directory / "episodes.csv").write_text("header\nrow\n", encoding="utf-8")
            (run_directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "episodes_planned": 1,
                        "episodes_completed": 1,
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(complete_runs(task_directory), [run_directory])


class Level0MonitorTests(unittest.TestCase):
    def test_nested_array_logs_and_running_task_status_are_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_directory = root / "logs" / "level0" / "test" / "original"
            data_directory = root / "data" / "level0" / "test" / "original" / "task_000003"
            log_directory.mkdir(parents=True)
            data_directory.mkdir(parents=True)
            output = log_directory / "level0-original-123_3.out"
            error = log_directory / "level0-original-123_3.err"
            output.write_text("running\n", encoding="utf-8")
            error.write_text("", encoding="utf-8")
            (data_directory / "task_status.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "sensor": "omnidirectional",
                        "slurm_job_id": "123_3",
                        "task": {
                            "method": "cost",
                            "map_path": "datasets/corner.pgm",
                            "seed": "4",
                            "sensor_range": "3.5",
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(discover_log_job_ids(root / "logs"), {"123_3"})
            self.assertEqual(
                log_paths(root / "logs", "123_3"),
                {"stdout": str(output), "stderr": str(error)},
            )
            record = load_data_records(root / "data")["123_3"]
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["method"], "cost")
            self.assertEqual(record["map"], "corner.pgm")

            self.assertEqual(
                log_paths(root / "logs", "22171_[8-59%8]"),
                {"stdout": "", "stderr": ""},
            )


if __name__ == "__main__":
    unittest.main()
