import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MERGE_DIRECTORY = REPOSITORY_ROOT / "hpc" / "jobs" / "merge"
sys.path.insert(0, str(MERGE_DIRECTORY))

from merge_level0 import merge_collection  # noqa: E402


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_attempt(task_directory, number, status, coverage="0.99"):
    run_directory = task_directory / "experiment" / "run{}".format(number)
    run_directory.mkdir(parents=True)
    manifest = {
        "status": status,
        "episodes_planned": 1,
        "episodes_completed": 1 if status == "complete" else 0,
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    write_csv(
        run_directory / "episodes.csv",
        (
            "episode_id",
            "success",
            "termination_reason",
            "steps_executed",
            "final_coverage_ratio",
            "coverage_90_reached",
            "coverage_90_step",
            "coverage_99_reached",
            "coverage_99_step",
        ),
        [
            {
                "episode_id": "episode_000001",
                "success": "True",
                "termination_reason": "target_coverage",
                "steps_executed": "10",
                "final_coverage_ratio": coverage,
                "coverage_90_reached": "True",
                "coverage_90_step": "7",
                "coverage_99_reached": "True",
                "coverage_99_step": "10",
            }
        ],
    )
    write_csv(run_directory / "steps.csv", ("episode_id", "step"), [{"episode_id": "episode_000001", "step": "0"}])
    write_csv(run_directory / "agents.csv", ("episode_id", "agent_id"), [{"episode_id": "episode_000001", "agent_id": "0"}])
    return run_directory


class Level0MergeTests(unittest.TestCase):
    def create_collection(self, root):
        task_root = root / "tasks" / "level0" / "demo"
        task_root.mkdir(parents=True)
        write_csv(
            task_root / "tasks.csv",
            ("task_id", "task_key", "method", "map_name", "seed"),
            [
                {
                    "task_id": "0",
                    "task_key": "cost__loop__seed_0001",
                    "method": "cost",
                    "map_name": "loop",
                    "seed": "1",
                }
            ],
        )
        return root / "data" / "level0" / "demo"

    def test_merge_selects_newest_complete_retry_and_adds_source_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = self.create_collection(root)
            original_task = data_root / "original" / "task_000000"
            four_task = data_root / "four_beam" / "task_000000"
            write_attempt(original_task, 1, "complete", coverage="0.991")
            write_attempt(four_task, 1, "running", coverage="0.2")
            selected_four = write_attempt(four_task, 2, "complete", coverage="0.875")
            output = root / "combined" / "demo"

            manifest = merge_collection(root, "demo", output)
            with (output / "episodes.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                episodes = list(csv.DictReader(handle))
            with (output / "run_inventory.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                inventory = list(csv.DictReader(handle))

        self.assertEqual(manifest["expected_runs"], 2)
        self.assertEqual(manifest["selected_complete_runs"], 2)
        self.assertEqual(manifest["row_counts"]["episodes.csv"], 2)
        self.assertEqual(len(episodes), 2)
        four_episode = next(row for row in episodes if row["variant"] == "four_beam")
        self.assertEqual(four_episode["source_attempt"], "2")
        self.assertEqual(four_episode["final_coverage_ratio"], "0.875")
        self.assertEqual(four_episode["source_run_directory"], str(selected_four))
        self.assertEqual(len(inventory), 2)

    def test_merge_refuses_incomplete_collection_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = self.create_collection(root)
            write_attempt(
                data_root / "original" / "task_000000", 1, "complete"
            )

            with self.assertRaisesRegex(ValueError, "1 of 2 expected runs"):
                merge_collection(root, "demo", root / "combined" / "demo")


if __name__ == "__main__":
    unittest.main()
