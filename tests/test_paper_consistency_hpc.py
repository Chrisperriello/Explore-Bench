import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = REPOSITORY_ROOT / "hpc" / "jobs" / "paper_consistency"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load_module("paper_consistency_generator", JOB_ROOT / "generate_tasks.py")
runner = load_module("paper_consistency_runner", JOB_ROOT / "run_task.py")


class PaperConsistencyTaskTests(unittest.TestCase):
    def setUp(self):
        self.suite = generator.load_suite(
            JOB_ROOT / "paper_consistency_suite.json", REPOSITORY_ROOT
        )
        self.tables = generator.build_rows(self.suite)

    def test_suite_has_three_matched_ten_task_tables(self):
        self.assertEqual(
            set(self.tables), {"legacy_3p5", "legacy_7p0", "perfect_3p5"}
        )
        for rows in self.tables.values():
            self.assertEqual(len(rows), 10)
            self.assertEqual([row["task_id"] for row in rows], list(range(10)))

        paired_fields = ("task_id", "method", "map_path", "team_size", "seed")
        reference = [
            tuple(row[field] for field in paired_fields)
            for row in self.tables["legacy_3p5"]
        ]
        for treatment in ("legacy_7p0", "perfect_3p5"):
            paired = [
                tuple(row[field] for field in paired_fields)
                for row in self.tables[treatment]
            ]
            self.assertEqual(paired, reference)

    def test_treatments_change_only_the_intended_setting(self):
        legacy = self.tables["legacy_3p5"][0]
        long_range = self.tables["legacy_7p0"][0]
        perfect = self.tables["perfect_3p5"][0]

        ignored = {"task_key", "treatment", "sensor_range", "communication_mode"}
        for key in legacy:
            if key not in ignored:
                self.assertEqual(legacy[key], long_range[key])
                self.assertEqual(legacy[key], perfect[key])
        self.assertEqual(long_range["sensor_range"], 7.0)
        self.assertEqual(long_range["communication_mode"], "legacy")
        self.assertEqual(perfect["sensor_range"], 3.5)
        self.assertEqual(perfect["communication_mode"], "perfect")

    def test_collector_command_contains_treatment_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            command = runner.collector_command(
                REPOSITORY_ROOT,
                self.tables["perfect_3p5"][0],
                Path(directory),
            )

        self.assertEqual(command[command.index("--sensor-types") + 1], "omnidirectional")
        self.assertEqual(command[command.index("--sensor-ranges") + 1], "3.5")
        self.assertEqual(command[command.index("--communication-mode") + 1], "perfect")


if __name__ == "__main__":
    unittest.main()
