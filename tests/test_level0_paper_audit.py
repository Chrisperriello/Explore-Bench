import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = REPOSITORY_ROOT / "hpc" / "jobs" / "level0_paper_audit"
ANALYSIS_ROOT = REPOSITORY_ROOT / "analysis" / "level0_paper_audit"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load_module("level0_audit_generator", JOB_ROOT / "generate_tasks.py")
collector = load_module("level0_audit_collector", JOB_ROOT / "collect_episode.py")
runner = load_module("level0_audit_runner", JOB_ROOT / "run_task.py")
analyzer = load_module("level0_audit_analyzer", ANALYSIS_ROOT / "analyze.py")


class FakeEnvironment:
    resolution = 0.5
    num_agents = 2
    gt_map = np.zeros((2, 3), dtype=np.uint8)
    built_map = [
        np.array([[0, 0, 0], [0, 205, 205]], dtype=np.uint8),
        np.array([[205, 205, 0], [0, 205, 205]], dtype=np.uint8),
    ]
    path_log = [[1, 2], [1]]
    agent_pos = [[0, 0], [1, 0]]
    agent_yaw = [0.0, 1.0]


class Level0PaperAuditTests(unittest.TestCase):
    def setUp(self):
        self.suite = generator.load_suite(
            JOB_ROOT / "level0_paper_audit_suite.json", REPOSITORY_ROOT
        )
        self.tables = generator.build_rows(self.suite)

    def test_frozen_design_has_four_paired_twenty_task_tables(self):
        self.assertEqual(set(self.tables), set(generator.EXPECTED_TREATMENTS))
        reference = None
        for rows in self.tables.values():
            self.assertEqual(len(rows), 20)
            self.assertEqual([row["task_id"] for row in rows], list(range(20)))
            paired = [
                (row["task_id"], row["method"], row["seed"], row["map_path"])
                for row in rows
            ]
            if reference is None:
                reference = paired
            else:
                self.assertEqual(paired, reference)

    def test_treatments_change_only_sensor_type_range_and_role(self):
        control = self.tables["omnidirectional_3p5"][0]
        ignored = {"task_key", "treatment", "treatment_role", "sensor_type", "sensor_range"}
        for treatment, rows in self.tables.items():
            row = rows[0]
            for name, value in control.items():
                if name not in ignored:
                    self.assertEqual(row[name], value, (treatment, name))

    def test_independent_metric_conventions_do_not_collapse(self):
        values = collector.measurement(FakeEnvironment(), np.array([1.0, 2.0]))

        self.assertEqual(values["agent_coverage_cells"], [4, 2])
        self.assertEqual(values["team_coverage_cells"], 4)
        self.assertAlmostEqual(values["paper_sigma_area_m2"], 0.25)
        self.assertAlmostEqual(values["released_sigma_ratio"], 1.0 / 6.0)
        self.assertAlmostEqual(values["paper_overlap_ratio_total_terrain"], 0.0)
        self.assertAlmostEqual(values["released_overlap_ratio"], 0.0)
        self.assertAlmostEqual(values["collector_overlap_ratio_total"], 1.0 / 3.0)

    def test_stuck_check_uses_pre_move_position(self):
        env = FakeEnvironment()
        env.last_goals = np.array([[0, 0], [1, 0]])
        env.positions_before_goal = np.array([[0, 1], [1, 1]])
        self.assertFalse(collector.all_agents_stuck(env))
        env.positions_before_goal = env.last_goals.copy()
        self.assertTrue(collector.all_agents_stuck(env))

    def test_completed_attempts_are_immutable_and_retry_uses_next_number(self):
        with tempfile.TemporaryDirectory() as directory:
            task_directory = Path(directory)
            run1 = task_directory / "run1"
            run1.mkdir()
            for filename in ("episodes.csv", "steps.csv", "agents.csv"):
                (run1 / filename).write_text("header\n", encoding="utf-8")
            (run1 / "manifest.json").write_text(
                json.dumps({"status": "complete", "episodes_completed": 1}),
                encoding="utf-8",
            )

            self.assertEqual(runner.complete_runs(task_directory), [run1])
            self.assertEqual(runner.next_run_directory(task_directory).name, "run2")

    def test_old_regression_layout_with_experiment_directory_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "legacy_3p5"
                / "task_000000"
                / "experiment"
                / "run1"
            )
            path.mkdir(parents=True)
            with (path / "episodes.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["method", "seed"])
                writer.writeheader()
                writer.writerow({"method": "cost", "seed": 1})

            rows = analyzer.previous_rows(Path(directory))
            self.assertEqual(rows, [{"method": "cost", "seed": "1"}])

    def test_verdict_closeness_uses_median_not_range_and_excludes_runtime(self):
        stats = {
            "n": 3,
            "mean": 200.0,
            "median": 200.0,
            "minimum": 100.0,
            "maximum": 300.0,
        }
        self.assertFalse(analyzer.compatible("Ttopo", 134.0, stats))
        self.assertFalse(analyzer.comparable_candidate("wall_seconds_at_90"))
        self.assertTrue(analyzer.comparable_candidate("decision_step_at_90"))

    def test_report_writes_markdown_and_html(self):
        summary = []
        for method in analyzer.METHODS:
            summary.append(
                {
                    "treatment": "omnidirectional_3p5",
                    "method": method,
                    "runs": 10,
                    "coverage_99_successes": 10,
                    "final_team_coverage_ratio_median": 0.99,
                    "coverage_90_step_median": 100,
                    "coverage_99_step_median": 125,
                }
            )
        verdicts = [
            {
                "method": method,
                "metric": metric,
                "paper_value": 1,
                "primary_median": 1,
                "best_interpretation": "primary",
                "best_median": 1,
                "verdict": "supported",
                "explanation": "within tolerance",
            }
            for method in analyzer.METHODS
            for metric in ("Ttopo", "Ttotal", "sigma", "r_o")
        ]
        checks = [{"check": "regression", "status": "pass", "detail": "ok"}]
        pair_summary = [
            {
                "sensor_range": 3.5,
                "method": "cost",
                "pairs": 10,
                "median_delta_final_coverage": 0,
                "complete_99_step_pairs": 10,
                "median_delta_99_step": 0,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            overall = analyzer.write_report(
                output, [], summary, pair_summary, verdicts, checks
            )
            self.assertTrue((output / "audit_report.md").is_file())
            html_text = (output / "audit_report.html").read_text(encoding="utf-8")

        self.assertTrue(overall.startswith("PASS"))
        self.assertIn("<!doctype html>", html_text)


if __name__ == "__main__":
    unittest.main()
