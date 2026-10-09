import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIRECTORY = REPOSITORY_ROOT / "analysis" / "level0_baseline"
sys.path.insert(0, str(ANALYSIS_DIRECTORY))

from analyze_level0 import (  # noqa: E402
    build_condition_summaries,
    build_paper_direct_comparison,
    build_paired_runs,
    validate_episode_pairing,
)


def episode(sensor, coverage, success, reached_90=True, reached_99=False):
    return {
        "sensor_variant": sensor,
        "method": "cost",
        "map_name": "square_loop.pgm",
        "team_size": "2",
        "seed": "1",
        "communication_mode": "legacy",
        "success": str(success),
        "termination_reason": "target_coverage" if success else "max_steps",
        "final_coverage_ratio": str(coverage),
        "steps_executed": "400" if success else "1000",
        "team_path_length_m": "100",
        "maximum_agent_path_length_m": "55",
        "overlap_ratio_total": "0.25",
        "coverage_std_area_m2": "2.5",
        "coverage_90_reached": str(reached_90),
        "coverage_90_step": "200" if reached_90 else "",
        "coverage_90_team_path_m": "50" if reached_90 else "",
        "coverage_98_reached": str(success),
        "coverage_98_step": "350" if success else "",
        "coverage_98_team_path_m": "85" if success else "",
        "coverage_99_reached": str(reached_99),
        "coverage_99_step": "400" if reached_99 else "",
        "coverage_99_team_path_m": "100" if reached_99 else "",
    }


class Level0BaselineAnalysisTests(unittest.TestCase):
    def test_pairing_requires_both_sensor_variants(self):
        with self.assertRaisesRegex(ValueError, "not paired"):
            validate_episode_pairing(
                [episode("omnidirectional", 0.99, True, reached_99=True)]
            )

    def test_paired_delta_uses_percentage_points_and_missing_milestones(self):
        rows = [
            episode("omnidirectional", 0.99, True, reached_99=True),
            episode("four_beam", 0.60, False, reached_99=False),
        ]

        paired = build_paired_runs(rows)

        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]["delta_final_coverage_ratio"], -39.0)
        self.assertFalse(paired[0]["both_coverage_99_reached"])
        self.assertEqual(paired[0]["delta_coverage_99_step"], "")

    def test_condition_summary_reports_reach_rate_before_conditional_time(self):
        rows = [
            episode("omnidirectional", 0.99, True, reached_99=True),
            episode("four_beam", 0.60, False, reached_99=False),
        ]

        summaries = build_condition_summaries(rows)
        four = next(row for row in summaries if row["sensor_variant"] == "four_beam")

        self.assertEqual(four["runs"], 1)
        self.assertEqual(four["coverage_99_reached_runs"], 0)
        self.assertEqual(four["coverage_99_reach_rate"], 0.0)
        self.assertEqual(four["coverage_99_step_conditional_mean"], "")

    def test_paper_comparison_separates_absolute_values_from_ranking(self):
        cost = episode("omnidirectional", 0.99, True, reached_99=True)
        cost.update(
            {
                "map_name": "room1_modified.pgm",
                "coverage_90_step": "200",
                "coverage_99_step": "400",
                "coverage_std_area_m2": "2.5",
                "coverage_std_ratio": "0.25",
                "overlap_ratio_total": "0.25",
            }
        )
        mmpf = episode("omnidirectional", 0.99, True, reached_99=True)
        mmpf.update(
            {
                "method": "mmpf",
                "map_name": "room1_modified.pgm",
                "coverage_90_step": "150",
                "coverage_99_step": "300",
                "coverage_std_area_m2": "1.0",
                "coverage_std_ratio": "0.10",
                "overlap_ratio_total": "0.10",
            }
        )

        comparison = build_paper_direct_comparison([cost, mmpf])

        self.assertEqual(len(comparison), 4)
        self.assertTrue(all(row["directional_ranking_matches"] for row in comparison))
        self.assertEqual(comparison[0]["paper_unit"], "s")
        self.assertEqual(comparison[0]["our_unit"], "decisions")
        self.assertAlmostEqual(comparison[3]["our_cost_mean"], 0.25)


if __name__ == "__main__":
    unittest.main()
