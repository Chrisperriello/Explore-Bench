import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPOSITORY_ROOT / "analysis" / "paper_consistency"
sys.path.insert(0, str(ANALYSIS_ROOT))

from analyze import build_pairs, build_summaries, find_summary  # noqa: E402


def episode(treatment, method, seed, coverage, success):
    return {
        "treatment": treatment,
        "method": method,
        "seed": str(seed),
        "success": str(success),
        "final_coverage_ratio": str(coverage),
        "steps_executed": "100" if success else "1000",
        "team_path_length_m": "10",
        "coverage_90_reached": str(coverage >= 0.90),
        "coverage_90_step": "50" if coverage >= 0.90 else "",
        "coverage_98_reached": str(coverage >= 0.98),
        "coverage_98_step": "80" if coverage >= 0.98 else "",
        "coverage_99_reached": str(success),
        "coverage_99_step": "100" if success else "",
    }


class PaperConsistencyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for method in ("cost", "mmpf"):
            for seed in range(1, 6):
                self.rows.extend(
                    [
                        episode("legacy_3p5", method, seed, 0.99, True),
                        episode("legacy_7p0", method, seed, 1.0, True),
                        episode("perfect_3p5", method, seed, 0.75, False),
                    ]
                )

    def test_summary_separates_finished_processes_from_success(self):
        summaries = build_summaries(self.rows)
        perfect = find_summary(summaries, "perfect_3p5")

        self.assertEqual(perfect["runs"], 10)
        self.assertEqual(perfect["successes"], 0)
        self.assertEqual(perfect["success_rate"], 0)
        self.assertAlmostEqual(perfect["final_coverage_mean"], 0.75)

    def test_pairs_measure_range_and_information_effects(self):
        pairs = build_pairs(self.rows)

        self.assertEqual(len(pairs), 10)
        self.assertAlmostEqual(pairs[0]["range_effect_coverage"], 0.01)
        self.assertAlmostEqual(pairs[0]["information_effect_coverage"], -0.24)


if __name__ == "__main__":
    unittest.main()
