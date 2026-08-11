import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_repository_metrics.py"
SPEC = importlib.util.spec_from_file_location("render_repository_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RepositoryMetricsTests(unittest.TestCase):
    def test_svg_is_white_privacy_safe_and_contains_requested_metrics(self) -> None:
        svg = MODULE.render_svg({
            "repository": "Example/project",
            "created_at": "2026-01-01T00:00:00Z",
            "generated_at": "2026-08-10T00:00:00Z",
            "stars": 3,
            "forks": 2,
            "commits": 42,
            "unique_visitors_14d": 7,
            "views_14d": 9,
            "unique_cloners_14d": 4,
            "clones_14d": 6,
            "starred_at": [
                "2026-02-01T00:00:00Z",
                "2026-04-01T00:00:00Z",
                "2026-07-01T00:00:00Z",
            ],
        })

        self.assertIn("Repository Pulse", svg)
        self.assertIn("Stars over time", svg)
        self.assertIn("Unique visitors", svg)
        self.assertIn("Unique cloners", svg)
        self.assertIn("Total clones", svg)
        self.assertEqual(svg.count('class="metric-period">14d'), 3)
        self.assertIn('fill: #ffffff', svg)
        self.assertIn('class="curve"', svg)
        self.assertNotIn("GITHUB_TOKEN", svg)


if __name__ == "__main__":
    unittest.main()
