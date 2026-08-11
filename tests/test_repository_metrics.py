import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_repository_metrics.py"
SPEC = importlib.util.spec_from_file_location("render_repository_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RepositoryMetricsTests(unittest.TestCase):
    def test_actions_uses_dated_aggregate_when_traffic_api_is_forbidden(self) -> None:
        def fake_request(path, _credential, *, accept="application/vnd.github+json"):
            if path == "/repos/Example/project":
                return ({
                    "created_at": "2026-01-01T00:00:00Z",
                    "stargazers_count": 1,
                    "forks_count": 2,
                }, {})
            if path.endswith("/traffic/views"):
                raise MODULE.HTTPError(path, 403, "Forbidden", {}, None)
            if path.endswith("/commits?per_page=1"):
                return ([{"sha": "example"}], {})
            self.fail(f"unexpected request: {path} ({accept})")

        fallback = {
            "unique_visitors_14d": 7,
            "views_14d": 9,
            "unique_cloners_14d": 4,
            "clones_14d": 6,
            "clone_series_14d": [
                {"timestamp": "2026-08-07T00:00:00Z", "count": 2},
                {"timestamp": "2026-08-08T00:00:00Z", "count": 1},
                {"timestamp": "2026-08-09T00:00:00Z", "count": 3},
            ],
            "traffic_as_of": "2026-08-09T00:00:00Z",
        }
        with patch.object(MODULE, "_request_json", side_effect=fake_request):
            snapshot = MODULE.fetch_snapshot("Example/project", "workflow-credential", fallback)

        self.assertFalse(snapshot["traffic_live"])
        self.assertEqual(snapshot["unique_visitors_14d"], 7)
        self.assertEqual(snapshot["unique_cloners_14d"], 4)
        self.assertEqual(sum(item["count"] for item in snapshot["clone_series_14d"]), 6)
        self.assertEqual(snapshot["traffic_as_of"], "2026-08-09T00:00:00Z")

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
            "clone_series_14d": [
                {"timestamp": "2026-08-07T00:00:00Z", "count": 2},
                {"timestamp": "2026-08-08T00:00:00Z", "count": 1},
                {"timestamp": "2026-08-09T00:00:00Z", "count": 3},
            ],
            "traffic_as_of": "2026-08-09T00:00:00Z",
            "traffic_live": False,
        })

        self.assertIn("Repository Pulse", svg)
        self.assertIn("Total clones over time · rolling 14 days", svg)
        self.assertNotIn("Stars over time", svg)
        self.assertIn("Unique visitors", svg)
        self.assertIn("Unique cloners", svg)
        self.assertIn("Total clones", svg)
        self.assertEqual(svg.count('class="metric-period">14d'), 3)
        self.assertIn("GitHub Traffic owner snapshot: 2026-08-09", svg)
        self.assertIn('fill: #ffffff', svg)
        self.assertIn('class="curve"', svg)
        self.assertNotIn("GITHUB_TOKEN", svg)


if __name__ == "__main__":
    unittest.main()
