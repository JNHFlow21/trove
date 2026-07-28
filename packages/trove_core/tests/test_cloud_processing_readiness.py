from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from trove_core.providers.config import ASR_SECRET_NAME, VISION_SECRET_NAME
from trove_core.providers.readiness import CloudReadinessInput, check_cloud_processing_readiness, redaction_issues


REPO_ROOT = Path.home() / 'Trove' / 'trove'


class CloudProcessingReadinessTests(unittest.TestCase):
    def test_ready_report_can_pass_without_secret_values_when_secret_names_exist(self):
        with tempfile.TemporaryDirectory() as d:
            report = check_cloud_processing_readiness(
                CloudReadinessInput(
                    repo_root=REPO_ROOT,
                    vault_root=Path(d),
                    cost_cap_rmb=10.0,
                    estimated_cost_rmb=1.25,
                    doc_verification_date=date.today().isoformat(),
                    provider_docs_ok=True,
                    selected_account_ids=['acct-a', 'acct-b'],
                    discovered_account_ids=['acct-a', 'acct-b'],
                    undecryptable_account_ids=['acct-c'],
                    coverage_gap_account_ids=['acct-c'],
                    require_clean_git=False,
                ),
                env={},
                agent_switch_names={ASR_SECRET_NAME, VISION_SECRET_NAME},
            )
        data = report.to_dict()
        self.assertTrue(data['ok'], data['hard_stops'])
        self.assertFalse(data['provider_status']['providers']['asr']['secret_value_included'])
        self.assertEqual(data['scope']['coverage_gap_count'], 1)

    def test_hard_stops_cover_cost_docs_scope_and_redaction(self):
        with tempfile.TemporaryDirectory() as d:
            report = check_cloud_processing_readiness(
                CloudReadinessInput(
                    repo_root=REPO_ROOT,
                    vault_root=Path(d),
                    selected_account_ids=['acct-a'],
                    discovered_account_ids=['acct-a', 'acct-x'],
                    undecryptable_account_ids=['acct-y'],
                    coverage_gap_account_ids=[],
                    redaction_probe='provider_payload transcript photo.jpg ' + '/Users/' + 'alice/Desktop/private.db',
                    require_clean_git=False,
                    require_usage_store=False,
                ),
                env={},
                agent_switch_names=set(),
            )
        codes = {issue['code'] for issue in report.to_dict()['hard_stops']}
        self.assertIn('provider_docs_stale_or_missing', codes)
        self.assertIn('cost_estimate_missing', codes)
        self.assertIn('asr_secret_missing', codes)
        self.assertIn('vision_secret_missing', codes)
        self.assertIn('unauthorized_account_in_scope', codes)
        self.assertIn('undecryptable_without_coverage_gap', codes)
        self.assertIn('redaction_probe_failed', codes)
        warning_codes = {issue['code'] for issue in report.to_dict()['warnings']}
        self.assertIn('asr_cost_unlimited', warning_codes)

    def test_redaction_detector_flags_private_material(self):
        issues = redaction_issues('raw transcript Authorization: abcdefghijklmnopqrst file.m4a ' + '/Users/' + 'me/Downloads/x')
        self.assertIn('private_path', issues)
        self.assertIn('secret_or_token', issues)
        self.assertIn('provider_payload_or_transcript_marker', issues)
        self.assertIn('media_filename', issues)


if __name__ == '__main__':
    unittest.main()
