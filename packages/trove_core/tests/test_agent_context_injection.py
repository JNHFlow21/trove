from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.agent_tools.context import build_agent_context
from trove_core.agent_tools.tools import model_status, source_inventory, vector_status
from trove_core.wechat.indexer import index_fixture_vault


class AgentContextInjectionTests(unittest.TestCase):
    def test_agent_context_excludes_raw_content_and_private_paths(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            ctx = build_agent_context(d)
            text = str(ctx)
            self.assertFalse(ctx['raw_content_included'])
            self.assertNotIn('价格太高', text)
            self.assertNotIn(d, text)
            self.assertIn('counts', ctx['vault'])

    def test_agent_tool_status_surfaces_runtime_without_content(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn('zvec', vector_status(d))
            self.assertIn('expected_dimensions', model_status(model_path=None))
            src = Path(d) / 'decrypted' / 'current'
            src.mkdir(parents=True)
            (src / 'message.db').write_text('x')
            manifest = source_inventory([str(src)])
            self.assertIn('canonical_source_ids', manifest)
            self.assertNotIn(str(src), str(manifest))
