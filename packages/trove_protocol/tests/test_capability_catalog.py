from __future__ import annotations

from dataclasses import replace
import json
import unittest
from pathlib import Path

from trove_protocol.capabilities import (
    CATALOG,
    CatalogValidationError,
    STANDARD_MCP_TOOLS,
    catalog_snapshot,
    capabilities_for_pack,
    validate_catalog,
    validate_input,
)


class CapabilityCatalogTests(unittest.TestCase):
    def test_standard_pack_is_the_reviewed_twelve_tool_surface(self):
        self.assertEqual(len(STANDARD_MCP_TOOLS), 12)
        self.assertEqual(STANDARD_MCP_TOOLS, {
            'trove_capabilities', 'trove_resolve', 'trove_recall', 'trove_group_summary',
            'trove_search', 'trove_context', 'trove_profile', 'trove_files_list',
            'trove_media_fetch', 'trove_media_enrich', 'trove_operation_status',
            'trove_operation_continue',
        })
        self.assertLess(
            set(spec.mcp_name for spec in capabilities_for_pack('standard')),
            set(spec.mcp_name for spec in capabilities_for_pack('operations')),
        )
        self.assertLess(
            set(spec.mcp_name for spec in capabilities_for_pack('operations')),
            set(spec.mcp_name for spec in capabilities_for_pack('admin')),
        )
        tool_surface = [
            {'name': spec.mcp_name, 'description': spec.description, 'inputSchema': spec.input_schema}
            for spec in capabilities_for_pack('standard')
        ]
        compact = json.dumps(tool_surface, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.assertLessEqual(len(compact), 24 * 1024)
        self.assertLessEqual((len(compact) * 11 + 39) // 40, 6000)

    def test_names_routes_and_source_neutrality_are_validated(self):
        validate_catalog(CATALOG)
        duplicate = (*CATALOG, replace(CATALOG[0], capability_id=CATALOG[1].capability_id))
        with self.assertRaises(CatalogValidationError):
            validate_catalog(duplicate)
        branded = tuple(
            replace(spec, description='source-specific brand') if index == 0 else spec
            for index, spec in enumerate(CATALOG)
        )
        branded = (replace(branded[0], description='wechat-specific command'), *branded[1:])
        with self.assertRaisesRegex(CatalogValidationError, 'source-specific'):
            validate_catalog(branded)

    def test_scoped_inputs_and_citations_require_account_id(self):
        for spec in CATALOG:
            if not spec.scoped:
                continue
            with self.subTest(capability=spec.capability_id):
                self.assertIn('account_id', spec.input_schema['properties'])
                citation = spec.output_schema['$defs']['citation']
                self.assertIn('account_id', citation['required'])

    def test_unknown_input_fields_fail_closed(self):
        recall = next(spec for spec in CATALOG if spec.mcp_name == 'trove_recall')
        validate_input(recall, {'conversation_id': 'fixture', 'limit': 5})
        with self.assertRaisesRegex(CatalogValidationError, 'unknown field'):
            validate_input(recall, {'conversation_id': 'fixture', 'forged': True})

    def test_snapshot_is_source_neutral_and_has_no_experimental_action_surface(self):
        encoded = json.dumps(catalog_snapshot(), ensure_ascii=False).lower()
        self.assertNotIn('wechat', encoded)
        self.assertNotIn('trove_action_', encoded)

    def test_reply_surface_is_admin_read_only_and_has_no_decision_authority(self):
        reply = [
            spec for spec in CATALOG
            if spec.capability_id.startswith('trove.reply_')
        ]
        self.assertEqual({
            spec.capability_id for spec in reply
        }, {
            'trove.reply_status',
            'trove.reply_reviews',
            'trove.reply_activity',
        })
        self.assertTrue(all(spec.pack == 'admin' for spec in reply))
        self.assertTrue(all(spec.risk == 'read' for spec in reply))
        self.assertTrue(all(spec.replay_policy == 'read' for spec in reply))

    def test_protocol_package_has_no_core_or_provider_import(self):
        root = Path('packages/trove_protocol/trove_protocol')
        source = '\n'.join(path.read_text(encoding='utf-8') for path in root.rglob('*.py'))
        self.assertNotIn('trove_core', source)
        self.assertNotIn('import trove_provider', source)
        self.assertNotIn('from trove_provider', source)

    def test_catalog_matches_the_reviewed_v1_golden(self):
        expected = json.loads(Path('tests/golden/trove_protocol_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(catalog_snapshot(), expected)


if __name__ == '__main__':
    unittest.main()
