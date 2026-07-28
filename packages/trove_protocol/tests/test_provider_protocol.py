from __future__ import annotations

import unittest

from trove_protocol.provider import (
    ProviderContractError,
    ProviderManifest,
    ProtocolRange,
    canonical_provider_schema_hash,
    validate_provider_hello,
)
from trove_protocol.target import AmbiguousTargetError, TargetRef, resolve_unique_display_target


class ProviderProtocolTests(unittest.TestCase):
    def test_protocol_ranges_must_intersect(self):
        self.assertTrue(ProtocolRange('trove/1', 'trove/1').intersects(ProtocolRange('trove/1', 'trove/2')))
        self.assertFalse(ProtocolRange('trove/2', 'trove/2').intersects(ProtocolRange('trove/1', 'trove/1')))

    def test_manifest_is_strict_secret_name_only_metadata(self):
        manifest = ProviderManifest.from_dict({
            'provider_id': 'fixture-source', 'version': '1.0.0',
            'protocol_min': 'trove/1', 'protocol_max': 'trove/1',
            'capabilities': ['read', 'media'], 'source_types': ['fixture'],
            'secret_names': ['FIXTURE_SECRET_NAME'],
            'package_sha256': 'a' * 64,
            'schema_sha256': canonical_provider_schema_hash(),
            'resource_class': 'bounded-local',
        })
        self.assertEqual(manifest.secret_names, ('FIXTURE_SECRET_NAME',))
        with self.assertRaises(ProviderContractError):
            ProviderManifest.from_dict(manifest.to_dict() | {'secret_value': 'forbidden'})

    def test_hello_must_match_manifest_and_schema_hash(self):
        manifest = ProviderManifest.from_dict({
            'provider_id': 'fixture-source', 'version': '1.0.0',
            'protocol_min': 'trove/1', 'protocol_max': 'trove/1',
            'capabilities': ['read'], 'source_types': ['fixture'],
            'secret_names': [], 'package_sha256': 'a' * 64,
            'schema_sha256': canonical_provider_schema_hash(),
            'resource_class': 'bounded-local',
        })
        validate_provider_hello(manifest, {
            'provider_id': 'fixture-source', 'protocol': 'trove/1',
            'schema_sha256': canonical_provider_schema_hash(), 'version': '1.0.0',
        })
        with self.assertRaises(ProviderContractError):
            validate_provider_hello(manifest, {
                'provider_id': 'other', 'protocol': 'trove/1',
                'schema_sha256': canonical_provider_schema_hash(), 'version': '1.0.0',
            })

    def test_target_ref_never_uses_display_name_as_identity(self):
        first = TargetRef(
            provider_id='fixture-source', account_id='account-a', kind='contact',
            stable_id='contact-1', conversation_id='conversation-1', peer_id='peer-1',
            display_hints={'name': 'Same Name'},
        )
        second = TargetRef(
            provider_id='fixture-source', account_id='account-b', kind='contact',
            stable_id='contact-2', conversation_id='conversation-2', peer_id='peer-2',
            display_hints={'name': 'Same Name'},
        )
        self.assertNotEqual(first.identity_key, second.identity_key)
        self.assertNotIn('Same Name', first.identity_key)
        with self.assertRaises(AmbiguousTargetError):
            resolve_unique_display_target([first, second], 'Same Name')

    def test_manifest_rejects_duplicate_capability_and_forged_source_type(self):
        base = {
            'provider_id': 'fixture-source', 'version': '1.0.0',
            'protocol_min': 'trove/1', 'protocol_max': 'trove/1',
            'capabilities': ['read'], 'source_types': ['fixture'],
            'secret_names': [], 'package_sha256': 'a' * 64,
            'schema_sha256': canonical_provider_schema_hash(),
            'resource_class': 'bounded-local',
        }
        with self.assertRaises(ProviderContractError):
            ProviderManifest.from_dict(base | {'capabilities': ['read', 'read']})
        with self.assertRaises(ProviderContractError):
            ProviderManifest.from_dict(base | {'source_types': ['../forged']})


if __name__ == '__main__':
    unittest.main()
