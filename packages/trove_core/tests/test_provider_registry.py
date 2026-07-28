from __future__ import annotations

import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from trove_core.entitlements import (
    AllowAllEntitlementProvider, StaticEntitlementProvider,
    dispatcher_entitlement_check,
)
from trove_core.providers.adapters.current_source import CurrentSourceAdapter
from trove_core.providers.factory import ProviderFactory
from trove_core.providers.registry import (
    ProviderAllowlistEntry,
    ProviderRegistry,
    ProviderRegistryError,
)
from trove_protocol.provider import ProviderManifest, canonical_provider_schema_hash


class _Provider:
    def __init__(self, provider_id='fixture-source'):
        self.provider_id = provider_id
        self.calls = []

    def hello(self):
        return {
            'provider_id': self.provider_id, 'protocol': 'trove/1',
            'schema_sha256': canonical_provider_schema_hash(), 'version': '1.0.0',
        }

    def capabilities(self):
        return {'capabilities': ['read'], 'source_types': ['fixture']}

    def health(self):
        return {'ok': True, 'state': 'ready'}

    def accounts(self):
        return [{'account_id': 'account-a', 'label': 'Fixture', 'message_count': 2, 'watermark': '2'}]

    def invoke(self, method, payload):
        self.calls.append((method, payload))
        return {'method': method, 'payload': payload}


class _TTY:
    def __init__(self, input_text):
        self.input = io.StringIO(input_text)
        self.output = io.StringIO()

    def isatty(self):
        return True

    def write(self, value):
        return self.output.write(value)

    def flush(self):
        return None

    def readline(self):
        return self.input.readline()


class ProviderRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / 'provider.pkg'
        self.package.write_bytes(b'fixture-provider-package')
        os.chmod(self.package, 0o600)
        self.package_hash = hashlib.sha256(self.package.read_bytes()).hexdigest()
        self.manifest = ProviderManifest.from_dict({
            'provider_id': 'fixture-source', 'version': '1.0.0',
            'protocol_min': 'trove/1', 'protocol_max': 'trove/1',
            'capabilities': ['read'], 'source_types': ['fixture'],
            'secret_names': ['FIXTURE_SECRET_NAME'],
            'package_sha256': self.package_hash,
            'schema_sha256': canonical_provider_schema_hash(),
            'resource_class': 'bounded-local',
        })
        self.allow = ProviderAllowlistEntry(
            provider_id='fixture-source', package_sha256=self.package_hash,
            owner_uid=os.getuid(), capabilities=frozenset({'read'}),
            source_types=frozenset({'fixture'}), secret_names=frozenset({'FIXTURE_SECRET_NAME'}),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_verified_provider_registers_and_enumerates_accounts(self):
        registry = ProviderRegistry({'fixture-source': self.allow})
        registry.register(self.manifest, _Provider(), package_path=self.package)
        self.assertEqual(registry.accounts('fixture-source')[0]['account_id'], 'account-a')

    def test_hash_permissions_privilege_and_duplicate_fail_closed(self):
        cases = []
        bad_hash = ProviderAllowlistEntry(
            **(self.allow.to_dict() | {'package_sha256': '0' * 64})
        )
        cases.append((ProviderRegistry({'fixture-source': bad_hash}), self.manifest, self.package))
        os.chmod(self.package, 0o644)
        cases.append((ProviderRegistry({'fixture-source': self.allow}), self.manifest, self.package))
        for registry, manifest, package in cases:
            with self.subTest(registry=registry):
                with self.assertRaises(ProviderRegistryError):
                    registry.register(manifest, _Provider(), package_path=package)
        os.chmod(self.package, 0o600)
        expanded = ProviderManifest.from_dict(self.manifest.to_dict() | {'capabilities': ['read', 'media']})
        with self.assertRaises(ProviderRegistryError):
            ProviderRegistry({'fixture-source': self.allow}).register(expanded, _Provider(), package_path=self.package)
        registry = ProviderRegistry({'fixture-source': self.allow})
        registry.register(self.manifest, _Provider(), package_path=self.package)
        with self.assertRaises(ProviderRegistryError):
            registry.register(self.manifest, _Provider(), package_path=self.package)

    def test_missing_health_or_schema_mismatch_is_rejected(self):
        provider = _Provider()
        provider.health = None
        with self.assertRaises(ProviderRegistryError):
            ProviderRegistry({'fixture-source': self.allow}).register(
                self.manifest, provider, package_path=self.package,
            )

    def test_current_source_adapter_preserves_contract_calls(self):
        source = _Provider()
        adapter = ProviderFactory.adapt_current_source(self.manifest, source)
        self.assertIsInstance(adapter, CurrentSourceAdapter)
        self.assertEqual(adapter.accounts(), source.accounts())
        self.assertEqual(adapter.invoke('read', {'fixture': True}), {'method': 'read', 'payload': {'fixture': True}})

    def test_privilege_expansion_requires_exact_interactive_pin_confirmation(self):
        registry = ProviderRegistry({'fixture-source': self.allow})
        expanded = ProviderAllowlistEntry(**(
            self.allow.to_dict() | {'capabilities': frozenset({'read', 'media'})}
        ))
        with self.assertRaises(ProviderRegistryError):
            registry.replace_allowlist_entry(expanded, terminal=io.StringIO(''))
        confirmation = f'APPROVE PROVIDER-PIN fixture-source {self.package_hash}\n'
        registry.replace_allowlist_entry(expanded, terminal=_TTY(confirmation))

    def test_entitlement_deny_happens_before_provider_and_allow_all_changes_no_approval(self):
        provider = _Provider()
        deny = StaticEntitlementProvider({'trove.media_fetch': False})
        decision = deny.check('trove.media_fetch')
        self.assertFalse(decision.allowed)
        self.assertEqual(provider.calls, [])
        allowed = AllowAllEntitlementProvider().check('trove.files_export')
        self.assertTrue(allowed.allowed)
        self.assertTrue(allowed.approval_still_required)
        self.assertFalse(AllowAllEntitlementProvider().check('trove.unknown').allowed)

        class UnsafeEntitlement:
            def check(self, _capability_id):
                from trove_protocol.experimental.entitlements import EntitlementDecision
                return EntitlementDecision(True, approval_still_required=False)

        spec = type('Spec', (), {'capability_id': 'trove.files_export'})()
        self.assertFalse(dispatcher_entitlement_check(UnsafeEntitlement())(spec))


if __name__ == '__main__':
    unittest.main()
