from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import tempfile
import time
import unittest

from trove_client import TroveClient, TroveClientError
from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.providers.registry import (
    ProviderAllowlistEntry,
    ProviderRegistry,
    ProviderRegistryError,
)
from trove_core.vault.config import VaultConfig
from trove_daemon.lifecycle import RuntimeIdentity, catalog_identity
from trove_daemon.provider_loader import ProviderLoader
from trove_daemon.runtime_owner import RuntimeOwner
from trove_daemon.server import DaemonServer
from trove_protocol.provider import ProviderManifest
from trove_protocol.codec import MAX_FRAME_BYTES


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'packages/trove_provider_wechat/trove_provider_wechat'


def registry(manifest: ProviderManifest, *, package_hash: str | None = None) -> ProviderRegistry:
    return ProviderRegistry({
        manifest.provider_id: ProviderAllowlistEntry(
            provider_id=manifest.provider_id,
            package_sha256=package_hash or manifest.package_sha256,
            owner_uid=os.getuid(),
            capabilities=frozenset(manifest.capabilities),
            source_types=frozenset(manifest.source_types),
            secret_names=frozenset(manifest.secret_names),
        ),
    })


class ProviderFaultRecoveryTests(unittest.TestCase):
    @staticmethod
    def _read_frame(connection: socket.socket) -> dict:
        header = connection.recv(4)
        if len(header) != 4:
            raise AssertionError('missing response header')
        length = struct.unpack('>I', header)[0]
        body = bytearray()
        while len(body) < length:
            chunk = connection.recv(length - len(body))
            if not chunk:
                raise AssertionError('truncated response frame')
            body.extend(chunk)
        return json.loads(body)

    def test_missing_and_hash_failed_provider_keep_vault_reads_and_typed_action(self):
        manifest = ProviderManifest.from_dict(json.loads((PACKAGE / 'manifest.json').read_text()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            config = VaultConfig.resolve(str(vault), env={})
            config.ensure()
            owner = RuntimeOwner(config, provider_factory=lambda: None)
            try:
                status = owner.status()['provider_load']
                self.assertFalse(status['ok'])
                self.assertTrue(status['pure_vault_read_available'])
                self.assertEqual(status['next_action']['capability'], 'trove.provider_status')

                copied = root / 'tampered-provider'
                shutil.copytree(PACKAGE, copied)
                (copied / 'main.py').write_text(
                    (copied / 'main.py').read_text(encoding='utf-8') + '\n# tampered\n',
                    encoding='utf-8',
                )
                failed = ProviderLoader(
                    registry(manifest), runtime_dir=root / 'provider-runtime',
                ).load(copied, module_name='trove_provider_wechat')
                payload = failed.to_dict()
                self.assertFalse(payload['ok'])
                self.assertEqual(payload['error']['code'], 'provider_package_hash_mismatch')
                self.assertEqual(payload['next_action']['action'], 'repair_or_reinstall_provider')
                self.assertTrue(payload['pure_vault_read_available'])
            finally:
                owner.close()

    def test_protocol_downgrade_and_entitlement_deny_fail_before_handler(self):
        package_bytes = b'provider-protocol-fixture'
        digest = hashlib.sha256(package_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'provider.pkg'
            package.write_bytes(package_bytes)
            package.chmod(0o600)
            manifest = ProviderManifest.from_dict({
                'provider_id': 'fixture-provider', 'version': '1.0.0',
                'protocol_min': 'trove/2', 'protocol_max': 'trove/2',
                'capabilities': ['read'], 'source_types': ['fixture'],
                'secret_names': [], 'package_sha256': digest,
                'schema_sha256': __import__(
                    'trove_protocol.provider', fromlist=['canonical_provider_schema_hash'],
                ).canonical_provider_schema_hash(),
                'resource_class': 'bounded-local',
            })
            allow = ProviderAllowlistEntry(
                'fixture-provider', digest, os.getuid(),
                frozenset({'read'}), frozenset({'fixture'}), frozenset(),
            )
            with self.assertRaisesRegex(ProviderRegistryError, 'protocol range'):
                ProviderRegistry({'fixture-provider': allow}).preflight(
                    manifest, package_path=package,
                )

            config = VaultConfig.resolve(str(Path(directory) / 'vault'), env={})
            config.ensure()
            denied = build_default_dispatcher(
                config, entitlement_check=lambda _spec: False,
            ).dispatch('trove.search', {'query': 'fixture'}, request_id='deny')
            self.assertEqual(denied['error']['code'], 'capability_unavailable')
            self.assertEqual(denied['next']['action'], 'inspect_entitlement')

    def test_slow_provider_request_times_out_without_wedging_daemon(self):
        class Dispatcher:
            slow = True

            def dispatch(self, _capability, _payload, *, request_id, response_budget=None):
                if self.slow:
                    time.sleep(0.2)
                return {
                    'protocol': 'trove/1', 'request_id': request_id,
                    'ok': True, 'data': {'ready': True},
                }

        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            vault.mkdir()
            identity = RuntimeIdentity.for_vault(
                vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
            )
            dispatcher = Dispatcher()
            server = DaemonServer(identity, dispatcher, max_workers=1, max_pending=1, idle_timeout=None)
            server.start()
            client = TroveClient(identity, autostart=None)
            try:
                with self.assertRaises(TroveClientError) as caught:
                    client.call(
                        'trove.provider_status', {}, request_id='slow-provider', timeout=0.05,
                    )
                self.assertEqual(caught.exception.code, 'timeout')
                time.sleep(0.2)
                dispatcher.slow = False
                self.assertTrue(client.call(
                    'trove.capabilities', {}, request_id='after-slow', timeout=1,
                )['ok'])
            finally:
                client.close()
                server.stop(timeout=1)

    def test_bad_frame_and_stale_socket_are_recovered_before_healthy_request(self):
        class Dispatcher:
            def dispatch(self, _capability, _payload, *, request_id, response_budget=None):
                return {
                    'protocol': 'trove/1', 'request_id': request_id,
                    'ok': True, 'data': {'ready': True},
                }

        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            vault.mkdir()
            identity = RuntimeIdentity.for_vault(
                vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
            )
            identity.prepare()
            stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stale.bind(str(identity.socket_path))
            stale.close()
            server = DaemonServer(identity, Dispatcher(), idle_timeout=None)
            server.start()
            try:
                malformed = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                malformed.connect(str(identity.socket_path))
                malformed.sendall(struct.pack('>I', MAX_FRAME_BYTES + 1))
                self.assertEqual(
                    self._read_frame(malformed)['error']['code'], 'frame_too_large',
                )
                malformed.close()
                partial = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                partial.connect(str(identity.socket_path))
                partial.sendall(b'\x00\x00\x00\x20{"half"')
                partial.close()
                with TroveClient(identity, autostart=None) as client:
                    self.assertTrue(client.call(
                        'trove.capabilities', {}, request_id='healthy-after-bad-frame',
                    )['ok'])
            finally:
                server.stop(timeout=1)

    def test_provider_candidate_health_failure_rolls_back_after_bounded_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            config = VaultConfig.resolve(str(Path(directory) / 'vault'), env={})
            config.ensure()
            identity = RuntimeIdentity.for_vault(
                config.root, build_hash='b' * 64, catalog_hash=catalog_identity(),
            )

            def runtime(status: dict):
                owner = RuntimeOwner(config, provider_factory=lambda: None)
                owner.attach_provider_registry(object(), status)
                return owner, DaemonServer(
                    identity, build_default_dispatcher(config, runtime_owner=owner),
                    idle_timeout=None,
                )

            old = {
                'ok': True, 'provider_id': 'fixture-provider',
                'generation': 'generation-old', 'pure_vault_read_available': True,
            }
            candidate = {
                'ok': False,
                'error': {'code': 'provider_candidate_health_failed', 'retryable': False},
                'next_action': {
                    'capability': 'trove.provider_status',
                    'action': 'rollback_previous_provider',
                },
                'pure_vault_read_available': True,
            }
            first_owner, first = runtime(old)
            first.start()
            try:
                with TroveClient(identity, autostart=None) as client:
                    self.assertEqual(client.call(
                        'trove.provider_status', {}, request_id='provider-before-upgrade',
                    )['data']['provider']['generation'], 'generation-old')
            finally:
                self.assertTrue(first.stop(timeout=2))
                first_owner.close()

            candidate_owner, candidate_server = runtime(candidate)
            candidate_server.start()
            try:
                with TroveClient(identity, autostart=None) as client:
                    status = client.call(
                        'trove.provider_status', {}, request_id='candidate-health',
                    )['data']['provider']
                    self.assertFalse(status['ok'])
                    self.assertEqual(status['next_action']['action'], 'rollback_previous_provider')
            finally:
                self.assertTrue(candidate_server.stop(timeout=2))
                candidate_owner.close()

            rollback_owner, rollback = runtime(old)
            rollback.start()
            try:
                with TroveClient(identity, autostart=None) as client:
                    status = client.call(
                        'trove.provider_status', {}, request_id='provider-after-rollback',
                    )['data']['provider']
                    self.assertTrue(status['ok'])
                    self.assertEqual(status['generation'], 'generation-old')
            finally:
                rollback.stop(timeout=2)
                rollback_owner.close()


if __name__ == '__main__':
    unittest.main()
