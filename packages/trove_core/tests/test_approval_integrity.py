from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import trove_core.approvals as approvals_module
from trove_core.vault.tracing import TraceStorageError, TraceTimeline

from trove_core.approvals import (
    ApprovalGrant,
    ApprovalManager,
    ApprovalRequired,
    ApprovalValidationError,
    canonical_payload_digest,
    claim_approval_grant,
)


class ApprovalIntegrityTests(unittest.TestCase):
    def _approved(self, manager: ApprovalManager, action: str, danger_class: str, payload: dict):
        record = manager.request(action, danger_class, payload)
        manager.decide(record.approval_id, 'approved')
        return record

    def test_exact_approval_is_bound_and_consumed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            payload = {'backend': 'sqlite', 'max_messages': 12}
            record = self._approved(manager, 'vector_rebuild', 'vector_purge_rebuild', payload)

            grant = manager.require(
                'vector_rebuild',
                'vector_purge_rebuild',
                payload,
                approval_id=record.approval_id,
            )

            self.assertIsInstance(grant, ApprovalGrant)
            self.assertEqual(grant['approval_status'], 'consumed')
            self.assertEqual(manager.load(record.approval_id).status, 'consumed')
            with self.assertRaises(ApprovalRequired) as replay:
                manager.require(
                    'vector_rebuild',
                    'vector_purge_rebuild',
                    payload,
                    approval_id=record.approval_id,
                )
            self.assertEqual(replay.exception.code, 'approval_replayed')

    def test_cross_action_payload_and_danger_class_fail_without_consuming(self):
        cases = [
            ('scope_rebuild', 'destructive_rebuild', {'scope': 'all'}, 'approval_action_mismatch'),
            ('reset_index_cache', 'delete_or_purge', {'scope': 'all'}, 'approval_danger_class_mismatch'),
            ('reset_index_cache', 'destructive_rebuild', {'scope': 'other'}, 'approval_payload_mismatch'),
        ]
        for action, danger_class, payload, code in cases:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                manager = ApprovalManager(directory)
                record = self._approved(
                    manager,
                    'reset_index_cache',
                    'destructive_rebuild',
                    {'scope': 'all'},
                )
                with self.assertRaises(ApprovalRequired) as mismatch:
                    manager.require(action, danger_class, payload, approval_id=record.approval_id)
                self.assertEqual(mismatch.exception.code, code)
                self.assertEqual(manager.load(record.approval_id).status, 'approved')

    def test_pending_rejected_expired_and_malformed_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            pending = manager.request('scope_rebuild', 'destructive_rebuild', {})
            with self.assertRaises(ApprovalRequired) as pending_error:
                manager.consume('scope_rebuild', 'destructive_rebuild', {}, approval_id=pending.approval_id)
            self.assertEqual(pending_error.exception.code, 'approval_pending')

            rejected = manager.request('scope_rebuild', 'destructive_rebuild', {})
            manager.decide(rejected.approval_id, 'rejected')
            with self.assertRaises(ApprovalRequired) as rejected_error:
                manager.consume('scope_rebuild', 'destructive_rebuild', {}, approval_id=rejected.approval_id)
            self.assertEqual(rejected_error.exception.code, 'approval_rejected')

            expired = self._approved(manager, 'scope_rebuild', 'destructive_rebuild', {})
            stale = replace(
                manager.load(expired.approval_id),
                expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace('+00:00', 'Z'),
            )
            manager._write(stale)
            with self.assertRaises(ApprovalRequired) as expired_error:
                manager.consume('scope_rebuild', 'destructive_rebuild', {}, approval_id=expired.approval_id)
            self.assertEqual(expired_error.exception.code, 'approval_expired')

            malformed_id = 'appr-../../outside'
            with self.assertRaises(ApprovalValidationError) as malformed:
                manager.load(malformed_id)
            self.assertEqual(malformed.exception.code, 'malformed_approval_id')

            broken = manager.request('scope_rebuild', 'destructive_rebuild', {})
            manager._path(broken.approval_id).write_text('{broken', encoding='utf-8')
            with self.assertRaises(ApprovalValidationError) as broken_error:
                manager.load(broken.approval_id)
            self.assertEqual(broken_error.exception.code, 'malformed_approval')

    def test_concurrent_consumers_have_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            payload = {'scope': 'private'}
            record = self._approved(manager, 'scope_rebuild', 'destructive_rebuild', payload)

            def consume_once(_index: int) -> str:
                try:
                    manager.consume(
                        'scope_rebuild',
                        'destructive_rebuild',
                        payload,
                        approval_id=record.approval_id,
                    )
                    return 'won'
                except ApprovalRequired as exc:
                    return exc.code

            with ThreadPoolExecutor(max_workers=12) as executor:
                results = list(executor.map(consume_once, range(24)))
            self.assertEqual(results.count('won'), 1)
            self.assertEqual(results.count('approval_replayed'), 23)

    def test_original_payload_digest_is_persisted_but_private_values_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            private_path = str(Path(directory) / 'private-customer' / 'chat.db')
            canary = 'PRIVATE_APPROVAL_CANARY'
            payload = {
                'sources': [private_path],
                'dest_dir': private_path + '-export',
                'secret_name': canary,
                'backend': 'sqlite',
            }
            record = manager.request('full_import', 'full_import', payload)
            persisted = manager._path(record.approval_id).read_text(encoding='utf-8')
            trace = (Path(directory) / 'logs' / 'trace-timeline.redacted.jsonl').read_text(encoding='utf-8')

            self.assertEqual(record.request_hash, canonical_payload_digest(payload))
            self.assertNotIn(private_path, persisted)
            self.assertNotIn(canary, persisted)
            self.assertNotIn(private_path, trace)
            self.assertNotIn(canary, trace)
            self.assertIn(record.request_hash, persisted)
            self.assertNotEqual(json.loads(persisted)['payload']['backend'], 'sqlite')
            self.assertEqual(Path(directory, 'approvals').stat().st_mode & 0o777, 0o700)
            self.assertEqual(manager._path(record.approval_id).stat().st_mode & 0o777, 0o600)
            self.assertEqual(Path(directory, 'logs').stat().st_mode & 0o777, 0o700)
            self.assertEqual(Path(directory, 'logs', 'trace-timeline.redacted.jsonl').stat().st_mode & 0o777, 0o600)

    def test_one_step_requires_explicit_flag_and_issues_authentic_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            with self.assertRaises(ApprovalRequired):
                manager.require('scope_rebuild', 'destructive_rebuild', {})
            grant = manager.require(
                'scope_rebuild',
                'destructive_rebuild',
                {},
                one_step_approval=True,
            )
            grant.validate_for(directory, action='scope_rebuild', danger_class='destructive_rebuild', payload={})
            with self.assertRaises(ValueError):
                ApprovalGrant(
                    approval_id=grant.approval_id,
                    action=grant.action,
                    danger_class=grant.danger_class,
                    request_hash=grant.request_hash,
                    expires_at=grant.expires_at,
                    consumed_at=grant.consumed_at,
                    consumption_id=grant.consumption_id,
                    vault_hash=grant.vault_hash,
                    _seal=object(),
                    _issued_pid=os.getpid(),
                )

            for invalid in ('false', 1, object()):
                with self.subTest(invalid=type(invalid).__name__), self.assertRaises(ApprovalValidationError) as raised:
                    manager.require(
                        'scope_rebuild',
                        'destructive_rebuild',
                        {},
                        one_step_approval=invalid,  # type: ignore[arg-type]
                    )
                self.assertEqual(raised.exception.code, 'invalid_one_step_approval')

    def test_grant_can_be_claimed_by_only_one_core_command(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            payload = {'scope': 'all'}
            grant = manager.require(
                'reset_index_cache',
                'destructive_rebuild',
                payload,
                one_step_approval=True,
            )

            claimed = grant.claim_for(
                directory,
                action='reset_index_cache',
                danger_class='destructive_rebuild',
                payload=payload,
            )
            self.assertIs(claimed, grant)
            with self.assertRaises(ApprovalValidationError) as replay:
                grant.claim_for(
                    directory,
                    action='reset_index_cache',
                    danger_class='destructive_rebuild',
                    payload=payload,
                )
            self.assertEqual(replay.exception.code, 'approval_grant_replayed')

    def test_copy_and_deepcopy_share_the_one_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {'backend': 'sqlite'}
            grant = ApprovalManager(directory).require(
                'vector_rebuild',
                'vector_purge_rebuild',
                payload,
                one_step_approval=True,
            )
            copies = [grant, copy.copy(grant), copy.deepcopy(grant)]

            def claim(item: ApprovalGrant) -> str:
                try:
                    item.claim_for(
                        directory,
                        action='vector_rebuild',
                        danger_class='vector_purge_rebuild',
                        payload=payload,
                    )
                    return 'won'
                except ApprovalValidationError as exc:
                    return exc.code

            with ThreadPoolExecutor(max_workers=24) as executor:
                results = list(executor.map(claim, [copies[index % 3] for index in range(24)]))
            self.assertEqual(results.count('won'), 1)
            self.assertEqual(results.count('approval_grant_replayed'), 23)

    def test_replaced_grant_fields_are_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            grant = ApprovalManager(directory).require(
                'scope_rebuild',
                'destructive_rebuild',
                {'scope': 'private'},
                one_step_approval=True,
            )
            tampered_grant = replace(grant, action='reset_index_cache')
            with self.assertRaises(ApprovalValidationError) as tampered:
                tampered_grant.claim_for(
                    directory,
                    action='reset_index_cache',
                    danger_class='destructive_rebuild',
                    payload={'scope': 'private'},
                )
            self.assertEqual(tampered.exception.code, 'invalid_grant')

    def test_slots_and_module_claim_reject_instance_method_shadowing(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {'scope': 'private'}
            grant = ApprovalManager(directory).require(
                'scope_rebuild',
                'destructive_rebuild',
                payload,
                one_step_approval=True,
            )
            self.assertFalse(hasattr(grant, '__dict__'))
            with self.assertRaises(TypeError):
                class InvalidGrantSubclass(ApprovalGrant):
                    pass
            object.__setattr__(grant, 'action', 'reset_index_cache')
            with self.assertRaises(ApprovalValidationError) as tampered:
                claim_approval_grant(
                    grant,
                    directory,
                    action='reset_index_cache',
                    danger_class='destructive_rebuild',
                    payload=payload,
                )
            self.assertEqual(tampered.exception.code, 'invalid_grant')

    def test_str_subclasses_and_omitted_payload_cannot_bypass_binding(self):
        class AlwaysEqual(str):
            def __eq__(self, _other):
                return True

            def __ne__(self, _other):
                return False

        with tempfile.TemporaryDirectory() as directory:
            payload = {'scope': 'private'}
            grant = ApprovalManager(directory).require(
                'scope_rebuild',
                'destructive_rebuild',
                payload,
                one_step_approval=True,
            )
            with self.assertRaises((ValueError, ApprovalValidationError)):
                forged = replace(grant, action=AlwaysEqual('reset_index_cache'))
                forged.claim_for(
                    directory,
                    action='reset_index_cache',
                    danger_class='destructive_rebuild',
                    payload=payload,
                )
            with self.assertRaises(ApprovalValidationError) as omitted:
                grant.claim_for(
                    directory,
                    action='scope_rebuild',
                    danger_class='destructive_rebuild',
                )
            self.assertEqual(omitted.exception.code, 'grant_payload_mismatch')

    @unittest.skipUnless(hasattr(os, 'fork'), 'fork is required')
    def test_grant_inherited_by_forked_child_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {'scope': 'private'}
            grant = ApprovalManager(directory).require(
                'scope_rebuild',
                'destructive_rebuild',
                payload,
                one_step_approval=True,
            )
            read_fd, write_fd = os.pipe()
            child = os.fork()
            if child == 0:
                os.close(read_fd)
                try:
                    grant.claim_for(
                        directory,
                        action='scope_rebuild',
                        danger_class='destructive_rebuild',
                        payload=payload,
                    )
                    result = 'won'
                except ApprovalValidationError as exc:
                    result = exc.code
                os.write(write_fd, result.encode('ascii'))
                os.close(write_fd)
                os._exit(0)
            os.close(write_fd)
            child_result = os.read(read_fd, 128).decode('ascii')
            os.close(read_fd)
            os.waitpid(child, 0)
            self.assertEqual(child_result, 'cross_process_grant')
            grant.claim_for(
                directory,
                action='scope_rebuild',
                danger_class='destructive_rebuild',
                payload=payload,
            )

    def test_expiry_is_rechecked_inside_atomic_claim_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            record = self._approved(manager, 'scope_rebuild', 'destructive_rebuild', {})
            near_expiry = replace(
                manager.load(record.approval_id),
                expires_at=(datetime.now(timezone.utc) + timedelta(milliseconds=150)).isoformat().replace('+00:00', 'Z'),
            )
            manager._write(near_expiry)
            grant = manager.consume('scope_rebuild', 'destructive_rebuild', {}, approval_id=record.approval_id)
            state = approvals_module._issued_grant_state(grant)
            outcome: list[str] = []
            state.lock.acquire()
            try:
                worker = threading.Thread(target=lambda: outcome.append(self._claim_code(grant, directory)))
                worker.start()
                time.sleep(0.2)
            finally:
                state.lock.release()
            worker.join(timeout=2)
            self.assertEqual(outcome, ['approval_grant_expired'])

    @staticmethod
    def _claim_code(grant: ApprovalGrant, directory: str) -> str:
        try:
            grant.claim_for(
                directory,
                action='scope_rebuild',
                danger_class='destructive_rebuild',
                payload={},
            )
            return 'won'
        except ApprovalValidationError as exc:
            return exc.code

    def test_authority_inode_replacement_cannot_mint_two_grants(self):
        with tempfile.TemporaryDirectory() as directory:
            manager_a = ApprovalManager(directory)
            manager_b = ApprovalManager(directory)
            payload = {'scope': 'private'}
            record = self._approved(manager_a, 'scope_rebuild', 'destructive_rebuild', payload)
            loaded = threading.Event()
            resume = threading.Event()
            original = manager_a._load_unlocked

            def paused_load(approval_id, lease):
                value = original(approval_id, lease)
                loaded.set()
                resume.wait(timeout=3)
                return value

            manager_a._load_unlocked = paused_load  # type: ignore[method-assign]

            def consume(manager):
                try:
                    manager.consume(
                        'scope_rebuild',
                        'destructive_rebuild',
                        payload,
                        approval_id=record.approval_id,
                    )
                    return 'won'
                except (ApprovalRequired, ApprovalValidationError) as exc:
                    return exc.code

            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(consume, manager_a)
                self.assertTrue(loaded.wait(timeout=2))
                authority = manager_a._lock_path(record.approval_id)
                saved_authority = manager_a.dir / '.saved-authority'
                os.link(authority, saved_authority)
                authority.unlink()
                second = executor.submit(consume, manager_b)
                second_result = second.result(timeout=3)
                authority.unlink()
                os.replace(saved_authority, authority)
                resume.set()
                first_result = first.result(timeout=3)
            self.assertEqual([first_result, second_result].count('won'), 1)
            self.assertIn('approval_replayed', [first_result, second_result])

    def test_unknown_payload_fields_and_values_are_marker_only(self):
        with tempfile.TemporaryDirectory() as directory:
            canary = 'PRIVATE_UNKNOWN_APPROVAL_CANARY'
            numeric_canary = 15551234567
            manager = ApprovalManager(directory)
            record = manager.request(
                'full_import',
                'full_import',
                {
                    'reason': canary,
                    'nested': {'memo': canary, 'customer_number': numeric_canary},
                    canary: [canary],
                },
            )
            persisted = manager._path(record.approval_id).read_text(encoding='utf-8')
            trace = Path(directory, 'logs', 'trace-timeline.redacted.jsonl').read_text(encoding='utf-8')
            self.assertNotIn(canary, persisted)
            self.assertNotIn(canary, trace)
            self.assertNotIn(str(numeric_canary), persisted)
            self.assertNotIn(str(numeric_canary), trace)

    def test_trace_failure_after_durable_consume_does_not_lose_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            payload = {'scope': 'private'}
            record = self._approved(manager, 'scope_rebuild', 'destructive_rebuild', payload)
            trace_path = Path(directory, 'logs', 'trace-timeline.redacted.jsonl')
            trace_path.unlink()
            outside = Path(directory, 'outside-trace')
            outside.write_text('unchanged\n', encoding='utf-8')
            outside.chmod(0o600)
            os.link(outside, trace_path)

            grant = manager.consume(
                'scope_rebuild',
                'destructive_rebuild',
                payload,
                approval_id=record.approval_id,
            )
            self.assertIsInstance(grant, ApprovalGrant)
            self.assertEqual(manager.load(record.approval_id).status, 'consumed')
            self.assertEqual(outside.read_text(encoding='utf-8'), 'unchanged\n')

    def test_authority_symlink_is_typed_and_has_no_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            manager.dir.mkdir(mode=0o700)
            outside = Path(directory, 'outside-authority')
            outside.write_text('unchanged', encoding='utf-8')
            outside.chmod(0o644)
            manager._lock_path('appr-' + 'b' * 16).symlink_to(outside)
            with self.assertRaises(ApprovalValidationError) as raised:
                manager.request('scope_rebuild', 'destructive_rebuild', {})
            self.assertEqual(raised.exception.code, 'approval_storage_identity_changed')
            self.assertEqual(outside.stat().st_mode & 0o777, 0o644)
            self.assertEqual(outside.read_text(encoding='utf-8'), 'unchanged')

    def test_trace_tail_handles_utf8_boundary_and_malformed_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = TraceTimeline(directory)
            trace.start('sync', {'count': 1})
            path = Path(directory, 'logs', 'trace-timeline.redacted.jsonl')
            line = json.dumps({'trace_id': 'trace-' + 'a' * 16, 'stage': 'sync', 'status': 'ok', 'created_at': '2026-01-01T00:00:00Z', 'payload': {'field': '中文'}} , ensure_ascii=False).encode('utf-8') + b'\n'
            with path.open('ab') as handle:
                while handle.tell() <= 5 * 1024 * 1024 + 32:
                    handle.write(line)
            rows = trace.list(limit=3)
            self.assertEqual(len(rows), 3)
            with self.assertRaises(TraceStorageError) as malformed:
                trace.append('sync', 'ok', {'count': 1 << 20000})
            self.assertEqual(malformed.exception.code, 'invalid_trace_event')

    def test_non_object_payloads_and_malformed_record_types_are_typed(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            for invalid in ([], '', 0, False, (), set()):
                with self.subTest(invalid=type(invalid).__name__), self.assertRaises(ApprovalValidationError) as raised:
                    manager.request('scope_rebuild', 'destructive_rebuild', invalid)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.code, 'invalid_payload')

            for field, invalid in (
                ('request_hash', 123),
                ('danger_class', []),
                ('status', []),
                ('action', []),
            ):
                with self.subTest(field=field):
                    record = manager.request('scope_rebuild', 'destructive_rebuild', {})
                    raw = json.loads(manager._path(record.approval_id).read_text(encoding='utf-8'))
                    raw[field] = invalid
                    manager._path(record.approval_id).write_text(json.dumps(raw), encoding='utf-8')
                    manager._path(record.approval_id).chmod(0o600)
                    with self.assertRaises(ApprovalValidationError) as malformed:
                        manager.load(record.approval_id)
                    self.assertEqual(malformed.exception.code, 'malformed_approval')

    def test_hardlinked_authority_is_rejected_before_chmod_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            manager.dir.mkdir(mode=0o700)
            outside = Path(directory, 'outside.txt')
            outside.write_text('unchanged', encoding='utf-8')
            outside.chmod(0o644)
            os.link(outside, manager._lock_path('appr-' + 'a' * 16))

            with self.assertRaises(ApprovalValidationError) as raised:
                manager.request('scope_rebuild', 'destructive_rebuild', {})
            self.assertEqual(raised.exception.code, 'approval_storage_identity_changed')
            self.assertEqual(outside.read_text(encoding='utf-8'), 'unchanged')
            self.assertEqual(outside.stat().st_mode & 0o777, 0o644)

    def test_insecure_approval_directory_permissions_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ApprovalManager(directory)
            with patch('trove_core.approvals.os.fchmod', side_effect=OSError('synthetic chmod failure')):
                with self.assertRaises(ApprovalValidationError) as raised:
                    manager.request('scope_rebuild', 'destructive_rebuild', {})
            self.assertEqual(raised.exception.code, 'approval_storage_permissions')


if __name__ == '__main__':
    unittest.main()
