#!/usr/bin/env python3
"""Run the six official Skill jobs through the real CLI and MCP adapters.

The emitted artifact is deliberately content-free: it contains only aggregate
counts, rates, a synthetic-fixture hash, and explicit privacy assertions.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Any, Callable, Mapping, Protocol


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime

ensure_project_runtime(__file__)

from trove_cli.v1_main import run as run_cli  # noqa: E402
from trove_client import TroveClient  # noqa: E402
from trove_core.application.dispatcher import build_default_dispatcher  # noqa: E402
from trove_core.store.repositories import (  # noqa: E402
    MediaAssetLinkRecord,
    MediaAssetRecord,
    MultimodalRepository,
)
from trove_core.store.sqlite_store import SQLiteStore  # noqa: E402
from trove_core.vault.config import VaultConfig  # noqa: E402
from trove_core.wechat.fixture_factory import generate_fixture  # noqa: E402
from trove_core.wechat.indexer import index_fixture_vault  # noqa: E402
from trove_daemon.lifecycle import RuntimeIdentity  # noqa: E402
from trove_daemon.runtime_owner import RuntimeOwner  # noqa: E402
from trove_daemon.server import DaemonServer  # noqa: E402
from trove_mcp.v1_server import create_server  # noqa: E402
from trove_protocol.capabilities import CATALOG_BY_ID  # noqa: E402


REPORT_KEYS = frozenset({
    'schema_version', 'artifact_type', 'ok', 'fixture_sha256', 'clients',
    'summary', 'privacy',
})
CLIENT_KEYS = frozenset({
    'client', 'tasks', 'tasks_succeeded', 'task_success_rate', 'calls',
    'wrong_tool_calls', 'operator_interventions', 'citation_count',
})
SUMMARY_KEYS = frozenset({
    'clients', 'tasks', 'tasks_succeeded', 'task_success_rate', 'calls',
    'wrong_tool_calls', 'operator_interventions', 'citation_count',
})
PRIVACY = {
    'content_included': False,
    'queries_included': False,
    'citations_included': False,
    'account_identifiers_included': False,
    'private_paths_included': False,
    'raw_errors_included': False,
    'secret_values_included': False,
}
PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
)


class Adapter(Protocol):
    name: str

    def call(self, capability: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _fixture_hash() -> str:
    digest = hashlib.sha256()
    fixture = generate_fixture()
    for message in fixture.messages:
        digest.update(json.dumps(
            message.safe_dict(), ensure_ascii=False, sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _create_vault(root: Path) -> VaultConfig:
    index_fixture_vault(root, reset=True)
    config = VaultConfig.resolve(str(root), env={})
    source = root / 'sources/fixture-card.png'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PNG_1X1)
    source.chmod(0o600)
    store = SQLiteStore(config.paths.sqlite_path)
    try:
        message = next(
            row for row in store.all_messages()
            if row['account_id'] == 'acct-work'
            and row['conversation_id'] == 'conv-sales-review'
        )
        repository = MultimodalRepository(store)
        repository.upsert_media_asset(MediaAssetRecord(
            asset_id='asset-product-acceptance-image',
            account_id='acct-work', source_type='message', source_id='fixture-card',
            modality='image', media_type='image', citation=message['citation'],
            path_ref='sources/fixture-card.png', cache_state='cached',
            processing_state='ready',
            metadata={
                'file_name': 'fixture-card.png',
                'message_citation': message['citation'],
                'conversation_id': 'conv-sales-review',
                'account_id': 'acct-work',
            },
        ))
        repository.upsert_media_asset_link(MediaAssetLinkRecord(
            'link-product-acceptance-image', 'asset-product-acceptance-image',
            'acct-work', 'message', message['citation'], 'group_chat', True,
            'fixture',
        ))
    finally:
        store.close()
    return config


def _citations(value: Any) -> list[str]:
    if isinstance(value, str) and value.startswith('trove://'):
        return [value]
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in _citations(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _citations(nested)]
    return []


def _cli_argv(capability: str, payload: Mapping[str, Any], vault: Path) -> list[str]:
    spec = CATALOG_BY_ID[capability]
    route = (
        ('media', 'annotate')
        if capability == 'trove.media_enrich' and payload.get('kind') == 'annotate'
        else spec.cli_route
    )
    argv = [
        '--vault', str(vault), '--request-id', f'acceptance-cli-{secrets.token_urlsafe(12)}',
        *route,
    ]
    fixed = {'kind'} if capability == 'trove.media_enrich' else set()
    for field, value in payload.items():
        if value is None or field in fixed:
            continue
        option = '--account' if field == 'account_id' else '--' + field.replace('_', '-')
        if isinstance(value, bool):
            argv.append(option if value else '--no-' + option[2:])
        elif isinstance(value, list):
            for item in value:
                argv.extend((option, str(item)))
        elif isinstance(value, Mapping):
            argv.extend((option, json.dumps(value, ensure_ascii=False, separators=(',', ':'))))
        else:
            argv.extend((option, str(value)))
    return argv


class CLIAdapter:
    name = 'cli'

    def __init__(self, vault: Path):
        self.vault = vault

    def call(self, capability: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        output = io.StringIO()
        code = run_cli(_cli_argv(capability, payload, self.vault), stdout=output)
        result = json.loads(output.getvalue())
        if code != 0 or result.get('ok') is not True:
            raise RuntimeError('CLI task call failed')
        return result

    def close(self) -> None:
        return None


class MCPAdapter:
    name = 'mcp'

    def __init__(self, vault: Path):
        self.server = create_server(pack='operations', vault=vault)

    def call(self, capability: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = asyncio.run(self.server._tool_manager.call_tool(
            CATALOG_BY_ID[capability].mcp_name, dict(payload),
        ))
        if not isinstance(result, dict) or result.get('ok') is not True:
            raise RuntimeError('MCP task call failed')
        return result

    def close(self) -> None:
        self.server._trove_runtime.close()


@dataclass
class TaskProbe:
    adapter: Adapter
    expected: list[str]
    calls: int = 0
    wrong_tool_calls: int = 0
    citations: int = 0

    def call(self, capability: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        expected = self.expected[self.calls] if self.calls < len(self.expected) else None
        if capability != expected:
            self.wrong_tool_calls += 1
        self.calls += 1
        result = self.adapter.call(capability, payload)
        self.citations += len(_citations(result.get('data')))
        return result


def _recall(probe: TaskProbe) -> bool:
    result = probe.call('trove.recall', {
        'account_id': 'acct-work', 'conversation_id': 'conv-sales-review', 'limit': 20,
    })
    return result.get('coverage', {}).get('state') == 'complete' and probe.citations > 0


def _group_summary(probe: TaskProbe) -> bool:
    result = probe.call('trove.group_summary', {
        'account_id': 'acct-work', 'conversation_id': 'conv-sales-review', 'limit': 20,
    })
    return result.get('coverage', {}).get('state') == 'complete' and probe.citations > 0


def _search_context(probe: TaskProbe) -> bool:
    search = probe.call('trove.search', {
        'query': '客户卡在哪', 'account_id': 'acct-work', 'semantic': 'off', 'limit': 5,
    })
    found = _citations(search.get('data'))
    if not found:
        return False
    context = probe.call('trove.context', {'citation': found[0], 'before': 2, 'after': 2})
    return bool(_citations(context.get('data')))


def _profile(probe: TaskProbe) -> bool:
    result = probe.call('trove.profile', {
        'target': '示例教育', 'account_id': 'acct-work', 'limit': 20,
    })
    return probe.citations > 0 and isinstance(result.get('data'), Mapping)


def _files(probe: TaskProbe) -> bool:
    listed = probe.call('trove.files_list', {
        'account_id': 'acct-work', 'conversation_id': 'conv-sales-review',
        'media_types': ['image'], 'limit': 20,
    })
    files = listed.get('data', {}).get('files') or []
    selected = next(
        (item for item in files if item.get('asset_id') == 'asset-product-acceptance-image'),
        None,
    )
    if selected is None:
        return False
    fetched = probe.call('trove.media_fetch', {
        'citation': selected['citation'], 'allow_remote': False,
    })
    return (
        fetched.get('data', {}).get('status') == 'available'
        and fetched.get('data', {}).get('evidence_ok') is True
        and probe.citations > 0
    )


def _media(probe: TaskProbe) -> bool:
    listed = probe.call('trove.files_list', {
        'account_id': 'acct-work', 'conversation_id': 'conv-sales-review',
        'media_types': ['image'], 'limit': 20,
    })
    files = listed.get('data', {}).get('files') or []
    selected = next(
        (item for item in files if item.get('asset_id') == 'asset-product-acceptance-image'),
        None,
    )
    if selected is None:
        return False
    started = probe.call('trove.media_enrich', {
        'citation': selected['citation'], 'kind': 'annotate',
    })
    operation = started.get('data', {}).get('operation') or {}
    status = probe.call('trove.operation_status', {
        'operation_id': operation.get('operation_id'),
    })
    return status.get('data', {}).get('operation', {}).get('state') == 'pending' and probe.citations > 0


TASKS: tuple[tuple[tuple[str, ...], Callable[[TaskProbe], bool]], ...] = (
    (('trove.recall',), _recall),
    (('trove.group_summary',), _group_summary),
    (('trove.search', 'trove.context'), _search_context),
    (('trove.profile',), _profile),
    (('trove.files_list', 'trove.media_fetch'), _files),
    (('trove.files_list', 'trove.media_enrich', 'trove.operation_status'), _media),
)


def _run_client(adapter: Adapter) -> dict[str, Any]:
    succeeded = calls = wrong = citation_count = 0
    for expected, task in TASKS:
        probe = TaskProbe(adapter, list(expected))
        try:
            succeeded += int(task(probe))
        except Exception:
            pass
        calls += probe.calls
        wrong += probe.wrong_tool_calls
        citation_count += probe.citations
    total = len(TASKS)
    return {
        'client': adapter.name,
        'tasks': total,
        'tasks_succeeded': succeeded,
        'task_success_rate': round(succeeded / total, 6),
        'calls': calls,
        'wrong_tool_calls': wrong,
        'operator_interventions': 0,
        'citation_count': citation_count,
    }


def validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != REPORT_KEYS or report.get('schema_version') != 1:
        raise ValueError('invalid acceptance report shape')
    if report.get('artifact_type') != 'agent_product_acceptance_redacted':
        raise ValueError('invalid acceptance artifact type')
    if report.get('privacy') != PRIVACY:
        raise ValueError('invalid acceptance privacy contract')
    clients = report.get('clients')
    summary = report.get('summary')
    if (
        not isinstance(clients, list) or len(clients) != 2
        or not all(isinstance(item, dict) and set(item) == CLIENT_KEYS for item in clients)
        or not isinstance(summary, dict) or set(summary) != SUMMARY_KEYS
    ):
        raise ValueError('invalid acceptance metrics')
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    forbidden = ('trove://', '/Users/', '客户', 'fixture-card', 'conv-sales', 'acct-work')
    if any(value in serialized for value in forbidden):
        raise ValueError('acceptance report contains evidence or private data')


def run() -> dict[str, Any]:
    os.environ.setdefault('TROVE_DISABLE_AUTO_MODEL_DISCOVERY', '1')
    with tempfile.TemporaryDirectory(prefix='trove-agent-acceptance-') as directory:
        config = _create_vault(Path(directory) / 'vault')
        owner = RuntimeOwner(config, provider_factory=lambda: None)
        dispatcher = build_default_dispatcher(config, runtime_owner=owner)
        identity = RuntimeIdentity.for_vault(config.root)
        server = DaemonServer(identity, dispatcher, idle_timeout=None)
        server.start()
        clients: list[dict[str, Any]] = []
        try:
            for adapter in (CLIAdapter(config.root), MCPAdapter(config.root)):
                try:
                    clients.append(_run_client(adapter))
                finally:
                    adapter.close()
        finally:
            server.stop(timeout=3.0)
            owner.close()
    total_tasks = sum(item['tasks'] for item in clients)
    succeeded = sum(item['tasks_succeeded'] for item in clients)
    report = {
        'schema_version': 1,
        'artifact_type': 'agent_product_acceptance_redacted',
        'ok': bool(
            len(clients) == 2
            and succeeded == total_tasks
            and all(item['wrong_tool_calls'] == 0 for item in clients)
            and all(item['operator_interventions'] == 0 for item in clients)
        ),
        'fixture_sha256': _fixture_hash(),
        'clients': clients,
        'summary': {
            'clients': len(clients),
            'tasks': total_tasks,
            'tasks_succeeded': succeeded,
            'task_success_rate': round(succeeded / total_tasks, 6) if total_tasks else 0.0,
            'calls': sum(item['calls'] for item in clients),
            'wrong_tool_calls': sum(item['wrong_tool_calls'] for item in clients),
            'operator_interventions': sum(item['operator_interventions'] for item in clients),
            'citation_count': sum(item['citation_count'] for item in clients),
        },
        'privacy': dict(PRIVACY),
    }
    validate_report(report)
    return report


def _atomic_write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.agent-acceptance-', dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run content-free Agent product acceptance.')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        report = run()
        if args.out:
            _atomic_write(args.out.expanduser(), report)
            output = {'ok': report['ok'], 'artifact_file': args.out.name}
        else:
            output = report
    except Exception:
        print(json.dumps({'ok': False, 'error_code': 'agent_product_acceptance_failed'}, sort_keys=True))
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
