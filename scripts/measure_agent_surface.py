#!/usr/bin/env python3
"""Freeze the pre-v1 public surface and measure its schema cost.

The generated inventory is intentionally a historical artifact.  Later cutover
tests consume it to prove that every legacy name was either replaced, made
internal, or deleted; they must not regenerate it from the post-cutover tree.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DISPOSITIONS = frozenset({
    'new_public', 'operations', 'admin', 'internal', 'intentional_delete',
})
PROPOSED_STANDARD_TOOLS = frozenset({
    'trove_capabilities',
    'trove_resolve',
    'trove_recall',
    'trove_group_summary',
    'trove_search',
    'trove_context',
    'trove_profile',
    'trove_files_list',
    'trove_media_fetch',
    'trove_media_enrich',
    'trove_operation_status',
    'trove_operation_continue',
})


def _compact_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')


def estimate_json_tokens(value: Any) -> int:
    """Return the frozen conservative schema estimator used by release gates.

    Four UTF-8 bytes per token is a useful English/JSON baseline.  The 10%
    reserve prevents mixed-language descriptions and punctuation from being
    systematically under-counted without adding a tokenizer dependency.
    """

    return math.ceil(len(_compact_json(value)) * 1.10 / 4.0)


def _git_sha() -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def _with_project_packages() -> None:
    for rel in ('packages/trove_protocol', 'packages/trove_cli', 'packages/trove_mcp'):
        path = str(ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def discover_cli_routes() -> list[str]:
    _with_project_packages()
    from trove_cli.parser import public_routes

    return sorted(' '.join(route.path) for route in public_routes())


def discover_mcp_tools() -> tuple[list[str], list[dict[str, Any]]]:
    _with_project_packages()
    from trove_mcp.catalog_adapter import descriptors_for_pack

    tools = descriptors_for_pack('standard')
    names = sorted(tool.name for tool in tools)
    schemas = [tool.to_dict() for tool in tools]
    return names, schemas


def discover_entry_points() -> list[str]:
    payload = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    return sorted((payload.get('project') or {}).get('scripts') or {})


def discover_legacy_skills() -> list[str]:
    names: set[str] = set()
    for parent in ('.agents/skills', '.codex/skills', '.claude/skills', '.hermes/skills'):
        root = ROOT / parent
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.name.startswith('trove-'):
                names.add(path.name)
    # The bridge may be absent in a clean checkout, but it is still part of the
    # product surface being frozen for cutover.
    names.add('trove-chat-recall')
    return sorted(names)


def _cli_disposition(name: str) -> tuple[str, str]:
    direct = {
        'version': 'trove version', 'doctor': 'trove doctor', 'status': 'trove status',
        'up': 'trove start', 'down': 'trove stop', 'ps': 'trove status',
        'chat-recall': 'trove recall', 'search': 'trove search',
        'context': 'trove context', 'customer-profile': 'trove profile show',
        'person-profile': 'trove profile show', 'person-profile show': 'trove profile show',
        'person-profile plan': 'trove profile build', 'files-list': 'trove files list',
        'media-fetch': 'trove media fetch', 'media transcribe': 'trove media transcribe',
        'media annotate': 'trove media annotate', 'media status': 'trove media status',
        'sync': 'trove sync', 'provider-status': 'trove provider status',
        'approvals': 'trove approval list', 'approval-request': 'trove approval request',
        'observe add': 'trove observe add', 'observe propose': 'trove observe propose',
        'observe list': 'trove observe list', 'files-archive': 'trove files export',
        'list-contacts': 'trove resolve', 'list-conversations': 'trove resolve',
    }
    if name in direct:
        replacement = direct[name]
        disposition = 'operations' if name in {
            'observe add', 'observe propose', 'files-archive', 'media transcribe', 'media annotate',
        } else 'new_public'
        return disposition, replacement
    if name.startswith('profile-enrichment '):
        return 'internal', ''
    if name in {'profile-enrichment', 'observe', 'media', 'person-profile'}:
        return 'internal', ''
    if name in {'ask', 'wiki', 'report', 'conversation-card', 'customer-card'}:
        return 'intentional_delete', ''
    if name in {'approval-decision', 'observe approve'}:
        return 'internal', 'trove operator approve|reject'
    if name.startswith(('decrypt', 'import', 'rebuild-', 'reset-', 'scope-', 'vector-')) or name in {
        'maintain', 'index', 'realtime-sync', 'source-inventory', 'source-manifest',
        'writer-marker-recovery', 'identity-reconcile', 'appmsg-backfill',
        'content-kind-backfill', 'media-reference-backfill', 'cloud-readiness',
        'eval', 'eval-matrix', 'trace', 'process-config', 'schedule',
        'schedule install', 'schedule status', 'schedule uninstall', 'embed-daemon',
        'provider-jobs', 'provider-pricing', 'model-status', 'media-inventory',
        'media-status', 'media understanding-status', 'profile-auto',
        'profile-snapshots', 'profile-snapshots status', 'profile-snapshots list',
        'profile-snapshots get', 'profile-snapshots diff',
    }:
        return 'admin', 'trove repair/provider/sync <catalog-defined-leaf>'
    if name.startswith(('profile-auto ', 'profile-snapshots ')):
        return 'admin', 'trove repair <catalog-defined-leaf>'
    if name in {'media observe', 'image-observe', 'voice-transcribe'}:
        return 'operations', 'trove media enrich'
    if name in {'observe retire', 'person-profile propose'}:
        return 'operations', 'trove observe add'
    return 'intentional_delete', ''


def _mcp_disposition(name: str) -> tuple[str, str]:
    mapping = {
        'trove_chat_recall': 'trove_recall',
        'trove_search': 'trove_search',
        'trove_context': 'trove_context',
        'trove_customer_profile': 'trove_profile',
        'trove_person_profile': 'trove_profile',
        'trove_files_list': 'trove_files_list',
        'trove_media_fetch': 'trove_media_fetch',
        'trove_observe_add': 'trove_observe_add',
        'trove_observe_propose': 'trove_observe_add',
        'trove_files_archive': 'trove_files_export',
        'trove_request_approval': 'trove_approval_request',
        'trove_list_approvals': 'trove_approval_status',
    }
    if name in mapping:
        disposition = 'operations' if name in {
            'trove_observe_add', 'trove_observe_propose', 'trove_files_archive',
            'trove_request_approval', 'trove_list_approvals',
        } else 'new_public'
        return disposition, mapping[name]
    if name.startswith('trove_profile_enrichment_'):
        return 'internal', ''
    if name in {'trove_wiki', 'trove_person_profile_claims_propose'}:
        return 'intentional_delete', ''
    if name.startswith(('trove_list_', 'trove_media_', 'trove_voice_')):
        return 'new_public', 'trove_resolve/trove_media_enrich'
    return 'admin', 'admin catalog capability'


def build_inventory() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cli = discover_cli_routes()
    mcp_names, mcp_schemas = discover_mcp_tools()
    entries = discover_entry_points()
    skills = discover_legacy_skills()
    items: list[dict[str, str]] = []
    for name in cli:
        disposition, replacement = _cli_disposition(name)
        items.append({'surface': 'cli', 'name': name, 'disposition': disposition, 'replacement': replacement})
    for name in mcp_names:
        disposition, replacement = _mcp_disposition(name)
        items.append({'surface': 'mcp', 'name': name, 'disposition': disposition, 'replacement': replacement})
    for name in entries:
        items.append({'surface': 'entry_point', 'name': name, 'disposition': 'new_public', 'replacement': name})
    for name in skills:
        items.append({
            'surface': 'skill', 'name': name, 'disposition': 'new_public',
            'replacement': 'trove-recall' if name == 'trove-chat-recall' else name,
        })
    items.sort(key=lambda item: (item['surface'], item['name']))
    inventory = {
        'schema_version': 1,
        'artifact_type': 'trove_legacy_surface_inventory',
        'captured_git_sha': _git_sha(),
        'dispositions': sorted(ALLOWED_DISPOSITIONS),
        'items': items,
    }
    validate_inventory(inventory)
    return inventory, mcp_schemas


def validate_inventory(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('inventory schema_version must be 1')
    sha = payload.get('captured_git_sha')
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('inventory requires a full captured_git_sha')
    if set(payload.get('dispositions') or ()) != ALLOWED_DISPOSITIONS:
        raise ValueError('inventory dispositions are not the frozen set')
    items = payload.get('items')
    if not isinstance(items, list) or not items:
        raise ValueError('inventory items must be a non-empty list')
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('inventory item must be an object')
        identity = (str(item.get('surface') or ''), str(item.get('name') or ''))
        if not all(identity) or identity in seen:
            raise ValueError('inventory items must have unique surface/name identities')
        seen.add(identity)
        if item.get('disposition') not in ALLOWED_DISPOSITIONS:
            raise ValueError(f'invalid disposition for {identity!r}')
        if not isinstance(item.get('replacement'), str):
            raise ValueError(f'replacement must be a string for {identity!r}')


def load_task_corpus(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open('r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
            required = {'id', 'goal', 'capability', 'input', 'legacy_adapter', 'expected', 'max_calls'}
            if not isinstance(case, dict) or not required.issubset(case):
                raise ValueError(f'{path}:{line_no}: incomplete task case')
            if case['id'] in seen:
                raise ValueError(f'{path}:{line_no}: duplicate id')
            seen.add(case['id'])
            if case['capability'] not in PROPOSED_STANDARD_TOOLS:
                raise ValueError(f'{path}:{line_no}: unknown standard capability')
            if not isinstance(case['input'], dict) or not isinstance(case['expected'], dict):
                raise ValueError(f'{path}:{line_no}: input/expected must be objects')
            if type(case['max_calls']) is not int or not 1 <= case['max_calls'] <= 4:
                raise ValueError(f'{path}:{line_no}: max_calls out of bounds')
            cases.append(case)
    if not cases:
        raise ValueError(f'{path}: no task cases')
    return cases


def write_inventory(path: Path) -> dict[str, Any]:
    inventory, _schemas = build_inventory()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Freeze legacy Agent surfaces and measure MCP schema size.')
    parser.add_argument('--out', default='tests/golden/trove_legacy_surface_inventory.json')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    inventory, schemas = build_inventory()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    result = {
        'inventory_items': len(inventory['items']),
        'tools_list_bytes': len(_compact_json(schemas)),
        'tools_list_estimated_tokens': estimate_json_tokens(schemas),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
