from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import unittest

from scripts.generate_capability_reference import render_capability_reference


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DOCS = (
    'README.md',
    'docs/architecture.md',
    'docs/architecture/application-boundary.md',
    'docs/architecture/reply-runtime.md',
    'docs/mcp.md',
    'docs/capability-map.md',
    'docs/protocol.md',
    'docs/provider-sdk.md',
    'docs/operations.md',
    'docs/providers/wechat.md',
    'docs/testing.md',
    'docs/release.md',
)
HISTORICAL_PREFIXES = ('docs/perf/', 'docs/plans/', 'docs/release-notes/')
REMOVED_ACTIVE_DOCS = frozenset({
    'docs/adr/0002-incremental-vector-ledger-and-watcher-recovery.md',
    'docs/adr/README.md',
    'docs/automation-ops.md',
    'docs/decrypt-key-capture.md',
    'docs/evaluation.md',
    'docs/incident-response.md',
    'docs/operator-acceptance-autosync.md',
    'docs/performance-audit-2026-07-10.md',
    'docs/person-profile.md',
    'docs/retrieval-evaluation-closeout.md',
    'docs/vault-mutation-coordination.md',
    'docs/writer-marker-recovery.md',
})
LEGACY_TOKENS = (
    'trove-api', 'trove_api', 'trove-wechat', 'trove_cli.main',
    'chat-recall', 'customer-profile', 'person-profile', 'files-list',
    'web_console', 'web console', 'npm ', 'localhost:', '127.0.0.1:',
    'trove up', 'trove down', 'trove ps',
)
LINK = re.compile(r'(?<!!)\[[^\]]+\]\(([^)]+)\)')
FENCE = re.compile(r'```(?:bash|sh)\n(.*?)```', re.DOTALL)


def repository_snapshot() -> dict[str, str]:
    needed = set(ACTIVE_DOCS) | {'skills/manifest.json'}
    needed.update({
        f'skills/{name}/SKILL.md'
        for name in (
            'trove-recall', 'trove-group-summary', 'trove-search',
            'trove-profile', 'trove-file-recall', 'trove-media-enrichment',
            'trove-moments', 'trove-triage',
        )
    })
    listed = subprocess.run(
        ['git', '-C', str(ROOT), 'ls-files', '-z'],
        check=False, capture_output=True,
    )
    if listed.returncode:
        result: dict[str, str] = {}
        for path in ROOT.rglob('*'):
            if not path.is_file() or '.git' in path.parts:
                continue
            name = path.relative_to(ROOT).as_posix()
            result[name] = path.read_text(encoding='utf-8') if name in needed else ''
        return result
    result: dict[str, str] = {}
    for raw in listed.stdout.split(b'\0'):
        if not raw:
            continue
        name = raw.decode('utf-8')
        result[name] = ''
        if name not in needed:
            continue
        shown = subprocess.run(
            ['git', '-C', str(ROOT), 'show', f':{name}'],
            check=False, capture_output=True,
        )
        if shown.returncode == 0:
            result[name] = shown.stdout.decode('utf-8', errors='replace')
    return result


def _relative_link(source: str, target: str) -> str:
    clean = target.split('#', 1)[0]
    if source == 'README.md':
        return PurePosixPath(clean).as_posix()
    return (PurePosixPath(source).parent / clean).as_posix()


class DocumentationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = repository_snapshot()
        cls.active = {name: cls.files[name] for name in ACTIVE_DOCS}

    def test_formal_documentation_is_minimal_and_legacy_free(self):
        self.assertTrue(set(ACTIVE_DOCS) <= set(self.files))
        self.assertFalse(REMOVED_ACTIVE_DOCS & set(self.files))
        formal = {
            name for name in self.files
            if name.endswith('.md')
            and (name == 'README.md' or name.startswith('docs/'))
            and not name.startswith(HISTORICAL_PREFIXES)
        }
        self.assertEqual(formal, set(ACTIVE_DOCS))
        for name, text in self.active.items():
            checked = text.lower()
            with self.subTest(name=name):
                self.assertNotIn('/users/', text.lower())
                for token in LEGACY_TOKENS:
                    self.assertNotIn(token, checked)

    def test_readme_is_ordered_artifact_walkthrough(self):
        readme = self.active['README.md']
        headings = ('## Install', '## Connect MCP', '## First call', '## Failure path')
        positions = [readme.index(item) for item in headings]
        self.assertEqual(positions, sorted(positions))
        for token in (
            'trove_runtime-1.0.0', 'trove version', 'trove-mcp',
            'trove --vault "$TROVE_VAULT_ROOT" doctor', 'trove_recall',
            'error.retryable', 'approval_required',
        ):
            self.assertIn(token, readme)
        for source_detail in ('cd ', './scripts/', '-m trove_', 'packages/'):
            self.assertNotIn(source_detail, readme)

    def test_generated_reference_is_byte_identical_to_catalog(self):
        self.assertEqual(
            self.active['docs/capability-map.md'],
            render_capability_reference(),
        )

    def test_links_resolve_inside_formal_or_historical_docs(self):
        for source, text in self.active.items():
            for target in LINK.findall(text):
                with self.subTest(source=source, target=target):
                    self.assertFalse(target.startswith(('http:', 'https:', '/')))
                    resolved = _relative_link(source, target)
                    self.assertIn(resolved, self.files)

    def test_command_examples_use_current_entrypoints(self):
        allowed = {
            'python3', 'export', 'mkdir', 'chmod', 'trove', 'trove-mcp',
            'agent-switch', './scripts/trove-python',
        }
        for name, content in self.active.items():
            for block in FENCE.findall(content):
                for raw in block.splitlines():
                    line = raw.strip()
                    if not line or line.startswith(('#', '--')):
                        continue
                    first = line.split()[0]
                    if first.startswith('"$HOME/'):
                        first = 'python3'
                    with self.subTest(name=name, line=line):
                        self.assertIn(first, allowed)

    def test_approval_is_human_only_without_bypass(self):
        combined = '\n'.join(self.active.values()).lower()
        self.assertNotIn('--yes', combined)
        self.assertIn('approval decision is not an agent capability', combined)
        self.assertIn('controlling terminal', combined)
        self.assertIn('trove --vault "$trove_vault_root" operator approve approval_id', combined)
        reference = self.active['docs/capability-map.md']
        self.assertNotIn('operator approve', reference.lower())
        self.assertNotIn('approval decision', reference.lower())

    def test_protocol_skill_and_document_budgets(self):
        for name, text in self.active.items():
            versions = set(re.findall(r'trove/[0-9]+', text))
            with self.subTest(name=name):
                self.assertLessEqual(versions, {'trove/1'})
        self.assertLessEqual(len(self.active['README.md'].encode()), 4_000)
        self.assertLessEqual(len(self.active['docs/mcp.md'].encode()), 4_000)

        manifest = json.loads(self.files['skills/manifest.json'])
        names = {item['name'] for item in manifest['skills']}
        self.assertEqual(names, {
            'trove-recall', 'trove-group-summary', 'trove-search',
            'trove-profile', 'trove-file-recall', 'trove-media-enrichment',
            'trove-moments', 'trove-triage',
        })
        for name in names:
            skill = self.files[f'skills/{name}/SKILL.md']
            self.assertLessEqual(len(skill.encode()), 5_000)
            self.assertIn('trove/1', skill)


if __name__ == '__main__':
    unittest.main()
