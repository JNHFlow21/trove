from __future__ import annotations

import ast
import inspect
from pathlib import Path
import unittest

from trove_core.agent_tools import tools as agent_tools
from trove_core.application import repositories
from trove_core.application import queries as application_queries
from trove_cli import v1_main as cli_main
from trove_mcp import v1_server as mcp_server


_MCP_SERVER_PATH = Path(mcp_server.__file__)


class ApplicationBoundaryArchitectureTests(unittest.TestCase):
    def test_migrated_agent_query_and_command_slices_only_delegate(self) -> None:
        query_functions = (
            'search',
            'fetch_context',
            'fetch_conversation_context',
            'list_contacts',
            'list_moments',
            'list_favorites',
            'list_conversations',
            'files_list',
        )
        command_functions = (
            'import_contacts',
            'import_moments',
            'import_favorites',
            'scope_rebuild',
            'sync',
            'maintain',
            'start_import',
            'reset_index_cache',
            'vector_index',
        )
        forbidden = ('SQLiteStore', '._filter_row', 'ContextService', 'build_search_engine(')
        for name in query_functions:
            with self.subTest(query=name):
                source = inspect.getsource(getattr(agent_tools, name))
                self.assertIn('TroveQueries', source)
                self.assertFalse(any(token in source for token in forbidden), source)
        for name in command_functions:
            with self.subTest(command=name):
                source = inspect.getsource(getattr(agent_tools, name))
                self.assertIn('TroveCommands', source)
                self.assertFalse(any(token in source for token in forbidden), source)

    def test_protocol_modules_do_not_reintroduce_migrated_leaf_bypasses(self) -> None:
        cli_source = inspect.getsource(cli_main)
        mcp_source = _MCP_SERVER_PATH.read_text(encoding='utf-8')
        forbidden = (
            'execute_full_import(',
            'execute_reset_index_cache(',
            'execute_scope_rebuild(',
            'execute_vector_mutation(',
            'index_vectors(',
            'run_sync(',
            'run_maintain(',
            '._filter_row(',
        )
        for label, source in (('cli', cli_source), ('mcp', mcp_source)):
            with self.subTest(adapter=label):
                for token in forbidden:
                    self.assertNotIn(token, source)
        self.assertNotIn('SQLiteStore', mcp_source)

    def test_vertical_repository_consumers_cannot_call_private_store_methods(self) -> None:
        source = inspect.getsource(repositories)
        tree = ast.parse(source)
        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == 'self'
                and owner.attr == 'store'
                and node.func.attr.startswith('_')
            ):
                violations.append(f'{node.lineno}:{node.func.attr}')
        self.assertEqual(violations, [])
        self.assertNotIn('def __getattr__', source)

    def test_no_protocol_adapter_uses_removed_private_filter(self) -> None:
        modules = (cli_main, agent_tools)
        for module in modules:
            with self.subTest(module=module.__name__):
                source = Path(module.__file__).read_text(encoding='utf-8')
                self.assertNotIn('._filter_row(', source)
        self.assertNotIn('._filter_row(', _MCP_SERVER_PATH.read_text(encoding='utf-8'))

    def test_every_public_application_query_owns_complete_generation_read(self) -> None:
        tree = ast.parse(Path(application_queries.__file__).read_text(encoding='utf-8'))
        class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'TroveQueries'
        )
        required = {
            'resolve_contact', 'search', 'context', 'conversation_context',
            'list_contacts', 'list_moments', 'list_favorites',
            'list_conversations', 'list_files', 'evidence',
        }
        guarded: set[str] = set()
        for node in class_node.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in required:
                continue
            if any(isinstance(decorator, ast.Name) and decorator.id == '_complete_generation_read' for decorator in node.decorator_list):
                guarded.add(node.name)
        self.assertEqual(guarded, required)

    def test_cli_read_commands_have_no_writable_sqlite_leaf_bypass(self) -> None:
        source = Path(cli_main.__file__).read_text(encoding='utf-8')
        for token in (
            'SQLiteStore',
            'build_wiki_page(',
            'build_customer_card(',
            'build_conversation_card(',
            'build_cited_report(',
        ):
            self.assertNotIn(token, source)


if __name__ == '__main__':
    unittest.main()
