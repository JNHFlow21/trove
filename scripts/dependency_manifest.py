#!/usr/bin/env python3
"""Build a conservative local import manifest for migration allowlists."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)
import argparse
import ast
import json
from pathlib import Path


def module_to_path(root: Path, module: str) -> Path | None:
    candidate = root / (module.replace('.', '/') + '.py')
    if candidate.exists():
        return candidate
    package = root / module.replace('.', '/') / '__init__.py'
    if package.exists():
        return package
    return None


def imports_for_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def build_manifest(root: Path, files: list[Path]) -> dict:
    root = root.resolve()
    entries = []
    missing = []
    for file in files:
        path = (root / file).resolve() if not file.is_absolute() else file.resolve()
        local_imports = []
        for mod in sorted(imports_for_file(path)):
            mod_path = module_to_path(root, mod)
            if mod_path:
                rel = mod_path.relative_to(root).as_posix()
                local_imports.append(rel)
                if rel not in [p.as_posix() for p in files]:
                    missing.append({"file": path.relative_to(root).as_posix(), "import": mod, "required_path": rel})
        entries.append({"file": path.relative_to(root).as_posix(), "local_imports": local_imports})
    return {"root": str(root), "entries": entries, "missing_local_imports": missing, "ok": not missing}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args(argv)
    manifest = build_manifest(Path(args.root), [Path(p) for p in args.files])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
