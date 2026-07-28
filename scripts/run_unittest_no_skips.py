#!/usr/bin/env python3
"""Run named unittest suites and fail when any test is skipped."""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime

ensure_project_runtime(__file__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('tests', nargs='+')
    args = parser.parse_args(argv)
    suite = unittest.defaultTestLoader.loadTestsFromNames(args.tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if result.skipped:
        names = ', '.join(test.id() for test, _reason in result.skipped)
        print(f'unexpected skipped tests: {names}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
