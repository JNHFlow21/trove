#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trove_core.reply.migration import migrate_legacy_reply_runtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Migrate the legacy WeChat reply runner into TROVE shadow mode.',
    )
    parser.add_argument('--vault', required=True)
    parser.add_argument('--legacy-config', required=True)
    parser.add_argument('--legacy-state', required=True)
    parser.add_argument('--work-root')
    args = parser.parse_args()
    kwargs = {}
    if args.work_root:
        kwargs['work_root'] = Path(args.work_root)
    try:
        report = migrate_legacy_reply_runtime(
            Path(args.vault),
            legacy_config_path=Path(args.legacy_config),
            legacy_state_path=Path(args.legacy_state),
            **kwargs,
        )
    except Exception as exc:
        print(json.dumps({
            'ok': False,
            'error_code': str(
                getattr(exc, 'code', 'reply_migration_failed'),
            ),
        }, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
