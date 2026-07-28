"""TROVE v1 CLI entry point."""

from .v1_main import main, run

__all__ = ['main', 'run']


if __name__ == '__main__':
    raise SystemExit(main())
