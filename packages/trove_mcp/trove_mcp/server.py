"""TROVE v1 MCP stdio entry point."""

from .v1_server import SERVER_NAME, create_server, main

__all__ = ['SERVER_NAME', 'create_server', 'main']


if __name__ == '__main__':
    raise SystemExit(main())
