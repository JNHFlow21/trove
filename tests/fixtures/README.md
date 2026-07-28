# Synthetic fixtures

TROVE tests generate WeChat-like fixture Vaults at runtime. They include multiple accounts, group chats, private chats, customer-sales blockers, team project decisions, duplicate local IDs across shards, and outgoing/incoming direction markers.

Committed fixtures must be synthetic only. Real `messages.jsonl`, SQLite databases, WAL/SHM files, or decrypted exports are forbidden.
