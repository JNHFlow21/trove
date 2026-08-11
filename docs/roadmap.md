# TROVE roadmap

TROVE's roadmap is organized around invariants rather than promised dates.
Issues and pull requests should preserve the local-first, bounded-evidence,
typed-protocol, Provider, and human-approval boundaries.

## Current line: `trove/1`

- Maintain stable bounded recall, search, context, profile, file, and media
  contracts.
- Keep the CLI and MCP adapter on the same generated capability catalog.
- Harden source-Provider verification, upgrade/rollback, and privacy gates.
- Improve local indexing quality and latency without changing coverage claims.
- Expand synthetic interoperability and failure-path fixtures.

## Candidate directions

- Additional independently packaged source Providers that satisfy the public
  Provider SDK and trust boundary.
- Clearer contributor tooling for synthetic Vault creation and contract tests.
- Better operator-visible diagnostics, citation inspection, and storage health.
- Reproducible, signed release artifacts after the release gate proves the exact
  source, dependency, protocol, catalog, and Provider set.

## Explicit non-goals

- A hosted service that receives users' private Vault contents.
- Silent upload of chats, contacts, media, embeddings, or diagnostic evidence.
- Ambient send authority for agents or approval decisions through MCP.
- Treating retrieved personal content as trusted instructions.
- Compatibility claims for platforms that are not covered by release gates.

See [Architecture](architecture.md), [Release](release.md), and
[Open-source privacy](../PRIVACY.md) for the invariants behind this roadmap.
