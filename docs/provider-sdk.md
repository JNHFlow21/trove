# TROVE Provider SDK contract

A Provider is an independently installable distribution that implements the
language-neutral `trove/1` Provider contract. v1 loads a verified Provider in
the daemon process; a future process boundary does not change the methods or
schemas.

## Distribution contract

Declare one entry point in group `trove.providers`. Ship `manifest.json` and
`package.seal` as package data. The manifest fixes:

- Provider id and semantic version;
- minimum and maximum protocol versions;
- requested `read`, `media`, or reserved `action` capability;
- source types and secret **names**;
- package and canonical schema SHA-256 values; and
- bounded resource class.

The runtime reads distribution metadata without import, then verifies owner and
permissions, the release-pinned package hash, seal, protocol range, capability,
source type, and secret-name allowlists. Only then may it import the entry point.
An upgrade drains bounded work, restarts, performs `hello`, capability and health
checks, and rolls back on failure.

## Methods

| Method | Purpose |
| --- | --- |
| `hello()` | exact Provider id, version, protocol, and schema hash |
| `capabilities()` | requested capabilities and source types |
| `health()` | bounded redacted health state |
| `accounts()` | account id, label, record count, and optional watermark |
| `invoke(method, payload)` | one bounded contract operation |

Records must be normalized and bounded before a Vault transaction. Scoped
records and citations retain `account_id`; display names are never identity
keys. Text and metadata remain untrusted evidence.

Secret values never belong in manifests, configuration, argv, environment,
logs, traces, reports, or tests. Declare names only; resolve values through
Agent Switch safe descriptor or stdin transport at the narrow consumer.

Provider decisions cannot waive core entitlement, approval, idempotency,
operation-journal, staging, or writer-coordination rules. A missing or rejected
Provider degrades only its dependent capabilities; existing Vault reads remain
available.

## Action contract

An `action` Provider accepts only a core-prepared intent. The intent binds an
opaque source target, account, expected source watermark, bounded payload
digest, idempotency key, and policy or approval reference. The Provider must
revalidate its exact client, process, account, target, and draft before any
external side effect.

The operation journal moves through `prepared`, `dispatched`, `reconciling`,
then `completed`, `failed`, or `unknown`. Only an exact source-side record plus
remote acknowledgement may complete delivery. A crash or lost response enters
reconciliation; `unknown` is terminal and is never automatically sent again.
