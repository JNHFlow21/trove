# TROVE release

Release `1.0.0` is the macOS-only `trove/1` line. A release set contains an
exact `trove-runtime` wheel, an independently installable Provider wheel, and a
distribution manifest binding source commit, protocol, catalog, runtime,
Provider, artifact hashes, and privacy flags.

## Build and verify

Use a clean approved commit:

```bash
./scripts/trove-python scripts/check.py release
./scripts/trove-python scripts/build_distribution.py --out "$TROVE_RELEASE_OUT"
./scripts/trove-python scripts/verify_distribution.py "$TROVE_RELEASE_OUT/distribution-manifest.json"
```

The manifest must report `source_dirty=false`. Verification fails on unknown or
missing files, unsafe permissions, hash drift, forbidden package layout, wrong
entry points or dependencies, a missing Provider seal, or privacy flags that do
not explicitly exclude private paths, secrets, and Vault content.

## Promotion

1. Install the set in a fresh environment with no source checkout on its path.
2. Run version, doctor, CLI recall, MCP list/call, independent Provider
   install/remove, upgrade, candidate failure, and rollback drills.
3. Run chaos, security, resource, response-size, and catalog drift gates.
4. Run real-Vault acceptance with redacted evidence outside the repository.
5. Cut over Agent Switch and project Skills only after candidate health passes.
6. Audit installed consumers and prove one daemon per canonical Vault.

Any executable, dependency, protocol, catalog, manifest, or Provider change
invalidates earlier receipts. Documentation-only evidence may follow a frozen
subject only when it does not change generated contract bytes.

Keep real data, source paths, queries, snippets, citations, media, databases,
model caches, provider payloads, command output, and secret values outside the
release artifact. A detected leak stops promotion and triggers credential and
artifact incident handling.
