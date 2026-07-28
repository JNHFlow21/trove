# Testing

Use the tiered runner from the source checkout:

```bash
./scripts/trove-python scripts/check.py unit
./scripts/trove-python scripts/check.py contract
./scripts/trove-python scripts/check.py package
./scripts/trove-python scripts/check.py e2e
./scripts/trove-python scripts/check.py perf
```

`release` runs those tiers in order. Contract tests bind CLI, MCP, Skills,
protocol, public surface, documentation, and generated reference to the same
catalog. Package tests build and inspect both wheels. End-to-end tests install
the artifact into a fresh isolated environment with no source path, exercise
CLI and MCP, independently install and remove the Provider, and drill activation
rollback.

Tests use synthetic fixtures. Real-Vault checks emit redacted hashes, counts,
timings, and booleans only, and their proof stays in the private Vault. Run the
privacy scan after every generated artifact or receipt.

Regenerate and verify the reference:

```bash
./scripts/trove-python scripts/generate_capability_reference.py
./scripts/trove-python scripts/generate_capability_reference.py --check
```
