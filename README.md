# TROVE

TROVE is a macOS-only, local-private capability runtime. It returns bounded
cited evidence to external Agents and can optionally run a local Reply Runtime
that generates, reviews, and delivers replies through verified source
Providers. Reply delivery is disabled by default. MCP is the primary Agent
interface and the CLI is the recovery and operator interface.

The product is **TROVE**. WeChat support is an optional source Provider rather
than the product identity.

## Install

From a verified release artifact directory:

```bash
python3 -m venv "$HOME/.local/share/trove/runtime"
"$HOME/.local/share/trove/runtime/bin/pip" install ./trove_runtime-1.0.0-py3-none-any.whl ./trove_provider_*.whl
export PATH="$HOME/.local/share/trove/runtime/bin:$PATH"
trove version
```

Keep the artifact directory and Vault owner-only. Create or select a Vault,
then run the redacted health check. The explicit path avoids hidden discovery.

```bash
export TROVE_VAULT_ROOT="$HOME/Trove/trove-vault"
mkdir -p "$TROVE_VAULT_ROOT"
chmod 700 "$TROVE_VAULT_ROOT"
trove --vault "$TROVE_VAULT_ROOT" doctor
```

## Connect MCP

Register `trove-mcp` through Agent Switch with these arguments:

```text
--pack standard --vault $TROVE_VAULT_ROOT
```

Run `agent-switch doctor` before changing its central configuration and
`agent-switch reconcile` afterward. Do not hand-edit generated client configs.
The standard pack is sufficient for ordinary recall and search.

## First call

Ask the Agent to call `trove_recall`, or use the exact CLI fallback:

```bash
trove --vault "$TROVE_VAULT_ROOT" recall --target "Example person" --limit 50
```

The JSON envelope states `ok`, typed errors, citations, and coverage. Follow an
opaque cursor only when the requested coverage needs another page.

## Failure path

Run `trove --vault "$TROVE_VAULT_ROOT" doctor`. Retry only when
`error.retryable` is true. For `ambiguous_target`, select one returned account.
For `approval_required`, stop: an Agent can request or inspect approval but only
a human at the controlling terminal can decide it.

See [MCP](docs/mcp.md), [operations](docs/operations.md), and the generated
[capability reference](docs/capability-map.md). Provider setup is separate;
see the [installed source Provider](docs/providers/wechat.md). The optional
Reply Runtime architecture and safety model are documented in
[Reply Runtime](docs/architecture/reply-runtime.md).

## Privacy boundary

Real chat databases, exports, media, transcripts, provider payloads, secrets,
logs, and local Vault data do not belong in this repository. Tests and checked-in
evidence must be synthetic or explicitly source-safe. Run the privacy scanner
before every commit.

See [open-source privacy](PRIVACY.md) and
[security policy](SECURITY.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). TROVE is licensed under
[Apache License 2.0](LICENSE).
