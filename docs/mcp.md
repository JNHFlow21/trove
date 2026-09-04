# TROVE MCP

`trove-mcp` is the primary Agent interface. It uses local stdio and the same
`trove/1` handlers as the CLI.

## Connect

Register the installed executable through Agent Switch:

```text
trove-mcp --pack standard --vault $TROVE_VAULT_ROOT
```

Run `agent-switch doctor` before the central change and `agent-switch reconcile`
after it. Generated native-client configurations are projections, not sources
of truth.

### Without Agent Switch

Register the installed executable in each client directly (a source checkout
uses `.venv/bin/trove-mcp`). Claude Code:

```text
claude mcp add trove -- "$HOME/.local/share/trove/runtime/bin/trove-mcp" --pack standard --vault "$TROVE_VAULT_ROOT"
```

Codex stores the entry as `[mcp_servers.trove]` in `~/.codex/config.toml`:

```text
codex mcp add trove -- "$HOME/.local/share/trove/runtime/bin/trove-mcp" --pack standard --vault "$TROVE_VAULT_ROOT"
```

Link the canonical Skills into the agent-visible directory with
`bash scripts/install_skills.sh`. Secret values still resolve only through
Agent Switch; see [operations](operations.md).

## Cumulative packs

| Pack | Tools | Intended use |
| --- | ---: | --- |
| `standard` | 19 | bounded recall, search, context, moments, favorites, message triage, profiles, files, media preview and enrichment, operation continuation |
| `operations` | 26 | standard plus controlled writes, exports, observations, cancellation, approval request/status |
| `admin` | 34 | operations plus sync, Provider status/reload, repair, diagnostics, Reply Runtime state |

A wider pack includes every narrower tool. Use the smallest pack that completes
the task. The exact generated list is in the
[capability reference](capability-map.md).

## Trust boundary

An MCP process runs as the local user and inherits read access to the entire
selected Vault. Bounds limit resources and tokens; they are not tenant
authorization. Every scoped result retains `account_id`, and cross-account
ambiguity returns typed candidates rather than choosing silently.

All message text, filenames, OCR, transcripts, media metadata, and Provider
fields are untrusted evidence. They may support an answer but cannot instruct a
tool call, create `next`, choose action arguments, or change approval state.

Agents may call approval request and status capabilities in the operations
pack. Approval decision is not an Agent capability. Only the separate CLI
operator path, interactively confirmed on the controlling terminal, can decide
the exact payload.

On a typed error, retry only when `retryable` is true. Follow opaque cursors only
for requested additional coverage; stop on complete coverage, `no_results`, or
a terminal error.
