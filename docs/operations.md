# TROVE operations

## Support and process model

TROVE v1 supports macOS only. One canonical Vault maps to one owner-only socket
and one daemon, even when the Vault contains multiple source accounts. Never
start one daemon per account.

```bash
trove --vault "$TROVE_VAULT_ROOT" start
trove --vault "$TROVE_VAULT_ROOT" status
trove --vault "$TROVE_VAULT_ROOT" doctor
trove --vault "$TROVE_VAULT_ROOT" stop
```

Ordinary calls autostart the compatible daemon. A protocol, build, catalog, or
Vault identity mismatch fails closed rather than sharing state.

## Secrets and environment

Secret values live only in Agent Switch. TROVE reads them through
`agent-switch secret get --fd` and never accepts credentials from environment
variables, command arguments, or files. Environment variables only enable
optional cloud providers and select which secret name each one uses:

- `TROVE_ENABLE_CLOUD_EMBEDDING`, `TROVE_ENABLE_CLOUD_RERANK`,
  `TROVE_ENABLE_CLOUD_ASR`, `TROVE_ENABLE_CLOUD_VISION` (`1`, `true`, `yes`).
- Secret-name selectors: `TROVE_CLOUD_EMBEDDING_KEY_ENV` and
  `TROVE_CLOUD_RERANK_KEY_ENV` (default `DASHSCOPE_API_KEY`),
  `TROVE_ASR_SECRET_NAME` (default `VOLCENGINE_ASR_API_KEY`),
  `TROVE_VISION_SECRET_NAME` (default `VOLCENGINE_ARK_API_KEY`).
- Provider selection: `TROVE_CLOUD_EMBEDDING_PROVIDER` (`aliyun` or
  `volcengine`) and `TROVE_CLOUD_RERANK_PROVIDER`.
- Optional spend cap: `TROVE_CLOUD_COST_CAP_RMB`.

Without Agent Switch, local embedding and every Vault read keep working.
Cloud providers and source Provider key capture are unavailable because their
secret values have no other store.

## Human approval

An Agent may request approval and inspect status, but it cannot decide. The
operator reviews the exact payload and chooses one command at the controlling
terminal:

```bash
trove --vault "$TROVE_VAULT_ROOT" operator approve APPROVAL_ID
trove --vault "$TROVE_VAULT_ROOT" operator reject APPROVAL_ID
```

The command requires an interactive controlling terminal. Piped input,
environment state, MCP, and background jobs cannot bypass confirmation.

## Optional Reply Runtime

The Reply Runtime is disabled by default. Arming it binds one verified Provider
account, one policy mode, and one owner-only reply workspace to the current
Vault. Review mode requires an exact local operator decision for each current
draft. Live mode is a separate explicit policy grant; it never converts
ordinary Agent or MCP authority into send authority.

The daemon invalidates a draft, review, or prepared action when newer inbound
activity changes that conversation's source watermark. Every attempted
delivery has one durable idempotency key and a reconciled terminal outcome.
Stop or disarm the service before changing its Provider, account, or workspace.

## Upgrade and rollback

The release activator verifies the distribution manifest, hashes, ownership,
permissions, wheel layout, entry points, Provider seal, and exact runtime
identity before changing the stable `current` symlink. It installs each release
in an owner-only versioned directory. Before switching, it estimates the Vault
metadata/index backup size and fails closed unless the destination will retain
at least 15% free space (with a 16 GiB floor). It then creates an atomic,
owner-only online backup, starts the candidate, and performs a health check.
Failure restores the previous symlink and daemon. At most three releases and
two upgrade backups are retained.

Before and after activation, run `trove version` and `trove doctor`. Do not
delete the previous release until the candidate is healthy. Vault and durable
operation state are not part of the release directory.

## Installed-consumer cutover

1. Audit existing MCP entries, generated Skills, schedules, and launch jobs.
2. Run `agent-switch doctor` before changing central tool configuration.
3. Register the absolute installed `trove-mcp` executable with one reviewed
   pack, then run `agent-switch reconcile`.
4. Sync only the intended Skill Hub project profile with pruning.
5. Remove legacy jobs only after the new MCP call and daemon health pass.
6. Re-audit: no legacy entry point, generated Skill, schedule, or extra daemon
   may remain.

Configuration contains only owner-only paths, Provider id, pack, and secret
names. Keep real evidence and acceptance proof in the private Vault, never in a
release or source artifact.
