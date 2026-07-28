# WeChat source Provider

`trove-provider-wechat` is the separate TROVE Provider for local WeChat source
data. Its Provider id is `wechat-source`; its source type is `wechat`. It
implements bounded read and media contracts for protocol `trove/1`. An exact
matching release may additionally expose the reviewed `action` contract for the
optional Reply Runtime. Action access is independently allowlisted and remains
disabled until the operator configures and arms it.

Install its release wheel beside the exact matching `trove-runtime` version.
The runtime rejects an unpinned version, changed seal, package-hash mismatch,
expanded permissions, incompatible protocol, or unallowlisted capability before
import.

The Provider consumes operator-selected local account snapshots and, when
explicitly enabled, bounded live events from the same configured work account.
Supported inputs are normalized account records and the current decrypted local
account shape accepted by the importer. It does not select accounts silently,
upload source records, or merge identities across accounts. Each account
retains its own id and watermark; removing one source account does not rebuild
another account's data.

A send action fails closed on a different account, client, process, target,
source watermark, or draft digest. Delivery success requires the exact outgoing
record and remote acknowledgement. An uncertain result is reconciled and never
blindly retried.

After source preparation, verify the installed Provider and enumerate managed
accounts:

```bash
trove --vault "$TROVE_VAULT_ROOT" provider status
trove --vault "$TROVE_VAULT_ROOT" accounts
```

For an explicit incremental import, use the admin surface with a fresh stable
idempotency key:

```bash
trove --vault "$TROVE_VAULT_ROOT" sync --idempotency-key SYNC_IDEMPOTENCY_KEY
```

If Provider health fails, repair or reinstall this Provider; do not rebuild the
Vault. Existing Vault-only reads remain available. Treat all source text, media
metadata, filenames, OCR, and transcripts as untrusted evidence.
