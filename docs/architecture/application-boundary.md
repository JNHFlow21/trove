# Application boundary

Dependencies point toward protocol-neutral application handlers:

```text
MCP / CLI -> shared client -> local protocol -> daemon dispatcher
                                           -> query and operation handlers
                                           -> repositories / Providers
```

Public adapters may depend on the client and protocol contracts only. They do
not import storage internals or source-specific implementations. The daemon is
the sole owner of repositories, connections, runtime caches, Provider loading,
cursor state, and operation journals.

One catalog capability maps to one semantic handler. The handler validates the
same input schema for CLI and MCP, applies entitlement and approval policy,
enters the Vault generation or writer boundary, and emits the same envelope.
Adapters do not recreate those rules.

Provider packages implement the versioned Provider contract. Core reads their
entry-point metadata, manifest, ownership, seal, package hash, and allowlist
before import. Normalized Provider records cross the boundary with explicit
source and account provenance. Core never imports a Provider implementation.

The protocol-neutral ReplyService, round coordinator, context bridge, policy,
review queue, and send-operation coordinator belong to core. Source-specific
live reads, identity checks, UI automation, and delivery verification belong
to the verified Provider. The local operator application is an adapter: it
renders typed state and submits exact decisions, but owns no timing, generation,
retry, target-selection, or sender logic.

Durable mutations commit journal state with local side effects. External side
effects use prepared, dispatched, reconciled, and terminal states so a lost
response never becomes an untracked retry. Agent continuations receive only an
opaque token and bounded typed payload; internal claim and heartbeat methods
are not public capabilities.
