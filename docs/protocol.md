# Protocol `trove/1`

TROVE clients use length-prefixed UTF-8 JSON over an owner-only Unix-domain
socket. The daemon verifies peer identity, Vault identity, runtime and catalog
hashes, deadline, request size, and response budget before dispatch.

Every response has `protocol`, `request_id`, `ok`, and exactly one of `data` or
`error`. An error contains a stable `code` and boolean `retryable`. Pagination
adds `page` and `coverage`; continuation or recovery may add structured `next`.
Absent concepts are omitted rather than emitted as empty placeholders.

```json
{"protocol":"trove/1","request_id":"opaque","ok":true,"data":{},"page":{"has_more":false},"coverage":{"state":"complete"}}
```

```json
{"protocol":"trove/1","request_id":"opaque","ok":false,"error":{"code":"ambiguous_target","retryable":false,"details":{}}}
```

Paginated reads use random daemon-side cursors. A cursor is bound to the
capability, normalized filters, keyset and high-water mark, Vault generation,
and TTL. Expired, stale, or mismatched cursors fail explicitly.

The compact response soft budget is 64 KiB and hard cap is 256 KiB. Large media
never enters JSON as base64; owner-only staging transfers carry path, size,
SHA-256, and TTL metadata.

Evidence envelopes carry `provenance.trust=untrusted_evidence`. Evidence keys
that resemble control fields are renamed. Only core code creates control,
approval, action, or continuation fields.

The schema source of truth is the capability catalog. See the generated
[capability reference](capability-map.md).
