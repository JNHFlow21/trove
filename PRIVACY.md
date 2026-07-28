# Open-source privacy boundary

The source repository contains product code, public documentation, schemas, and
synthetic tests. It must not contain any user's real data.

## Never commit

- chat databases, exports, messages, contact books, account identifiers, or
  relationship profiles;
- names, phone numbers, email addresses, locations, or other personal details
  copied from a real account;
- images, voice messages, videos, transcripts, OCR output, or provider payloads;
- Vault files, caches, logs, proof from real runs, credentials, tokens, or
  machine-specific absolute paths.

## Synthetic fixture rule

Fixtures must use unmistakably synthetic labels such as `Sample Contact`,
`wxid_fixturea`, and `acct-a`. Synthetic conversations should test only the
minimum behavior required by the contract; they should not paraphrase a real
conversation or profile.

## Local storage

Runtime data belongs in a Vault outside the repository. The default product
location is `$HOME/Trove/trove-vault`; callers may instead pass `--vault` or set
`TROVE_VAULT_ROOT`.

## Required checks

```bash
./scripts/trove-python scripts/privacy_scan.py .
./scripts/trove-python scripts/check.py contract
```

The scanner is a guardrail, not proof that content is non-personal. Review test
fixtures and documentation as data before publishing.
