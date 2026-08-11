# Open-source privacy boundary

The public source repository contains product code, public documentation,
schemas, and synthetic tests. It must not contain any user's real data—not in
the current tree, generated artifacts, Git objects, branches, tags, or deleted
history.

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
minimum behavior required by the contract; they must not quote or paraphrase a
real conversation or profile.

## Public-history rule

Never make a private development repository public merely by changing its
visibility. Deleted files remain recoverable from Git history. The public
repository must contain only reviewed, source-safe commits; private branches,
tags, reflogs, release evidence, and development artifacts are not publication
inputs.

## Local storage

Runtime data belongs in a Vault outside the repository. The default product
location is `$HOME/Trove/trove-vault`; callers may instead pass `--vault` or set
`TROVE_VAULT_ROOT`. Keep the Vault and release acceptance evidence owner-only.

Provider credentials belong in Agent Switch's private secret store. TROVE
configuration may reference secret **names**, never secret values.

## Required checks

```bash
./scripts/trove-python scripts/privacy_scan.py .
./scripts/trove-python scripts/check.py contract
gitleaks git --redact
```

CI performs both the project scanner and a full-history Gitleaks scan. The
scanners are guardrails, not proof that content is non-personal. Review fixtures,
documentation, binary artifacts, Git history, and screenshots as data before
publishing.
