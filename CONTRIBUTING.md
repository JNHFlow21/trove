# Contributing to TROVE

Thank you for contributing.

## Development

TROVE currently supports macOS and Python 3.11 or newer.

```bash
bash scripts/bootstrap_runtime.sh
./scripts/trove-python scripts/check.py contract
./scripts/trove-python scripts/privacy_scan.py .
```

Before opening a pull request, run:

```bash
./scripts/trove-python scripts/check.py release
```

## Privacy requirements

- Never commit real chats, contacts, media, transcripts, exports, provider
  payloads, local paths, logs, Vault files, credentials, or secret values.
- Use clearly synthetic identities and content in tests.
- Keep runtime Vaults outside the repository.
- Put provider credentials only in Agent Switch.

Pull requests that weaken these boundaries will not be accepted.

## Change scope

Keep changes focused, add the smallest failing regression test, and preserve
the public `trove/1` protocol unless the change explicitly versions it.

Report security or privacy vulnerabilities through GitHub private vulnerability
reporting rather than a public issue.
