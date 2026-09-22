# Security policy

## Reporting a vulnerability or leaked credential

Do not open a public issue containing a secret. Contact the maintainer through
the repository owner's private GitHub contact channel or GitHub Security
Advisories when enabled.

If a credential is committed:

1. Revoke or rotate it immediately at the issuing provider. Deleting a file or
   rewriting Git history does not invalidate a credential.
2. Replace it with an environment-variable lookup.
3. Rewrite all reachable Git history and force-push the affected refs.
4. Ask collaborators to re-clone or carefully rebase onto the rewritten
   history.
5. Review CI logs, release assets, forks, caches, and provider access logs.

This project reads LLM credentials only from `LLM_API_KEY` or
`DEEPSEEK_API_KEY`. Never add a real value to source code, configuration,
examples, screenshots, test fixtures, or issue reports.

## Supported versions

Until the research prototype reaches version 1.0, only the latest tagged
release receives security fixes.
