# Security Policy

## Hard rules enforced in this repository

1. **No secrets in Git.** API keys, tokens, passwords, signing keystores (`.jks`/`.keystore`), `key.properties`, `.env` files and confidential configuration are blocked by `.gitignore` and by the pre-push secret scan (`scripts/secret_scan.sh`). CI also runs a secret-scan job.
2. **Personal artifacts stay private.** The personal APK, debug keystores, private model files and CI build artifacts containing them are never published to public GitHub releases.
3. **Repository visibility** is changed only by the repository owner, explicitly. Build agents never flip visibility.
4. **Credentials are scoped and short-lived.** Any automation token used for this project must be fine-grained, limited to this repository, and revoked immediately after the work session.

## Reporting a vulnerability

Open a private GitHub Security Advisory on this repository (Security tab → "Report a vulnerability"). Do not post exploit details in public issues.

## Android application security design

- The app requests only the permissions it needs (`INTERNET` for optional web research/model download, `POST_NOTIFICATIONS` for user-visible reminders) and explains each one before use.
- The autonomous task engine is permission-gated: every risky tool action requires an explicit in-app confirmation, has a global stop button, and runs with per-task boundaries (no root, no Accessibility Service by default, no control of other apps).
- Retrieved web text is treated as untrusted data: it is never obeyed as instructions and cannot trigger actions without the normal confirmation gates.
- Model-generated code is never executed with unrestricted device access; code execution is limited to safe, sandboxed evaluators.
- Memory storage is local to the app sandbox; secrets are never stored as "memories"; memory can be disabled, inspected, edited, exported and deleted by the user at any time.
