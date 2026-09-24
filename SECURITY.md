# Security Policy

## Reporting a vulnerability

Do not open a public issue for security problems. Report them
privately through GitHub Security Advisories on this repository
(Security → Report a vulnerability), which goes directly to the
maintainer. Please allow reasonable time for a fix before
disclosure.

## Surface area to be aware of

Genesis is a local-first system, but it has real surfaces:

- **Unix socket IPC** between the CLI, daemon, and retina.
- **Ambient audio capture** (optional) — microphone input via Vosk.
- **Web fetching** — the autonomous learner fetches pages and stores
  their content in a local cache.
- **Privileged helper** — `scripts/cpufreq_helper.sh` runs via a scoped
  sudoers rule; review `scripts/genesis-sudoers` before installing.
- **Memory-mapped state file** — the core state is a file; file-system
  permissions are the security boundary.
- **Self-modification paths** — it can learn from and reason about its
  own source. Treat prompt-injection-style input as untrusted content
  flowing into a system with reflexive access.

If you find a problem in any of these — or anywhere else — please report
it privately first.
