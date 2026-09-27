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
  their content in a local cache. There is **no domain allow-list**:
  `ALLOW_ALL_DOMAINS` is `True` and both allow-lists are empty, so any
  URL the learner reaches is fetched and ingested. The per-site
  approval flow (`/requests`, `/approve`, `/deny`) is dead code today.
  The one real filter is an adult/malware *content* filter on the
  separate `tools/` fetch path. Use `./run.sh --offline` to close the
  network, and treat any fetched page as untrusted input.
- **Two privileged helpers** — `scripts/cpufreq_helper.sh` and
  `scripts/rtc_wake_helper.sh` run via a scoped `NOPASSWD` sudoers
  rule; review `scripts/genesis-sudoers` and
  `scripts/install_sudoers.sh` before installing. Both validate their
  arguments against a strict allow-list, but sudoers rules key on the
  *file*, and these files live inside the operator's own checkout —
  the same tree Genesis can write to. A `NOPASSWD` rule on a
  user-writable script is a root-escalation primitive. Install the
  helpers root-owned outside the checkout.
- **Memory-mapped state file** — the core state is a file; file-system
  permissions are the security boundary.
- **Self-modification paths** — it can learn from and reason about its
  own source. Treat prompt-injection-style input as untrusted content
  flowing into a system with reflexive access.

If you find a problem in any of these — or anywhere else — please report
it privately first.
