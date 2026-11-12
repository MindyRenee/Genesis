# Security policy

Genesis is a research system with an embodied host boundary. Treat a running
instance like a user-level process that can read its permitted inputs, use CPU,
and hold persistent state — not like a sandboxed model.

## Supported boundary

The supported startup path is `./genesis` (or the packaged `genesis` command).
State belongs in `~/.local/share/genesis`, or the directory
selected by `GENESIS_DATA_DIR` (the only override).

Do not point a live instance at accounts, files, microphones, cameras, webhook
URLs, or hardware controls you are not willing to let it use under the
permissions of the user running it.

## Network access

Genesis can autonomously fetch public web content when network access is
enabled. A blocklist (`frontal_lobe/tools/web_search.py`) refuses known
adult, malware, phishing, and tracking domains and adult-themed queries —
it is a known-bad filter, not an allow-list: the rest of the open web is
reachable. Run offline when that is not acceptable:

```bash
./genesis --offline
```

## Persistence and shutdown

The developmental record is cumulative and load-bearing. Stop Genesis with
Ctrl-C, `./genesis --stop`, or `genesis --stop`. Do not use `kill -9`, edit the
memory-mapped state directly, or delete state as an operational shortcut.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could expose data,
credentials, or a running instance. Use GitHub's private vulnerability reporting
flow if it is enabled for the repository; otherwise contact the maintainer
through the GitHub account linked from the repository.

Include:

- the commit or version tested,
- whether the instance was online or offline,
- the affected boundary (filesystem, network, IPC, sensors, ARC adapter, etc.),
- a minimal reproduction,
- and whether any persistent state was modified.

Do not include real secrets, recordings, or personal state dumps in a report.

## Scope

In scope:

- unintended secret disclosure from the repository or runtime,
- state-file corruption or unsafe recovery behavior,
- IPC privilege or protocol confusion,
- unsafe autonomous network or tool use beyond the documented boundary,
- unsafe interaction with system sensors, CPU controls, or ARC submission
  credentials.

Out of scope:

- the documented ability to fetch public web pages when online,
- CPU/disk use within the documented bounds,
- the fact that a non-sandboxed process can access what its user can access,
- or benchmark failures that do not cross a security boundary.
