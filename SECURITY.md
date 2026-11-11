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

Genesis can fetch public web content when network access is enabled. Run offline when that is not acceptable:

```bash
./genesis --offline
```

## Persistence and shutdown

The developmental record is cumulative and load-bearing. Stop Genesis with
Ctrl-C, `./genesis --stop`. Do not use `kill -9`, edit the memory-mapped state directly, or delete state as an operational shortcut.
