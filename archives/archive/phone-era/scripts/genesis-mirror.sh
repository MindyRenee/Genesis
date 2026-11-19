#!/usr/bin/env bash
# Mirror a Genesis data dir into the Genesis-Data-Directory repo and
# push. Snapshots are consistent: SQLite stores go through the backup
# API (never a raw copy under WAL), everything else is a plain file
# read. Run it on demand or from a watcher; it is a no-op when nothing
# changed.
#
#   genesis-mirror.sh [DATA_DIR] [COMMIT_MSG]
#
# Defaults: DATA_DIR=~/.local/share/genesis-october

set -euo pipefail

DATA_DIR="${1:-$HOME/.local/share/genesis-october}"
MSG="${2:-mirror $(basename "$DATA_DIR") $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
REPO="$HOME/Genesis-Data-Directory"
DEST="$REPO/runtime/$(basename "$DATA_DIR")"

mkdir -p "$DEST"

# SQLite stores — backup API only; a raw cp under WAL tears.
for db in "$DATA_DIR"/*.db; do
    [ -e "$db" ] || continue
    sqlite3 "$db" ".backup '$DEST/$(basename "$db")'"
done

# Everything else durable — skip locks, pidfiles, sockets, logs,
# session sentinels, re-installable weights, and anything that can
# hold a secret (env files, keys, certs) — pushing those trips secret
# scanning and revokes the credential.
find "$DATA_DIR" -maxdepth 1 -type f \
    ! -name '*.db' ! -name '*.db-*' \
    ! -name '*.pid' ! -name '*.lock' ! -name '*.sock' \
    ! -name '*.log' ! -name 'continuity.seen' ! -name 'mind.session' \
    ! -name '*.env' ! -name '*.env.*' ! -name 'agent.env' \
    ! -name '*.key' ! -name '*.pem' ! -name '*.secret' \
    ! -name 'id_*' ! -name 'credentials*' \
    -exec cp -p {} "$DEST/" \;

# Durable subtrees (projects, drawings, experiments) — copy whole.
for sub in projects drawings experiments; do
    [ -d "$DATA_DIR/$sub" ] || continue
    mkdir -p "$DEST/$sub"
    cp -a "$DATA_DIR/$sub/." "$DEST/$sub/"
done

cd "$REPO"
git add -A "$DEST"
if git diff --cached --quiet; then
    echo "mirror: no changes"
    exit 0
fi
git -c user.name="Genesis" \
    -c user.email="genesis@localhost" \
    commit -q -m "$MSG"
git push -q origin HEAD
echo "mirror: pushed $(git rev-parse --short HEAD)"
