#!/usr/bin/env bash
# Publish paper/genesis.pdf as a new Zenodo version of record 22817337.
#
# Requires a personal access token with scopes deposit:write and
# deposit:actions (create at https://zenodo.org/account/settings/applications).
#
# Usage:
#   ZENODO_TOKEN=... ./publish-zenodo.sh            # draft + upload + publish
#   ZENODO_TOKEN=... ./publish-zenodo.sh --draft-only  # stop before publishing
#
# Steps: new-version draft -> clear carried-over files -> upload
# genesis.pdf -> PUT zenodo-metadata.json -> publish. Prints the new
# version DOI on success.
set -euo pipefail

RECORD=22817337
API=https://zenodo.org/api
HERE="$(cd "$(dirname "$0")" && pwd)"
PDF="$HERE/genesis.pdf"
META="$HERE/zenodo-metadata.json"

: "${ZENODO_TOKEN:?set ZENODO_TOKEN (zenodo.org -> account -> applications)}"
AUTH="access_token=$ZENODO_TOKEN"

[ -f "$PDF" ]  || { echo "missing $PDF — run make -C paper first" >&2; exit 1; }
[ -f "$META" ] || { echo "missing $META" >&2; exit 1; }

echo "== creating new-version draft of record $RECORD"
DRAFT_LINK=$(curl -fsS -X POST \
  "$API/deposit/depositions/$RECORD/actions/newversion?$AUTH" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['links']['latest_draft'])")
DRAFT_ID="${DRAFT_LINK##*/}"
echo "   draft id: $DRAFT_ID"

echo "== clearing carried-over files"
curl -fsS "$API/deposit/depositions/$DRAFT_ID/files?$AUTH" \
| python3 -c "import json,sys
for f in json.load(sys.stdin):
    print(f['links']['self'])" \
| while read -r file_url; do
    echo "   delete $file_url"
    curl -fsS -X DELETE "$file_url?$AUTH" -o /dev/null
done

echo "== uploading genesis.pdf"
BUCKET=$(curl -fsS "$API/deposit/depositions/$DRAFT_ID?$AUTH" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['links']['bucket'])")
curl -fsS -X PUT "$BUCKET/genesis.pdf?$AUTH" \
  --data-binary "@$PDF" -H 'Content-Type: application/pdf' -o /dev/null

echo "== updating metadata"
curl -fsS -X PUT "$API/deposit/depositions/$DRAFT_ID?$AUTH" \
  -H 'Content-Type: application/json' --data-binary "@$META" -o /dev/null

if [ "${1:-}" = "--draft-only" ]; then
  echo "== draft staged (not published):"
  echo "   https://zenodo.org/deposit/$DRAFT_ID"
  exit 0
fi

echo "== publishing"
curl -fsS -X POST "$API/deposit/depositions/$DRAFT_ID/actions/publish?$AUTH" \
| python3 -c "import json,sys
r = json.load(sys.stdin)
print('published:', r['doi_url'])
print('record:   https://zenodo.org/records/%s' % r['id'])"
