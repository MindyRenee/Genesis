#!/usr/bin/env bash
# Install the official ARC-AGI-1 dataset and ARC-AGI-3 toolkit.
#
# Usage:
#   bash scripts/install_arc_resources.sh
#
# Run from the Genesis repository root. Set GENESIS_ARC1_DIR to choose a
# different dataset checkout location. The script updates an existing official
# checkout with fast-forward-only semantics, refuses to overwrite a non-Git
# directory, and selects a Termux-specific dependency installation path.
set -euo pipefail
ROOT="$(pwd)"
DEST="${GENESIS_ARC1_DIR:-data/arc/ARC-AGI-1}"
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required." >&2
  exit 1
fi
mkdir -p "$(dirname -- "$DEST")"
if [[ -d "$DEST/.git" ]]; then
  echo "Updating official ARC-AGI-1 repository at $DEST"
  git -C "$DEST" pull --ff-only
elif [[ -e "$DEST" ]]; then
  echo "ERROR: $DEST exists but is not a Git checkout; move it or set GENESIS_ARC1_DIR." >&2
  exit 1
else
  git clone --depth 1 https://github.com/fchollet/ARC-AGI.git "$DEST"
fi
# Termux cannot reliably install every optional desktop dependency in the
# project's base dependency set (notably Cairo/OpenCV wheels). Install the
# official toolkit directly there; desktop environments use the project extra.
if [[ -n "${PREFIX:-}" && "${PREFIX}" == *com.termux* ]]; then
  python3 -m pip install "arc-agi==0.9.9"
else
  python3 -m pip install -e "$ROOT[arc]"
fi
echo "ARC-AGI-1 data: $DEST/data/{training,evaluation}"
echo "ARC-AGI-3 toolkit installed. ARC_API_KEY is optional for additional games."
echo "No model or LLM is installed by this script."
