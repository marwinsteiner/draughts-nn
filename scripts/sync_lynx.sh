#!/usr/bin/env bash
# Copy the working tree (tracked + untracked, minus .gitignore'd files) to the
# training box. Host comes from $DNN_HOST or the ssh alias "lynx".
set -euo pipefail
HOST="${DNN_HOST:-lynx}"
DEST="${DNN_DEST:-draughts/draughts-nn}"
cd "$(git rev-parse --show-toplevel)"
git ls-files -co --exclude-standard -z | tar --null -T - -czf - | ssh "$HOST" "mkdir -p $DEST && tar xzf - -C $DEST"
echo "synced to $HOST:$DEST"
