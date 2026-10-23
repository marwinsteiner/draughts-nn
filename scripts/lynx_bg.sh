#!/usr/bin/env bash
# Run a command detached on the training box, logging to logs/<name>.log.
#   scripts/lynx_bg.sh <name> <command...>
set -euo pipefail
HOST="${DNN_HOST:-lynx}"
DEST="${DNN_DEST:-draughts/draughts-nn}"
name="$1"; shift
ssh -n "$HOST" "cd $DEST && mkdir -p logs && (setsid nohup bash -c '$*' > logs/$name.log 2>&1 < /dev/null &) ; sleep 1; echo started $name"
