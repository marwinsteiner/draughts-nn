#!/usr/bin/env bash
# Start (detached) the lidraughts bot, the learn-from-loss watcher and the
# matchmaker on the training box. Needs ~/.lidraughts_token (bot:play scope)
# and DNN_BOT_NAME. Logs go to draughts-nn/logs/.
#   DNN_BOT_NAME=MyBot bash draughts-nn/scripts/run_bot.sh [--upgrade]
set -euo pipefail
ROOT="${DNN_ROOT:-$HOME/draughts}"
BOT_DIR="$ROOT/lidraughts-bot-main"
REPO="$ROOT/draughts-nn"
: "${DNN_BOT_NAME:?set DNN_BOT_NAME}"
export LIDRAUGHTS_BOT_TOKEN="$(cat "$HOME/.lidraughts_token")"
mkdir -p "$REPO/logs"

if [ "${1:-}" = "--upgrade" ]; then  # one-time, irreversible: turn the account into a BOT account
  (cd "$BOT_DIR" && .venv/bin/python lidraughts-bot.py -u)
  exit 0
fi

(cd "$BOT_DIR" && setsid nohup .venv/bin/python lidraughts-bot.py > "$REPO/logs/bot.log" 2>&1 < /dev/null &)
(cd "$REPO" && setsid nohup .venv/bin/python -m dnn.learn watch --bot "$DNN_BOT_NAME" > "$REPO/logs/learn.log" 2>&1 < /dev/null &)
(cd "$REPO" && setsid nohup .venv/bin/python -m dnn.matchmaker --bot "$DNN_BOT_NAME" > "$REPO/logs/matchmaker.log" 2>&1 < /dev/null &)
sleep 2
pgrep -af "lidraughts-bot.py|dnn.learn|dnn.matchmaker" | grep -v pgrep
