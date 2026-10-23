#!/usr/bin/env bash
# Set up lidraughts-bot next to the repo on the training box and write its config.
# Run ON the box (after scripts/sync_lynx.sh):  bash draughts-nn/scripts/deploy_bot.sh
# Then start with:                               bash draughts-nn/scripts/run_bot.sh
set -euo pipefail
ROOT="${DNN_ROOT:-$HOME/draughts}"
BOT_DIR="$ROOT/lidraughts-bot-main"
UV="${UV:-$HOME/.local/bin/uv}"

if [ ! -d "$BOT_DIR" ]; then
  curl -sL https://github.com/AttackingOrDefending/lidraughts-bot/archive/refs/heads/main.tar.gz | tar xz -C "$ROOT"
fi
if [ ! -d "$BOT_DIR/.venv" ]; then
  (cd "$BOT_DIR" && "$UV" venv -q --python 3.12 .venv && "$UV" pip install -q --python .venv/bin/python -r requirements.txt)
fi
sed "s|@ROOT@|$ROOT|g" "$ROOT/draughts-nn/bot/config.yml.template" > "$BOT_DIR/config.yml"
make -C "$ROOT/draughts-nn/engine" -j4 >/dev/null
test -e "$ROOT/draughts-nn/nets/current.bin" || { echo "no nets/current.bin - train and promote a net first"; exit 1; }
echo "lidraughts-bot ready in $BOT_DIR"
