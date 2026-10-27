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
sed -e "s|@ROOT@|$ROOT|g" -e "s|@BOT@|${DNN_BOT_NAME:-valhalla}|g" \
  "$ROOT/draughts-nn/bot/config.yml.template" > "$BOT_DIR/config.yml"
cp "$ROOT/draughts-nn/bot/streak_guard.py" "$BOT_DIR/streak_guard.py"
cp "$ROOT/draughts-nn/bot/game_budget.py" "$BOT_DIR/game_budget.py"
# No chat commands: !name reveals the engine, !eval shows spectators (and so opponents) our evaluation.
if ! grep -q "dnn: no chat commands" "$BOT_DIR/conversation.py"; then
  sed -i '/^    def command(self, line, game, cmd):$/a\        return  # dnn: no chat commands' "$BOT_DIR/conversation.py"
fi
# Block list and human streak limit (bot/streak_guard.py): decline challenges, incl. rematch offers.
"$BOT_DIR/.venv/bin/python" - "$BOT_DIR/model.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "    def is_supported(self, config):\n        try:\n"
check = ('            if self.challenger_name.lower() in [n.lower() for n in config.get("block_list") or []]:'
         '  # dnn: block_list\n                return False, "later"\n')
streak = ('            from streak_guard import over_limit  # dnn: streak_guard\n'
          '            if over_limit(config, self.challenger_name, self.challenger_is_bot):\n'
          '                return False, "later"\n')
if "dnn: block_list" not in s:
    assert anchor in s, "lidraughts-bot model.py changed; update the patches"
    s = s.replace(anchor, anchor + check, 1)
if "dnn: streak_guard" not in s:
    s = s.replace(check, check + streak, 1)
open(p, "w").write(s)
PY
# A failed decline (the challenger withdrew meanwhile: 404) must not stop the bot, and
# no challenge is accepted once the game limit is reached (bot/game_budget.py).
"$BOT_DIR/.venv/bin/python" - "$BOT_DIR/lidraughts-bot.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
decline = "                    li.decline_challenge(chlng.id, reason=decline_reason)\n"
safe_decline = ("                    try:  # dnn: decline race\n"
                "                        li.decline_challenge(chlng.id, reason=decline_reason)\n"
                "                    except Exception as e:  # the challenger may have withdrawn it meanwhile\n"
                "                        logger.info(f\"Could not decline {chlng}: {e!r}\")\n")
pop = "                chlng = challenge_queue.pop(0)\n"
budget = ("                import game_budget  # dnn: game_budget\n"
          "                if not game_budget.may_accept(challenge_config, queued_processes):\n"
          "                    logger.info(f\"Game limit reached; declining {chlng}\")\n"
          "                    try:\n"
          "                        li.decline_challenge(chlng.id, reason=\"later\")\n"
          "                    except Exception as e:\n"
          "                        logger.info(f\"Could not decline {chlng}: {e!r}\")\n"
          "                    continue\n")
if "dnn: decline race" not in s:
    assert s.count(decline) == 1, "lidraughts-bot.py changed; update the patches"
    s = s.replace(decline, safe_decline, 1)
if "dnn: game_budget" not in s:
    assert s.count(pop) == 1, "lidraughts-bot.py changed; update the patches"
    s = s.replace(pop, pop + budget, 1)
open(p, "w").write(s)
PY
make -C "$ROOT/draughts-nn/engine" -j4 >/dev/null
test -e "$ROOT/draughts-nn/nets/current.bin" || { echo "no nets/current.bin - train and promote a net first"; exit 1; }
echo "lidraughts-bot ready in $BOT_DIR"
