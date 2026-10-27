"""Stop the bot for good once the study's game limit has been played.

    python -m dnn.freeze --bot valhalla

Every 30 seconds counts the finished blitz games (bot/game_budget.py). Once
LIMIT games have finished and none is in progress, stops lidraughts-bot, its
engines, the matchmaker, the human challenge watcher and the learn-from-loss
watcher, and records the moment in var/frozen.json. No new game can start in the
meantime: every program that starts games checks the same budget first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bot.game_budget import LIMIT, finished_games, playing

ROOT = Path(__file__).parents[1]
FROZEN = ROOT / "var" / "frozen.json"
# Challengers first, so nothing new starts while the bot itself shuts down.
STOP = ["dnn[.]matchmaker", "dnn[.]challenge", "dnn[.]learn", "lidraughts-bot[.]py", "engine/dn[n]"]


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True)
    ap.add_argument("--interval", type=float, default=30)
    args = ap.parse_args()
    token = (Path.home() / ".lidraughts_token").read_text().strip()
    last = None
    while True:
        try:
            done, now = finished_games(args.bot.lower()), playing(token)
            if (done, now) != last:
                log(f"{done}/{LIMIT} games finished, {now} in progress")
                last = (done, now)
            if done >= LIMIT and now == 0:
                break
        except Exception as e:  # network trouble: look again next time
            log(f"error: {e!r}")
        time.sleep(args.interval)
    for pattern in STOP:
        subprocess.run(["pkill", "-f", pattern])
    FROZEN.write_text(json.dumps({"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "games": done}, indent=1))
    log(f"limit reached: stopped {', '.join(STOP)}")


if __name__ == "__main__":
    main()
