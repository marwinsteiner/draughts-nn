"""Streak limit for human opponents, used by the patched lidraughts-bot (see
scripts/deploy_bot.sh).

A human's challenge (or rematch offer) is declined if the bot's last
`max_human_streak` games were all against that same player, until
`human_streak_cooldown_min` minutes have passed since the last of them. Playing
anyone else in between resets the streak. Bots are not limited.
"""

from __future__ import annotations

import json
import time
import urllib.request


def over_limit(config: dict, challenger: str, challenger_is_bot: bool) -> bool:
    max_streak = int(config.get("max_human_streak") or 0)
    bot = config.get("bot_name")
    if challenger_is_bot or max_streak <= 0 or not bot:
        return False
    try:
        req = urllib.request.Request(
            f"https://lidraughts.org/api/games/user/{bot}?max={max_streak}",
            headers={"Accept": "application/x-ndjson", "User-Agent": "draughts-nn streak guard"})
        with urllib.request.urlopen(req, timeout=10) as r:
            games = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    except Exception:
        return False  # never refuse a game because the check itself failed
    if len(games) < max_streak:
        return False
    me, them = bot.lower(), challenger.lower()

    def opponent(g: dict) -> str:
        ids = [g["players"][s].get("user", {}).get("id", "") for s in ("white", "black")]
        return next((i for i in ids if i != me), "")

    if not all(opponent(g) == them for g in games):
        return False
    last_ms = max(g.get("lastMoveAt") or g.get("createdAt") or 0 for g in games)
    cooldown_ms = float(config.get("human_streak_cooldown_min", 120)) * 60_000
    return time.time() * 1000 - last_ms < cooldown_ms
