"""Hard cap on the number of blitz games the bot plays: the study is frozen at LIMIT.

Finished games are counted from the public game export (aborted games excluded),
games in progress from /api/account/playing, and a challenge we sent that may
still be accepted from a marker file that dnn.challenge and dnn.matchmaker keep
while they wait (lidraughts has no API to list our own open challenges).

Everything that can start a game asks first: the patched lidraughts-bot before
accepting a challenge (may_accept), dnn.challenge and dnn.matchmaker before
sending one (budget_left). Stdlib only: deploy_bot.sh copies this file next to
lidraughts-bot as well.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

LIMIT = 200
SPEED = "blitz"
BASE = "https://lidraughts.org"
UNFINISHED = {"created", "started", "aborted", "noStart"}
MARKER_MAX_AGE = 300  # seconds; a marker older than this is left over from a crash


def _get(url: str, token: str | None = None, accept: str = "application/json") -> str:
    headers = {"Accept": accept, "User-Agent": "draughts-nn game budget"}
    if token:
        headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
        return r.read().decode()


def account_id(token: str) -> str:
    return json.loads(_get(f"{BASE}/api/account", token))["id"]


def finished_games(bot: str) -> int:
    text = _get(f"{BASE}/api/games/user/{bot}?moves=false", accept="application/x-ndjson")
    games = [json.loads(line) for line in text.splitlines() if line.strip()]
    return sum(g.get("speed") == SPEED and g.get("status") not in UNFINISHED for g in games)


def playing(token: str) -> int:
    return len(json.loads(_get(f"{BASE}/api/account/playing", token)).get("nowPlaying", []))


def pending(marker: Path) -> int:
    try:
        return int(time.time() - marker.stat().st_mtime < MARKER_MAX_AGE)
    except FileNotFoundError:
        return 0


def budget_left(bot: str, token: str, marker: Path, count_marker: bool = True) -> int:
    used = finished_games(bot) + playing(token) + (pending(marker) if count_marker else 0)
    return LIMIT - used


@contextmanager
def outgoing(marker: Path):
    """Held while a challenge we sent may still be accepted."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def may_accept(config: dict, accepted_not_started: int = 0) -> bool:
    """For lidraughts-bot: accept a challenge only while games are left. Declines if the
    check itself fails, so the limit holds even when the API is unreachable."""
    try:
        left = budget_left(config["bot_name"], os.environ["LIDRAUGHTS_BOT_TOKEN"],
                           Path(config["outgoing_marker"]))
        return left - accepted_not_started > 0
    except Exception:
        return False
