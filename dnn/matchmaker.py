"""Challenge other online bots when idle, so the bot keeps playing rated games
(BOT accounts cannot use the lobby; they only play challenges).

    python -m dnn.matchmaker --bot MyBot [--tc 180+2] [--idle 10]

Etiquette: only BOT accounts are challenged, at most --per-bot-daily times a
day each, and a bot that declines is left alone for a day.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

from dnn import lidraughts

STATE = Path(__file__).parents[1] / "var" / "matchmaker.json"


STANDARD_PERFS = ("ultraBullet", "bullet", "blitz", "rapid", "classical")


def known_bots() -> list[str]:
    """Bots listed on /player/bots that have played international (standard) draughts."""
    html = lidraughts._get(f"{lidraughts.BASE}/player/bots").text
    out = []
    for name in sorted(set(re.findall(r'href="/@/([A-Za-z0-9_-]+)"', html))):
        perfs = lidraughts._get(f"{lidraughts.BASE}/api/user/{name}").json().get("perfs", {})
        if any(perfs.get(p, {}).get("games", 0) > 0 for p in STANDARD_PERFS):
            out.append(name)
    return out


def online(ids: list[str]) -> set[str]:
    out = set()
    for i in range(0, len(ids), 50):
        r = lidraughts._get(f"{lidraughts.BASE}/api/users/status", params={"ids": ",".join(ids[i:i + 50])})
        out |= {u["id"].lower() for u in r.json() if u.get("online")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True)
    ap.add_argument("--tc", default="180+2")
    ap.add_argument("--idle", type=float, default=10, help="minutes without a game before challenging")
    ap.add_argument("--per-bot-daily", type=int, default=6)
    args = ap.parse_args()
    token = (Path.home() / ".lidraughts_token").read_text().strip()
    auth = {"Authorization": f"Bearer {token}", **lidraughts.UA}
    base, _, inc = args.tc.partition("+")
    state = json.loads(STATE.read_text()) if STATE.exists() else {"log": [], "declined": {}}
    last_busy = time.time()
    while True:
        try:
            playing = requests.get(f"{lidraughts.BASE}/api/account/playing", headers=auth, timeout=30).json()
            if playing.get("nowPlaying"):
                last_busy = time.time()
            elif time.time() - last_busy > args.idle * 60:
                now = time.time()
                recent = [e for e in state["log"] if now - e["t"] < 86400]
                counts: dict[str, int] = {}
                for e in recent:
                    counts[e["bot"]] = counts.get(e["bot"], 0) + 1
                bots = [b for b in known_bots() if b.lower() != args.bot.lower()]
                live = online(bots)
                cands = [b for b in bots if b.lower() in live and counts.get(b, 0) < args.per_bot_daily
                         and now - state["declined"].get(b, 0) > 86400]
                if cands:
                    opp = random.choice(cands)
                    r = requests.post(f"{lidraughts.BASE}/api/challenge/{opp}", headers=auth, timeout=30,
                                      data={"rated": "true", "clock.limit": base, "clock.increment": inc or "0",
                                            "variant": "standard", "color": "random"})
                    state["log"] = recent + [{"t": now, "bot": opp, "status": r.status_code}]
                    print(f"{time.strftime('%H:%M:%S')} challenged {opp}: {r.status_code} {r.text[:200]}", flush=True)
                    if r.status_code >= 400:
                        state["declined"][opp] = now
                    STATE.parent.mkdir(parents=True, exist_ok=True)
                    STATE.write_text(json.dumps(state))
                    last_busy = now  # give the challenge time to be accepted
        except Exception as e:  # keep running through network errors
            print(f"{time.strftime('%H:%M:%S')} error: {e!r}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
