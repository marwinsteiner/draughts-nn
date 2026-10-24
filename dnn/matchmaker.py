"""Challenge other online bots when idle, so the bot keeps playing rated games
(BOT accounts cannot use the lobby; they only play challenges).

    python -m dnn.matchmaker --bot MyBot [--idle 2]

Etiquette and bookkeeping:
- only BOT accounts that play international draughts are challenged;
- a challenge not accepted within --wait seconds is cancelled and that bot is
  left alone for --cooldown minutes;
- bots that ignore us are retried with rotating time controls; the controls a
  bot accepted are remembered and preferred;
- at most --per-bot-daily games per bot per day (counted from games actually
  played, not challenges sent).
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
TIME_CONTROLS = ["180+2", "60+1", "120+1", "180+0", "300+3"]


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


def games_today(bot: str) -> dict[str, int]:
    since = int((time.time() - 86400) * 1000)
    counts: dict[str, int] = {}
    for g in lidraughts.user_games(bot, since=since, rated_only=False):
        for side in ("white", "black"):
            u = g["players"][side].get("user", {})
            if u.get("id") and u["id"] != bot.lower():
                counts[u["id"]] = counts.get(u["id"], 0) + 1
    return counts


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True)
    ap.add_argument("--idle", type=float, default=2, help="minutes without a game before challenging")
    ap.add_argument("--wait", type=float, default=90, help="seconds to wait for acceptance")
    ap.add_argument("--cooldown", type=float, default=120, help="minutes to leave a bot alone after it ignored us")
    ap.add_argument("--per-bot-daily", type=int, default=40)
    args = ap.parse_args()
    token = (Path.home() / ".lidraughts_token").read_text().strip()
    auth = {"Authorization": f"Bearer {token}", **lidraughts.UA}
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state.setdefault("ignored_until", {})  # bot -> unix time
    state.setdefault("accepted_tc", {})    # bot -> [tc, ...]
    state.setdefault("tried_tc", {})       # bot -> index into TIME_CONTROLS

    def save() -> None:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=1))

    def opponents_now() -> list[str]:
        r = requests.get(f"{lidraughts.BASE}/api/account/playing", headers=auth, timeout=30)
        return [(g.get("opponent") or {}).get("username", "?").lower() for g in r.json().get("nowPlaying", [])]

    last_busy = time.time()
    bots_cache: tuple[float, list[str]] = (0.0, [])
    while True:
        time.sleep(30)
        try:
            if opponents_now():
                last_busy = time.time()
            elif time.time() - last_busy > args.idle * 60:
                now = time.time()
                if now - bots_cache[0] > 3600:
                    bots_cache = (now, [b for b in known_bots() if b.lower() != args.bot.lower()])
                bots = bots_cache[1]
                live = online(bots)
                played = games_today(args.bot)
                cands = [b for b in bots if b.lower() in live and played.get(b.lower(), 0) < args.per_bot_daily
                         and now >= state["ignored_until"].get(b, 0)]
                if not cands:
                    last_busy = now  # nothing to do; look again after another idle period
                    continue
                # Prefer bots that have accepted before.
                accepting = [b for b in cands if state["accepted_tc"].get(b)]
                opp = random.choice(accepting or cands)
                if state["accepted_tc"].get(opp):
                    tc = random.choice(state["accepted_tc"][opp])
                else:
                    k = state["tried_tc"].get(opp, 0)
                    tc = TIME_CONTROLS[k % len(TIME_CONTROLS)]
                    state["tried_tc"][opp] = k + 1
                base, _, inc = tc.partition("+")
                r = requests.post(f"{lidraughts.BASE}/api/challenge/{opp}", headers=auth, timeout=30,
                                  data={"rated": "true", "clock.limit": base, "clock.increment": inc or "0",
                                        "variant": "standard", "color": "random"})
                if r.status_code >= 400:
                    log(f"challenge {opp} {tc} rejected by API: {r.status_code} {r.text[:150]}")
                    state["ignored_until"][opp] = now + args.cooldown * 60
                    save()
                    continue
                cid = r.json().get("challenge", {}).get("id") or r.json().get("id")
                log(f"challenged {opp} {tc} ({cid})")
                started = False
                t0 = time.time()
                while time.time() - t0 < args.wait:
                    time.sleep(10)
                    now_opps = opponents_now()
                    if now_opps:
                        started = opp.lower() in now_opps  # a human may have challenged us meanwhile
                        break
                if started:
                    log(f"{opp} accepted {tc}")
                    acc = state["accepted_tc"].setdefault(opp, [])
                    if tc not in acc:
                        acc.append(tc)
                    last_busy = time.time()
                else:
                    if cid:
                        requests.post(f"{lidraughts.BASE}/api/challenge/{cid}/cancel", headers=auth, timeout=30)
                    log(f"{opp} ignored {tc}; cancelled, cooling down {args.cooldown:.0f} min")
                    state["ignored_until"][opp] = time.time() + args.cooldown * 60
                save()
        except Exception as e:  # keep running through network errors
            log(f"error: {e!r}")


if __name__ == "__main__":
    main()
