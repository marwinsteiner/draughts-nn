"""Challenge strong idle humans now, one at a time (manual, on request).

    python -m dnn.challenge [--max 3] [--tc 180+2] [--perf blitz] [--min-rating 2150]
    python -m dnn.challenge --watch --min-rating 2150   # wait for strong idle players

With --watch it polls every --interval minutes and runs a round whenever a
player at or above --min-rating is online and idle, sending at most --daily
human challenges per 24 hours in total.

Takes the online, idle players from the --perf leaderboard, strongest first,
skipping anyone asked in the last 24 hours (var/challenged_humans.json). Each
challenge is cancelled if not accepted within --wait seconds; stops at the
first acceptance, after --max challenges, or if the bot is already playing.
The matchmaker is paused meanwhile so it cannot start a bot game in between.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

from dnn import lidraughts

STATE = Path(__file__).parents[1] / "var" / "challenged_humans.json"
LOCK = STATE.with_suffix(".lock")


def run_round(args: argparse.Namespace) -> int:
    """One round; returns the number of challenges sent. Rounds are serialised by a
    file lock (manual runs and --watch must never ask the same person twice)."""
    import fcntl
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        return _run_round(args)


def _save(asked: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(asked, indent=1))
    tmp.replace(STATE)


def _run_round(args: argparse.Namespace) -> int:
    B = lidraughts.BASE
    auth = {"Authorization": "Bearer " + (Path.home() / ".lidraughts_token").read_text().strip(), **lidraughts.UA}
    asked = json.loads(STATE.read_text()) if STATE.exists() else {}
    now = time.time()
    asked = {k: t for k, t in asked.items() if now - t < 86400}

    r = lidraughts._get(f"{B}/player/top/200/{args.perf}", headers={"Accept": "application/vnd.lidraughts.v3+json"})
    board = {u["id"]: (u["perfs"][args.perf]["rating"], u["username"]) for u in r.json()["users"]
             if u["id"] not in asked and u["perfs"][args.perf]["rating"] >= args.min_rating}
    ids = list(board)
    idle = []
    for i in range(0, len(ids), 50):
        st = lidraughts._get(f"{B}/api/users/status", params={"ids": ",".join(ids[i:i + 50])}).json()
        idle += [board[s["id"]] for s in st if s.get("online") and not s.get("playing") and s.get("title") != "BOT"]
    idle.sort(reverse=True)
    if not idle:
        return 0
    print(time.strftime("%H:%M:%S"), "idle:", idle[:8], flush=True)

    def games_now() -> list[str]:
        res = requests.get(f"{B}/api/account/playing", headers=auth, timeout=30).json()
        return [g.get("gameId") for g in res.get("nowPlaying", [])]

    base, _, inc = args.tc.partition("+")
    tried = 0
    subprocess.run(["pkill", "-STOP", "-f", "dnn[.]matchmaker"])
    try:
        for rating, name in idle:
            if tried >= args.max:
                break
            if games_now():
                print("bot is already playing; stopping")
                break
            asked[name.lower()] = time.time()
            _save(asked)  # immediately, before the request goes out
            r = requests.post(f"{B}/api/challenge/{name}", headers=auth, timeout=30,
                              data={"rated": "true", "clock.limit": base, "clock.increment": inc or "0",
                                    "variant": "standard", "color": "random"})
            if r.status_code >= 400:
                print(f"{name} ({rating}) not challengeable: {r.text[:120]}", flush=True)
                continue
            tried += 1
            cid = r.json().get("challenge", {}).get("id") or r.json().get("id")
            print(f"{time.strftime('%H:%M:%S')} challenged {name} ({rating}) {cid}", flush=True)
            t0 = time.time()
            ok = False
            while time.time() - t0 < args.wait:
                time.sleep(8)
                if cid in games_now():
                    ok = True
                    break
            if ok:
                print(f"{name} ACCEPTED: {B}/{cid}")
                break
            requests.post(f"{B}/api/challenge/{cid}/cancel", headers=auth, timeout=30)
            print(f"{name}: no answer in {args.wait:.0f}s, cancelled", flush=True)
    finally:
        subprocess.run(["pkill", "-CONT", "-f", "dnn[.]matchmaker"])
        _save(asked)
    return tried


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=3)
    ap.add_argument("--tc", default="180+2")
    ap.add_argument("--perf", default="blitz")
    ap.add_argument("--wait", type=float, default=90)
    ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=10, help="minutes between checks (--watch)")
    ap.add_argument("--daily", type=int, default=8, help="max human challenges per 24h (--watch)")
    args = ap.parse_args()
    if not args.watch:
        run_round(args)
        return
    per_round = args.max
    while True:
        # The daily cap counts from the shared record (survives restarts and
        # includes manual rounds): everyone asked in the last 24 hours.
        asked = json.loads(STATE.read_text()) if STATE.exists() else {}
        sent = sum(time.time() - t < 86400 for t in asked.values())
        if sent < args.daily:
            args.max = min(per_round, args.daily - sent)
            try:
                run_round(args)
            except Exception as e:  # network trouble: try again next time
                print(f"{time.strftime('%H:%M:%S')} error: {e!r}", flush=True)
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
