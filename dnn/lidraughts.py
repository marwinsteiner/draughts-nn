"""lidraughts.org API helpers and the human-game harvester.

    python -m dnn.lidraughts harvest --out data/human/games.jsonl --top 100 --per-user 1000

Games are stored one per line (JSON) with id, speed, ratings, winner and the
move list in lidraughts' PDN form ("32-28 ... 24x33"); only rated games of the
standard variant from the initial position are kept. Resumable: game ids
already in the output file are skipped and users already harvested are noted
in <out>.users.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import requests

BASE = "https://lidraughts.org"
UA = {"User-Agent": "draughts-nn research harvester (github.com/marwinsteiner/draughts-nn)"}


def _get(url: str, *, params: dict | None = None, headers: dict | None = None, stream: bool = False,
         token: str | None = None) -> requests.Response:
    h = dict(UA, **(headers or {}))
    if token:
        h["Authorization"] = f"Bearer {token}"
    while True:
        r = requests.get(url, params=params, headers=h, stream=stream, timeout=60)
        if r.status_code == 429:
            time.sleep(61)
            continue
        r.raise_for_status()
        return r


def top_players(perf: str, n: int = 200) -> list[str]:
    r = _get(f"{BASE}/player/top/{n}/{perf}", headers={"Accept": "application/vnd.lidraughts.v3+json"})
    return [u["id"] for u in r.json()["users"]]


def user_games(user: str, *, max_games: int | None = None, since: int | None = None,
               token: str | None = None) -> Iterator[dict]:
    params = {"rated": "true", "moves": "true", "perfType": "ultraBullet,bullet,blitz,rapid,classical"}
    if max_games:
        params["max"] = str(max_games)
    if since:
        params["since"] = str(since)
    r = _get(f"{BASE}/api/games/user/{user}", params=params, headers={"Accept": "application/x-ndjson"},
             stream=True, token=token)
    for line in r.iter_lines():
        if line:
            yield json.loads(line)


def compact(g: dict) -> dict | None:
    if g.get("variant") != "standard" or g.get("initialFen") or not g.get("moves"):
        return None
    pl = g["players"]
    return {
        "id": g["id"], "speed": g.get("speed"), "status": g.get("status"), "winner": g.get("winner"),
        "white": pl["white"].get("user", {}).get("id"), "black": pl["black"].get("user", {}).get("id"),
        "white_rating": pl["white"].get("rating"), "black_rating": pl["black"].get("rating"),
        "created": g.get("createdAt"), "moves": g["moves"],
    }


def harvest(out: Path, top: int, per_user: int, perfs: list[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        with out.open() as f:
            seen = {json.loads(line)["id"] for line in f if line.strip()}
    users_file = out.with_suffix(".users")
    done_users = set(users_file.read_text().split()) if users_file.exists() else set()
    users: list[str] = []
    for perf in perfs:
        for u in top_players(perf, top):
            if u not in users:
                users.append(u)
    print(f"{len(users)} users, {len(done_users)} done, {len(seen)} games on disk", flush=True)
    with out.open("a") as f:
        for i, u in enumerate(users, 1):
            if u in done_users:
                continue
            n_new = 0
            for attempt in range(3):  # streams time out now and then; already-seen games are skipped
                try:
                    for g in user_games(u, max_games=per_user):
                        c = compact(g)
                        if c is None or c["id"] in seen:
                            continue
                        seen.add(c["id"])
                        f.write(json.dumps(c) + "\n")
                        n_new += 1
                    break
                except requests.RequestException as e:
                    print(f"[{i}/{len(users)}] {u}: {e!r}, retry {attempt + 1}", flush=True)
                    time.sleep(30)
            else:
                f.flush()
                continue  # leave the user unmarked so a later run retries
            f.flush()
            with users_file.open("a") as uf:
                uf.write(u + "\n")
            print(f"[{i}/{len(users)}] {u}: +{n_new} (total {len(seen)})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--top", type=int, default=100)
    h.add_argument("--per-user", type=int, default=1000)
    h.add_argument("--perfs", nargs="+", default=["blitz", "bullet", "rapid", "classical"])
    args = ap.parse_args()
    if args.cmd == "harvest":
        harvest(args.out, args.top, args.per_user, args.perfs)


if __name__ == "__main__":
    main()
