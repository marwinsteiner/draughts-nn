"""Score every position of valhalla's lidraughts games with Scan, for the Results section.

    python paper/analysis/score_games.py --games var/results/games.ndjson --out var/results/scored.jsonl

Input is the API export (/api/games/user/valhalla?moves=true). For each game we
replay the moves under lidraughts' rules and ask Scan for its evaluation of every
position, including the final one. Scores are in men, from valhalla's point of
view. One output line per game; games already in --out are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dnn.arena import EngineSpec  # noqa: E402
from dnn.rules import Game, replay_lidraughts  # noqa: E402

BOT = "valhalla"


def score_game(args: tuple) -> dict:
    g, nodes = args
    me = "W" if g["players"]["white"].get("user", {}).get("id") == BOT else "B"
    plies = replay_lidraughts(g.get("moves", ""))
    game = Game()
    hubs = []
    for hub, mv in plies:
        hubs.append(hub)
        game.push(mv)
    hubs.append(game.hub)
    eng = EngineSpec.parse("scan").start()
    eng.new_game()
    scores = []
    for hub in hubs:
        res = eng.search(hub, nodes=nodes, mode="analyze")
        s = res.score or 0.0
        scores.append(s if hub[0] == me else -s)
    eng.close()
    draw = {"clock": game.clock, "limit": game.limit,
            "repetition": game.since_reset.count(game.hub) >= 3}
    return {"id": g["id"], "me": me, "hubs": hubs, "moves": [m for _, m in plies], "scan": scores,
            "final_draw_state": draw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nodes", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.open()}
    games = [json.loads(line) for line in args.games.open()]
    todo = [(g, args.nodes) for g in games if g["id"] not in done and g.get("moves")]
    print(f"{len(todo)} games to score", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex, args.out.open("a") as f:
        for i, rec in enumerate(ex.map(score_game, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {rec['id']} {len(rec['hubs'])} positions", flush=True)


if __name__ == "__main__":
    main()
