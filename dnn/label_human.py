"""Label positions from harvested human games with teacher scores.

    python -m dnn.label_human --games data/human/games.jsonl --out data/human_n100k --min-rating 1800

Every quiet position (no capture available, more than one legal move) of games
where both players are rated >= --min-rating is scored by Scan at --nodes; the
game result is attached (time forfeits count as decided). Output chunks have the
same format as dnn.datagen.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dnn.arena import EngineSpec
from dnn.datagen import SCORE_CLIP
from dnn.rules import MoveGen, replay_lidraughts


def worker(args: tuple) -> str:
    games, out_path, nodes = args
    eng = EngineSpec.parse("scan").start()
    gen = MoveGen()
    pos, score, result, ply = [], [], [], []
    for g in games:
        try:
            plies = replay_lidraughts(g["moves"], gen)
        except ValueError:
            continue
        white_result = {"white": 1.0, "black": 0.0}.get(g.get("winner") or "", 0.5)
        eng.new_game()
        for i, (hub, _mv) in enumerate(plies):
            legal = gen.legal(hub)
            if len(legal) < 2 or any("x" in m for m in legal):
                continue
            res = eng.search(hub, nodes=nodes, mode="analyze")
            s = max(-SCORE_CLIP, min(SCORE_CLIP, int(round((res.score or 0.0) * 100))))
            r = white_result if hub[0] == "W" else 1 - white_result
            pos.append(np.frombuffer(hub.encode(), np.uint8))
            score.append(s)
            result.append(int(round(2 * r - 1)))
            ply.append(i)
    eng.close()
    gen.close()
    tmp = Path(out_path).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pos=np.stack(pos), score=np.array(score, np.int16),
                        result=np.array(result, np.int8), ply=np.array(ply, np.int16))
    os.replace(tmp, out_path)
    return f"{Path(out_path).name}: {len(pos)} positions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-rating", type=int, default=1800)
    ap.add_argument("--nodes", type=int, default=100000)
    ap.add_argument("--games-per-chunk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    games = []
    with args.games.open() as f:
        for line in f:
            g = json.loads(line)
            if min(g.get("white_rating") or 0, g.get("black_rating") or 0) >= args.min_rating:
                games.append(g)
    # file order is append-only, so chunk i always holds the same games across runs
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in range(0, len(games), args.games_per_chunk):
        path = args.out / f"chunk_{i // args.games_per_chunk:05d}.npz"
        if not path.exists() and len(games[i:i + args.games_per_chunk]) == args.games_per_chunk:
            jobs.append((games[i:i + args.games_per_chunk], str(path), args.nodes))
    print(f"{len(games)} games >= {args.min_rating}, {len(jobs)} chunks to label", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, msg in enumerate(ex.map(worker, jobs), 1):
            print(f"[{i}/{len(jobs)}] {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
