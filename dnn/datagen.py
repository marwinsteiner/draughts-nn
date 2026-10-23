"""Training data from teacher self-play.

Scan plays itself from randomised openings at a fixed node budget. Every quiet
position (side to move has no capture and more than one move) is stored with
Scan's search score and, once the game ends, the game result - both from the
side to move's point of view.

    python -m dnn.datagen --out data/scan_sp --games 200000 --workers 18 --nodes 100000

Output: one .npz per chunk with
    pos    (N, 51) uint8   hub string bytes
    score  (N,)    int16   teacher score, centi-men, side to move
    result (N,)    int8    +1 win / 0 draw / -1 loss, side to move
    ply    (N,)    int16
"""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dnn.arena import EngineSpec
from dnn.notation import START_HUB, hub_move_key
from dnn.rules import Game, MoveGen

SCORE_CLIP = 3000  # centi-men
ADJUDICATE_SCORE = 1500  # |score| this large for ADJUDICATE_PLIES consecutive plies ends the game
ADJUDICATE_PLIES = 6
MAX_PLIES = 300


def play_one(engine, gen: MoveGen, rng: random.Random, nodes: int, random_plies: int,
             random_move_prob: float) -> tuple[list[tuple[str, int, int]], float]:
    """Returns ([(hub, score_cp_stm, ply)], white_result in {1, .5, 0})."""
    game = Game()
    for _ in range(random_plies):
        legal = gen.legal(game.hub)
        if not legal:
            break
        game.push(rng.choice(legal))
    engine.new_game()
    records: list[tuple[str, int, int]] = []
    streak, streak_sign = 0, 0
    while True:
        legal = gen.legal(game.hub)
        if not legal:
            return records, (0.0 if game.hub[0] == "W" else 1.0)
        if game.is_draw() or len(game.moves) >= MAX_PLIES:
            return records, 0.5
        pos, hist = game.hub_moves_since_reset()
        res = engine.search(pos, hist, nodes=nodes, mode="analyze")
        score = int(round((res.score or 0.0) * 100))
        score = max(-SCORE_CLIP, min(SCORE_CLIP, score))
        quiet = all("x" not in m for m in legal)
        if quiet and len(legal) > 1:
            records.append((game.hub, score, len(game.moves)))
        # adjudicate clear wins
        sign = (1 if score > 0 else -1) * (1 if game.hub[0] == "W" else -1)
        if abs(score) >= ADJUDICATE_SCORE:
            streak = streak + 1 if sign == streak_sign else 1
            streak_sign = sign
            if streak >= ADJUDICATE_PLIES:
                return records, 1.0 if sign > 0 else 0.0
        else:
            streak = 0
        if rng.random() < random_move_prob:
            move = rng.choice(legal)
        else:
            key = hub_move_key(res.move)
            match = [m for m in legal if hub_move_key(m)[:2] == key[:2] and key[2] <= hub_move_key(m)[2]]
            move = match[0] if len(match) == 1 else rng.choice(legal)
        game.push(move)


def worker(args: tuple) -> str:
    out_dir, chunk_id, games, nodes, seed, teacher = args
    rng = random.Random(seed)
    engine = EngineSpec.parse(teacher).start()
    gen = MoveGen()
    pos, score, result, ply = [], [], [], []
    for _ in range(games):
        records, white_result = play_one(engine, gen, rng, nodes, rng.randint(2, 8), 0.01)
        for hub, s, p in records:
            r = white_result if hub[0] == "W" else 1 - white_result
            pos.append(np.frombuffer(hub.encode(), dtype=np.uint8))
            score.append(s)
            result.append(int(round(2 * r - 1)))
            ply.append(p)
    engine.close()
    gen.close()
    path = Path(out_dir) / f"chunk_{chunk_id:05d}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pos=np.stack(pos), score=np.array(score, np.int16),
                        result=np.array(result, np.int8), ply=np.array(ply, np.int16))
    os.replace(tmp, path)
    return f"{path.name}: {len(pos)} positions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--games-per-chunk", type=int, default=200)
    ap.add_argument("--nodes", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--teacher", default="scan")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    existing = {int(p.stem.split("_")[1].split(".")[0]) for p in args.out.glob("chunk_*.npz")}
    n_chunks = args.games // args.games_per_chunk
    jobs = [(str(args.out), i, args.games_per_chunk, args.nodes, args.seed * 1_000_003 + i, args.teacher)
            for i in range(n_chunks) if i not in existing]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, msg in enumerate(ex.map(worker, jobs), 1):
            print(f"[{i}/{len(jobs)}] {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
