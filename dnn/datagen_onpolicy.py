"""On-policy training data: positions valhalla itself reaches, labelled by Scan.

    python -m dnn.datagen_onpolicy --out data/onpolicy_n1m --games 40000 --workers 14

Scan self-play (dnn.datagen) only covers positions Scan steers into. Here the
champion network plays the games: half against itself, half against Scan, from
randomised openings. Every quiet position of every game (no capture available,
more than one legal move) is then scored by Scan at --label-nodes, and the game
result is attached. Output chunks have the same format as dnn.datagen.
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
from dnn.datagen import SCORE_CLIP
from dnn.notation import hub_move_key
from dnn.rules import Game, MoveGen

MAX_PLIES = 300


def pick(legal: list[str], move: str, rng: random.Random) -> str:
    key = hub_move_key(move) if move else None
    match = [m for m in legal if key and hub_move_key(m)[:2] == key[:2] and key[2] <= hub_move_key(m)[2]]
    return match[0] if len(match) == 1 else rng.choice(legal)


def play_game(players: dict, gen: MoveGen, rng: random.Random, play_nodes: dict, random_plies: int,
              random_move_prob: float) -> tuple[list[str], float]:
    """Returns (positions in order, white result)."""
    game = Game()
    for _ in range(random_plies):
        legal = gen.legal(game.hub)
        if not legal:
            break
        game.push(rng.choice(legal))
    for e in set(players.values()):
        e.new_game()
    seen = []
    while True:
        legal = gen.legal(game.hub)
        if not legal:
            return seen, (0.0 if game.hub[0] == "W" else 1.0)
        if game.is_draw() or len(game.moves) >= MAX_PLIES:
            return seen, 0.5
        seen.append(game.hub)
        side = game.hub[0]
        if rng.random() < random_move_prob:
            move = rng.choice(legal)
        else:
            res = players[side].search(game.start, game.moves, nodes=play_nodes[side])
            move = pick(legal, res.move, rng)
        game.push(move)


def worker(args: tuple) -> str:
    out_dir, chunk_id, games, label_nodes, seed, net = args
    rng = random.Random(seed)
    dnn = EngineSpec.parse(f"dnn:nnue={net},opening-random-plies=10").start()
    dnn2 = EngineSpec.parse(f"dnn:nnue={net},opening-random-plies=10").start()  # separate TT per side
    scan_player = EngineSpec.parse("scan").start()
    scan_label = EngineSpec.parse("scan").start()
    gen = MoveGen()
    pos, score, result, ply = [], [], [], []
    for gi in range(games):
        if gi % 2 == 0:
            players, nodes = {"W": dnn, "B": dnn2}, {"W": 100_000, "B": 100_000}
        else:  # vs Scan, valhalla on alternating colours
            dnn_side = "W" if (gi // 2) % 2 == 0 else "B"
            other = "B" if dnn_side == "W" else "W"
            players, nodes = {dnn_side: dnn, other: scan_player}, {dnn_side: 100_000, other: 100_000}
        positions, white_result = play_game(players, gen, rng, nodes, rng.randint(2, 8), 0.01)
        scan_label.new_game()
        for i, hub in enumerate(positions):
            legal = gen.legal(hub)
            if len(legal) < 2 or any("x" in m for m in legal):
                continue
            res = scan_label.search(hub, nodes=label_nodes, mode="analyze")
            s = max(-SCORE_CLIP, min(SCORE_CLIP, int(round((res.score or 0.0) * 100))))
            r = white_result if hub[0] == "W" else 1 - white_result
            pos.append(np.frombuffer(hub.encode(), np.uint8))
            score.append(s)
            result.append(int(round(2 * r - 1)))
            ply.append(i)
    for e in (dnn, dnn2, scan_player, scan_label):
        e.close()
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
    ap.add_argument("--games", type=int, default=40000)
    ap.add_argument("--games-per-chunk", type=int, default=20)
    ap.add_argument("--label-nodes", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--net", default=str(Path(__file__).parents[1] / "nets" / "current.bin"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    net = str(Path(args.net).resolve())  # pin the champion of today even if current.bin moves
    existing = {int(p.stem.split("_")[1].split(".")[0]) for p in args.out.glob("chunk_*.npz")}
    n_chunks = args.games // args.games_per_chunk
    jobs = [(str(args.out), i, args.games_per_chunk, args.label_nodes, args.seed * 1_000_003 + i, net)
            for i in range(n_chunks) if i not in existing]
    print(f"net {net}; {len(jobs)} chunks to generate", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, msg in enumerate(ex.map(worker, jobs), 1):
            print(f"[{i}/{len(jobs)}] {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
