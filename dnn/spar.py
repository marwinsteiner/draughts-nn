"""Sparring: learn from losses against Scan, continuously.

    python -m dnn.spar --workers 8

Each round the champion plays a batch of fast games against Scan (colour-
swapped random openings). Lost games are analysed exactly like losses to
humans (dnn.learn: deep Scan scores, mistakes, lesson data with branches from
each mistake); the champion is fine-tuned on the round's lessons and promoted
only if it beats the current champion in an arena match (score >=
SPAR_MIN_SCORE). Rounds, scores vs Scan and gate results go to
var/learn/log.jsonl with "spar_*" events, so progress vs Scan is tracked over time.
"""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dnn.arena import EngineSpec, Limits, _pair, make_openings, summarise
from dnn.learn import (CURRENT, ROOT, analyse_plies, champion_dir, finetune, gate, lesson_data, log, promote,
                       promotion_lock)
from dnn.rules import Game

SPAR_DATA = ROOT / "data" / "spar"
SPAR_MIN_SCORE = 0.51
MAX_LESSONS = 12  # lost games turned into lessons per round
OPENING_PLIES = 4


def plies_of(moves: list[str]) -> list[tuple[str, str]]:
    game, out = Game(), []
    for mv in moves:
        out.append((game.hub, mv))
        game.push(mv)
    return out


def play_round(champ_bin: Path, games: int, tc: str, workers: int, seed: int) -> list[tuple[float, dict]]:
    a = EngineSpec.parse(f"dnn:nnue={champ_bin}")
    b = EngineSpec.parse("scan")
    base, _, inc = tc.partition("+")
    limits = Limits(tc_base=float(base), tc_inc=float(inc or 0))
    openings = make_openings(games // 2, OPENING_PLIES, seed=seed)
    out: list[tuple[float, dict]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for pair in ex.map(_pair, [a] * len(openings), [b] * len(openings), openings, [limits] * len(openings)):
            out += pair
    return out


def run(rounds: int, games: int, tc: str, workers: int, dry_run: bool = False) -> None:
    for r in range(rounds):
        tag = time.strftime("%Y%m%d_%H%M%S")
        champ = champion_dir()
        results = play_round(champ / "net.bin", games, tc, workers, seed=int(time.time()))
        scores = [s for s, _ in results]
        log({"event": "spar_round", "round": tag, "champion": champ.name, "vs_scan": summarise(scores)})
        losses = [g for s, g in results if s == 0.0][:MAX_LESSONS]
        lesson_dirs = []
        for i, g in enumerate(losses):
            me = "W" if g["white"].startswith("dnn") else "B"
            an = analyse_plies(plies_of(g["moves"]), me, workers)
            an["mistakes"] = [m for m in an["mistakes"] if m["ply"] >= OPENING_PLIES]  # not the random opening
            if not an["mistakes"]:
                continue
            white_result = g["result"]
            lesson_dirs.append(lesson_data(f"{tag}_{i:02d}", white_result, an, workers, root=SPAR_DATA))
        log({"event": "spar_lessons", "round": tag, "losses": len(losses), "lessons": len(lesson_dirs)})
        if not lesson_dirs:
            continue
        with promotion_lock():
            champ = champion_dir()  # a human lesson may have promoted meanwhile
            cand = finetune(lesson_dirs, f"spar_{tag}")
            res = gate(cand, champ, workers)
            ok = res["score"] >= SPAR_MIN_SCORE and not dry_run
            log({"event": "spar_gate", "round": tag, "candidate": cand.name, "champion": champ.name, **res,
                 "promoted": ok, "dry_run": dry_run})
            if ok:
                dest = promote(cand)
                log({"event": "promoted", "champion": dest.name, "source": "spar"})
            else:
                shutil.rmtree(cand, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10**9)
    ap.add_argument("--games", type=int, default=100, help="games vs Scan per round")
    ap.add_argument("--tc", default="5+0.05")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="never promote (pipeline test)")
    args = ap.parse_args()
    assert CURRENT.exists(), "no champion: promote a net first"
    run(args.rounds, args.games, args.tc, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
