"""Learn from losses: watch the bot's games and retrain after every loss to a human.

    python -m dnn.learn watch --bot <BotName>             # poll forever
    python -m dnn.learn lesson --game <lidraughts id>     # run one lesson by hand

For each lost game against a human:
 1. analyse  - Scan (deep, LESSON_NODES) scores every position of the game; a bot
               move that drops Scan's evaluation by >= MISTAKE_MEN is a mistake.
 2. data     - every quiet position of the game, plus Scan self-play
               continuations branching from each mistake position, labelled at
               the usual teacher depth -> data/lessons/<game id>/.
 3. finetune - warm-start from the champion, train on a random slice of the base
               data plus all lesson data oversampled, same K as the champion.
 4. gate     - arena candidate vs champion; promote only if it is not weaker
               (score >= GATE_MIN_SCORE) - a single game must not make the bot worse.
 5. promote  - nets/current.bin (what the bot loads at the start of each game)
               is atomically repointed to the new champion.

State lives in var/learn/: state.json, log.jsonl, games.jsonl (every bot game).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from dnn import lidraughts
from dnn.arena import EngineSpec, Limits, _pair, make_openings, summarise
from dnn.datagen import play_one
from dnn.notation import hub_move_key
from dnn.rules import Game, MoveGen, replay_lidraughts

ROOT = Path(__file__).parents[1]
NETS = ROOT / "nets"
CURRENT = NETS / "current.bin"
VAR = ROOT / "var" / "learn"
LESSONS = ROOT / "data" / "lessons"

LESSON_NODES = 2_000_000
TEACHER_NODES = 100_000
MISTAKE_MEN = 0.3
BRANCH_GAMES = 40
BASE_DATA = os.environ.get("DNN_BASE_DATA", "data/scan_sp_n100k")
BASE_CHUNKS = 200  # ~4.5M positions
LESSON_REPEAT = 8
GATE_GAMES = 200
GATE_TC = "5+0.05"
GATE_MIN_SCORE = 0.49
GATE_WORKERS = 12

if os.environ.get("DNN_LEARN_QUICK"):  # pipeline smoke test
    LESSON_NODES, BRANCH_GAMES, BASE_CHUNKS, GATE_GAMES = 100_000, 6, 5, 20


def log(event: dict) -> None:
    VAR.mkdir(parents=True, exist_ok=True)
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with (VAR / "log.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def load_state() -> dict:
    p = VAR / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"since": None, "seen": [], "pending": []}


def save_state(s: dict) -> None:
    VAR.mkdir(parents=True, exist_ok=True)
    tmp = VAR / "state.json.tmp"
    tmp.write_text(json.dumps(s, indent=1))
    os.replace(tmp, VAR / "state.json")


def champion_dir() -> Path:
    return CURRENT.resolve().parent


# ---------------------------------------------------------------------------
# game replay

def replay(pdn_moves: str) -> list[tuple[str, str]]:
    return replay_lidraughts(pdn_moves)


def bot_color(g: dict, bot: str) -> str:
    return "W" if (g.get("white") or "").lower() == bot.lower() else "B"


# ---------------------------------------------------------------------------
# 1-2. analysis and lesson data

def _scan_scores(args: tuple) -> list[tuple[int, float, str]]:
    items, nodes = args
    eng = EngineSpec.parse("scan").start()
    out = []
    for i, hub in items:
        res = eng.search(hub, nodes=nodes, mode="analyze")
        out.append((i, res.score or 0.0, res.move))
    eng.close()
    return out


def analyse(g: dict, bot: str, workers: int) -> dict:
    plies = replay(g["moves"])
    final = Game()
    for _, mv in plies:
        final.push(mv)
    hubs = [h for h, _ in plies] + [final.hub]
    items = list(enumerate(hubs))
    shards = [(items[i::workers], LESSON_NODES) for i in range(workers)]
    scores: dict[int, tuple[float, str]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_scan_scores, shards):
            for i, s, mv in part:
                scores[i] = (s, mv)
    me = bot_color(g, bot)
    mistakes = []
    for i, (hub, mv) in enumerate(plies):
        if hub[0] != me:
            continue
        before = scores[i][0]
        after = -scores[i + 1][0]  # next position is scored for the opponent
        drop = before - after
        # A drop after Scan's own choice is search noise (deeper horizon), not a mistake.
        same = hub_move_key(mv)[:2] == hub_move_key(scores[i][1])[:2] if scores[i][1] else False
        if drop >= MISTAKE_MEN and not same:
            mistakes.append({"ply": i, "hub": hub, "played": mv, "scan_best": scores[i][1],
                             "before": before, "after": after, "drop": round(drop, 2)})
    return {"plies": plies, "hubs": hubs, "scores": {i: s for i, (s, _) in scores.items()}, "mistakes": mistakes}


def _branch_worker(args: tuple) -> list[tuple[str, int, int, float]]:
    starts, seed = args
    rng = random.Random(seed)
    eng = EngineSpec.parse("scan").start()
    gen = MoveGen()
    out = []
    for start in starts:
        records, white_result = play_one(eng, gen, rng, TEACHER_NODES, 0, 0.03, start=start)
        for hub, s, p in records:
            out.append((hub, s, p, white_result if hub[0] == "W" else 1 - white_result))
    eng.close()
    gen.close()
    return out


def lesson_data(game_id: str, g: dict, an: dict, workers: int) -> Path:
    out_dir = LESSONS / game_id
    out_dir.mkdir(parents=True, exist_ok=True)
    white_result = {"white": 1.0, "black": 0.0}.get(g.get("winner") or "", 0.5)
    rows: list[tuple[str, int, int, float]] = []
    gen = MoveGen()
    for i, hub in enumerate(an["hubs"]):
        legal = gen.legal(hub)
        if len(legal) > 1 and all("x" not in m for m in legal):
            s = int(round(max(-30.0, min(30.0, an["scores"][i])) * 100))
            rows.append((hub, s, i, white_result if hub[0] == "W" else 1 - white_result))
    gen.close()
    # Branch from each mistake and the two plies before it (the plan that led there).
    starts = []
    for m in an["mistakes"]:
        for k in (m["ply"] - 2, m["ply"] - 1, m["ply"]):
            if k >= 0:
                starts += [an["hubs"][k]] * (BRANCH_GAMES // 3)
    if starts:
        jobs = [(starts[i::workers], hash((game_id, i)) & 0xFFFFFFFF) for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for part in ex.map(_branch_worker, jobs):
                rows += part
    np.savez_compressed(
        out_dir / "chunk_00000.npz",
        pos=np.stack([np.frombuffer(h.encode(), np.uint8) for h, *_ in rows]),
        score=np.array([s for _, s, _, _ in rows], np.int16),
        result=np.array([int(round(2 * r - 1)) for *_, r in rows], np.int8),
        ply=np.array([p for _, _, p, _ in rows], np.int16))
    (out_dir / "mistakes.json").write_text(json.dumps(an["mistakes"], indent=1))
    return out_dir


# ---------------------------------------------------------------------------
# 3-5. finetune, gate, promote

def finetune(lesson_dirs: list[Path], tag: str) -> Path:
    champ = champion_dir()
    meta = json.loads((champ / "meta.json").read_text())
    out = NETS / f"cand_{tag}"
    data = [f"{BASE_DATA}:{BASE_CHUNKS}"] + [f"{d}:0:{LESSON_REPEAT}" for d in lesson_dirs]
    cmd = [sys.executable, "-m", "dnn.train", "--data", *data, "--out", str(out), "--init", str(champ / "net.pt"),
           "--k", str(meta["k"]), "--hidden", str(meta["hidden"]), "--l2", str(meta["l2"]), "--l3", str(meta["l3"]),
           "--epochs", "3", "--lr", "2e-4", "--lam", str(meta.get("lam", 0.75))]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def mistake_fix_rate(net: Path, mistakes: list[dict]) -> float:
    """Share of mistake positions where the net (depth-8 search) avoids the played move."""
    if not mistakes:
        return 1.0
    eng = EngineSpec.parse(f"dnn:nnue={net}").start()
    fixed = 0
    for m in mistakes:
        res = eng.search(m["hub"], depth=8, mode="analyze")
        fixed += hub_move_key(res.move)[:2] != hub_move_key(m["played"])[:2]
    eng.close()
    return fixed / len(mistakes)


def gate(cand: Path, champ: Path) -> dict:
    a, b = EngineSpec.parse(f"dnn:nnue={cand / 'net.bin'}"), EngineSpec.parse(f"dnn:nnue={champ / 'net.bin'}")
    base, _, inc = GATE_TC.partition("+")
    limits = Limits(tc_base=float(base), tc_inc=float(inc))
    openings = make_openings(GATE_GAMES // 2, 4, seed=int(time.time()))
    scores = []
    with ProcessPoolExecutor(max_workers=GATE_WORKERS) as ex:
        for pair in ex.map(_pair, [a] * len(openings), [b] * len(openings), openings, [limits] * len(openings)):
            scores += [s for s, _ in pair]
    return {"score": sum(scores) / len(scores), "summary": summarise(scores)}


def promote(cand: Path) -> Path:
    n = len(list(NETS.glob("champ_*")))
    dest = NETS / f"champ_{n:03d}"
    shutil.move(str(cand), dest)
    tmp = NETS / "current.bin.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(dest.relative_to(NETS) / "net.bin")
    os.replace(tmp, CURRENT)
    return dest


@contextmanager
def paused_datagen():
    """SIGSTOP background data generation (and its Scan children) during a lesson,
    so the lesson and the live bot get the CPU; always resumed afterwards."""
    out = subprocess.run(["pgrep", "-f", "dnn[.]datagen"], capture_output=True, text=True).stdout.split()
    pids = [int(p) for p in out]
    for p in list(pids):
        kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        pids += [int(k) for k in kids]
    stopped = []
    try:
        for p in pids:
            try:
                os.kill(p, signal.SIGSTOP)
                stopped.append(p)
            except ProcessLookupError:
                pass
        yield
    finally:
        for p in stopped:
            try:
                os.kill(p, signal.SIGCONT)
            except ProcessLookupError:
                pass


def promote_dir(src: Path) -> Path:
    """Install a trained net directory (net.bin, net.pt, meta.json) as the next champion."""
    n = len(list(NETS.glob("champ_*")))
    dest = NETS / f"champ_{n:03d}"
    shutil.copytree(src, dest)
    tmp = NETS / "current.bin.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(dest.relative_to(NETS) / "net.bin")
    os.replace(tmp, CURRENT)
    return dest


def run_lesson(game_id: str, g: dict, bot: str, workers: int) -> None:
    with paused_datagen():
        _run_lesson(game_id, g, bot, workers)


def _run_lesson(game_id: str, g: dict, bot: str, workers: int) -> None:
    log({"event": "lesson_start", "game": game_id, "opponent": g["black"] if bot_color(g, bot) == "W" else g["white"]})
    an = analyse(g, bot, workers)
    log({"event": "analysed", "game": game_id, "mistakes": [(m["ply"], m["played"], m["scan_best"], m["drop"])
                                                             for m in an["mistakes"]]})
    d = lesson_data(game_id, g, an, workers)
    champ = champion_dir()
    cand = finetune([d], game_id)
    fix_before = mistake_fix_rate(champ / "net.bin", an["mistakes"])
    fix_after = mistake_fix_rate(cand / "net.bin", an["mistakes"])
    res = gate(cand, champ)
    ok = res["score"] >= GATE_MIN_SCORE
    log({"event": "gate", "game": game_id, "candidate": cand.name, "champion": champ.name, **res,
         "mistake_fix_before": fix_before, "mistake_fix_after": fix_after, "promoted": ok})
    if ok:
        dest = promote(cand)
        log({"event": "promoted", "champion": dest.name})
    else:
        shutil.rmtree(cand, ignore_errors=True)


# ---------------------------------------------------------------------------
# watcher

def is_human(g: dict, bot: str, raw: dict) -> bool:
    opp = "black" if bot_color(g, bot) == "W" else "white"
    return raw["players"][opp].get("user", {}).get("title") != "BOT" and "aiLevel" not in raw["players"][opp]


def watch(bot: str, interval: int, workers: int) -> None:
    token_file = Path.home() / ".lidraughts_token"
    token = token_file.read_text().strip() if token_file.exists() else None
    state = load_state()
    while True:
        try:
            raws = list(lidraughts.user_games(bot, since=state["since"], token=token, rated_only=False,
                                              clocks=True))
        except Exception as e:  # network trouble: try again later
            log({"event": "poll_error", "error": repr(e)})
            time.sleep(interval)
            continue
        for raw in sorted(raws, key=lambda r: r.get("createdAt", 0)):
            if raw["id"] in state["seen"] or raw.get("status") in ("created", "started"):
                continue
            g = lidraughts.compact(raw)
            state["seen"].append(raw["id"])
            with (VAR / "games_raw.jsonl").open("a") as f:  # full API record (clocks, status, players)
                f.write(json.dumps(raw) + "\n")
            if g is None:
                continue
            with (VAR / "games.jsonl").open("a") as f:
                f.write(json.dumps(g) + "\n")
            me = "white" if bot_color(g, bot) == "W" else "black"
            lost = g["winner"] is not None and g["winner"] != me
            if lost and is_human(g, bot, raw) and raw.get("status") not in ("aborted", "noStart"):
                state["pending"].append(g)
            state["since"] = max(state["since"] or 0, raw.get("createdAt", 0) + 1)
        save_state(state)
        while state["pending"]:
            g = state["pending"][0]
            try:
                run_lesson(g["id"], g, bot, workers)
            except Exception as e:
                log({"event": "lesson_error", "game": g["id"], "error": repr(e)})
            state["pending"].pop(0)
            save_state(state)
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--bot", required=True)
    w.add_argument("--interval", type=int, default=120)
    w.add_argument("--workers", type=int, default=12)
    les = sub.add_parser("lesson")
    les.add_argument("--game", required=True)
    les.add_argument("--bot", required=True)
    les.add_argument("--workers", type=int, default=12)
    pr = sub.add_parser("promote", help="install a trained net dir as the next champion")
    pr.add_argument("--net", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "watch":
        watch(args.bot, args.interval, args.workers)
    elif args.cmd == "promote":
        dest = promote_dir(args.net)
        log({"event": "promoted", "champion": dest.name, "source": str(args.net)})
    else:
        raw = lidraughts._get(f"{lidraughts.BASE}/game/export/{args.game}", params={"moves": "true"},
                              headers={"Accept": "application/json"}).json()
        g = lidraughts.compact(raw)
        if g is None:
            sys.exit("not a standard game from the initial position")
        run_lesson(args.game, g, args.bot, args.workers)


if __name__ == "__main__":
    main()
