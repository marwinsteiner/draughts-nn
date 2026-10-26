"""Engine-vs-engine matches with real clocks, colour-swapped opening pairs and
lidraughts draw rules.

    python -m dnn.arena --a dnn --b scan --games 200 --tc 10+0.1 --workers 16
    python -m dnn.arena --a dnn:nnue=nets/v1.bin --b dnn --games 400 --nodes 20000

Engine specs: "dnn" or "scan", optionally followed by ":key=value,key=value"
Hub parameters (for dnn: nnue=..., tt-size=...; for scan: book=false, ...).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dnn.hub import HubEngine
from dnn.notation import START_HUB, hub_move_key
from dnn.rules import ENGINE, Game, MoveGen, apply_move

ROOT = Path(__file__).parents[1]
SCAN_DIR = Path(os.environ.get("SCAN_DIR", Path.home() / "draughts" / "scan-master"))
MAX_PLIES = 300


@dataclass(frozen=True)
class EngineSpec:
    kind: str
    params: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def parse(s: str) -> "EngineSpec":
        kind, _, rest = s.partition(":")
        params = tuple(tuple(kv.split("=", 1)) for kv in rest.split(",") if kv)
        return EngineSpec(kind, params)  # type: ignore[arg-type]

    def label(self) -> str:
        return self.kind + ("" if not self.params else ":" + ",".join(f"{k}={v}" for k, v in self.params))

    def start(self) -> HubEngine:
        params = dict(self.params)
        if self.kind == "dnn":
            if "nnue" in params:
                params["nnue"] = str((ROOT / params["nnue"]).resolve()) if not os.path.isabs(params["nnue"]) else params["nnue"]
            binary = ROOT / params.pop("bin") if "bin" in params else ENGINE  # e.g. bin=var/bins/dnn_base
            return HubEngine([str(binary), "hub"], params=params)
        if self.kind == "scan":
            params.setdefault("book", "false")
            return HubEngine(["./scan", "hub"], cwd=SCAN_DIR, params=params)
        raise ValueError(f"unknown engine kind {self.kind}")


def make_openings(n: int, plies: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    gen = MoveGen()
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        hub, moves = START_HUB, []
        for _ in range(plies):
            legal = gen.legal(hub)
            mv = rng.choice(legal)
            moves.append(mv)
            hub = apply_move(hub, mv)
        if hub not in seen and gen.legal(hub):
            seen.add(hub)
            out.append(moves)
    gen.close()
    return out


@dataclass
class Limits:
    tc_base: float | None = None
    tc_inc: float = 0.0
    nodes: int | None = None
    depth: int | None = None
    move_time: float | None = None


_engines: dict[str, HubEngine] = {}
_gen: MoveGen | None = None


def _get(spec: EngineSpec) -> HubEngine:
    key = spec.label()
    if key not in _engines:
        _engines[key] = spec.start()
    return _engines[key]


def play_game(white: EngineSpec, black: EngineSpec, opening: list[str], limits: Limits) -> dict:
    """Returns {'result': 1/0.5/0 for White, 'moves': [...], 'reason': str}."""
    global _gen
    if _gen is None:
        _gen = MoveGen()
    engines = {"W": _get(white), "B": _get(black)}
    if white == black:  # same spec: separate instances so TTs don't mix
        key = black.label() + "#2"
        if key not in _engines:
            _engines[key] = black.start()
        engines["B"] = _engines[key]
    for e in engines.values():
        e.new_game()
    game = Game()
    for mv in opening:
        game.push(mv)
    clocks = {"W": limits.tc_base, "B": limits.tc_base}
    while True:
        side = game.hub[0]
        legal = _gen.legal(game.hub)
        if not legal:
            return {"result": 0.0 if side == "W" else 1.0, "moves": game.moves, "reason": "no moves"}
        if game.is_draw():
            return {"result": 0.5, "moves": game.moves, "reason": "draw rule"}
        if len(game.moves) >= MAX_PLIES:
            return {"result": 0.5, "moves": game.moves, "reason": "max plies"}
        pos, hist = game.start, game.moves  # full game from the start, as lidraughts-bot sends it
        t0 = time.monotonic()
        res = engines[side].search(pos, hist, time=clocks[side], inc=limits.tc_inc, nodes=limits.nodes,
                                   depth=limits.depth, move_time=limits.move_time) if limits.tc_base is None \
            else engines[side].search(pos, hist, time=clocks[side], inc=limits.tc_inc)
        dt = time.monotonic() - t0
        if limits.tc_base is not None:
            clocks[side] = clocks[side] - dt + limits.tc_inc
            if clocks[side] < -0.5:
                return {"result": 0.0 if side == "W" else 1.0, "moves": game.moves, "reason": "time"}
        key = hub_move_key(res.move) if res.move else None
        match = [m for m in legal if key and hub_move_key(m)[:2] == key[:2] and key[2] <= hub_move_key(m)[2]]
        if len(match) != 1:
            return {"result": 0.0 if side == "W" else 1.0, "moves": game.moves,
                    "reason": f"illegal move {res.move!r}"}
        game.push(match[0])


def _pair(a: EngineSpec, b: EngineSpec, opening: list[str], limits: Limits) -> list[tuple[float, dict]]:
    """Play opening twice with colours swapped. Returns (score for A, game) pairs."""
    g1 = play_game(a, b, opening, limits) | {"white": a.label(), "black": b.label()}
    g2 = play_game(b, a, opening, limits) | {"white": b.label(), "black": a.label()}
    return [(g1["result"], g1), (1 - g2["result"], g2)]


def elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def summarise(scores: list[float]) -> str:
    n = len(scores)
    w = sum(s == 1 for s in scores)
    d = sum(s == 0.5 for s in scores)
    l_ = n - w - d
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    se = math.sqrt(var / n)
    lo, hi = elo(mean - 1.96 * se), elo(mean + 1.96 * se)
    return f"n={n} +{w} ={d} -{l_} score={mean:.3f} elo={elo(mean):+.0f} [{lo:+.0f},{hi:+.0f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=100, help="total games (rounded to pairs)")
    ap.add_argument("--tc", default=None, help="base+inc in seconds, e.g. 10+0.1")
    ap.add_argument("--nodes", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--move-time", type=float)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="append games as JSON lines")
    ap.add_argument("--openings-file", type=Path, help="JSON list of move lists (hub notation) to start from")
    ap.add_argument("--repeat", type=int, default=1, help="plays of each --openings-file entry (x2 colours)")
    args = ap.parse_args()

    a, b = EngineSpec.parse(args.a), EngineSpec.parse(args.b)
    limits = Limits(nodes=args.nodes, depth=args.depth, move_time=args.move_time)
    if args.tc:
        base, _, inc = args.tc.partition("+")
        limits.tc_base, limits.tc_inc = float(base), float(inc or 0)
    if args.openings_file:  # fixed start positions (game prefixes), each played --repeat times
        openings = json.loads(args.openings_file.read_text()) * args.repeat
    else:
        openings = make_openings(args.games // 2, args.opening_plies, args.seed)

    scores: list[float] = []
    reasons: dict[str, int] = {}
    games = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_pair, a, b, op, limits) for op in openings]
        for i, f in enumerate(as_completed(futs), 1):
            for s, g in f.result():
                scores.append(s)
                games.append(g)
                reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
            if i % max(1, len(futs) // 10) == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {a.label()} vs {b.label()}: {summarise(scores)}", flush=True)
    print("termination:", reasons)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            for g in games:
                f.write(json.dumps(g) + "\n")


if __name__ == "__main__":
    main()
