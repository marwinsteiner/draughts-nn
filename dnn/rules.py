"""Game referee: positions as hub strings, legal moves from the engine binary,
lidraughts draw rules (mirrors engine/src/rules.h)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dnn.notation import START_HUB, hub_move_key

ENGINE = Path(os.environ.get("DNN_ENGINE", Path(__file__).parents[1] / "engine" / "dnn"))


class MoveGen:
    """Persistent `dnn moves` process: hub position in, legal hub moves out."""

    def __init__(self, engine: Path = ENGINE):
        self.p = subprocess.Popen([str(engine), "moves"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)

    def legal(self, hub: str) -> list[str]:
        assert self.p.stdin and self.p.stdout
        self.p.stdin.write(hub + "\n")
        self.p.stdin.flush()
        return self.p.stdout.readline().split()

    def close(self) -> None:
        if self.p.stdin:
            self.p.stdin.close()
        self.p.wait()


def apply_move(hub: str, move: str) -> str:
    """Apply a legal hub move ('32-28' or '28x19x23') to a hub position."""
    frm, to, caps = hub_move_key(move)
    sq = list(hub)
    piece = sq[frm]
    sq[frm] = "e"
    for c in caps:
        sq[c] = "e"
    if piece == "w" and to <= 5:
        piece = "W"
    elif piece == "b" and to >= 46:
        piece = "B"
    sq[to] = piece
    sq[0] = "B" if hub[0] == "W" else "W"
    return "".join(sq)


def draw_limit(hub: str) -> int:
    body = hub[1:]
    if sum(c != "e" for c in body) > 4:
        return 50
    for mine, theirs in (("wW", "bB"), ("bB", "wW")):
        n_mine = sum(c in mine for c in body)
        if n_mine == 1 and mine[1] in body:
            if theirs[1] not in body:
                return 50
            n_theirs = sum(c in theirs for c in body)
            return 10 if n_theirs <= 2 else 32
    return 50


@dataclass
class Game:
    """A game in progress with lidraughts' draw rules applied."""

    start: str = START_HUB
    moves: list[str] = field(default_factory=list)
    hub: str = ""
    clock: int = 0
    limit: int = 50
    since_reset: list[str] = field(default_factory=list)  # positions since last reset, current last

    def __post_init__(self):
        self.hub = self.hub or self.start
        self.limit = draw_limit(self.hub)
        self.since_reset = [self.hub]

    def push(self, move: str) -> None:
        before = self.hub
        frm, to, caps = hub_move_key(move)
        mover_king = before[frm] in "WB"
        after = apply_move(before, move)
        promotes = not mover_king and after[to] in "WB"
        limit = draw_limit(after)
        if limit == 50:
            reset = bool(caps) or not mover_king
        else:
            mover_kings = sum(c == before[frm].upper() for c in after[1:])
            first_promotion = promotes and mover_kings == 1
            pieces_before = sum(c != "e" for c in before[1:])
            reset = first_promotion or (limit == 10 and bool(caps) and pieces_before > 3) or (
                limit == 32 and bool(caps) and pieces_before > 4)
        self.clock = 0 if reset else self.clock + 1
        self.limit = limit
        self.hub = after
        self.moves.append(move)
        self.since_reset = [after] if reset else self.since_reset + [after]

    def is_draw(self) -> bool:
        if self.clock >= self.limit:
            return True
        return self.since_reset.count(self.hub) >= 3

    def hub_moves_since_reset(self) -> tuple[str, list[str]]:
        """(position at last reset, moves since) - what a Hub engine needs for repetition checks."""
        n = len(self.since_reset) - 1
        if n == 0:
            return self.hub, []
        # replay from the reset position
        return self.since_reset[0], self.moves[-n:]
