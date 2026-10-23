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


def square_rc(sq: int) -> tuple[int, int]:
    row = (sq - 1) // 5
    return row, 2 * ((sq - 1) % 5) + (1 if row % 2 == 0 else 0)


def rc_square(row: int, col: int) -> int:
    return row * 5 + col // 2 + 1


def between(a: int, b: int) -> list[int]:
    """Squares strictly between a and b on their common diagonal."""
    (r0, c0), (r1, c1) = square_rc(a), square_rc(b)
    dr, dc = (r1 > r0) - (r1 < r0), (c1 > c0) - (c1 < c0)
    if abs(r1 - r0) != abs(c1 - c0):
        raise ValueError(f"{a} and {b} are not on a diagonal")
    return [rc_square(r0 + k * dr, c0 + k * dc) for k in range(1, abs(r1 - r0))]


def replay_lidraughts(moves: str, gen: MoveGen | None = None) -> list[tuple[str, str]]:
    """lidraughts move list -> [(hub position before, hub move played)].

    The API's move list writes a multi-capture as one token per jump
    ("24x33 33x22" for 24x22); consecutive chained capture tokens are merged and
    the jump path disambiguates captures with equal origin and destination."""
    toks = moves.split()
    paths: list[list[int]] = []
    for tok in toks:
        sep = "x" if "x" in tok else "-"
        sqs = [int(t) for t in tok.split(sep)]
        if sep == "x" and paths and paths[-1][0] < 0 and paths[-1][-1] == sqs[0]:
            paths[-1] += sqs[1:]
            continue
        paths.append(([-1] if sep == "x" else [0]) + sqs)  # marker: -1 capture, 0 quiet
    own = gen is None
    gen = gen or MoveGen()
    game, out = Game(), []
    try:
        for p in paths:
            capture, sqs = p[0] < 0, p[1:]
            legal = gen.legal(game.hub)
            match = [m for m in legal if hub_move_key(m)[:2] == (sqs[0], sqs[-1])]
            if len(match) > 1 and capture:
                caps = set()
                for a, b in zip(sqs, sqs[1:]):
                    caps |= {s for s in between(a, b) if game.hub[s] != "e" and game.hub[s].lower() != game.hub[sqs[0]].lower()}
                match = [m for m in match if hub_move_key(m)[2] == frozenset(caps)]
            if not match and capture and p is paths[-1]:
                break  # game ended mid-capture (flag/resign between jumps)
            if len(match) != 1:
                raise ValueError(f"cannot resolve {'x'.join(map(str, sqs))} in {game.hub}: {match}")
            out.append((game.hub, match[0]))
            game.push(match[0])
    finally:
        if own:
            gen.close()
    return out
