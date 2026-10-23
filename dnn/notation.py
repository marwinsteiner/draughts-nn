"""Conversions between position formats.

hub:  'W'/'B' + 50 chars (w b W B e) in square order, as used by the Hub protocol.
fen:  PDN FEN as used by lidraughts / pydraughts, e.g. "W:W31,32,K45:B1,2".
"""

from __future__ import annotations

START_HUB = "W" + "b" * 20 + "e" * 10 + "w" * 20


def fen_to_hub(fen: str) -> str:
    fen = fen.strip().strip('"').rstrip(".")
    parts = fen.split(":")
    side = parts[0].upper()
    squares = ["e"] * 50
    for part in parts[1:]:
        if not part:
            continue
        colour, body = part[0].upper(), part[1:]
        for tok in filter(None, body.split(",")):
            king = tok.startswith("K")
            tok = tok.lstrip("K")
            if "-" in tok:  # range "31-35"
                lo, hi = map(int, tok.split("-"))
                sqs = range(lo, hi + 1)
            else:
                sqs = [int(tok)]
            ch = "w" if colour == "W" else "b"
            for s in sqs:
                squares[s - 1] = ch.upper() if king else ch
    return side + "".join(squares)


def hub_to_fen(hub: str) -> str:
    side = hub[0].upper()
    white, black = [], []
    for i, ch in enumerate(hub[1:], start=1):
        if ch in "wW":
            white.append(("K" if ch == "W" else "") + str(i))
        elif ch in "bB":
            black.append(("K" if ch == "B" else "") + str(i))
    return f"{side}:W{','.join(white)}:B{','.join(black)}"


def hub_move_key(move: str) -> tuple[int, int, frozenset[int]]:
    """Canonical (from, to, captured-set) for a hub move like '28x19x23'."""
    sep = "x" if "x" in move else "-"
    nums = [int(t) for t in move.split(sep)]
    return nums[0], nums[1], frozenset(nums[2:])
