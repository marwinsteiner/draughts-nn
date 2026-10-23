"""Referee: lidraughts move-list replay and draw rules."""

import pytest

from dnn.notation import hub_move_key
from dnn.rules import Game, between, replay_lidraughts
from test_movegen import ENGINE

pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="engine not built")

# lidraughts game zGzjswGM (API move list: multi-captures are split per jump)
GAME = ("32-28 18-23 38-32 12-18 42-38 7-12 47-42 20-24 33-29 24x33 33x22 34-30 15-20 39-33 10-15 44-39 1-7 "
        "30-25 20-24 50-44 14-20 25x14 9x20 32-28 23x32 37x28 4-9 40-34 5-10 34-29 10-14 44-40 16-21 40-34 "
        "21-27 41-37 18-23 29x18 12x23 23x32 32x41 46x37 7-12 49-44 2-7 31-26 13-18 37-32 8-13 32x21 22-28")
# the same game's PDN export, one token per move
PDN = ("32-28 18-23 38-32 12-18 42-38 7-12 47-42 20-24 33-29 24x22 34-30 15-20 39-33 10-15 44-39 1-7 30-25 "
       "20-24 50-44 14-20 25x14 9x20 32-28 23x32 37x28 4-9 40-34 5-10 34-29 10-14 44-40 16-21 40-34 21-27 "
       "41-37 18-23 29x18 12x41 46x37 7-12 49-44 2-7 31-26 13-18 37-32 8-13 32x21 22-28")


def test_between():
    assert between(13, 24) == [19]
    assert between(46, 5) == [41, 37, 32, 28, 23, 19, 14, 10]


def test_replay_merges_split_captures():
    plies = replay_lidraughts(GAME)
    pdn = []
    for _, m in plies:
        a, b, caps = hub_move_key(m)
        pdn.append(f"{a}{'x' if caps else '-'}{b}")
    assert pdn == PDN.split()


def test_threefold_repetition_and_clock():
    sq = ["e"] * 50
    for s_, c in ((46, "W"), (5, "B"), (31, "w"), (32, "w"), (19, "b"), (20, "b")):
        sq[s_ - 1] = c
    g = Game(start="W" + "".join(sq))
    cycle = ["46-41", "5-10", "41-46", "10-5"]
    for m in cycle * 2:
        assert not g.is_draw()
        g.push(m)
    assert g.clock == 8 and g.limit == 50
    assert g.is_draw()  # start position seen for the third time


def test_lone_king_five_move_rule():
    sq = ["e"] * 50
    sq[46 - 1], sq[50 - 1], sq[5 - 1] = "W", "W", "B"  # two kings vs one: 5 moves each
    g = Game(start="W" + "".join(sq))
    assert g.limit == 10
    moves = ["46-41", "5-10", "50-45", "10-14", "41-37", "14-10", "45-40", "10-5", "37-41", "5-10"]
    for m in moves:
        assert not g.is_draw()
        g.push(m)
    assert g.is_draw()
