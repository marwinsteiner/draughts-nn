"""The engine must reach the same position as the referee when replaying a move list
(as lidraughts-bot sends it), including king captures that end on their start square."""

import random
import subprocess

from dnn.rules import ENGINE, Game, MoveGen
from test_movegen import pytestmark  # noqa: F401  (skip when the engine isn't built)


def test_engine_replay_matches_referee():
    rng = random.Random(5)
    gen = MoveGen()
    p = subprocess.Popen([str(ENGINE), "hub"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write("hub\ninit\n")
    p.stdin.flush()
    while p.stdout.readline().strip() != "ready":
        pass
    checked = 0
    for _ in range(150):  # game 113 of this seed contains 27x27x31x32x41x42
        g = Game()
        for _ in range(250):
            legal = gen.legal(g.hub)
            if not legal or g.is_draw():
                break
            g.push(rng.choice(legal))
        p.stdin.write(f'pos pos={g.start} moves="{" ".join(g.moves)}"\nprint-pos\n')
        p.stdin.flush()
        line = p.stdout.readline().strip()
        assert line.startswith("pos pos="), line
        assert line.split()[1][4:] == g.hub, " ".join(g.moves)
        checked += 1
    p.stdin.write("quit\n")
    p.stdin.flush()
    gen.close()
    assert checked == 150
