"""The engine must stay within its clock whatever the GUI sends: zero or negative
increment (lidraughts-bot subtracts its lag allowance) and a clock at or below zero."""

import subprocess
import time

from dnn.rules import ENGINE
from test_movegen import pytestmark  # noqa: F401  (skip when the engine isn't built)

POS = "Wbbbbbbbbbbbbbbbbbbbbeeeeeeeeeewwwwwwwwwwwwwwwwwwww"


def think(level: str) -> float:
    p = subprocess.Popen([str(ENGINE), "hub"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(f"hub\ninit\npos pos={POS}\n{level}\n")
    p.stdin.flush()
    while p.stdout.readline().strip() != "ready":
        pass
    t0 = time.monotonic()
    p.stdin.write("go think\n")
    p.stdin.flush()
    while not p.stdout.readline().startswith("done"):
        pass
    dt = time.monotonic() - t0
    p.stdin.write("quit\n")
    p.stdin.flush()
    p.wait(timeout=5)
    return dt


def test_negative_increment_respects_clock():
    assert think("level time=2.15 inc=-0.1") < 0.86  # budget <= 40% of the clock


def test_clock_at_or_below_zero_moves_at_once():
    assert think("level time=0 inc=-0.1") < 0.3
    assert think("level time=-0.5 inc=-0.1") < 0.3


def test_normal_clock_still_thinks():
    assert think("level time=60 inc=2") > 0.5


def test_no_increment_banks_time_for_overhead():
    # 3+0 with 12s left: the per-move lag reserve (0.3s x 40) leaves nothing to spend
    assert think("level time=12 inc=0") < 0.2
    # early in a 3+0 game: at most ~(180-12)/50 = 3.4s
    assert think("level time=180 inc=0") < 3.6
