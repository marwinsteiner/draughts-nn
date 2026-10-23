"""Engine move generator vs pydraughts on random-game positions (kings, long captures)."""

import os
import random
import subprocess
from pathlib import Path

import pytest

from dnn.notation import fen_to_hub, hub_move_key

ENGINE = Path(os.environ.get("DNN_ENGINE", Path(__file__).parents[1] / "engine" / "dnn"))
pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="engine not built")


def engine_moves(positions: list[str]) -> list[set]:
    out = subprocess.run([str(ENGINE), "moves"], input="\n".join(positions) + "\n",
                         capture_output=True, text=True, check=True).stdout
    return [{hub_move_key(m) for m in line.split()} for line in out.splitlines()]


def random_positions(n_games: int, seed: int) -> list[tuple[str, set]]:
    from draughts import Board

    rng = random.Random(seed)
    samples = []
    for _ in range(n_games):
        board = Board(variant="standard")
        for _ply in range(300):
            moves = board.legal_moves()
            if not moves or board.is_over():
                break
            samples.append((fen_to_hub(board.fen), {hub_move_key(m.hub_move) for m in moves}))
            board.push(rng.choice(moves))
    return samples


def test_start_position_perft():
    out = subprocess.run([str(ENGINE), "perft", "6"], capture_output=True, text=True, check=True).stdout
    counts = [int(line.split()[2]) for line in out.splitlines()]
    assert counts == [9, 81, 658, 4265, 27117, 167140]


def test_random_positions_match_pydraughts():
    samples = random_positions(n_games=60, seed=1)
    ours = engine_moves([hub for hub, _ in samples])
    kings_seen = sum(any(c in "WB" for c in hub[1:]) for hub, _ in samples)
    assert kings_seen > 100, "test should cover king positions"
    for (hub, theirs), mine in zip(samples, ours, strict=True):
        assert mine == theirs, hub
