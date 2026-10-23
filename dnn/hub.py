"""Minimal Hub protocol client (drives dnn or Scan as a subprocess)."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def parse(line: str) -> tuple[str, dict[str, str]]:
    toks = shlex.split(line)
    if not toks:
        return "", {}
    args = {}
    for t in toks[1:]:
        k, _, v = t.partition("=")
        args[k] = v
    return toks[0], args


@dataclass
class SearchResult:
    move: str
    ponder: str | None = None
    score: float | None = None  # in men, side to move
    depth: int | None = None
    nodes: int | None = None
    pv: list[str] = field(default_factory=list)


class HubEngine:
    def __init__(self, cmd: list[str], cwd: str | Path | None = None, params: dict[str, str] | None = None):
        self.p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("hub")
        self.name = "?"
        while True:
            cmd_, args = parse(self.recv())
            if cmd_ == "id":
                self.name = args.get("name", "?")
            if cmd_ == "wait":
                break
        for k, v in (params or {}).items():
            self.send(f'set-param name={k} value="{v}"')
        self.send("init")
        while parse(self.recv())[0] != "ready":
            pass

    def send(self, line: str) -> None:
        assert self.p.stdin
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def recv(self) -> str:
        assert self.p.stdout
        line = self.p.stdout.readline()
        if not line:
            raise EOFError("engine exited")
        return line.strip()

    def new_game(self) -> None:
        self.send("new-game")

    def search(self, pos: str, moves: list[str] | None = None, *, time: float | None = None,
               inc: float = 0.0, move_time: float | None = None, depth: int | None = None,
               nodes: int | None = None, mode: str = "think") -> SearchResult:
        if moves:
            self.send(f'pos pos={pos} moves="{" ".join(moves)}"')
        else:
            self.send(f"pos pos={pos}")
        if depth is not None:
            self.send(f"level depth={depth}")
        elif nodes is not None:
            self.send(f"level nodes={nodes}")
        elif move_time is not None:
            self.send(f"level move-time={move_time:.3f}")
        elif time is not None:
            self.send(f"level time={time:.3f} inc={inc:.3f}")
        else:
            raise ValueError("no search limit given")
        self.send(f"go {mode}")
        res = SearchResult(move="")
        while True:
            cmd, args = parse(self.recv())
            if cmd == "info":
                if "score" in args:
                    try:
                        res.score = float(args["score"])
                    except ValueError:
                        pass
                if "depth" in args:
                    res.depth = int(float(args["depth"]))
                if "nodes" in args:
                    res.nodes = int(args["nodes"])
                if args.get("pv"):
                    res.pv = args["pv"].split()
            elif cmd == "done":
                res.move = args.get("move", "")
                res.ponder = args.get("ponder")
                return res
            elif cmd == "error":
                raise RuntimeError(f"{self.name}: {args.get('message')}")

    def close(self) -> None:
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
