#!/usr/bin/env bash
# A/B of the drawn-ending recogniser (draw-scale=1) vs off, 2026-10-26:
#  1. conversion: 12 start positions where valhalla was >= +1 man in the three
#     2+5 draws vs Virtual-Draughts, each played 3x per colour at 10s+0.2s;
#  2. regression: normal 400-game match at 5s+0.05.
cd "$(dirname "$0")/.."
N=nets/current.bin
echo "== conversion"
.venv/bin/python -m dnn.arena --a dnn:nnue=$N,draw-scale=1 --b dnn:nnue=$N,draw-scale=0 \
  --openings-file var/conversion_openings.json --repeat 3 --tc 10+0.2 --workers 14 2>&1 | tail -2
echo "== regression"
.venv/bin/python -m dnn.arena --a dnn:nnue=$N,draw-scale=1 --b dnn:nnue=$N,draw-scale=0 \
  --games 400 --tc 5+0.05 --workers 14 --seed 51 2>&1 | tail -2 | head -1
echo "== done"
