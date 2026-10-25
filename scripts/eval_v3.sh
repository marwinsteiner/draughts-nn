#!/usr/bin/env bash
# Benchmarks after the v3 retrain (2026-10-25): champ_005 vs Scan, v3 vs champ_004, v3 vs Scan.
# Scan benchmark = fixed 400 games (opening seed 20261024), comparable across champions.
cd "$(dirname "$0")/.."
A=".venv/bin/python -m dnn.arena --games 400 --tc 5+0.05 --workers 14"
echo "== champ_005 vs scan";     $A --a dnn:nnue=nets/champ_005/net.bin --b scan --seed 20261024 2>&1 | tail -2 | head -1
echo "== v3 vs champ_004";       $A --a dnn:nnue=nets/v3/net.bin --b dnn:nnue=nets/champ_004/net.bin --seed 41 2>&1 | tail -2 | head -1
echo "== v3 vs scan";            $A --a dnn:nnue=nets/v3/net.bin --b scan --seed 20261024 2>&1 | tail -2 | head -1
echo "== done"
