#!/usr/bin/env bash
# Pause/resume background data jobs (datagen, label_human) and their Scan children
# on this box, e.g. around timed arena matches:  pause_bg.sh stop|cont
set -uo pipefail
sig=$([ "${1:-}" = "cont" ] && echo CONT || echo STOP)
for p in $(pgrep -f "dnn[.](datagen|label_human)"); do
  kill -$sig $p $(pgrep -P $p) 2>/dev/null
done
echo "$sig sent; stopped processes now: $(ps -eo stat= | grep -c '^T')"
