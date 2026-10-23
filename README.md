# draughts-nn

A neural-network engine for international draughts (10x10, FMJD rules) that plays
on [lidraughts.org](https://lidraughts.org) as a BOT account and retrains after
every loss to a human. Target: a lidraughts rating of 2800+.

The original 8x8 checkers experiments (Blondie24-style evolution, an AlphaZero
notebook) are kept under `legacy/`.

## Design

- **Engine (`engine/`, C++20).** Bitboard move generator (54-bit ghost-square
  layout; perft verified to depth 9), alpha-beta search (PVS, aspiration windows,
  transposition table, killers/history, late-move reductions, a shallow-search
  cut, capture quiescence with a horizon-threat check), lidraughts' draw rules,
  Lazy SMP, and the Hub protocol so that lidraughts-bot, pydraughts and the
  tooling here can drive it like any other draughts engine.
- **Evaluation.** An NNUE-style network: 200 binary features per perspective
  (own/their man/king x 50 squares, board rotated for Black), a shared first
  layer with incrementally updated accumulators, then 2x256 -> 32 -> 32 -> 1.
  The second layer runs in int8 with AVX-VNNI. The output is a win-probability
  logit, reported in centi-men via a scale K fitted to game results.
- **Teacher.** [Scan](https://github.com/rhalbersma/scan) (Fabien Letouzey,
  GPLv3) labels training positions and serves as a benchmark opponent. It is not
  part of the bot: the bot only ever runs `dnn` with its own trained weights.
- **Data.** Scan self-play from randomised openings (scores at 100k nodes plus
  game results), and rated games of top lidraughts players harvested via the
  public API.
- **Learning from losses (`dnn/learn.py`).** Each loss to a human triggers a
  lesson: Scan analyses the game deeply and marks the bot's mistakes, positions
  from the game plus Scan self-play branching from each mistake become lesson
  data, the champion net is fine-tuned on base data plus the oversampled lesson,
  and the candidate replaces the champion only if it does not lose an arena
  match against it. A single game must not make the bot weaker.

## Layout

```
engine/            C++ engine (make -C engine)
dnn/               Python tooling
  rules.py         referee: positions, lidraughts draw rules, API move-list replay
  hub.py           Hub protocol client
  arena.py         engine matches with clocks, colour-swapped opening pairs, Elo
  datagen.py       teacher self-play data
  lidraughts.py    lidraughts API helpers, human game harvester
  train.py         network training (PyTorch) and export
  learn.py         learn-from-loss watcher and lesson pipeline
  matchmaker.py    challenges other online bots when idle
bot/               lidraughts-bot config template
scripts/           sync/deploy/run helpers for the training box
tests/             movegen vs pydraughts, inference vs PyTorch, rules
legacy/            8x8 checkers prototypes
```

## Usage

```
make -C engine                                  # build ./engine/dnn
engine/dnn perft 9                              # move generator check
python -m dnn.datagen --out data/scan_sp --games 100000
python -m dnn.train --data data/scan_sp --out nets/v1
python -m dnn.arena --a dnn:nnue=nets/v1/net.bin --b scan --games 200 --tc 10+0.1
```

Running the bot needs a fresh lidraughts account (no games played), a token
with the `bot:play` scope in `~/.lidraughts_token` on the box, and a promoted
net at `nets/current.bin`:

```
bash scripts/deploy_bot.sh
DNN_BOT_NAME=<name> bash scripts/run_bot.sh --upgrade   # once, irreversible
DNN_BOT_NAME=<name> bash scripts/run_bot.sh
```

## Licence

The code in this repository is the author's. Scan is used as an external
program for labelling and benchmarking only and is not distributed here.
