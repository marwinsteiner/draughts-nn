// Alpha-beta search: iterative deepening PVS with aspiration windows, a shared
// transposition table, killer/history ordering, late-move reductions, forced-move
// extensions and a capture quiescence search. Draws follow lidraughts' rules.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

#include "eval.h"
#include "position.h"
#include "rules.h"

constexpr int MAX_PLY = 128;
constexpr int SCORE_WIN = 10000;  // a win found at ply p scores SCORE_WIN - p
constexpr int SCORE_INF = 32000;
constexpr int SCORE_WIN_MIN = SCORE_WIN - 1000;

struct Limits {
  double time = -1, inc = 0, move_time = -1;  // seconds
  int moves_to_go = 0;
  int depth = MAX_PLY - 1;
  uint64_t nodes = 0;
  bool infinite = false;
};

// Tunable search switches (Hub params of the same name), for A/B testing.
struct SearchOptions {
  int lmr = 1;            // 0: log-formula LMR from depth 3, move 3; 1: index-based (1-2 plies)
  int probcut = 1;        // shallow-search cut on/off
  int probcut_margin = 10;  // centi-men per ply of depth
};
extern SearchOptions search_options;

struct SearchInfo {
  int depth = 0;
  int score = 0;
  uint64_t nodes = 0;
  double time = 0;
  std::vector<Move> pv;
};

struct TTEntry {
  uint32_t key;
  int16_t score;
  int16_t eval;
  uint16_t move;  // 1 + index in the generated move list, 0 = none
  uint8_t depth;
  uint8_t bound;  // 2 bits bound | 6 bits generation
};

class TT {
 public:
  void resize(size_t mb);
  void clear();
  void new_search() { gen_ = (gen_ + 1) & 63; }
  bool probe(uint64_t key, TTEntry& out) const;
  void store(uint64_t key, int score, int eval, int move, int depth, int bound);

 private:
  struct alignas(64) Bucket {
    TTEntry e[5];
    uint32_t pad;
  };
  std::vector<Bucket> table_;
  uint8_t gen_ = 0;
};

enum Bound { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct Root {
  Position pos;
  std::vector<uint64_t> history;  // keys of earlier positions since the last reset, oldest first
  DrawState draw;
};

class Searcher {
 public:
  explicit Searcher(TT& tt) : tt_(tt) {}

  // Runs to completion (or until stop); returns the principal variation of the
  // last completed iteration. `on_info` is called after each iteration.
  SearchInfo run(const Root& root, const Limits& limits, const std::function<void(const SearchInfo&)>& on_info);

  std::atomic<bool> stop{false};
  std::atomic<bool> pondering{false};
  void ponderhit();

 private:
  int search(int ply, int depth, int alpha, int beta, bool pv, bool prune);
  int qsearch(int ply, int alpha, int beta, int qdepth);
  bool is_draw(int ply) const;
  void make(int ply, const Move& m);
  void check_time();
  double elapsed() const;
  void order(int ply, MoveList& list, int tt_move, int* scores) const;

  TT& tt_;
  Limits limits_;
  std::atomic<int64_t> start_ns_{0};  // steady_clock ns; reset by ponderhit from another thread
  std::atomic<double> soft_ = 0, hard_ = 0;
  uint64_t nodes_ = 0;
  int root_depth_ = 0;
  bool aborted_ = false;

  Position pos_[MAX_PLY + 2];
  Accumulator acc_[MAX_PLY + 2];
  DrawState draw_[MAX_PLY + 2];
  std::vector<uint64_t> keys_;  // history + one key per ply
  int hist_len_ = 0;            // number of pre-root keys in keys_

  Move pv_[MAX_PLY + 2][MAX_PLY + 2];
  int pv_len_[MAX_PLY + 2];
  Move killers_[MAX_PLY + 2][2];
  int history_[2][NUM_BITS][NUM_BITS];
};
