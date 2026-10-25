#include "search.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>

SearchOptions search_options;

// ---------------------------------------------------------------------------
// Transposition table

void TT::resize(size_t mb) {
  const size_t n = std::max<size_t>(1, mb * 1024 * 1024 / sizeof(Bucket));
  table_.assign(n, Bucket{});
}

void TT::clear() { std::fill(table_.begin(), table_.end(), Bucket{}); }

bool TT::probe(uint64_t key, TTEntry& out) const {
  if (table_.empty()) return false;
  const Bucket& b = table_[(key >> 32) % table_.size()];
  const uint32_t k = uint32_t(key);
  for (const TTEntry& e : b.e)
    if (e.key == k && e.bound) {
      out = e;
      return true;
    }
  return false;
}

void TT::store(uint64_t key, int score, int eval, int move, int depth, int bound) {
  if (table_.empty()) return;
  Bucket& b = table_[(key >> 32) % table_.size()];
  const uint32_t k = uint32_t(key);
  TTEntry* victim = &b.e[0];
  int worst = 1 << 30;
  for (TTEntry& e : b.e) {
    if (e.key == k || !e.bound) {
      victim = &e;
      break;
    }
    const int age = (gen_ - (e.bound >> 2)) & 63;
    const int value = e.depth - 4 * age;
    if (value < worst) {
      worst = value;
      victim = &e;
    }
  }
  if (victim->key == k && victim->bound && move == 0) move = victim->move;  // keep old best move
  victim->key = k;
  victim->score = int16_t(score);
  victim->eval = int16_t(eval);
  victim->move = uint16_t(move);
  victim->depth = uint8_t(std::clamp(depth, 0, 255));
  victim->bound = uint8_t(bound | (gen_ << 2));
}

namespace {

int score_to_tt(int s, int ply) {
  if (s >= SCORE_WIN_MIN) return s + ply;
  if (s <= -SCORE_WIN_MIN) return s - ply;
  return s;
}
int score_from_tt(int s, int ply) {
  if (s >= SCORE_WIN_MIN) return s - ply;
  if (s <= -SCORE_WIN_MIN) return s + ply;
  return s;
}

bool opponent_can_capture(const Position& p) {
  Position q = p;
  q.side = ~q.side;
  return has_capture(q);
}

}  // namespace

// ---------------------------------------------------------------------------
// Search

namespace {
int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

double Searcher::elapsed() const { return (now_ns() - start_ns_) * 1e-9; }

void Searcher::ponderhit() {
  start_ns_ = now_ns();
  pondering = false;
}

void Searcher::check_time() {
  if (stop) {
    aborted_ = true;
    return;
  }
  if (limits_.nodes && nodes_ >= limits_.nodes) aborted_ = true;
  if (pondering || limits_.infinite) return;
  if (hard_ > 0 && root_depth_ > 1 && elapsed() >= hard_) aborted_ = true;
}

bool Searcher::is_draw(int ply) const {
  const DrawState& d = draw_[ply];
  if (d.clock >= d.limit) return true;
  // Any repetition since the last reset (same side to move) counts as a draw.
  const int cur = hist_len_ + ply;
  const int back = std::min(d.clock, cur);
  for (int i = 2; i <= back; i += 2)
    if (keys_[cur - i] == keys_[cur]) return true;
  return false;
}

void Searcher::make(int ply, const Move& m) {
  const Position& p = pos_[ply];
  Position& c = pos_[ply + 1];
  c = p;
  c.make(m);
  eval::update(acc_[ply], p, m, acc_[ply + 1]);
  draw_[ply + 1] = next_draw_state(p, m, c, draw_[ply]);
  const size_t idx = hist_len_ + ply + 1;
  if (keys_.size() <= idx) keys_.resize(idx + 1);
  keys_[idx] = c.key;
}

void Searcher::order(int ply, MoveList& list, int tt_move, int* scores) const {
  const Position& p = pos_[ply];
  for (int i = 0; i < list.size; ++i) {
    const Move& m = list[i];
    int s;
    if (i + 1 == tt_move)
      s = 1 << 30;
    else if (m.is_capture())
      s = (1 << 24) + 64 * popcount(m.captured & p.kings);
    else if (m == killers_[ply][0])
      s = (1 << 22);
    else if (m == killers_[ply][1])
      s = (1 << 22) - 1;
    else
      s = history_[p.side][m.from][m.to];
    scores[i] = s;
  }
}

int Searcher::qsearch(int ply, int alpha, int beta, int qdepth) {
  ++nodes_;
  if ((nodes_ & 1023) == 0) check_time();
  if (aborted_) return 0;
  pv_len_[ply] = ply;

  MoveList list;
  generate_moves(pos_[ply], list);
  if (list.size == 0) return -SCORE_WIN + ply;
  if (ply >= MAX_PLY) return eval::evaluate(pos_[ply], &acc_[ply]);

  const bool forced = list.capture_count > 0;
  int best = -SCORE_INF;
  if (!forced) {
    best = eval::evaluate(pos_[ply], &acc_[ply]);
    // Quiet position: stand pat unless the opponent threatens a capture right
    // at the horizon, in which case look one ply further at every reply.
    if (best >= beta || qdepth > 0 || !opponent_can_capture(pos_[ply])) return best;
    alpha = std::max(alpha, best);
  }
  for (const Move& m : list) {
    make(ply, m);
    const int s = -qsearch(ply + 1, -beta, -alpha, forced ? qdepth : qdepth + 1);
    if (aborted_) return 0;
    if (s > best) {
      best = s;
      if (s > alpha) {
        alpha = s;
        if (s >= beta) break;
      }
    }
  }
  return best;
}

int Searcher::search(int ply, int depth, int alpha, int beta, bool pv, bool prune) {
  pv_len_[ply] = ply;
  if (ply > 0 && is_draw(ply)) return 0;
  if (depth <= 0 || ply >= MAX_PLY - 1) return qsearch(ply, alpha, beta, 0);

  ++nodes_;
  if ((nodes_ & 1023) == 0) check_time();
  if (aborted_) return 0;

  // Mate distance pruning.
  alpha = std::max(alpha, -SCORE_WIN + ply);
  beta = std::min(beta, SCORE_WIN - ply - 1);
  if (alpha >= beta) return alpha;

  const Position& p = pos_[ply];
  const uint64_t key = p.key;
  TTEntry te{};
  const bool hit = tt_.probe(key, te);
  int tt_move = hit ? te.move : 0;
  if (hit && !pv && te.depth >= depth) {
    const int s = score_from_tt(te.score, ply);
    const int b = te.bound & 3;
    if ((b == BOUND_EXACT) || (b == BOUND_LOWER && s >= beta) || (b == BOUND_UPPER && s <= alpha)) return s;
  }

  MoveList list;
  generate_moves(p, list);
  if (list.size == 0) return -SCORE_WIN + ply;
  if (tt_move > list.size) tt_move = 0;

  // Shallow-search cut: if a much shallower search of this node clears beta by
  // a depth-scaled margin, assume the full search would too. The verification
  // search and its subtree do not prune again.
  if (search_options.probcut && prune && !pv && depth >= 3 && std::abs(beta) < SCORE_WIN_MIN) {
    const int margin = search_options.probcut_margin * depth;
    const int rbeta = beta + margin;
    const int s = search(ply, depth * 2 / 5, rbeta - 1, rbeta, false, false);
    if (aborted_) return 0;
    if (s >= rbeta) return s - margin;
  }

  // Forced moves (single legal move) cost nothing: extend.
  const int ext = list.size == 1 ? 1 : 0;

  int scores[MAX_MOVES], gen_idx[MAX_MOVES];
  order(ply, list, tt_move, scores);
  for (int i = 0; i < list.size; ++i) gen_idx[i] = i;

  const int orig_alpha = alpha;
  int best = -SCORE_INF, best_idx = -1;
  for (int n = 0; n < list.size; ++n) {
    // selection sort step
    int bi = n;
    for (int j = n + 1; j < list.size; ++j)
      if (scores[j] > scores[bi]) bi = j;
    std::swap(list[n], list[bi]);
    std::swap(scores[n], scores[bi]);
    std::swap(gen_idx[n], gen_idx[bi]);
    const Move m = list[n];

    make(ply, m);
    const int new_depth = depth - 1 + ext;
    int s;
    if (n == 0) {
      s = -search(ply + 1, new_depth, -beta, -alpha, pv, prune);
    } else {
      // Late-move reductions: quiet, non-promoting moves after the first few.
      int r = 0;
      const bool promotion = !(p.kings & bit(m.from)) && (bit(m.to) & (p.side == WHITE ? ROW_TOP : ROW_BOTTOM));
      if (search_options.lmr == 1) {
        if (depth >= 2 && n >= (pv ? 3 : 1) && !m.is_capture() && !promotion && ext == 0)
          r = (!pv && n >= 4) ? 2 : 1;
      } else if (depth >= 3 && n >= 3 && !m.is_capture() && !promotion && ext == 0) {
        r = int(0.5 + std::log(double(depth)) * std::log(double(n)) / 2.5);
        if (pv) r = std::max(0, r - 1);
        r = std::clamp(r, 0, new_depth - 1);
      }
      s = -search(ply + 1, new_depth - r, -alpha - 1, -alpha, false, prune);
      if (s > alpha && r > 0) s = -search(ply + 1, new_depth, -alpha - 1, -alpha, false, prune);
      if (s > alpha && s < beta && pv) s = -search(ply + 1, new_depth, -beta, -alpha, true, prune);
    }
    if (aborted_) return 0;

    if (s > best) {
      best = s;
      best_idx = n;
      if (s > alpha) {
        alpha = s;
        pv_[ply][ply] = m;
        for (int i = ply + 1; i < pv_len_[ply + 1]; ++i) pv_[ply][i] = pv_[ply + 1][i];
        pv_len_[ply] = std::max(ply + 1, pv_len_[ply + 1]);
        if (s >= beta) {
          if (!m.is_capture()) {
            if (!(killers_[ply][0] == m)) {
              killers_[ply][1] = killers_[ply][0];
              killers_[ply][0] = m;
            }
            int& h = history_[p.side][m.from][m.to];
            h += depth * depth;
            if (h > (1 << 20)) {
              for (auto& a : history_)
                for (auto& b : a)
                  for (int& x : b) x /= 2;
            }
          }
          break;
        }
      }
    }
  }

  // The TT stores the move by its index in generation order.
  const int store_move = best_idx >= 0 ? gen_idx[best_idx] + 1 : 0;
  const int bound = best >= beta ? BOUND_LOWER : (best > orig_alpha ? BOUND_EXACT : BOUND_UPPER);
  tt_.store(key, score_to_tt(best, ply), 0, store_move, depth, bound);
  return best;
}

SearchInfo Searcher::run(const Root& root, const Limits& limits,
                         const std::function<void(const SearchInfo&)>& on_info) {
  start_ns_ = now_ns();
  limits_ = limits;
  nodes_ = 0;
  aborted_ = false;  // `stop` is reset by the caller before launching (helpers may be stopped early)
  tt_.new_search();
  std::memset(killers_, 0, sizeof killers_);
  for (auto& a : history_)
    for (auto& b : a)
      for (int& x : b) x /= 8;

  // Time budget.
  double soft = 0, hard = 0;
  if (limits.move_time > 0) {
    soft = hard = std::max(0.01, limits.move_time - 0.02);
  } else if (limits.has_clock) {
    // The GUI may pass a clock at or below zero, or a negative increment (lidraughts-bot
    // subtracts its lag allowance from both): always end with a small positive budget.
    const double overhead = 0.15;  // network lag reserve
    const double left = std::max(0.02, limits.time - overhead);
    const double inc = std::max(0.0, limits.inc);
    const int mtg = limits.moves_to_go > 0 ? limits.moves_to_go : 30;
    soft = std::max(0.005, std::min(left / mtg + 0.75 * inc, left * 0.25));
    hard = std::max(soft, std::min(soft * 3.5, left * 0.4));
  }
  soft_ = soft;
  hard_ = hard;

  pos_[0] = root.pos;
  draw_[0] = root.draw;
  eval::refresh(pos_[0], acc_[0]);
  keys_ = root.history;
  hist_len_ = int(keys_.size());
  keys_.push_back(root.pos.key);

  MoveList root_moves;
  generate_moves(root.pos, root_moves);
  SearchInfo best;
  if (root_moves.size > 0) best.pv = {root_moves[0]};
  if (root_moves.size <= 1 && !limits.infinite && !pondering) {
    best.depth = 0;
    return best;
  }

  int prev = 0;
  for (int depth = 1; depth <= limits.depth; ++depth) {
    root_depth_ = depth;
    int delta = 30;
    int alpha = -SCORE_INF, beta = SCORE_INF;
    if (depth >= 4) {
      alpha = std::max(prev - delta, -SCORE_INF);
      beta = std::min(prev + delta, int(SCORE_INF));
    }
    int s;
    while (true) {
      s = search(0, depth, alpha, beta, true, true);
      if (aborted_) break;
      if (s <= alpha) {
        alpha = std::max(s - delta, -SCORE_INF);
        delta *= 2;
      } else if (s >= beta) {
        beta = std::min(s + delta, int(SCORE_INF));
        delta *= 2;
      } else {
        break;
      }
    }
    if (aborted_) break;
    prev = s;
    best.depth = depth;
    best.score = s;
    best.nodes = nodes_;
    best.time = elapsed();
    best.pv.assign(pv_[0], pv_[0] + pv_len_[0]);
    if (on_info) on_info(best);

    if (std::abs(s) >= SCORE_WIN_MIN && depth > 8 && !limits.infinite) {
      const int dist = SCORE_WIN - std::abs(s);
      if (depth > dist + 2) break;
    }
    if (!pondering && !limits.infinite && soft_ > 0 && elapsed() > soft_ * 0.6) break;
  }
  // Opening variety. Bots that replay the same opening against the same opponent
  // get those games ignored for rating (lidraughts BotFarming), and fixed openings
  // are easy for humans to prepare against: early in the game, pick at random
  // among root moves scoring within a small margin of the best.
  if (search_options.opening_random_plies > 0 && root.game_ply < search_options.opening_random_plies &&
      !limits.infinite && !pondering && !stop && best.depth >= 4 && root_moves.size > 1 &&
      (soft_ <= 0 || elapsed() < soft_ * 0.8)) {  // only if the move's time budget allows
    const Limits saved = limits_;
    limits_.infinite = true;  // short fixed-depth pass; not subject to the clock
    limits_.nodes = 0;
    aborted_ = false;
    const int d = std::min(best.depth - 2, 8);
    std::vector<std::pair<int, Move>> scored;
    for (const Move& m : root_moves) {
      make(0, m);
      const int s = -search(1, d, -SCORE_INF, SCORE_INF, true, true);
      if (aborted_) break;
      scored.emplace_back(s, m);
    }
    limits_ = saved;
    if (!aborted_ && scored.size() == size_t(root_moves.size)) {
      int top = -SCORE_INF;
      for (const auto& [s, m] : scored) top = std::max(top, s);
      std::vector<Move> ok;
      for (const auto& [s, m] : scored)
        if (s >= top - search_options.opening_random_margin) ok.push_back(m);
      static std::mt19937_64 rng(std::random_device{}());
      const Move pick = ok[rng() % ok.size()];
      if (!(pick == best.pv[0])) best.pv = {pick};
    }
    aborted_ = false;
  }
  // Infinite/ponder searches only return when told to.
  while ((limits.infinite || pondering) && !stop) {
    if (!pondering && soft_ > 0 && elapsed() > soft_) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  best.nodes = nodes_;
  best.time = elapsed();
  return best;
}
