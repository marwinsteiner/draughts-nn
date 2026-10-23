#include "position.h"

#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace zobrist {
uint64_t piece[2][2][NUM_BITS];
uint64_t side;

void init() {
  std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
  for (auto& c : piece)
    for (auto& k : c)
      for (auto& b : k) b = rng();
  side = rng();
}
}  // namespace zobrist

Position Position::start() {
  Position p;
  for (int sq = 1; sq <= 20; ++sq) p.pieces[BLACK] |= bit(square_to_bit(sq));
  for (int sq = 31; sq <= 50; ++sq) p.pieces[WHITE] |= bit(square_to_bit(sq));
  p.side = WHITE;
  p.key = p.compute_key();
  return p;
}

Position Position::from_hub(const std::string& s) {
  if (s.size() != 51) throw std::invalid_argument("hub position must have 51 chars: " + s);
  Position p;
  if (s[0] == 'W' || s[0] == 'w')
    p.side = WHITE;
  else if (s[0] == 'B' || s[0] == 'b')
    p.side = BLACK;
  else
    throw std::invalid_argument("bad side to move in: " + s);
  for (int sq = 1; sq <= 50; ++sq) {
    const BB b = bit(square_to_bit(sq));
    switch (s[sq]) {
      case 'w': p.pieces[WHITE] |= b; break;
      case 'b': p.pieces[BLACK] |= b; break;
      case 'W': p.pieces[WHITE] |= b; p.kings |= b; break;
      case 'B': p.pieces[BLACK] |= b; p.kings |= b; break;
      case 'e': case '.': break;
      default: throw std::invalid_argument("bad square char in: " + s);
    }
  }
  p.key = p.compute_key();
  return p;
}

std::string Position::to_hub() const {
  std::string s(51, 'e');
  s[0] = side == WHITE ? 'W' : 'B';
  for (int sq = 1; sq <= 50; ++sq) {
    const BB b = bit(square_to_bit(sq));
    const bool k = kings & b;
    if (pieces[WHITE] & b) s[sq] = k ? 'W' : 'w';
    if (pieces[BLACK] & b) s[sq] = k ? 'B' : 'b';
  }
  return s;
}

uint64_t Position::compute_key() const {
  uint64_t k = side == BLACK ? zobrist::side : 0;
  for (int c = 0; c < 2; ++c) {
    BB b = pieces[c];
    while (b) {
      const int sq = pop_lsb(b);
      k ^= zobrist::piece[c][(kings >> sq) & 1][sq];
    }
  }
  return k;
}

void Position::make(const Move& m) {
  const Color us = side, them = ~side;
  const BB from = bit(m.from), to = bit(m.to);
  const bool was_king = kings & from;

  pieces[us] ^= from | to;
  key ^= zobrist::piece[us][was_king][m.from];
  if (was_king) kings ^= from | to;

  BB cap = m.captured;
  while (cap) {
    const int sq = pop_lsb(cap);
    key ^= zobrist::piece[them][(kings >> sq) & 1][sq];
  }
  pieces[them] &= ~m.captured;
  kings &= ~m.captured;

  const bool promotes = !was_king && (to & (us == WHITE ? ROW_TOP : ROW_BOTTOM));
  if (promotes) kings |= to;
  key ^= zobrist::piece[us][was_king || promotes][m.to];

  side = them;
  key ^= zobrist::side;
}

// ---------------------------------------------------------------------------
// Move generation

namespace {

struct CaptureGen {
  MoveList& list;
  BB opp;      // opponent pieces
  BB occ;      // all pieces minus the moving one (captured pieces stay on board)
  int from;

  void add(int to, BB captured) {
    const int n = popcount(captured);
    if (n < list.capture_count) return;
    if (n > list.capture_count) {
      list.size = 0;
      list.capture_count = n;
    }
    const Move m{uint8_t(from), uint8_t(to), captured};
    for (int i = 0; i < list.size; ++i)
      if (list[i] == m) return;
    if (list.size < MAX_MOVES) list[list.size++] = m;
  }

  void man(int sq, BB captured) {
    bool extended = false;
    for (int d : DIRS) {
      const int over = sq + d, land = over + d;
      if (!on_board(over) || !on_board(land)) continue;
      if (!(opp & bit(over)) || (captured & bit(over)) || (occ & bit(land))) continue;
      extended = true;
      man(land, captured | bit(over));
    }
    if (!extended && captured) add(sq, captured);
  }

  void king(int sq, BB captured) {
    bool extended = false;
    for (int d : DIRS) {
      int over = sq + d;
      while (on_board(over) && !(occ & bit(over))) over += d;
      if (!on_board(over) || !(opp & bit(over)) || (captured & bit(over))) continue;
      for (int land = over + d; on_board(land) && !(occ & bit(land)); land += d) {
        extended = true;
        king(land, captured | bit(over));
      }
    }
    if (!extended && captured) add(sq, captured);
  }
};

}  // namespace

void generate_moves(const Position& pos, MoveList& list) {
  list.size = 0;
  list.capture_count = 0;
  const Color us = pos.side;
  const BB own = pos.pieces[us], opp = pos.pieces[~us];

  // Captures (mandatory, maximum count).
  {
    BB b = own;
    while (b) {
      const int sq = pop_lsb(b);
      CaptureGen g{list, opp, pos.occupied() & ~bit(sq), sq};
      if (pos.kings & bit(sq))
        g.king(sq, 0);
      else
        g.man(sq, 0);
    }
    if (list.capture_count > 0) return;
  }

  const BB empty = pos.empty();
  // Men.
  const BB men = pos.men(us);
  const int fwd[2] = {us == WHITE ? DIR_UL : DIR_DL, us == WHITE ? DIR_UR : DIR_DR};
  for (int d : fwd) {
    BB targets = shift(men, d) & empty;
    while (targets) {
      const int to = pop_lsb(targets);
      list[list.size++] = Move{uint8_t(to - d), uint8_t(to), 0};
    }
  }
  // Kings.
  BB k = pos.kings_of(us);
  while (k) {
    const int sq = pop_lsb(k);
    for (int d : DIRS)
      for (int to = sq + d; on_board(to) && (empty & bit(to)); to += d)
        list[list.size++] = Move{uint8_t(sq), uint8_t(to), 0};
  }
}

bool has_capture(const Position& pos) {
  // Cheap test: any piece adjacent to an opponent with an empty square behind,
  // or a king seeing one along a diagonal.
  const Color us = pos.side;
  const BB opp = pos.pieces[~us], empty = pos.empty();
  const BB men = pos.men(us);
  for (int d : DIRS)
    if (shift(shift(men, d) & opp, d) & empty) return true;
  BB k = pos.kings_of(us);
  const BB occ = pos.occupied();
  while (k) {
    const int sq = pop_lsb(k);
    for (int d : DIRS) {
      int over = sq + d;
      while (on_board(over) && !(occ & bit(over))) over += d;
      if (on_board(over) && (opp & bit(over)) && on_board(over + d) && (empty & bit(over + d))) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Notation

std::string move_to_pdn(const Move& m) {
  return std::to_string(bit_to_square(m.from)) + (m.is_capture() ? "x" : "-") +
         std::to_string(bit_to_square(m.to));
}

std::string move_to_hub(const Move& m) {
  std::string s = move_to_pdn(m);
  BB c = m.captured;
  while (c) s += "x" + std::to_string(bit_to_square(pop_lsb(c)));
  return s;
}

bool parse_move(const Position& pos, const std::string& s, Move& out) {
  std::vector<int> nums;
  std::string cur;
  for (char ch : s) {
    if (ch >= '0' && ch <= '9')
      cur += ch;
    else if (ch == '-' || ch == 'x' || ch == 'X' || ch == ':') {
      if (!cur.empty()) nums.push_back(std::stoi(cur));
      cur.clear();
    }
  }
  if (!cur.empty()) nums.push_back(std::stoi(cur));
  if (nums.size() < 2) return false;
  for (int n : nums)
    if (n < 1 || n > 50) return false;

  MoveList list;
  generate_moves(pos, list);
  const int from = square_to_bit(nums[0]), to = square_to_bit(nums[1]);
  BB caps = 0;
  for (size_t i = 2; i < nums.size(); ++i) caps |= bit(square_to_bit(nums[i]));

  const Move* found = nullptr;
  int matches = 0;
  for (const Move& m : list) {
    if (m.from != from || m.to != to) continue;
    if (caps && (m.captured & caps) != caps) continue;
    found = &m;
    ++matches;
  }
  if (matches == 0) {
    // PDN path form "a x b x c": first and last are origin/destination.
    if (nums.size() > 2) {
      const int to2 = square_to_bit(nums.back());
      for (const Move& m : list)
        if (m.from == from && m.to == to2) {
          found = &m;
          ++matches;
        }
    }
    if (matches == 0) return false;
  }
  if (matches > 1) return false;
  out = *found;
  return true;
}

uint64_t perft(const Position& pos, int depth) {
  MoveList list;
  generate_moves(pos, list);
  if (depth <= 1) return depth == 1 ? list.size : 1;
  uint64_t n = 0;
  for (const Move& m : list) {
    Position p = pos;
    p.make(m);
    n += perft(p, depth - 1);
  }
  return n;
}
