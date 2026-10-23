// Position, moves and legal move generation for international draughts (FMJD rules):
// men move forward and capture in all four directions, kings fly, capturing is
// mandatory and the sequence taking the most pieces must be chosen; captured
// pieces are removed only after the move (Turkish strike rule: a piece cannot be
// jumped twice); a man promotes only if it ends its move on the last row.
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "bitboard.h"

enum Color : int { WHITE = 0, BLACK = 1 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

struct Move {
  uint8_t from = 0, to = 0;  // bit indices
  BB captured = 0;           // captured pieces (bit set)

  bool is_capture() const { return captured != 0; }
  bool operator==(const Move& o) const { return from == o.from && to == o.to && captured == o.captured; }
};

constexpr int MAX_MOVES = 512;

struct MoveList {
  std::array<Move, MAX_MOVES> moves;
  int size = 0;
  int capture_count = 0;  // pieces captured by each move (all equal by the max rule)

  Move* begin() { return moves.data(); }
  Move* end() { return moves.data() + size; }
  const Move* begin() const { return moves.data(); }
  const Move* end() const { return moves.data() + size; }
  Move& operator[](int i) { return moves[i]; }
  const Move& operator[](int i) const { return moves[i]; }
};

namespace zobrist {
extern uint64_t piece[2][2][NUM_BITS];  // [color][is_king][bit]
extern uint64_t side;
void init();
}  // namespace zobrist

struct Position {
  BB pieces[2] = {0, 0};
  BB kings = 0;
  Color side = WHITE;
  uint64_t key = 0;

  static Position start();
  // Hub format: 'W'/'B' then 50 chars from {w,b,W,B,e} in square order.
  static Position from_hub(const std::string& s);
  std::string to_hub() const;

  BB occupied() const { return pieces[WHITE] | pieces[BLACK]; }
  BB empty() const { return BOARD & ~occupied(); }
  BB men(Color c) const { return pieces[c] & ~kings; }
  BB kings_of(Color c) const { return pieces[c] & kings; }

  void make(const Move& m);
  uint64_t compute_key() const;
};

void generate_moves(const Position& pos, MoveList& list);
bool has_capture(const Position& pos);

// Notation. Quiet: "32-28". Captures: "28x19" (PDN), hub form adds the captured
// squares: "28x19x23".
std::string move_to_pdn(const Move& m);
std::string move_to_hub(const Move& m);
// Parses PDN/hub notation (including multi-step "a x b x c" PDN paths) against the
// legal moves of `pos`. Returns false if no unique legal move matches.
bool parse_move(const Position& pos, const std::string& s, Move& out);

uint64_t perft(const Position& pos, int depth);
