// Bitboard layout for international draughts (10x10, 50 playable squares).
//
// Square s (1..50, standard numbering, square 1 top-left from White's view)
// maps to bit  b = (s-1) + ((s-1)/5)/2.  This inserts a ghost bit after every
// two rows (bits 10, 21, 32, 43), which makes all four diagonal directions
// constant shifts: -6/-5 towards row 0 (White's promotion side), +5/+6 towards
// row 9 (Black's promotion side).  Shifting onto a ghost or outside bits 0..53
// means leaving the board.
#pragma once

#include <bit>
#include <cstdint>

using BB = uint64_t;

constexpr int NUM_SQUARES = 50;
constexpr int NUM_BITS = 54;
constexpr BB GHOSTS = (1ULL << 10) | (1ULL << 21) | (1ULL << 32) | (1ULL << 43);
constexpr BB BOARD = ((1ULL << NUM_BITS) - 1) & ~GHOSTS;

// Directions in bit space. UL/UR go towards square 1..5.
constexpr int DIR_UL = -6, DIR_UR = -5, DIR_DL = 5, DIR_DR = 6;
constexpr int DIRS[4] = {DIR_UL, DIR_UR, DIR_DL, DIR_DR};

constexpr BB ROW_TOP = 0x1FULL;              // squares 1..5   (White promotes)
constexpr BB ROW_BOTTOM = 0x1FULL << 49;     // squares 46..50 (Black promotes)

constexpr int square_to_bit(int sq) {
  const int i = sq - 1;
  return i + (i / 5) / 2;
}

constexpr int bit_to_square(int b) { return b - b / 11 + 1; }

constexpr BB bit(int b) { return 1ULL << b; }

constexpr bool on_board(int b) { return b >= 0 && b < NUM_BITS && ((GHOSTS >> b) & 1) == 0; }

inline int popcount(BB b) { return std::popcount(b); }
inline int lsb(BB b) { return std::countr_zero(b); }
inline int pop_lsb(BB& b) {
  const int s = std::countr_zero(b);
  b &= b - 1;
  return s;
}

constexpr BB shift(BB b, int d) { return (d > 0 ? (b << d) : (b >> -d)) & BOARD; }

static_assert(square_to_bit(1) == 0 && square_to_bit(5) == 4);
static_assert(square_to_bit(11) == 11 && square_to_bit(50) == 53);
static_assert(bit_to_square(53) == 50 && bit_to_square(11) == 11 && bit_to_square(9) == 10);
static_assert(std::popcount(BOARD) == NUM_SQUARES);
