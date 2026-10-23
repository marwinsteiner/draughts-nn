// Static evaluation: an NNUE-style network when weights are loaded, a small
// handcrafted material/structure evaluation otherwise. Scores are in centi-men
// (100 = one man) from the side to move's point of view.
#pragma once

#include <string>
#include <vector>

#include "position.h"

constexpr int NUM_FEATURES = 200;  // {own man, own king, their man, their king} x 50 squares
constexpr int MAX_HIDDEN = 1024;

// Feature index for a piece on bit `b`, seen from `persp` (board rotated for Black
// so that each side's men always move towards relative square 1..5).
inline int feature_index(Color persp, Color owner, bool king, int b) {
  const int sq = bit_to_square(b);
  const int rel = persp == WHITE ? sq - 1 : 50 - sq;
  const int type = (owner == persp ? 0 : 2) + (king ? 1 : 0);
  return type * 50 + rel;
}

struct Network {
  int hidden = 0, l2 = 0, l3 = 0;
  float out_scale = 1.0f;  // network output -> centi-men
  std::vector<float> w1, b1;  // [NUM_FEATURES][hidden], [hidden]
  std::vector<float> w2, b2;  // [l2][2*hidden], [l2]
  std::vector<float> w3, b3;  // [l3][l2], [l3]
  std::vector<float> w4;      // [l3]
  float b4 = 0;

  bool load(const std::string& path, std::string& err);
  bool loaded() const { return hidden > 0; }
};

struct alignas(64) Accumulator {
  float v[2][MAX_HIDDEN];  // [perspective][hidden]
};

namespace eval {
extern Network net;

void refresh(const Position& pos, Accumulator& acc);
// child accumulator after `m` played from `parent_pos`
void update(const Accumulator& parent, const Position& parent_pos, const Move& m, Accumulator& child);
int evaluate(const Position& pos, const Accumulator* acc);
int handcrafted(const Position& pos);
}  // namespace eval
