// Draw rules as implemented by lidraughts (modules/draughts/variant/Standard.scala):
//  - threefold repetition (same position, same side to move) since the last reset;
//  - 25-move rule: 50 plies of king moves without capture;
//  - one lone king vs <=3 pieces incl. a king: drawn after 16 moves each (32 plies);
//  - one lone king vs <=2 pieces incl. a king: drawn after 5 moves each (10 plies).
// `clock` counts plies since the last reset; the game is drawn when clock >= limit.
#pragma once

#include "position.h"

struct DrawState {
  int clock = 0;
  int limit = 50;
};

// Ply limit for the material on the board (50, 32 or 10).
inline int draw_limit(const Position& p) {
  const BB all = p.occupied();
  if (popcount(all) > 4) return 50;
  for (Color c : {WHITE, BLACK}) {
    const BB mine = p.pieces[c], theirs = p.pieces[~c];
    if (popcount(mine) == 1 && (mine & p.kings)) {
      if (!(theirs & p.kings)) return 50;
      return popcount(theirs) <= 2 ? 10 : 32;
    }
  }
  return 50;
}

// State after `m` was played from `before` (giving `after`).
inline DrawState next_draw_state(const Position& before, const Move& m, const Position& after, DrawState s) {
  const Color mover = before.side;
  const bool king_move = before.kings & bit(m.from);
  const bool promotes = !king_move && (after.kings & bit(m.to));
  DrawState n;
  n.limit = draw_limit(after);
  if (n.limit == 50) {
    n.clock = (m.is_capture() || !king_move) ? 0 : s.clock + 1;
    return n;
  }
  const int pieces_before = popcount(before.occupied());
  const bool first_promotion = promotes && popcount(after.kings_of(mover)) == 1;
  const bool creates = first_promotion || (n.limit == 10 && m.is_capture() && pieces_before > 3) ||
                       (n.limit == 32 && m.is_capture() && pieces_before > 4);
  n.clock = creates ? 0 : s.clock + 1;
  return n;
}
