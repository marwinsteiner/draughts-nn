#include "eval.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

#if defined(__AVX2__) || defined(__AVXVNNI__)
#include <immintrin.h>
#endif

namespace eval {
Network net;
}

// Weight file: "DNN1" magic, int32 hidden/l2/l3, float out_scale, then float32
// arrays w1 b1 w2 b2 w3 b3 w4 b4 in the layouts documented in eval.h.
bool Network::load(const std::string& path, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open " + path;
    return false;
  }
  char magic[4];
  int32_t dims[3];
  f.read(magic, 4);
  f.read(reinterpret_cast<char*>(dims), sizeof dims);
  f.read(reinterpret_cast<char*>(&out_scale), sizeof out_scale);
  if (!f || std::memcmp(magic, "DNN1", 4) != 0) {
    err = "bad header in " + path;
    return false;
  }
  hidden = dims[0], l2 = dims[1], l3 = dims[2];
  if (hidden <= 0 || hidden > MAX_HIDDEN || l2 <= 0 || l2 > 256 || l3 <= 0 || l3 > 256) {
    err = "bad dimensions in " + path;
    hidden = 0;
    return false;
  }
  auto rd = [&](std::vector<float>& v, size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
  };
  rd(w1, size_t(NUM_FEATURES) * hidden);
  rd(b1, hidden);
  rd(w2, size_t(l2) * 2 * hidden);
  rd(b2, l2);
  rd(w3, size_t(l3) * l2);
  rd(b3, l3);
  rd(w4, l3);
  f.read(reinterpret_cast<char*>(&b4), sizeof b4);
  if (!f) {
    err = "truncated weights in " + path;
    hidden = 0;
    return false;
  }
  w2t.resize(w2.size());
  for (int j = 0; j < l2; ++j)
    for (int i = 0; i < 2 * hidden; ++i) w2t[size_t(i) * l2 + j] = w2[size_t(j) * 2 * hidden + i];
  // int8 copy of w2 for the VNNI path, layout [group of 4 inputs][32 outputs][4]
  w2q_scale = 0;
  if (l2 == 32 && (2 * hidden) % 4 == 0) {
    float mx = 1e-12f;
    for (float v : w2) mx = std::max(mx, std::abs(v));
    w2q_scale = 127.0f / mx;
    const int groups = 2 * hidden / 4;
    w2q.assign(size_t(groups) * 128, 0);
    for (int g = 0; g < groups; ++g)
      for (int j = 0; j < 32; ++j)
        for (int k = 0; k < 4; ++k)
          w2q[(size_t(g) * 32 + j) * 4 + k] =
              int8_t(std::lround(w2[size_t(j) * 2 * hidden + 4 * g + k] * w2q_scale));
  }
  w3t.resize(w3.size());
  for (int j = 0; j < l3; ++j)
    for (int i = 0; i < l2; ++i) w3t[size_t(i) * l3 + j] = w3[size_t(j) * l2 + i];
  return true;
}

namespace eval {

namespace {

inline void add_feature(float* acc, int idx) {
  const float* w = &net.w1[size_t(idx) * net.hidden];
  for (int i = 0; i < net.hidden; ++i) acc[i] += w[i];
}
inline void sub_feature(float* acc, int idx) {
  const float* w = &net.w1[size_t(idx) * net.hidden];
  for (int i = 0; i < net.hidden; ++i) acc[i] -= w[i];
}

inline float crelu(float x) { return std::clamp(x, 0.0f, 1.0f); }

}  // namespace

void refresh(const Position& pos, Accumulator& acc) {
  if (!net.loaded()) return;
  for (Color persp : {WHITE, BLACK}) {
    float* a = acc.v[persp];
    std::memcpy(a, net.b1.data(), sizeof(float) * net.hidden);
    for (Color owner : {WHITE, BLACK}) {
      BB b = pos.pieces[owner];
      while (b) {
        const int s = pop_lsb(b);
        add_feature(a, feature_index(persp, owner, pos.kings & bit(s), s));
      }
    }
  }
}

void update(const Accumulator& parent, const Position& parent_pos, const Move& m, Accumulator& child) {
  if (!net.loaded()) return;
  const Color us = parent_pos.side, them = ~us;
  const bool was_king = parent_pos.kings & bit(m.from);
  const bool promotes = !was_king && (bit(m.to) & (us == WHITE ? ROW_TOP : ROW_BOTTOM));
  for (Color persp : {WHITE, BLACK}) {
    float* a = child.v[persp];
    std::memcpy(a, parent.v[persp], sizeof(float) * net.hidden);
    sub_feature(a, feature_index(persp, us, was_king, m.from));
    add_feature(a, feature_index(persp, us, was_king || promotes, m.to));
    BB c = m.captured;
    while (c) {
      const int s = pop_lsb(c);
      sub_feature(a, feature_index(persp, them, parent_pos.kings & bit(s), s));
    }
  }
}

int draw_scale = 0;  // off until validated; Hub param draw-scale

namespace {
int evaluate_unscaled(const Position& pos, const Accumulator* acc);

// Lone king against at most 3 pieces is usually a draw in international draughts
// (the lone king holds the long diagonal; lidraughts ends it after 16 or 5 moves).
// The teacher (Scan without endgame databases) overrates these endings, so the
// network does too, and the engine would happily trade a winning middlegame down
// into them. Scale such evaluations towards zero.
int scale_drawish(const Position& pos, int s) {
  if (!draw_scale) return s;
  for (Color c : {WHITE, BLACK}) {
    const BB mine = pos.pieces[c];
    if (popcount(mine) == 1 && (mine & pos.kings) && popcount(pos.pieces[~c]) <= 3) return s / 16;
  }
  return s;
}
}  // namespace

int evaluate(const Position& pos, const Accumulator* acc) {
  return scale_drawish(pos, evaluate_unscaled(pos, acc));
}

namespace {
int evaluate_unscaled(const Position& pos, const Accumulator* acc) {
  if (!net.loaded() || !acc) return handcrafted(pos);
  // Generic float path: column-wise accumulation over the transposed weights
  // (vectorises across outputs without reassociating float sums) and skips the
  // inputs that the clipped ReLU zeroes.
  const int H = net.hidden, L2 = net.l2, L3 = net.l3;
  alignas(64) float h2[256], h3[256];
  const float* halves[2] = {acc->v[pos.side], acc->v[~pos.side]};
#if defined(__AVXVNNI__)
  if (L2 == 32 && net.w2q_scale > 0) {
    // int8 path: inputs quantised to u8 (0..255), weights to s8 with a per-network
    // scale, 32 int32 outputs accumulated by VNNI dot products over groups of 4
    // inputs. Two accumulator sets hide the dpbusd latency.
    alignas(32) uint8_t x[2 * MAX_HIDDEN + 8];
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f), q = _mm256_set1_ps(255.0f);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    auto to_i32 = [&](const float* p) {
      return _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(p), zero), one), q));
    };
    for (int half = 0; half < 2; ++half) {
      const float* a = halves[half];
      uint8_t* xo = x + half * H;
      int i = 0;
      for (; i + 32 <= H; i += 32) {
        const __m256i ab = _mm256_packs_epi32(to_i32(a + i), to_i32(a + i + 8));
        const __m256i cd = _mm256_packs_epi32(to_i32(a + i + 16), to_i32(a + i + 24));
        const __m256i v = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(ab, cd), order);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(xo + i), v);
      }
      for (; i < H; ++i) xo[i] = uint8_t(std::lround(std::clamp(a[i], 0.0f, 1.0f) * 255.0f));
    }
    const int groups = (2 * H) / 4;
    __m256i s[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
    __m256i t[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
    const int8_t* w = net.w2q.data();
    int g = 0;
    for (; g + 1 < groups; g += 2) {
      int32_t va, vb;
      std::memcpy(&va, x + 4 * g, 4);
      std::memcpy(&vb, x + 4 * g + 4, 4);
      const __m256i xa = _mm256_set1_epi32(va), xb = _mm256_set1_epi32(vb);
      const int8_t* wa = w + size_t(g) * 128;
      for (int k = 0; k < 4; ++k) {
        s[k] = _mm256_dpbusd_avx_epi32(s[k], xa, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wa + 32 * k)));
        t[k] = _mm256_dpbusd_avx_epi32(t[k], xb, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wa + 128 + 32 * k)));
      }
    }
    if (g < groups) {
      int32_t va;
      std::memcpy(&va, x + 4 * g, 4);
      const __m256i xa = _mm256_set1_epi32(va);
      for (int k = 0; k < 4; ++k)
        s[k] = _mm256_dpbusd_avx_epi32(s[k], xa, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + size_t(g) * 128 + 32 * k)));
    }
    const __m256 deq = _mm256_set1_ps(1.0f / (255.0f * net.w2q_scale));
    for (int k = 0; k < 4; ++k) {
      const __m256 v = _mm256_cvtepi32_ps(_mm256_add_epi32(s[k], t[k]));
      _mm256_store_ps(h2 + 8 * k, _mm256_fmadd_ps(v, deq, _mm256_loadu_ps(&net.b2[8 * k])));
    }
  } else
#endif
  {
    std::memcpy(h2, net.b2.data(), sizeof(float) * L2);
    for (int half = 0; half < 2; ++half) {
      const float* a = halves[half];
      const float* wbase = &net.w2t[size_t(half) * H * L2];
      for (int i = 0; i < H; ++i) {
        const float x = crelu(a[i]);
        if (x == 0.0f) continue;
        const float* w = wbase + size_t(i) * L2;
        for (int j = 0; j < L2; ++j) h2[j] += x * w[j];
      }
    }
  }
  std::memcpy(h3, net.b3.data(), sizeof(float) * L3);
  for (int i = 0; i < L2; ++i) {
    const float x = crelu(h2[i]);
    if (x == 0.0f) continue;
    const float* w = &net.w3t[size_t(i) * L3];
    for (int j = 0; j < L3; ++j) h3[j] += x * w[j];
  }
  float out = net.b4;
  for (int i = 0; i < L3; ++i) out += net.w4[i] * crelu(h3[i]);
  const int cp = int(out * net.out_scale);
  return std::clamp(cp, -5000, 5000);
}
}  // namespace

int handcrafted(const Position& pos) {
  // Material plus mild advancement and centre terms; a placeholder until the
  // network is trained.
  int score[2] = {0, 0};
  for (Color c : {WHITE, BLACK}) {
    score[c] += 100 * popcount(pos.men(c)) + 320 * popcount(pos.kings_of(c));
    BB men = pos.men(c);
    while (men) {
      const int s = pop_lsb(men);
      const int sq = bit_to_square(s);
      const int row = (sq - 1) / 5;              // 0 = top
      const int adv = c == WHITE ? 9 - row : row;  // rows advanced
      const int col = 2 * ((sq - 1) % 5) + (row % 2 == 0 ? 1 : 0);
      score[c] += 2 * adv + (col >= 3 && col <= 6 ? 4 : 0);
    }
  }
  return score[pos.side] - score[~pos.side];
}

}  // namespace eval
