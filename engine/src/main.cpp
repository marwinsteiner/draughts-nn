// dnn: draughts engine entry point.
//
//   dnn [hub]                     Hub protocol on stdin/stdout (default)
//   dnn perft <depth> [hubpos]    count leaf nodes (move generator check)
//   dnn moves                     read hub positions from stdin, print legal moves (hub notation)
//   dnn bench [depth] [nnue]      fixed-depth search over a few positions; prints nodes and speed
//   dnn eval [nnue]               read hub positions from stdin, print static eval (centi-men, stm)
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "eval.h"
#include "hub.h"
#include "position.h"
#include "search.h"

namespace {

int cmd_perft(int argc, char** argv) {
  const int depth = argc > 2 ? std::stoi(argv[2]) : 6;
  const Position pos = argc > 3 ? Position::from_hub(argv[3]) : Position::start();
  for (int d = 1; d <= depth; ++d) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t n = perft(pos, d);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "perft " << d << " " << n << " " << s << "s\n" << std::flush;
  }
  return 0;
}

int cmd_moves() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    const Position pos = Position::from_hub(line);
    MoveList list;
    generate_moves(pos, list);
    for (int i = 0; i < list.size; ++i) std::cout << (i ? " " : "") << move_to_hub(list[i]);
    std::cout << "\n" << std::flush;
  }
  return 0;
}

int cmd_eval(int argc, char** argv) {
  eval::draw_scale = 0;  // raw network output (compared against PyTorch in tests)
  if (argc > 2) {
    std::string err;
    if (!eval::net.load(argv[2], err)) {
      std::cerr << err << "\n";
      return 1;
    }
  }
  auto acc = std::make_unique<Accumulator>();
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    const Position pos = Position::from_hub(line);
    eval::refresh(pos, *acc);
    std::cout << eval::evaluate(pos, acc.get()) << "\n" << std::flush;
  }
  return 0;
}

// For each stdin position and each legal move: max |incremental - refreshed| accumulator difference.
int cmd_acccheck(int argc, char** argv) {
  std::string err;
  if (argc < 3 || !eval::net.load(argv[2], err)) {
    std::cerr << "usage: dnn acccheck <nnue> " << err << "\n";
    return 1;
  }
  auto parent = std::make_unique<Accumulator>(), child = std::make_unique<Accumulator>(),
       fresh = std::make_unique<Accumulator>();
  float worst = 0;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    const Position pos = Position::from_hub(line);
    eval::refresh(pos, *parent);
    MoveList list;
    generate_moves(pos, list);
    for (const Move& m : list) {
      Position c = pos;
      c.make(m);
      eval::update(*parent, pos, m, *child);
      eval::refresh(c, *fresh);
      for (int p = 0; p < 2; ++p)
        for (int i = 0; i < eval::net.hidden; ++i)
          worst = std::max(worst, std::abs(child->v[p][i] - fresh->v[p][i]));
    }
  }
  std::cout << worst << "\n";
  return 0;
}

// Micro-benchmark of network inference: ns per evaluate() and per accumulator update.
int cmd_evalbench(int argc, char** argv) {
  std::string err;
  if (argc < 3 || !eval::net.load(argv[2], err)) {
    std::cerr << "usage: dnn evalbench <nnue> " << err << "\n";
    return 1;
  }
  const Position pos = Position::from_hub("Wbbbbbbbbbbebbebbbbbbbeeebewweeewwwwwwwwwwwewwwwwww");
  auto a = std::make_unique<Accumulator>(), b = std::make_unique<Accumulator>();
  eval::refresh(pos, *a);
  MoveList list;
  generate_moves(pos, list);
  const int n = 2'000'000;
  volatile int sink = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) sink = sink + eval::evaluate(pos, a.get());
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) eval::update(*a, pos, list[i % list.size], *b);
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "evaluate " << std::chrono::duration<double, std::nano>(t1 - t0).count() / n << " ns, update "
            << std::chrono::duration<double, std::nano>(t2 - t1).count() / n << " ns\n";
  return 0;
}

int cmd_bench(int argc, char** argv) {
  const int depth = argc > 2 ? std::stoi(argv[2]) : 12;
  if (argc > 3) {
    std::string err;
    if (!eval::net.load(argv[3], err)) {
      std::cerr << err << "\n";
      return 1;
    }
  }
  const char* positions[] = {
      "Wbbbbbbbbbbbbbbbbbbbbeeeeeeeeeewwwwwwwwwwwwwwwwwwww",
      "Wbbbbbbbbbbbbbbbbbbbbeeeeeeeeeewwwwwwwwwwwwwwwwwwww",
      "Bbbbbbbbbbbbbebbbbbbbebeeeeeeewwwwwwwwwwwwwwewwwwwe",
      "Weeeeebbbbbbbbbbbebbbbebbewbwewweewwwwwwweeewwewwee",
      "Beebbebbbbbebbbbeeebbweebbewewwwbwewwwewwweweeewwee",
  };
  auto tt = std::make_unique<TT>();
  tt->resize(64);
  auto s = std::make_unique<Searcher>(*tt);
  uint64_t nodes = 0;
  double time = 0;
  for (const char* fen : positions) {
    Root r{Position::from_hub(fen), {}, {}};
    r.draw.limit = draw_limit(r.pos);
    Limits l;
    l.depth = depth;
    tt->clear();
    const SearchInfo si = s->run(r, l, nullptr);
    nodes += si.nodes;
    time += si.time;
    std::cout << fen << " depth " << si.depth << " score " << si.score << " best "
              << (si.pv.empty() ? "-" : move_to_hub(si.pv[0])) << " nodes " << si.nodes << "\n";
  }
  std::cout << "total nodes " << nodes << " time " << std::fixed << std::setprecision(2) << time
            << "s nps " << uint64_t(nodes / std::max(time, 1e-9)) << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  zobrist::init();
  const std::string cmd = argc > 1 ? argv[1] : "hub";
  if (cmd == "hub") return run_hub();
  if (cmd == "perft") return cmd_perft(argc, argv);
  if (cmd == "moves") return cmd_moves();
  if (cmd == "bench") return cmd_bench(argc, argv);
  if (cmd == "eval") return cmd_eval(argc, argv);
  if (cmd == "acccheck") return cmd_acccheck(argc, argv);
  if (cmd == "evalbench") return cmd_evalbench(argc, argv);
  std::cerr << "unknown command: " << cmd << "\n";
  return 1;
}
