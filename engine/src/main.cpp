// dnn: draughts engine entry point.
//
//   dnn [hub]                     Hub protocol on stdin/stdout (default)
//   dnn perft <depth> [hubpos]    count leaf nodes (move generator check)
//   dnn moves                     read hub positions from stdin, print legal moves (hub notation)
//   dnn bench [depth] [nnue]      fixed-depth search over a few positions; prints nodes and speed
#include <chrono>
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
  std::cerr << "unknown command: " << cmd << "\n";
  return 1;
}
