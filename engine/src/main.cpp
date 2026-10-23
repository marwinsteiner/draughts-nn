// dnn: draughts engine entry point.
//
//   dnn perft <depth> [hubpos]   count leaf nodes (move generator check)
//   dnn moves                    read hub positions from stdin, print legal moves (hub notation)
#include <chrono>
#include <iostream>
#include <string>

#include "position.h"

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
    std::cout << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  zobrist::init();
  const std::string cmd = argc > 1 ? argv[1] : "hub";
  if (cmd == "perft") return cmd_perft(argc, argv);
  if (cmd == "moves") return cmd_moves();
  std::cerr << "unknown command: " << cmd << "\n";
  return 1;
}
