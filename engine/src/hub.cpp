// Hub protocol v2 (see Scan's protocol.txt): the interface lidraughts-bot and
// pydraughts use to drive engines.
#include "hub.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "search.h"

namespace {

std::mutex out_mutex;

void send(const std::string& s) {
  std::lock_guard<std::mutex> lock(out_mutex);
  std::cout << s << "\n" << std::flush;
}

// "cmd a=1 b="x y" flag" -> cmd, {a:1, b:"x y", flag:""}
std::string parse_line(const std::string& line, std::map<std::string, std::string>& args) {
  size_t i = 0;
  auto skip = [&] {
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  };
  auto word = [&](bool stop_at_eq) {
    std::string w;
    if (i < line.size() && line[i] == '"') {
      ++i;
      while (i < line.size() && line[i] != '"') w += line[i++];
      ++i;
      return w;
    }
    while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i])) && !(stop_at_eq && line[i] == '='))
      w += line[i++];
    return w;
  };
  skip();
  const std::string cmd = word(false);
  while (true) {
    skip();
    if (i >= line.size()) break;
    const std::string key = word(true);
    std::string val;
    if (i < line.size() && line[i] == '=') {
      ++i;
      val = word(false);
    }
    args[key] = val;
  }
  return cmd;
}

std::string fmt_score(int s) {
  std::ostringstream o;
  if (s >= SCORE_WIN_MIN)
    o << "+" << 100.0;  // Hub has no mate syntax; report a decisive score in men
  else if (s <= -SCORE_WIN_MIN)
    o << -100.0;
  else
    o << std::fixed << std::setprecision(2) << s / 100.0;
  return o.str();
}

std::string pv_string(const std::vector<Move>& pv) {
  std::string s;
  for (const Move& m : pv) {
    if (!s.empty()) s += ' ';
    s += move_to_hub(m);
  }
  return s;
}

}  // namespace

int run_hub() {
  auto tt = std::make_unique<TT>();
  auto searcher = std::make_unique<Searcher>(*tt);
  std::vector<std::unique_ptr<Searcher>> helpers;  // Lazy SMP: extra threads sharing the TT
  int threads = 1;
  std::string nnue_path, nnue_error;
  std::string search_log;  // if set, one JSON line per finished search is appended here
  int tt_mb = 64;
  Root root{Position::start(), {}, {}};
  root.draw.limit = draw_limit(root.pos);
  Limits limits;
  std::thread worker;

  auto join = [&] {
    if (worker.joinable()) worker.join();
  };

  std::string line;
  while (std::getline(std::cin, line)) {
    std::map<std::string, std::string> a;
    const std::string cmd = parse_line(line, a);
    if (cmd.empty()) continue;

    if (cmd == "hub") {
      send("id name=dnn version=0.2 author=\"Marwin Steiner\" country=\"United Kingdom\"");
      send("param name=nnue value=\"" + nnue_path + "\" type=string");
      send("param name=search-log value=\"" + search_log + "\" type=string");
      send("param name=tt-size value=" + std::to_string(tt_mb) + " type=int min=1 max=16384");
      send("param name=draw-scale value=" + std::to_string(eval::draw_scale) + " type=int min=0 max=1");
      send("param name=threads value=" + std::to_string(threads) + " type=int min=1 max=64");
      send("param name=lmr value=" + std::to_string(search_options.lmr) + " type=int min=0 max=1");
      send("param name=probcut value=" + std::to_string(search_options.probcut) + " type=int min=0 max=1");
      send("param name=opening-random-plies value=" + std::to_string(search_options.opening_random_plies) +
           " type=int min=0 max=40");
      send("param name=opening-random-margin value=" + std::to_string(search_options.opening_random_margin) +
           " type=int min=0 max=200");
      send("param name=probcut-margin value=" + std::to_string(search_options.probcut_margin) +
           " type=int min=0 max=200");
      send("wait");
    } else if (cmd == "set-param") {
      if (a["name"] == "nnue") nnue_path = a["value"];
      if (a["name"] == "search-log") search_log = a["value"];
      if (a["name"] == "tt-size") tt_mb = std::stoi(a["value"]);
      if (a["name"] == "draw-scale") eval::draw_scale = std::stoi(a["value"]);
      if (a["name"] == "threads") threads = std::clamp(std::stoi(a["value"]), 1, 64);
      if (a["name"] == "lmr") search_options.lmr = std::stoi(a["value"]);
      if (a["name"] == "probcut") search_options.probcut = std::stoi(a["value"]);
      if (a["name"] == "probcut-margin") search_options.probcut_margin = std::stoi(a["value"]);
      if (a["name"] == "opening-random-plies") search_options.opening_random_plies = std::stoi(a["value"]);
      if (a["name"] == "opening-random-margin") search_options.opening_random_margin = std::stoi(a["value"]);
    } else if (cmd == "init") {
      tt->resize(tt_mb);
      helpers.clear();
      for (int i = 1; i < threads; ++i) helpers.push_back(std::make_unique<Searcher>(*tt));
      if (!nnue_path.empty() && !eval::net.load(nnue_path, nnue_error))
        send("error message=\"" + nnue_error + "\"");
      send("ready");
    } else if (cmd == "new-game") {
      join();
      tt->clear();
    } else if (cmd == "ping") {
      send("pong");
    } else if (cmd == "pos") {
      join();
      try {
        root.pos = a.count("pos") ? Position::from_hub(a["pos"]) : Position::start();
      } catch (const std::exception& e) {
        send(std::string("error message=\"") + e.what() + "\"");
        continue;
      }
      root.history.clear();
      root.draw = DrawState{0, draw_limit(root.pos)};
      root.game_ply = 0;
      std::istringstream ms(a["moves"]);
      std::string tok;
      while (ms >> tok) {
        Move m;
        if (!parse_move(root.pos, tok, m)) {
          send("error message=\"illegal move " + tok + "\"");
          break;
        }
        const Position before = root.pos;
        root.pos.make(m);
        root.draw = next_draw_state(before, m, root.pos, root.draw);
        ++root.game_ply;
        if (root.draw.clock == 0)
          root.history.clear();
        else
          root.history.push_back(before.key);
      }
    } else if (cmd == "level") {
      limits = Limits{};
      if (a.count("depth")) limits.depth = std::clamp(std::stoi(a["depth"]), 1, MAX_PLY - 1);
      if (a.count("nodes")) limits.nodes = std::stoull(a["nodes"]);
      if (a.count("move-time")) limits.move_time = std::stod(a["move-time"]);
      if (a.count("time")) {
        limits.time = std::stod(a["time"]);
        limits.has_clock = true;
      }
      if (a.count("inc")) limits.inc = std::stod(a["inc"]);
      if (a.count("moves")) limits.moves_to_go = std::stoi(a["moves"]);
      if (a.count("infinite")) limits.infinite = true;
    } else if (cmd == "go") {
      join();
      searcher->pondering = a.count("ponder") > 0;
      searcher->stop = false;
      for (auto& h : helpers) h->stop = false;
      const Root r = root;
      const Limits l = limits;
      worker = std::thread([&searcher, &helpers, &search_log, r, l] {
        Limits hl = l;  // helpers search until the main thread is done
        hl.infinite = true;
        hl.nodes = 0;
        hl.depth = MAX_PLY - 1;
        std::vector<std::thread> pool;
        for (auto& h : helpers) pool.emplace_back([&h, &r, hl] { h->run(r, hl, nullptr); });
        const SearchInfo res = searcher->run(r, l, [](const SearchInfo& si) {
          std::ostringstream o;
          o << "info depth=" << si.depth << " score=" << fmt_score(si.score) << " nodes=" << si.nodes
            << " time=" << std::fixed << std::setprecision(3) << si.time << " nps="
            << std::setprecision(2) << (si.time > 0 ? si.nodes / si.time / 1e6 : 0.0) << " pv=\""
            << pv_string(si.pv) << "\"";
          send(o.str());
        });
        for (auto& h : helpers) h->stop = true;
        for (auto& t : pool) t.join();
        std::string done = "done";
        if (!res.pv.empty()) {
          done += " move=" + move_to_hub(res.pv[0]);
          if (res.pv.size() > 1) done += " ponder=" + move_to_hub(res.pv[1]);
        }
        send(done);
        if (!search_log.empty()) {
          // what the bot thought at every move: joined later with the game record by position
          std::ostringstream o;
          const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count();
          o << "{\"t\": " << ms << ", \"pos\": \"" << r.pos.to_hub() << "\", \"time_left\": " << l.time
            << ", \"inc\": " << l.inc << ", \"depth\": " << res.depth << ", \"score\": " << res.score
            << ", \"nodes\": " << res.nodes << ", \"secs\": " << res.time << ", \"pv\": \"" << pv_string(res.pv)
            << "\"}";
          std::ofstream(search_log, std::ios::app) << o.str() << '\n';
        }
      });
    } else if (cmd == "print-pos") {  // debugging: the root position after `pos`
      send("pos pos=" + root.pos.to_hub() + " game-ply=" + std::to_string(root.game_ply));
    } else if (cmd == "stop") {
      searcher->stop = true;
    } else if (cmd == "ponder-hit") {
      searcher->ponderhit();
    } else if (cmd == "quit") {
      searcher->stop = true;
      join();
      return 0;
    }
  }
  searcher->stop = true;
  join();
  return 0;
}
