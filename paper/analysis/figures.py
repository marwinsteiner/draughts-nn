"""Figures and numbers for the Results section.

    python paper/analysis/figures.py --data var/results --out paper/figures

Reads, from --data:
  games.ndjson  API export of valhalla's games (/api/games/user/valhalla?moves=true)
  scored.jsonl  Scan's evaluation of every position (paper/analysis/score_games.py)
  freq.json     lidraughts' monthly blitz rating distribution (players per 25-point bin from 600)
Writes PDF figures and stats.json to --out.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BOT = "valhalla"
TOP_HUMAN = ("SatoruGojo", 2506)  # first place on the human blitz leaderboard, 26 Oct 2026
CLIP = 5.0  # men; Scan reports a won position as about +100
LEAD = 1.0  # men; a game is "decided" once one side stays this far ahead
BLUNDER = 1.0  # men lost by a single move
OPEN = 2.0  # men; move errors are counted only while neither side is further ahead than this
MIN_RATING, BIN = 600, 25

WIN, DRAW, LOSS = "#2a78d6", "#8a8984", "#e34948"  # diverging poles, neutral draw
HUMAN, BOTC = "#2a78d6", "#eb6834"  # categorical slots 1-2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUTCOME_COLOR = {"W": WIN, "D": DRAW, "L": LOSS}
OUTCOME_NAME = {"W": "Wins", "D": "Draws", "L": "Losses"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def pieces(hub: str) -> int:
    return sum(c != "e" for c in hub[1:])


def phase(n: int) -> str:
    return "opening" if n > 30 else ("middlegame" if n > 15 else "endgame")


def load(data: Path) -> list[dict]:
    raw = sorted((json.loads(line) for line in (data / "games.ndjson").open()), key=lambda g: g["createdAt"])
    scored = {}
    if (data / "scored.jsonl").exists():
        scored = {r["id"]: r for r in map(json.loads, (data / "scored.jsonl").open())}
    games = []
    for g in raw:
        me = "white" if g["players"]["white"].get("user", {}).get("id") == BOT else "black"
        op = "black" if me == "white" else "white"
        opp = g["players"][op]
        w = g.get("winner")
        rec = {
            "id": g["id"], "speed": g["speed"], "rated": g["rated"], "status": g["status"],
            "color": me[0].upper(), "opponent": opp.get("user", {}).get("name", "?"),
            "bot": opp.get("user", {}).get("title") == "BOT" or "aiLevel" in opp,
            "opp_rating": opp.get("rating"), "rating": g["players"][me].get("rating"),
            "diff": g["players"][me].get("ratingDiff"), "provisional": g["players"][me].get("provisional", False),
            "outcome": "W" if w == me else ("L" if w == op else "D"),
            "tc": f"{g['clock']['initial'] // 60}+{g['clock']['increment']}" if "clock" in g else "-",
            "created": g["createdAt"],
        }
        s = scored.get(g["id"])
        if s:
            rec["hubs"] = s["hubs"]
            ev = np.clip(np.array(s["scan"], float), -CLIP, CLIP)
            if g["status"] == "mate":  # the final position has no legal move: Scan returns no score
                ev[-1] = CLIP if rec["outcome"] == "W" else -CLIP
            rec["eval"] = ev
            rec["final_draw_state"] = s["final_draw_state"]
        games.append(rec)
    return games


def draw_cause(g: dict) -> str:
    st = g.get("final_draw_state")
    if not st:
        return "unknown"
    if st["repetition"]:
        return "threefold repetition"
    if st["clock"] >= st["limit"]:
        return {50: "25-move rule", 32: "16-move rule", 10: "5-move rule"}[st["limit"]]
    return "other"


def decisive_ply(g: dict) -> int | None:
    """Index of the first position from which the winner stayed >= LEAD men ahead (valhalla's view).
    The evaluation is first smoothed by a running median over 5 plies, so that a single noisy
    score does not move the decisive point."""
    e = g["eval"] if g["outcome"] == "W" else -g["eval"]
    pad = np.pad(e, 2, mode="edge")
    e = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)
    below = np.nonzero(e < LEAD)[0]
    if len(below) == 0:
        return 0
    t = int(below[-1]) + 1
    return t if t < len(e) else None  # None: never decided on the board (e.g. won on time while level)


def move_errors(g: dict) -> list[tuple[int, bool, float]]:
    """(pieces on board, valhalla moved?, eval lost by the mover in men) for every move made while
    the game was still open (|eval| <= OPEN before the move); in decided positions Scan's
    score saturates and differences say nothing about the quality of a move."""
    e, hubs, out = g["eval"], g["hubs"], []
    me = g["color"]
    for i in range(len(hubs) - 1):
        if abs(e[i]) > OPEN:
            continue
        mine = hubs[i][0] == me
        before, after = (e[i], e[i + 1]) if mine else (-e[i], -e[i + 1])
        out.append((pieces(hubs[i]), mine, max(0.0, before - after)))
    return out


def percentile_rating(freq: list[int], q: float) -> float:
    c = np.cumsum(freq) / sum(freq)
    i = int(np.searchsorted(c, q))
    lo = c[i - 1] if i else 0.0
    return MIN_RATING + BIN * (i + (q - lo) / (c[i] - lo))


def share_below(freq: list[int], r: float) -> float:
    edges = MIN_RATING + BIN * np.arange(len(freq) + 1)
    below = 0.0
    for i, n in enumerate(freq):
        below += n * np.clip((r - edges[i]) / BIN, 0, 1)
    return below / sum(freq)


def fig_rating(games: list[dict], freq: list[int], out: Path, stats: dict) -> None:
    blitz = [g for g in games if g["speed"] == "blitz" and g["rated"]]
    ratings = [1500.0]
    for g in blitz:
        ratings.append(g["rating"] + (g["diff"] or 0))
    ratings = np.array(ratings)
    n = np.arange(len(ratings))
    refs = [(0.5, "median player"), (0.9, "90th percentile"), (0.99, "99th percentile")]
    stats["rating"] = {
        "final": float(ratings[-1]), "games": len(blitz), "players_in_distribution": int(sum(freq)),
        "share_below_final": share_below(freq, ratings[-1]),
        "provisional_until_game": max((i + 1 for i, g in enumerate(blitz) if g["provisional"]), default=0),
        "passed": {},
    }
    for q, name in refs:
        r = percentile_rating(freq, q)
        k = int(np.argmax(ratings >= r)) if (ratings >= r).any() else None
        stats["rating"]["passed"][name] = {"rating": round(r), "after_game": k}
    stats["rating"]["share_below_after"] = [round(share_below(freq, r), 4) for r in ratings[:30]]
    k_top = [i for i in range(len(ratings)) if share_below(freq, ratings[i]) >= 0.999]
    stats["rating"]["passed"]["99.9th percentile"] = {"rating": round(percentile_rating(freq, 0.999)),
                                                     "after_game": k_top[0] if k_top else None}

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.plot(n, ratings, color=INK, lw=1.2, zorder=2)
    for i, g in enumerate(blitz, 1):
        ax.scatter(i, ratings[i], s=14 if not g["bot"] else 13, marker="o" if not g["bot"] else "s",
                   color=OUTCOME_COLOR[g["outcome"]], edgecolor="white", linewidth=0.5, zorder=3)
    for q, name in refs + [(0.999, "99.9th percentile")]:
        r = percentile_rating(freq, q)
        ax.axhline(r, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(1.01, r, f"{name} ({r:.0f})", transform=ax.get_yaxis_transform(), va="center", fontsize=7,
                color=MUTED)
    first_human = next(i for i, g in enumerate(blitz, 1) if not g["bot"])
    stats["rating"]["first_human_game"] = {"game": first_human, "rating_before": float(ratings[first_human - 1])}
    ax.axvline(first_human - 0.5, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    ax.text(first_human + 0.5, 1250, "first game\nagainst a human", fontsize=7, color=MUTED)
    ax.axhline(TOP_HUMAN[1], color=MUTED, lw=0.7, zorder=1)
    ax.text(1.01, TOP_HUMAN[1] + 25, f"no. 1 human ({TOP_HUMAN[1]})", transform=ax.get_yaxis_transform(),
            va="center", fontsize=7, color=MUTED)
    ax.set_xlabel("Rated blitz games played")
    ax.set_ylabel("valhalla's blitz rating")
    ax.set_xlim(-1, len(ratings) + 1)
    ax.set_ylim(1150, 2650)
    handles = [plt.Line2D([], [], ls="", marker="o", color=WIN, label="win"),
               plt.Line2D([], [], ls="", marker="o", color=DRAW, label="draw"),
               plt.Line2D([], [], ls="", marker="o", color=LOSS, label="loss"),
               plt.Line2D([], [], ls="", marker="o", mfc="none", color=MUTED, label="vs human"),
               plt.Line2D([], [], ls="", marker="s", mfc="none", color=MUTED, label="vs bot")]
    ax.legend(handles=handles, loc="lower right", ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out / "rating.pdf")
    plt.close(fig)


def fig_distribution(freq: list[int], rating: float, out: Path) -> None:
    """Players per rating bin (bars, left axis) and cumulative share (line, right axis)."""
    left = MIN_RATING + BIN * np.arange(len(freq))
    cum = 100 * np.cumsum(freq) / sum(freq)
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.bar(left, freq, width=BIN * 0.85, align="edge", color="#9ec5f4", edgecolor="none")
    ax.set_xlabel("Blitz rating")
    ax.set_ylabel("Players (bars)")
    ax.grid(axis="x", visible=False)
    ax2 = ax.twinx()
    ax2.plot(left + BIN, cum, color=HUMAN, lw=1.5)
    ax2.set_ylabel("Cumulative share of players, % (line)")
    ax2.set_ylim(0, 102)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    share = 100 * share_below(freq, rating)
    ax.axvline(rating, color=INK, lw=1.0)
    ax.axvline(TOP_HUMAN[1], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax2.text(rating - 15, 60, f"valhalla {rating:.0f}\nabove {share:.2f}%", ha="right", fontsize=7, color=INK)
    ax2.text(TOP_HUMAN[1] + 15, 30, f"no. 1 human\n{TOP_HUMAN[1]}", ha="left", fontsize=7, color=MUTED)
    ax.set_xlim(MIN_RATING, 2800)
    fig.tight_layout()
    fig.savefig(out / "distribution.pdf")
    plt.close(fig)


def fig_trajectories(games: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
    for ax, oc in zip(axes, "WDL"):
        gs = [g for g in games if g["outcome"] == oc and "eval" in g]
        for g in gs:
            ax.plot(np.arange(len(g["eval"])), g["eval"], color=BOTC if g["bot"] else HUMAN, lw=0.6, alpha=0.45)
        nh = sum(not g["bot"] for g in gs)
        ax.set_title(f"{OUTCOME_NAME[oc]} ({nh} vs humans, {len(gs) - nh} vs bots)")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("Ply")
        ax.set_xlim(0, 250)
    axes[0].set_ylabel("Scan's evaluation for valhalla (men)")
    axes[0].set_ylim(-CLIP - 0.3, CLIP + 0.3)
    axes[2].legend(handles=[plt.Line2D([], [], color=HUMAN, label="vs human"),
                            plt.Line2D([], [], color=BOTC, label="vs bot")], loc="upper right")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "trajectories.pdf")
    plt.close(fig)


def fig_decided(games: list[dict], out: Path, stats: dict) -> None:
    """Where in the game wins and losses were decided, and where draws ended."""
    rows = []
    for g in games:
        if "eval" not in g:
            continue
        if g["outcome"] == "D":
            rows.append(("D", g, pieces(g["hubs"][-1]), len(g["hubs"]) - 1))
            continue
        t = decisive_ply(g)
        rows.append((g["outcome"], g, pieces(g["hubs"][t]) if t is not None else None, t))
        if t:  # who played the move into the decided position, and how much it changed the evaluation
            mover_is_me = g["hubs"][t - 1][0] == g["color"]
            swing = abs(g["eval"][t] - g["eval"][t - 1])
            stats.setdefault("decisive_moves", []).append(
                {"id": g["id"], "outcome": g["outcome"], "bot": g["bot"], "ply": t, "pieces": pieces(g["hubs"][t]),
                 "by": "valhalla" if mover_is_me else "opponent", "swing": round(float(swing), 2),
                 "status": g["status"]})
    groups = [("wins vs humans", lambda r: r[0] == "W" and not r[1]["bot"], WIN),
              ("wins vs bots", lambda r: r[0] == "W" and r[1]["bot"], WIN),
              ("losses (all vs bots)", lambda r: r[0] == "L", LOSS),
              ("draws vs humans: end", lambda r: r[0] == "D" and not r[1]["bot"], DRAW),
              ("draws vs bots: end", lambda r: r[0] == "D" and r[1]["bot"], DRAW)]
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    rng = np.random.default_rng(0)
    st = stats.setdefault("decided", {})
    labels = []
    for y, (name, f, col) in enumerate(groups):
        sel = [r for r in rows if f(r)]
        xs = [r[2] for r in sel if r[2] is not None]
        undecided = [r for r in sel if r[2] is None]
        ax.scatter(xs, y + rng.uniform(-0.18, 0.18, len(xs)), s=12, color=col, edgecolor="white", linewidth=0.4)
        labels.append(f"{name} ({len(sel)})" + (f"\n{len(undecided)} not decided on the board" if undecided else ""))
        st[name] = {"n": len(sel), "median_pieces": float(np.median(xs)) if xs else None,
                    "never_decided_on_board": len(undecided),
                    "phase": dict(Counter(phase(x) for x in xs))}
    for b, label in ((30.5, "opening"), (15.5, "middlegame")):
        ax.axvline(b, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    for x, label in ((35.5, "opening"), (23, "middlegame"), (8, "endgame")):
        ax.text(x, len(groups) - 0.35, label, ha="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(groups)), labels)
    ax.set_ylim(len(groups) - 0.3, -0.6)
    ax.set_xlim(42.5, 0)
    ax.set_xticks([40, 35, 30, 25, 20, 15, 10, 5, 2])
    ax.set_xlabel("Pieces on the board")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "decided.pdf")
    plt.close(fig)


def fig_errors(games: list[dict], out: Path, stats: dict) -> None:
    """Mean evaluation lost per move, by game phase, for valhalla and its opponents."""
    acc = defaultdict(list)
    for g in games:
        if "eval" not in g:
            continue
        side = "bots" if g["bot"] else "humans"
        for n, mine, err in move_errors(g):
            who = f"valhalla vs {side}" if mine else side
            acc[(who, phase(n))].append(err)
    phases = ["opening", "middlegame", "endgame"]
    who_order = [("humans", HUMAN, "human opponents"), ("valhalla vs humans", INK, "valhalla (vs humans)"),
                 ("bots", BOTC, "bot opponents"), ("valhalla vs bots", MUTED, "valhalla (vs bots)")]
    st = stats.setdefault("errors", {})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3))
    width = 0.2
    for j, (who, col, label) in enumerate(who_order):
        means = [np.mean(acc[(who, p)]) if acc[(who, p)] else 0 for p in phases]
        rates = [100 * np.mean(np.array(acc[(who, p)]) >= BLUNDER) if acc[(who, p)] else 0 for p in phases]
        x = np.arange(3) + (j - 1.5) * width
        axes[0].bar(x, means, width * 0.9, color=col, label=label)
        axes[1].bar(x, rates, width * 0.9, color=col)
        st[label] = {p: {"moves": len(acc[(who, p)]), "mean_loss": round(float(m), 3), "blunder_pct": round(float(r), 1)}
                     for p, m, r in zip(phases, means, rates)}
    for ax, yl in zip(axes, ("Mean evaluation lost per move (men)", f"Moves losing $\\geq${BLUNDER:g} man (%)")):
        ax.set_xticks(range(3), phases)
        ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "errors.pdf")
    plt.close(fig)


def tables(games: list[dict], stats: dict) -> None:
    rec = Counter((g["speed"], "bot" if g["bot"] else "human", g["outcome"]) for g in games)
    stats["record"] = {f"{s}/{o}/{r}": n for (s, o, r), n in sorted(rec.items())}
    how = Counter(("bot" if g["bot"] else "human", g["outcome"], g["status"] if g["outcome"] != "D" else draw_cause(g))
                  for g in games)
    stats["how_ended"] = {f"{o}/{r}/{s}": n for (o, r, s), n in sorted(how.items())}
    humans = [g for g in games if not g["bot"]]
    stats["humans"] = {"games": len(humans), "players": len({g["opponent"] for g in humans}),
                       "opp_rating_mean": round(float(np.mean([g["opp_rating"] for g in humans]))),
                       "opp_rating_max": max(g["opp_rating"] for g in humans),
                       "plies_median_wins": float(np.median([len(g["hubs"]) - 1 for g in humans
                                                             if g["outcome"] == "W" and "hubs" in g] or [0]))}
    bots = [g for g in games if g["bot"]]
    stats["bots"] = {"games": len(bots), "names": dict(Counter(g["opponent"] for g in bots)),
                     "plies_median": float(np.median([len(g["hubs"]) - 1 for g in bots if "hubs" in g] or [0]))}
    lost = [g for g in games if g["outcome"] == "L"]
    stats["losses"] = [{"id": g["id"], "opponent": g["opponent"], "speed": g["speed"], "tc": g["tc"],
                        "status": g["status"], "plies": len(g.get("hubs", [])) - 1,
                        "created": datetime.fromtimestamp(g["created"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "final_eval": float(g["eval"][-2]) if "eval" in g and len(g["eval"]) > 1 else None,
                        "min_eval": float(g["eval"].min()) if "eval" in g else None}
                       for g in lost]
    dr = [g for g in games if g["outcome"] == "D" and "eval" in g]
    stats["draws"] = [{"id": g["id"], "opponent": g["opponent"], "bot": g["bot"], "cause": draw_cause(g),
                       "plies": len(g["hubs"]) - 1, "max_eval": float(g["eval"].max()),
                       "min_eval": float(g["eval"].min()), "end_pieces": pieces(g["hubs"][-1])} for g in dr]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("var/results"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    games = load(args.data)
    freq = json.loads((args.data / "freq.json").read_text())
    stats: dict = {"games": len(games), "scored": sum("eval" in g for g in games)}
    fig_rating(games, freq, args.out, stats)
    fig_distribution(freq, stats["rating"]["final"], args.out)
    blitz = [g for g in games if g["speed"] == "blitz"]  # the only speed we meant to play
    fig_trajectories(blitz, args.out)
    fig_decided(blitz, args.out, stats)
    fig_errors(blitz, args.out, stats)
    tables(games, stats)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("draws",)}, indent=1))


if __name__ == "__main__":
    main()
