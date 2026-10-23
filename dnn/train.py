"""Train the NNUE-style evaluation network on teacher-labelled positions.

    python -m dnn.train --data data/scan_sp_n100k --out nets/v1 --epochs 20

Inputs: per perspective, 200 binary features {own man, own king, their man,
their king} x 50 squares (board rotated 180 degrees for Black). Network:
shared L1 200->H for both perspectives, concat [stm, other] -> clipped ReLU ->
L2 -> CReLU -> L3 -> CReLU -> 1. Output o is a logit: win probability =
sigmoid(o), and the engine reports o * K centi-men, where K is fitted so that
sigmoid(score / K) best predicts game results.

Target: lam * sigmoid(teacher_score / K) + (1 - lam) * result01.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

C_W, C_B, C_WK, C_BK = ord("w"), ord("b"), ord("W"), ord("B")


def resolve_specs(specs: list[str], seed: int = 0) -> list[Path]:
    """Data specs "DIR[:max_chunks[:repeat]]": max_chunks picks a random subset of the
    directory's chunks (0 = all), repeat oversamples them (e.g. lesson data)."""
    rng = np.random.default_rng(seed)
    files: list[Path] = []
    for spec in specs:
        path, *rest = spec.split(":")
        max_chunks = int(rest[0]) if rest and rest[0] else 0
        repeat = int(rest[1]) if len(rest) > 1 else 1
        fs = sorted(f for f in Path(path).glob("chunk_*.npz") if ".tmp" not in f.name)
        if max_chunks and len(fs) > max_chunks:
            fs = [fs[i] for i in sorted(rng.choice(len(fs), max_chunks, replace=False))]
        files += fs * repeat
    return files


def load_chunks(files: list[Path], max_positions: int | None = None) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {"pos": [], "score": [], "result": [], "chunk": []}
    total = 0
    for i, f in enumerate(files):
        z = np.load(f)
        parts["pos"].append(z["pos"])
        parts["score"].append(z["score"])
        parts["result"].append(z["result"])
        parts["chunk"].append(np.full(len(z["score"]), i, np.int32))
        total += len(z["score"])
        if max_positions and total >= max_positions:
            break
    return {k: np.concatenate(v) for k, v in parts.items()}


def features(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pos: (N, 51) uint8 hub bytes on device -> (x_stm, x_other), each (N, 200) float."""
    sq = pos[:, 1:]
    wm, wk = (sq == C_W), (sq == C_WK)
    bm, bk = (sq == C_B), (sq == C_BK)
    x_white = torch.cat([wm, wk, bm, bk], dim=1)
    x_black = torch.cat([bm.flip(1), bk.flip(1), wm.flip(1), wk.flip(1)], dim=1)
    white_to_move = (pos[:, 0] == C_WK).unsqueeze(1)
    x_stm = torch.where(white_to_move, x_white, x_black).float()
    x_oth = torch.where(white_to_move, x_black, x_white).float()
    return x_stm, x_oth


class Net(nn.Module):
    def __init__(self, hidden: int = 256, l2: int = 32, l3: int = 32):
        super().__init__()
        self.l1 = nn.Linear(200, hidden)
        self.l2 = nn.Linear(2 * hidden, l2)
        self.l3 = nn.Linear(l2, l3)
        self.l4 = nn.Linear(l3, 1)

    def forward(self, x_stm: torch.Tensor, x_oth: torch.Tensor) -> torch.Tensor:
        a = torch.cat([self.l1(x_stm), self.l1(x_oth)], dim=1).clamp(0, 1)
        h = self.l2(a).clamp(0, 1)
        h = self.l3(h).clamp(0, 1)
        return self.l4(h).squeeze(1)


def fit_k(score: np.ndarray, result: np.ndarray) -> float:
    """K maximising the likelihood of results (as expected score) given sigmoid(score/K)."""
    s = score.astype(np.float64)
    y = (result.astype(np.float64) + 1) / 2
    best, best_k = math.inf, 200.0
    for k in np.arange(50, 800, 5):
        p = 1 / (1 + np.exp(-s / k))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if ll < best:
            best, best_k = ll, float(k)
    return best_k


def export(net: Net, k: float, path: Path) -> None:
    sd = {n: p.detach().float().cpu().numpy() for n, p in net.state_dict().items()}
    hidden, l2, l3 = sd["l1.weight"].shape[0], sd["l2.weight"].shape[0], sd["l3.weight"].shape[0]
    with path.open("wb") as f:
        f.write(b"DNN1")
        f.write(struct.pack("<iiif", hidden, l2, l3, k))
        for arr in (sd["l1.weight"].T, sd["l1.bias"], sd["l2.weight"], sd["l2.bias"], sd["l3.weight"],
                    sd["l3.bias"], sd["l4.weight"].reshape(-1), sd["l4.bias"]):
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="DIR[:max_chunks[:repeat]]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--l2", type=int, default=32)
    ap.add_argument("--l3", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.75, help="weight of teacher score vs game result")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-positions", type=int)
    ap.add_argument("--init", type=Path, help="warm-start from a .pt state dict")
    ap.add_argument("--k", type=float, help="fixed K (default: fit on data)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = load_chunks(resolve_specs(args.data), args.max_positions)
    n = len(d["score"])
    k = args.k or fit_k(d["score"][:2_000_000], d["result"][:2_000_000])
    print(f"loaded {n} positions from {d['chunk'].max() + 1} chunks in {time.time() - t0:.0f}s; K={k}", flush=True)

    # Split by chunk (whole games stay on one side).
    n_chunks = int(d["chunk"].max()) + 1
    rng = np.random.default_rng(0)
    val_chunks = set(rng.choice(n_chunks, max(1, int(n_chunks * args.val_frac)), replace=False).tolist())
    is_val = np.isin(d["chunk"], list(val_chunks))
    pos = torch.from_numpy(d["pos"]).to(dev)
    target = torch.from_numpy(
        args.lam / (1 + np.exp(-d["score"].astype(np.float32) / k))
        + (1 - args.lam) * (d["result"].astype(np.float32) + 1) / 2).to(dev)
    tr_idx = torch.from_numpy(np.flatnonzero(~is_val)).to(dev)
    va_idx = torch.from_numpy(np.flatnonzero(is_val)).to(dev)
    print(f"train {len(tr_idx)} / val {len(va_idx)}", flush=True)

    net = Net(args.hidden, args.l2, args.l3).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(tr_idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.05)

    def evaluate() -> float:
        net.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_idx), 65536):
                idx = va_idx[i:i + 65536]
                p = torch.sigmoid(net(*features(pos[idx])))
                tot += F.mse_loss(p, target[idx], reduction="sum").item()
                cnt += len(idx)
        net.train()
        return tot / max(cnt, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            p = torch.sigmoid(net(*features(pos[idx])))
            loss = F.mse_loss(p, target[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        va = evaluate()
        history.append({"epoch": epoch + 1, "train": tot / len(perm), "val": va})
        print(f"epoch {epoch + 1}: train {tot / len(perm):.6f} val {va:.6f} ({time.time() - t0:.0f}s)", flush=True)
        if va < best:
            best = va
            torch.save(net.state_dict(), args.out / "net.pt")
            export(net, k, args.out / "net.bin")
    (args.out / "meta.json").write_text(json.dumps({
        "data": args.data, "positions": n, "k": k, "lam": args.lam, "hidden": args.hidden,
        "l2": args.l2, "l3": args.l3, "epochs": args.epochs, "best_val": best, "history": history,
    }, indent=1))


if __name__ == "__main__":
    main()
