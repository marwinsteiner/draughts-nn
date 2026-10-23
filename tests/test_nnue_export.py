"""The engine's float inference must reproduce the PyTorch network exactly
(up to rounding to centi-men), including the incremental accumulator path."""

import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dnn.train import Net, export, features  # noqa: E402
from test_movegen import ENGINE, random_positions  # noqa: E402

pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="engine not built")


def test_engine_matches_torch(tmp_path):
    torch.manual_seed(0)
    net = Net(64, 16, 16)
    with torch.no_grad():  # make outputs non-trivial
        for p in net.parameters():
            p.mul_(3)
    k = 150.0
    export(net, k, tmp_path / "net.bin")

    hubs = [h for h, _ in random_positions(n_games=5, seed=3)][:300]
    pos = torch.from_numpy(np.stack([np.frombuffer(h.encode(), np.uint8) for h in hubs]))
    with torch.no_grad():
        ref = (net(*features(pos)) * k).numpy()
    ref = np.clip(ref.astype(np.int64), -5000, 5000)  # engine truncates toward zero then clamps

    out = subprocess.run([str(ENGINE), "eval", str(tmp_path / "net.bin")], input="\n".join(hubs) + "\n",
                         capture_output=True, text=True, check=True).stdout
    ours = np.array([int(x) for x in out.split()])
    assert np.abs(ours - ref).max() <= 1


def test_incremental_accumulator_matches_refresh(tmp_path):
    torch.manual_seed(1)
    export(Net(64, 16, 16), 150.0, tmp_path / "net.bin")
    hubs = [h for h, _ in random_positions(n_games=5, seed=4)]
    out = subprocess.run([str(ENGINE), "acccheck", str(tmp_path / "net.bin")], input="\n".join(hubs) + "\n",
                         capture_output=True, text=True, check=True).stdout
    assert float(out) < 1e-4
