from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rare.models.weight_averaging import average_state_dicts, last_k_checkpoints, recompute_bn


def _net(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))


def test_average_of_two_is_the_midpoint():
    a, b = _net(0).state_dict(), _net(1).state_dict()
    out = average_state_dicts([a, b])
    for k in a:
        if torch.is_floating_point(a[k]):
            assert torch.allclose(out[k], (a[k] + b[k]) / 2, atol=1e-6)


def test_integer_buffers_are_not_averaged():
    a, b = _net(0).state_dict(), _net(1).state_dict()
    a["1.num_batches_tracked"] = torch.tensor(10)
    b["1.num_batches_tracked"] = torch.tensor(20)
    assert average_state_dicts([a, b])["1.num_batches_tracked"].item() == 10


def test_mismatched_architectures_are_rejected():
    a = _net(0).state_dict()
    b = dict(a)
    b.pop("0.weight")
    with pytest.raises(ValueError, match="different parameter set"):
        average_state_dicts([a, b])


def test_last_k_picks_the_highest_epochs(tmp_path):
    for e in (21, 22, 23, 24):
        torch.save({"model": {}}, tmp_path / f"epoch_{e:03d}.pt")
    assert [p.name for p in last_k_checkpoints(tmp_path, 3)] == ["epoch_022.pt", "epoch_023.pt", "epoch_024.pt"]


def test_recompute_bn_updates_stats_and_restores_state():
    m = _net(0)
    m[1].momentum = 0.1
    before = m[1].running_mean.clone()
    recompute_bn(m, [(torch.randn(8, 3, 8, 8) * 5 + 3,) for _ in range(4)], torch.device("cpu"), max_batches=4)
    assert not torch.allclose(before, m[1].running_mean)
    assert m[1].momentum == 0.1
    assert not m.training
