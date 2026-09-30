from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submission"))

import ensemble as sub
from rare.data.transforms import build_eval_transform
from rare.fusion import mean_by_family
from rare.models.backbones import RareClassifier
from rare.models.delta_soup import mean_raw_delta, merge_lora_souped


@pytest.fixture
def sample_images():
    return np.random.default_rng(0).integers(0, 256, size=(4, 512, 637, 3), dtype=np.uint8)


def test_preprocess_matches_training_transform(sample_images):
    transform = build_eval_transform(size=384)
    expected = torch.stack([transform(image=img)["image"] for img in sample_images])
    actual = sub.preprocess(sample_images, 384)
    assert actual.shape == expected.shape == (4, 3, 384, 384)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_fusion_matches_training_path():
    matrix = np.random.default_rng(1).normal(size=(9, 300))
    families = ["a"] * 5 + ["b"] * 2 + ["c"] * 2
    np.testing.assert_array_equal(sub.fuse(matrix, families), mean_by_family(matrix, families))


def test_fusion_gives_each_family_one_vote():
    small = np.full((2, 40), 4.0)
    big = np.full((25, 40), -4.0)
    fused = sub.fuse(np.concatenate([small, big]), ["x"] * 2 + ["y"] * 25)
    assert np.allclose(fused, 0.5)


def test_fusion_is_invariant_to_member_offsets():
    rng = np.random.default_rng(0)
    families = [f"f{i}" for i in range(3) for _ in range(3)]
    signal = rng.normal(size=600)
    rows = np.stack([signal + 0.3 * rng.normal(size=600) for _ in families])
    shifted = rows + np.repeat([0.0, -0.5, -1.6], 3).reshape(-1, 1)
    assert np.array_equal(np.argsort(sub.fuse(rows, families)), np.argsort(sub.fuse(shifted, families)))


def test_training_checkpoint_loads_into_container_member():
    trained = RareClassifier("gastronet_rn50_dino", pretrained=False, img_size=224).eval()
    member = sub.Member("resnet50", img_size=224).eval()
    missing, unexpected = member.load_state_dict(trained.state_dict(), strict=False)
    assert not missing and not unexpected
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(trained(x), member(x), atol=1e-5)


def test_ensemble_directory_round_trip(tmp_path):
    torch.manual_seed(0)
    members = []
    for i, family in enumerate(["a", "b", "a"]):
        model = sub.Member("resnet18", img_size=64).eval()
        name = f"m{i}"
        (tmp_path / name).mkdir()
        save_file(model.state_dict(), tmp_path / name / "model.safetensors")
        (tmp_path / name / "meta.json").write_text(json.dumps(
            {"backbone": "resnet18", "img_size": 64, "family": family, "weights": "model.safetensors"}))
        members.append({"dir": name, "temperature": 0.5, "bias": 0.1, "weight": 1.0, "family": family})
    (tmp_path / "ensemble.json").write_text(json.dumps({"fusion": "mean_by_family", "members": members}))
    ens = sub.RareEnsemble.from_directory(tmp_path, torch.device("cpu"))
    scores = ens.predict(np.random.default_rng(0).integers(0, 256, (5, 80, 90, 3), dtype=np.uint8),
                         batch_size=2)
    assert scores.shape == (5,)
    assert ((scores > 0) & (scores < 1)).all()


def _fake_lora_state(seed: int, rank: int = 4, d_in: int = 8, d_out: int = 6) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "backbone.blk.attn.qkv.base.weight": torch.randn(d_out, d_in, generator=g),
        "backbone.blk.attn.qkv.base.bias": torch.randn(d_out, generator=g),
        "backbone.blk.attn.qkv.lora_a": torch.randn(rank, d_in, generator=g),
        "backbone.blk.attn.qkv.lora_b": torch.randn(d_out, rank, generator=g),
        "head.weight": torch.randn(1, d_out, generator=g),
    }


def test_soup_lambda_one_is_a_plain_merge():
    state = _fake_lora_state(0)
    expected = (state["backbone.blk.attn.qkv.base.weight"]
                + 1.5 * state["backbone.blk.attn.qkv.lora_b"] @ state["backbone.blk.attn.qkv.lora_a"])
    merged = merge_lora_souped(state, 1.5, 1.0, None)
    assert "backbone.blk.attn.qkv.lora_a" not in merged
    assert torch.allclose(merged["backbone.blk.attn.qkv.weight"], expected, atol=0)


def test_soup_lambda_zero_uses_the_run_mean():
    states = [_fake_lora_state(s) for s in range(3)]
    base = states[0]["backbone.blk.attn.qkv.base.weight"]
    for st in states:
        st["backbone.blk.attn.qkv.base.weight"] = base.clone()
    mean_raw = mean_raw_delta(states)
    got = merge_lora_souped(states[0], 1.5, 0.0, mean_raw)
    other = merge_lora_souped(states[1], 1.5, 0.0, mean_raw)
    expected = base + 1.5 * mean_raw["backbone.blk.attn.qkv"]
    assert torch.allclose(got["backbone.blk.attn.qkv.weight"], expected, atol=1e-6)
    assert torch.allclose(got["backbone.blk.attn.qkv.weight"], other["backbone.blk.attn.qkv.weight"], atol=1e-6)


def test_soup_refuses_a_mismatched_base():
    with pytest.raises(ValueError, match="different base weight"):
        mean_raw_delta([_fake_lora_state(s) for s in range(2)])


def _timed_ensemble(costs, deadline, monkeypatch, n_images=8):
    specs = [sub.MemberSpec(path=Path(f"/nonexistent/m{i}"), backbone=bb, img_size=64,
                            family=["a", "b", "c"][i % 3]) for i, (bb, _) in enumerate(costs)]
    ens = sub.RareEnsemble(specs, torch.device("cpu"))
    clock = {"t": 0.0}
    monkeypatch.setattr(sub.time, "time", lambda: clock["t"])
    ens._start_time = 0.0
    price = {spec.path: seconds for spec, (_, seconds) in zip(specs, costs)}
    current = {}

    def fake_load(self, spec):
        current["path"] = spec.path
        return object()

    def fake_logits(self, model, images, batch_size):
        clock["t"] += price[current["path"]]
        return np.zeros(len(images), dtype=np.float64)

    monkeypatch.setattr(sub.RareEnsemble, "_load", fake_load)
    monkeypatch.setattr(sub.RareEnsemble, "_member_logits", fake_logits)
    scores = ens.predict(np.zeros((n_images, 8, 8, 3), dtype=np.uint8), batch_size=4, deadline_s=deadline)
    return scores, clock["t"]


def test_deadline_uses_the_budget(monkeypatch):
    scores, elapsed = _timed_ensemble([("rn", 30.0)] * 25, 1100.0, monkeypatch)
    assert 715.0 < elapsed <= 1100.0
    assert scores.shape == (8,)


def test_deadline_is_never_overrun(monkeypatch):
    costs = [("vit", 120.0) if i % 3 == 0 else ("rn", 40.0) for i in range(25)]
    _, elapsed = _timed_ensemble(costs, 1100.0, monkeypatch)
    assert elapsed <= 1100.0


def test_at_least_one_member_runs(monkeypatch):
    scores, _ = _timed_ensemble([("rn", 5000.0)] * 25, 1100.0, monkeypatch)
    assert np.isfinite(scores).all()


def test_no_deadline_runs_everything(monkeypatch):
    _, elapsed = _timed_ensemble([("rn", 30.0)] * 25, None, monkeypatch)
    assert elapsed == pytest.approx(750.0)
