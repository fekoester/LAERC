import pytest
import torch

from laerc import TrainConfig
from laerc.config import make_run_name
from laerc.model import ReservoirBlock, ReservoirFFNLanguageModel, build_model


def test_default_growth_leaves_model_unchanged():
    torch.manual_seed(0)
    a = ReservoirFFNLanguageModel(50, 16, 3, res_leak_min=1e-2)
    torch.manual_seed(0)
    b = ReservoirFFNLanguageModel(50, 16, 3, res_leak_min=1e-2, res_depth_growth=1.0)
    for x, y in zip(a.blocks, b.blocks):
        assert torch.equal(x.leak, y.leak)
    # the plain reservoir keeps the fused nn.RNN path
    assert not any(blk.leaky for blk in ReservoirFFNLanguageModel(50, 16, 3).blocks)


@pytest.mark.parametrize("leak_min", [1.0, 0.1])
def test_geometric_schedule(leak_min):
    m = ReservoirFFNLanguageModel(50, 16, 4, res_leak_min=leak_min, res_depth_growth=4.0)
    taus = m.reservoir_timescales()
    for k, (lo, hi) in enumerate(taus):
        assert lo == pytest.approx(4.0**k, rel=1e-5)
        assert hi == pytest.approx(4.0**k / leak_min, rel=1e-5)
    # with leak_min = 1 the first block stays the plain (fused) reservoir, the deeper ones become leaky
    assert [blk.leaky for blk in m.blocks] == [leak_min < 1.0, True, True, True]


def test_chunked_equals_full_with_growth():
    torch.manual_seed(0)
    m = ReservoirFFNLanguageModel(100, 16, 3, res_leak_min=0.1, res_depth_growth=8.0).eval()
    idx = torch.randint(0, 100, (2, 20))
    full, _ = m(idx)
    a, _, st = m(idx[:, :7], return_state=True)
    b, _, _ = m(idx[:, 7:], state=st, return_state=True)
    assert torch.allclose(full, torch.cat([a, b], dim=1), atol=1e-5)


def _memory(blk, T=200):
    """How much of token 0 is still in the reservoir state after T tokens."""
    torch.manual_seed(1)
    x = torch.randn(1, T, blk.D)
    x2 = x.clone()
    x2[:, 0] += 1.0
    with torch.no_grad():
        run = (lambda z: blk._leaky_scan(z, None)[1]) if blk.leaky else (lambda z: blk.rnn(z, None)[1])
        return (run(x) - run(x2)).norm().item()


def test_slower_leak_remembers_longer():
    torch.manual_seed(0)
    fast = ReservoirBlock(16)
    torch.manual_seed(0)
    slow = ReservoirBlock(16, res_leak_scale=1 / 64)
    assert _memory(slow) > 1e3 * _memory(fast)


def test_deep_layers_hold_older_context():
    torch.manual_seed(0)
    m = ReservoirFFNLanguageModel(100, 16, 4, res_depth_growth=4.0).eval()
    idx = torch.randint(0, 100, (1, 150))
    idx2 = idx.clone()
    idx2[0, 0] = (idx[0, 0] + 1) % 100
    with torch.no_grad():
        _, _, s1 = m(idx, return_state=True)
        _, _, s2 = m(idx2, return_state=True)
    mem = [(a - b).norm().item() for a, b in zip(s1, s2)]
    assert mem[-1] > 100 * mem[0]


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        ReservoirFFNLanguageModel(50, 16, 2, res_depth_growth=0.5)
    with pytest.raises(ValueError):
        ReservoirBlock(16, res_leak_scale=0.0)


def test_config_and_old_checkpoints():
    cfg = TrainConfig(vocab=50, emb=16, layers=3, res_leak_min=0.1, res_depth_growth=2.0)
    m = build_model(cfg)
    assert m.reservoir_timescales()[2][0] == pytest.approx(4.0)
    assert make_run_name(cfg).endswith("_leak0.1_G2")
    # config dicts saved before this option existed still load, with growth 1
    old = {k: v for k, v in cfg.as_dict().items() if k != "res_depth_growth"}
    assert build_model(old).reservoir_timescales()[2][0] == pytest.approx(1.0)
    assert make_run_name(TrainConfig()) == "laerc_v50257_seq1024_d512_L16_R4_RMLP1_FF4"
