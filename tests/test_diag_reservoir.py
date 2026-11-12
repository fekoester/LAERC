import math

import pytest
import torch

from laerc import TrainConfig
from laerc.config import make_run_name
from laerc.model import DiagReservoir, ReservoirFFNLanguageModel, build_model, diag_scan


def _loop(u, log_lam, h0):
    lam, h, out = torch.exp(log_lam), h0, []
    for t in range(u.shape[1]):
        h = lam * h + u[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("T,chunk", [(1, 32), (31, 8), (64, 32), (100, 32)])
def test_diag_scan_matches_sequential_recurrence(T, chunk):
    torch.manual_seed(0)
    u = torch.randn(3, T, 5, dtype=torch.cfloat)
    log_lam = torch.complex(-torch.rand(5) * 0.5 - 1e-3, torch.rand(5) * math.pi)
    h0 = torch.randn(3, 5, dtype=torch.cfloat)
    h, h_T = diag_scan(u, log_lam, h0, chunk=chunk)
    ref = _loop(u, log_lam, h0)
    assert torch.allclose(h, ref, atol=1e-5) and torch.allclose(h_T, ref[:, -1], atol=1e-5)


def test_diag_scan_gradients():
    torch.manual_seed(0)
    u = torch.randn(2, 20, 3, dtype=torch.cdouble, requires_grad=True)
    log_lam = torch.complex(-torch.rand(3, dtype=torch.double) - 0.1, torch.rand(3, dtype=torch.double))
    assert torch.autograd.gradcheck(lambda z: diag_scan(z, log_lam, chunk=8)[0], (u,))


def test_diag_reservoir_is_a_linear_reservoir():
    """Features are Re/Im of h_t = lam h_{t-1} + gamma B x_t, and the modes have unit stationary variance."""
    torch.manual_seed(0)
    res = DiagReservoir(8, 64, tau_min=1.0, tau_max=100.0)
    x = torch.randn(2, 30, 8)
    feats, h_T = res(x)
    p = x @ res.B_in
    u = torch.complex(p[..., : res.N], p[..., res.N:])
    ref = _loop(u, torch.complex(res.log_mag, res.theta), torch.zeros(2, res.N, dtype=torch.cfloat))
    assert torch.allclose(feats, torch.cat([ref.real, ref.imag], -1), atol=1e-5)
    lo, hi = res.timescales
    assert lo == pytest.approx(1.0, rel=1e-4) and hi == pytest.approx(100.0, rel=1e-3)
    # stationary variance: Var Re h + Var Im h = (|B_re|^2 + |B_im|^2) / (1 - |lam|^2) for unit-variance x, and the
    # gamma = sqrt(1 - |lam|^2) in B_in makes that independent of the timescale (about 1 for every mode)
    long = res(torch.randn(64, 3000, 8))[0][:, 2000:]
    var = long.var(dim=(0, 1))
    var = var[: res.N] + var[res.N:]
    col = res.B_in.pow(2).sum(0)
    expected = (col[: res.N] + col[res.N:]) / (-torch.expm1(2 * res.log_mag))
    fast = slice(0, res.N // 2)  # the fast half has converged after 2000 steps
    assert torch.allclose(var[fast], expected[fast], rtol=0.15)
    assert 0.3 < expected.mean() < 3

def test_diag_model_chunked_equals_full_and_is_frozen():
    torch.manual_seed(0)
    m = ReservoirFFNLanguageModel(100, 16, 3, res_mode="diag", res_leak_min=0.1, res_depth_growth=10.0).eval()
    assert not any(p.requires_grad for blk in m.blocks for p in blk.diag.buffers())
    idx = torch.randint(0, 100, (2, 45))
    full, _ = m(idx)
    a, _, st = m(idx[:, :13], return_state=True)
    b, _, st = m(idx[:, 13:30], state=st, return_state=True)
    c, _, _ = m(idx[:, 30:], state=st, return_state=True)
    assert torch.allclose(full, torch.cat([a, b, c], dim=1), atol=1e-5)
    taus = m.reservoir_timescales()
    assert taus[2][1] == pytest.approx(10 * taus[1][1], rel=1e-3)


def test_diag_trains_input_side():
    torch.manual_seed(0)
    m = ReservoirFFNLanguageModel(50, 16, 2, res_mode="diag")
    idx = torch.randint(0, 50, (2, 12))
    _, loss = m(idx, idx)
    loss.backward()
    assert m.embed.weight.grad is not None and m.blocks[0].res_mlp[1].weight.grad.abs().sum() > 0


def test_diag_timescale_band():
    """Fastest mode at 1 token, slowest at tau_rho / leak_min, both shifted by growth**k."""
    m = ReservoirFFNLanguageModel(50, 16, 3, res_mode="diag", reservoir_radius=0.9, res_leak_min=0.01,
                                  res_depth_growth=4.0)
    tau_rho = 1 / math.log(1 / 0.9)
    for k, (lo, hi) in enumerate(m.reservoir_timescales()):
        assert lo == pytest.approx(4.0**k, rel=1e-4)
        assert hi == pytest.approx(4.0**k * tau_rho / 0.01, rel=1e-4)


def test_diag_config():
    cfg = TrainConfig(vocab=50, emb=16, layers=2, res_mode="diag")
    assert build_model(cfg).blocks[0].diag is not None
    assert "_diag" in make_run_name(cfg)
    with pytest.raises(ValueError):
        ReservoirFFNLanguageModel(50, 16, 2, res_mode="linear")
    with pytest.raises(ValueError):
        ReservoirFFNLanguageModel(50, 16, 2, res_mode="diag", reservoir_radius=1.0)
