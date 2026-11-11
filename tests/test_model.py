import importlib.util
import pathlib
import subprocess
import sys

import pytest
import torch

from laerc.model import ReservoirBlock, ReservoirFFNLanguageModel, build_optimizer

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _original_model():
    try:
        first = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.split()[0]
        code = subprocess.run(["git", "show", f"{first}:laerc/model.py"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None
    path = pathlib.Path(__file__).parent / "_orig_model_tmp.py"
    path.write_text(code)
    try:
        spec = importlib.util.spec_from_file_location("_orig_model_tmp", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        path.unlink()
    return mod


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.99])
def test_true_spectral_radius(rho):
    torch.manual_seed(0)
    W = ReservoirBlock(32, reservoir_radius=rho).rnn.weight_hh_l0
    assert torch.linalg.eigvals(W.double()).abs().max().item() == pytest.approx(rho, rel=1e-4)


def test_legacy_norm_scaling_matches_original_model():
    orig = _original_model()
    if orig is None:
        pytest.skip("needs git history")
    torch.manual_seed(0)
    a = orig.ReservoirFFNLanguageModel(500, 32, 3)
    torch.manual_seed(0)
    b = ReservoirFFNLanguageModel(500, 32, 3, res_scale="norm")
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    with torch.no_grad():
        for p in a.parameters():
            if p.requires_grad:
                p.normal_(std=0.05)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 500, (2, 17))
    torch.testing.assert_close(a(x)[0], b(x)[0])


@pytest.mark.parametrize("use_reservoir", [True, False])
def test_carried_state_equals_full_sequence(use_reservoir):
    torch.manual_seed(0)
    m = ReservoirFFNLanguageModel(300, 32, 2, use_reservoir=use_reservoir).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.normal_(std=0.05)
    x = torch.randint(0, 300, (3, 20))
    full, _ = m(x)
    parts, state = [], None
    for chunk in (x[:, :7], x[:, 7:8], x[:, 8:]):
        logits, _, state = m(chunk, state=state, return_state=True)
        parts.append(logits)
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-5, rtol=1e-5)


def test_beta2_is_used():
    m = ReservoirFFNLanguageModel(100, 16, 1)
    opt = build_optimizer(m, lr=1e-3, beta2=0.95)
    assert all(g["betas"] == (0.9, 0.95) for g in opt.param_groups)


def test_reservoir_is_frozen_and_tied():
    m = ReservoirFFNLanguageModel(100, 16, 2)
    assert not any(p.requires_grad for n, p in m.named_parameters() if ".rnn." in n)
    assert m.out.weight is m.embed.weight
