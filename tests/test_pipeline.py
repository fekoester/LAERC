import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from laerc import TrainConfig, finalize_config
from laerc.data import eval_windows, get_batch, load_meta
from laerc.train import train
from laerc.utils import load_checkpoint

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEXT = "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n" * 60


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    pytest.importorskip("tiktoken")
    d = tmp_path_factory.mktemp("data")
    raw = d / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text(TEXT)
    (raw / "b.txt").write_text(TEXT.upper())
    out = d / "tok"
    try:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prepare_gpt2_tokens.py"), "--raw_dir",
                        str(raw), "--out_dir", str(out), "--val_fraction", "0.2", "--compact_vocab"], check=True)
    except subprocess.CalledProcessError:
        pytest.skip("GPT-2 encoding not available (offline?)")
    return str(out)


def test_prepare_writes_compact_vocab_and_eot(data_dir):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    tr = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    va = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
    assert meta["vocab_size"] == len(meta["id_map"]) < 200
    assert max(tr.max(), va.max()) < meta["vocab_size"]
    assert meta["id_map"][meta["eot_id"]] == 50256
    assert (np.concatenate([tr, va]) == meta["eot_id"]).sum() == 2  # one per document
    assert len(tr) + len(va) == meta["n_train"] + meta["n_val"]


def test_batches():
    toks = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(toks, 16, 4, torch.device("cpu"), rng=np.random.default_rng(0))
    assert x.shape == (4, 15) and torch.equal(x[:, 1:], y[:, :-1])
    w1 = [b[0] for b in eval_windows(toks, 16, 4, 3)]
    w2 = [b[0] for b in eval_windows(toks, 16, 4, 3)]
    assert all(torch.equal(a, b) for a, b in zip(w1, w2))


def test_train_resume_and_sample(data_dir, tmp_path):
    def cfg(steps, **kw):
        c = TrainConfig(data_dir=data_dir, seq=16, emb=16, layers=2, batch=4, accum=1, lr=3e-3, epochs=1,
                        steps_per_epoch=steps, eval_interval_steps=3, eval_batches=2, log_interval_steps=2,
                        ckpt_dir=str(tmp_path / "ck"), flush_every=1, **kw)
        return finalize_config(c)

    out = train(cfg(6))
    assert out["steps"] == 6 and np.isfinite(out["best_val_loss"])
    assert os.path.exists(tmp_path / "ck" / "best.pt") and os.path.exists(tmp_path / "ck" / "model_step6.pt")
    ck = load_checkpoint(str(tmp_path / "ck" / "model_step6.pt"))
    assert ck["config"]["vocab"] == ck["meta"]["vocab_size"]

    out2 = train(cfg(9, resume=str(tmp_path / "ck" / "model_step6.pt")))
    assert out2["steps"] == 9

    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "sample.py"), "--ckpt_dir",
                          str(tmp_path / "ck"), "--prompt", "To be", "--max_tokens", "10", "--seed", "0",
                          "--top_p", "0.9"], capture_output=True, text=True, check=True)
    assert res.stdout.startswith("To be")


def test_train_and_sample_diag_reservoir(data_dir, tmp_path):
    c = finalize_config(TrainConfig(data_dir=data_dir, seq=16, emb=16, layers=2, batch=4, accum=2, lr=3e-3,
                                    epochs=1, steps_per_epoch=4, eval_interval_steps=2, eval_batches=2,
                                    log_interval_steps=2, ckpt_dir=str(tmp_path / "ck"), flush_every=1,
                                    res_mode="diag", res_leak_min=0.1, res_depth_growth=4.0))
    out = train(c)
    assert out["steps"] == 4 and np.isfinite(out["best_val_loss"])
    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "sample.py"), "--ckpt_dir",
                          str(tmp_path / "ck"), "--prompt", "To be", "--max_tokens", "10", "--seed", "0"],
                         capture_output=True, text=True, check=True)
    assert res.stdout.startswith("To be")



def test_train_accepts_non_laerc_model(data_dir, tmp_path):
    """The benchmark passes its GPT baseline to train(); it has no reservoir_timescales()."""
    from benchmarks.gpt_baseline import GPT

    c = finalize_config(TrainConfig(data_dir=data_dir, seq=16, emb=16, layers=1, batch=4, accum=1, lr=3e-3,
                                    epochs=1, steps_per_epoch=2, eval_interval_steps=2, eval_batches=1,
                                    log_interval_steps=1, ckpt_dir=str(tmp_path / "ck"), flush_every=1))
    vocab = load_meta(data_dir)["vocab_size"]
    assert train(c, model=GPT(vocab, 16, 1, heads=2, max_len=16))["steps"] == 2
