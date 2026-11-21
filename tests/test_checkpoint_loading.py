"""
A missing/renamed checkpoint key fails silently in load_pretrained (it just
drops any key that isn't in model_dict or has the wrong shape - see
scGPT-spatial/scgpt_spatial/tasks/cell_emb.py load_pretrained). This test
catches that: it loads the real checkpoint and asserts zero keys are missing,
so a bad future patch to the model or a checkpoint swap gets caught instead
of silently producing a model that's partially randomly-initialized.

Needs a real checkpoint on disk, so it's skipped (not failed) if config.yaml
isn't set up yet - see README setup steps 1-3.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config  # noqa: E402


def _get_config():
    try:
        return load_config(os.environ.get("SCGPT_CONFIG", "config.yaml"))
    except (FileNotFoundError, ValueError):
        return None


def test_checkpoint_loads_with_no_missing_params(capsys):
    cfg = _get_config()
    if cfg is None:
        pytest.skip("config.yaml not set up (see README setup) - skipping checkpoint test")

    sys.path.insert(0, cfg["repo"])
    import gene_embed

    gene_embed.load_model_and_vocab(cfg["checkpoint"])
    out = capsys.readouterr().out
    assert "missing from checkpoint: 0" in out, (
        "checkpoint failed to fully load into the model - "
        "a weight-loading mismatch will silently produce partially "
        "random-initialized embeddings. Full output:\n" + out
    )
